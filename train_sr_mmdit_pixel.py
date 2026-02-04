"""
MeanFlow-MMDiT for Super-Resolution (Pixel Space)

Key improvements:
    1. Pixel space - no VAE upper bound limitation
    2. Dual-stream MMDiT with joint attention
    3. Two conditional flow modes supported

Dual-stream design:
    - Stream 1 (img): z_t being refined
    - Stream 2 (cond): LR condition (spatial guidance)
    - Joint attention: each stream attends to both

Conditional flow modes:
    - 'lr_to_hr': z_t = (1-t)*HR + t*LR, v = LR - HR
                  (LR in both interpolation AND condition)
    
    - 'noise_to_hr': z_t = (1-t)*HR + t*noise, v = noise - HR
                     (LR ONLY as condition - cleaner separation)

Usage:
    python meanflow/train_sr_mmdit_pixel.py \
        --hr_dir meanflow/Data/DIV2K/DIV2K_train_HR \
        --lr_dir meanflow/Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --scale 4 --model_size small --cond_mode lr_to_hr  
"""

import os
import math
import random
import argparse
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ============================================================================
# Dataset
# ============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=128, scale=4, augment=True, repeat=5):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.repeat = repeat
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.hr_files) == len(self.lr_files)
        print(f"[Dataset] {len(self.hr_files)} pairs (repeat={repeat})")
    
    def __len__(self):
        return len(self.hr_files) * self.repeat
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.hr_files)
        
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[real_idx])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[real_idx])).convert('RGB')
        
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        hr_img, lr_up = self._random_crop(hr_img, lr_up)
        
        if self.augment:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_up = lr_up.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
                lr_up = lr_up.transpose(Image.FLIP_TOP_BOTTOM)
        
        hr = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 127.5 - 1.0
        lr = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1.0
        
        return {'hr': hr, 'lr': lr}
    
    def _random_crop(self, hr_img, lr_up):
        w, h = hr_img.size
        ps = self.patch_size
        
        if w < ps or h < ps:
            scale = max(ps / w, ps / h) * 1.1
            hr_img = hr_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            lr_up = lr_up.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            w, h = hr_img.size
        
        x, y = random.randint(0, w - ps), random.randint(0, h - ps)
        return hr_img.crop((x, y, x + ps, y + ps)), lr_up.crop((x, y, x + ps, y + ps))


# ============================================================================
# MMDiT Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MMDiTBlock(nn.Module):
    """
    MMDiT Block: Dual-stream with joint attention.
    
    Key difference from single-stream DiT:
        - Two streams with separate weights
        - Joint K,V: each stream attends to BOTH streams
        - Better cross-modal interaction
    """
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Norms
        self.norm1_img = RMSNorm(hidden_size)
        self.norm2_img = RMSNorm(hidden_size)
        self.norm1_cond = RMSNorm(hidden_size)
        self.norm2_cond = RMSNorm(hidden_size)
        
        # QKV
        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3)
        self.qkv_cond = nn.Linear(hidden_size, hidden_size * 3)
        self.proj_img = nn.Linear(hidden_size, hidden_size)
        self.proj_cond = nn.Linear(hidden_size, hidden_size)
        
        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_img = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden_size))
        self.mlp_cond = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden_size))
        
        # AdaLN
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
    
    def forward(self, x_img, x_cond, c):
        B = x_img.shape[0]
        
        # AdaLN params
        shift_i, scale_i, gate_i, shift_c, scale_c, gate_c = self.adaLN(c).chunk(6, dim=-1)
        
        # Norm + modulate
        xi = self.norm1_img(x_img) * (1 + scale_i.unsqueeze(1)) + shift_i.unsqueeze(1)
        xc = self.norm1_cond(x_cond) * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)
        
        # QKV
        qkv_i = self.qkv_img(xi).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_c = self.qkv_cond(xc).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_i, k_i, v_i = qkv_i[0], qkv_i[1], qkv_i[2]
        q_c, k_c, v_c = qkv_c[0], qkv_c[1], qkv_c[2]
        
        # Joint attention
        k_joint = torch.cat([k_i, k_c], dim=2)
        v_joint = torch.cat([v_i, v_c], dim=2)
        
        attn_i = F.scaled_dot_product_attention(q_i, k_joint, v_joint).transpose(1, 2).reshape(B, -1, self.hidden_size)
        attn_c = F.scaled_dot_product_attention(q_c, k_joint, v_joint).transpose(1, 2).reshape(B, -1, self.hidden_size)
        
        # Residual
        x_img = x_img + gate_i.unsqueeze(1) * self.proj_img(attn_i)
        x_cond = x_cond + gate_c.unsqueeze(1) * self.proj_cond(attn_c)
        
        # MLP
        x_img = x_img + gate_i.unsqueeze(1) * self.mlp_img(self.norm2_img(x_img))
        x_cond = x_cond + gate_c.unsqueeze(1) * self.mlp_cond(self.norm2_cond(x_cond))
        
        return x_img, x_cond


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.freq_dim = freq_dim
    
    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = torch.cat([torch.cos(t[:, None] * freqs), torch.sin(t[:, None] * freqs)], dim=-1)
        return self.mlp(emb)


class PixelMMDiT(nn.Module):
    """
    MMDiT for Pixel-Space SR.
    
    Dual-stream:
        - img stream: z_t (being refined)
        - cond stream: LR (spatial guidance)
    """
    
    def __init__(self, img_size=128, patch_size=4, in_channels=3,
                 hidden_size=512, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        
        # Embeddings
        self.patch_embed_img = nn.Linear(patch_dim, hidden_size)
        self.patch_embed_cond = nn.Linear(patch_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Time
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        # Blocks
        self.blocks = nn.ModuleList([MMDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        
        # Output
        self.norm_out = RMSNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, patch_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        return x.reshape(B, C, H//p, p, W//p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*p*p)
    
    def unpatchify(self, x):
        B, N, _ = x.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        return x.reshape(B, h, w, self.in_channels, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, self.in_channels, h*p, w*p)
    
    def forward(self, z_t, lr_cond, t, h):
        x_img = self.patch_embed_img(self.patchify(z_t)) + self.pos_embed
        x_cond = self.patch_embed_cond(self.patchify(lr_cond)) + self.pos_embed
        c = self.t_embedder(t) + self.h_embedder(h)
        
        for block in self.blocks:
            x_img, x_cond = block(x_img, x_cond, c)
        
        return self.unpatchify(self.proj_out(self.norm_out(x_img)))


# ============================================================================
# Model Configs
# ============================================================================

MODEL_CONFIGS = {
    'xs': lambda s: PixelMMDiT(s, hidden_size=256, depth=8, num_heads=4),
    'small': lambda s: PixelMMDiT(s, hidden_size=384, depth=12, num_heads=6),
    'base': lambda s: PixelMMDiT(s, hidden_size=512, depth=12, num_heads=8),
    'large': lambda s: PixelMMDiT(s, hidden_size=768, depth=24, num_heads=12),
}


# ============================================================================
# Trainer
# ============================================================================

class PixelMMDiTTrainer:
    """
    Trainer with two conditional flow modes:
    
    'lr_to_hr': z_t = (1-t)*HR + t*LR
                LR in both interpolation AND condition
    
    'noise_to_hr': z_t = (1-t)*HR + t*noise
                   LR ONLY as condition (cleaner separation)
    """
    
    def __init__(self, model, device, lr=1e-4, cond_mode='lr_to_hr'):
        self.model = model.to(device)
        self.device = device
        self.cond_mode = cond_mode
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
        self.ema_decay = 0.9999
        self.ema_params = {n: p.clone().detach() for n, p in model.named_parameters()}
        
        print(f"[Trainer] Mode: {cond_mode}")
    
    def update_ema(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.ema_params[n].mul_(self.ema_decay).add_(p.data, alpha=1-self.ema_decay)
    
    def sample_time(self, bs):
        # 60% h=0, 20% h>0, 20% t=1,h=1
        n0, n1 = int(bs * 0.6), int(bs * 0.2)
        n2 = bs - n0 - n1
        
        t = torch.zeros(bs, device=self.device)
        h = torch.zeros(bs, device=self.device)
        
        t[:n0] = torch.sigmoid(torch.randn(n0, device=self.device) - 0.4)
        
        if n1 > 0:
            t1 = torch.sigmoid(torch.randn(n1, device=self.device) - 0.4)
            r1 = torch.sigmoid(torch.randn(n1, device=self.device) - 0.4)
            t[n0:n0+n1] = torch.maximum(t1, r1)
            h[n0:n0+n1] = torch.abs(t1 - r1)
        
        t[n0+n1:] = 1.0
        h[n0+n1:] = 1.0
        
        perm = torch.randperm(bs, device=self.device)
        return t[perm], h[perm]
    
    def train_step(self, batch):
        self.model.train()
        hr, lr = batch['hr'].to(self.device), batch['lr'].to(self.device)
        bs = hr.shape[0]
        
        t, h = self.sample_time(bs)
        t_exp = t[:, None, None, None]
        
        if self.cond_mode == 'lr_to_hr':
            z_t = (1 - t_exp) * hr + t_exp * lr
            v_target = lr - hr
        else:  # noise_to_hr
            noise = torch.randn_like(hr)
            z_t = (1 - t_exp) * hr + t_exp * noise
            v_target = noise - hr
        
        v_pred = self.model(z_t, lr, t, h)
        loss = F.mse_loss(v_pred, v_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.update_ema()
        
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def inference(self, lr, num_steps=1, use_ema=True):
        self.model.eval()
        
        if use_ema:
            orig = {n: p.clone() for n, p in self.model.named_parameters()}
            for n, p in self.model.named_parameters():
                p.data.copy_(self.ema_params[n])
        
        bs = lr.shape[0]
        dt = 1.0 / num_steps
        x = lr.clone() if self.cond_mode == 'lr_to_hr' else torch.randn_like(lr)
        
        for i in range(num_steps):
            t = torch.full((bs,), 1.0 - i * dt, device=self.device)
            h = torch.full((bs,), dt, device=self.device)
            x = x - dt * self.model(x, lr, t, h)
        
        if use_ema:
            for n, p in self.model.named_parameters():
                p.data.copy_(orig[n])
        
        return x
    
    @torch.no_grad()
    def validate(self, loader, num_steps=1):
        psnrs = []
        for batch in loader:
            hr, lr = batch['hr'].to(self.device), batch['lr'].to(self.device)
            pred = self.inference(lr, num_steps)
            mse = (((hr + 1) / 2 - ((pred + 1) / 2).clamp(0, 1)) ** 2).mean(dim=[1, 2, 3])
            psnrs.extend((10 * torch.log10(1 / (mse + 1e-8))).cpu().tolist())
        return np.mean(psnrs)
    
    def save(self, path, epoch, loss, psnr1=0, psnr10=0):
        torch.save({'epoch': epoch, 'model': self.model.state_dict(), 'ema': self.ema_params,
                    'optimizer': self.optimizer.state_dict(), 'cond_mode': self.cond_mode,
                    'psnr_1step': psnr1, 'psnr_10step': psnr10}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt: self.ema_params = ckpt['ema']
        if 'optimizer' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--cond_mode', type=str, default='lr_to_hr', choices=['lr_to_hr', 'noise_to_hr'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_base', type=str, default='./checkpoints_mmdit_pixel')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    save_dir = os.path.join(args.save_base, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model_size}_x{args.scale}_{args.cond_mode}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print(f"MMDiT-Pixel | {args.model_size} | {args.scale}x | {args.cond_mode}")
    print("="*60)
    
    train_loader = DataLoader(SRDataset(args.hr_dir, args.lr_dir, args.patch_size, args.scale),
                              args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_loader = DataLoader(SRDataset(args.val_hr_dir, args.val_lr_dir, args.patch_size, args.scale, False, 1),
                                4, shuffle=False, num_workers=2)
    
    model = MODEL_CONFIGS[args.model_size](args.patch_size)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = PixelMMDiTTrainer(model, device, args.lr, args.cond_mode)
    
    if args.resume:
        trainer.load(args.resume)
    
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    best_psnr = 0
    
    for epoch in range(args.epochs):
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            losses.append(trainer.train_step(batch)['loss'])
        
        psnr1 = psnr10 = 0
        if val_loader:
            psnr1 = trainer.validate(val_loader, 1)
            psnr10 = trainer.validate(val_loader, 10)
            print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}, 1-step={psnr1:.2f}dB, 10-step={psnr10:.2f}dB")
        
        scheduler.step()
        
        if psnr1 > best_psnr:
            best_psnr = psnr1
            trainer.save(os.path.join(save_dir, 'best.pt'), epoch, np.mean(losses), psnr1, psnr10)
            print("  Saved best!")
    
    print(f"Done! Best: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()