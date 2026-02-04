"""
MeanFlow-MMDiT for Super-Resolution (Unified: Pixel + Latent)

Modes:
    - pixel: Direct pixel space, no VAE limitation
    - latent: SD3/FLUX VAE latent space

Usage:
    # Pixel mode
    python train_sr_mmdit.py --mode pixel \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --scale 4 --model_size small

    # Latent mode
    python train_sr_mmdit.py --mode latent --vae_type flux \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X8 \
        --scale 8 --model_size small
"""

import os, math, random, argparse
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
        self.hr_dir, self.lr_dir = hr_dir, lr_dir
        self.patch_size, self.scale = patch_size, scale
        self.augment, self.repeat = augment, repeat
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(self.hr_files) == len(self.lr_files)
        print(f"[Dataset] {len(self.hr_files)} pairs (repeat={repeat})")
    
    def __len__(self):
        return len(self.hr_files) * self.repeat
    
    def __getitem__(self, idx):
        i = idx % len(self.hr_files)
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[i])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[i])).convert('RGB')
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        hr_img, lr_up = self._crop(hr_img, lr_up)
        
        if self.augment:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_up = lr_up.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
                lr_up = lr_up.transpose(Image.FLIP_TOP_BOTTOM)
        
        hr = torch.from_numpy(np.array(hr_img)).float().permute(2,0,1) / 127.5 - 1.0
        lr = torch.from_numpy(np.array(lr_up)).float().permute(2,0,1) / 127.5 - 1.0
        return {'hr': hr, 'lr': lr}
    
    def _crop(self, hr, lr):
        w, h = hr.size
        ps = self.patch_size
        if w < ps or h < ps:
            s = max(ps/w, ps/h) * 1.1
            hr = hr.resize((int(w*s), int(h*s)), Image.BICUBIC)
            lr = lr.resize((int(w*s), int(h*s)), Image.BICUBIC)
            w, h = hr.size
        x, y = random.randint(0, w-ps), random.randint(0, h-ps)
        return hr.crop((x,y,x+ps,y+ps)), lr.crop((x,y,x+ps,y+ps))


# ============================================================================
# VAE Wrapper
# ============================================================================

class VAEWrapper:
    def __init__(self, vae_type='flux', device='cuda'):
        self.vae_type, self.device = vae_type, device
        self.scale_factor, self.latent_channels = 8, 16
        from diffusers import AutoencoderKL
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers" if vae_type == 'sd3' else "black-forest-labs/FLUX.1-dev"
        print(f"Loading {vae_type.upper()} VAE...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        for p in self.vae.parameters(): p.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, z):
        return self.vae.decode(z / self.vae.config.scaling_factor).sample


# ============================================================================
# MMDiT Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MMDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm1_img, self.norm2_img = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.norm1_cond, self.norm2_cond = RMSNorm(hidden_size), RMSNorm(hidden_size)
        
        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3)
        self.qkv_cond = nn.Linear(hidden_size, hidden_size * 3)
        self.proj_img = nn.Linear(hidden_size, hidden_size)
        self.proj_cond = nn.Linear(hidden_size, hidden_size)
        
        mlp_h = int(hidden_size * mlp_ratio)
        self.mlp_img = nn.Sequential(nn.Linear(hidden_size, mlp_h), nn.GELU(), nn.Linear(mlp_h, hidden_size))
        self.mlp_cond = nn.Sequential(nn.Linear(hidden_size, mlp_h), nn.GELU(), nn.Linear(mlp_h, hidden_size))
        
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        nn.init.zeros_(self.adaLN[-1].weight); nn.init.zeros_(self.adaLN[-1].bias)
    
    def forward(self, x_img, x_cond, c):
        B = x_img.shape[0]
        shift_i, scale_i, gate_i, shift_c, scale_c, gate_c = self.adaLN(c).chunk(6, dim=-1)
        
        xi = self.norm1_img(x_img) * (1 + scale_i.unsqueeze(1)) + shift_i.unsqueeze(1)
        xc = self.norm1_cond(x_cond) * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)
        
        qkv_i = self.qkv_img(xi).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_c = self.qkv_cond(xc).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_i, k_i, v_i = qkv_i[0], qkv_i[1], qkv_i[2]
        q_c, k_c, v_c = qkv_c[0], qkv_c[1], qkv_c[2]
        
        k_j, v_j = torch.cat([k_i, k_c], dim=2), torch.cat([v_i, v_c], dim=2)
        attn_i = F.scaled_dot_product_attention(q_i, k_j, v_j).transpose(1,2).reshape(B, -1, self.hidden_size)
        attn_c = F.scaled_dot_product_attention(q_c, k_j, v_j).transpose(1,2).reshape(B, -1, self.hidden_size)
        
        x_img = x_img + gate_i.unsqueeze(1) * self.proj_img(attn_i)
        x_cond = x_cond + gate_c.unsqueeze(1) * self.proj_cond(attn_c)
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
        emb = torch.cat([torch.cos(t[:,None] * freqs), torch.sin(t[:,None] * freqs)], dim=-1)
        return self.mlp(emb)


class MMDiT(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_channels=3, hidden_size=512, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_size, self.in_channels, self.hidden_size = patch_size, in_channels, hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        
        self.patch_embed_img = nn.Linear(patch_dim, hidden_size)
        self.patch_embed_cond = nn.Linear(patch_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([MMDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        
        self.norm_out = RMSNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, patch_dim)
        nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)
    
    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        return x.reshape(B, C, H//p, p, W//p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*p*p)
    
    def unpatchify(self, x, h, w):
        B, N, _ = x.shape
        p = self.patch_size
        return x.reshape(B, h//p, w//p, self.in_channels, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, self.in_channels, h, w)
    
    def forward(self, z_t, z_cond, t, h):
        B, C, H, W = z_t.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        pos = self._interp_pos(H, W) if num_patches != self.num_patches else self.pos_embed
        
        x_img = self.patch_embed_img(self.patchify(z_t)) + pos
        x_cond = self.patch_embed_cond(self.patchify(z_cond)) + pos
        c = self.t_embedder(t) + self.h_embedder(h)
        
        for blk in self.blocks:
            x_img, x_cond = blk(x_img, x_cond, c)
        return self.unpatchify(self.proj_out(self.norm_out(x_img)), H, W)
    
    def _interp_pos(self, h, w):
        old = int(self.num_patches ** 0.5)
        nh, nw = h // self.patch_size, w // self.patch_size
        pos = self.pos_embed.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(nh, nw), mode='bilinear', align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, nh * nw, -1)


# ============================================================================
# Model Configs
# ============================================================================

CONFIGS = {
    'xs': (256, 8, 4), 'small': (384, 12, 6), 'base': (512, 12, 8), 'large': (768, 24, 12)
}

def create_model(model_size, mode, img_size):
    h, d, n = CONFIGS[model_size]
    ps, ch = (4, 3) if mode == 'pixel' else (2, 16)
    return MMDiT(img_size=img_size, patch_size=ps, in_channels=ch, hidden_size=h, depth=d, num_heads=n)


# ============================================================================
# Trainer
# ============================================================================

class MMDiTTrainer:
    def __init__(self, model, device, mode='pixel', vae=None, lr=1e-4):
        self.model = model.to(device)
        self.device, self.mode, self.vae = device, mode, vae
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
        self.ema_decay = 0.9999
        self.ema = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    def _ema_update(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.ema[n].mul_(self.ema_decay).add_(p.data, alpha=1-self.ema_decay)
    
    def _sample_time(self, bs):
        n0, n1 = int(bs * 0.6), int(bs * 0.2)
        t, h = torch.zeros(bs, device=self.device), torch.zeros(bs, device=self.device)
        t[:n0] = torch.sigmoid(torch.randn(n0, device=self.device) - 0.4)
        if n1 > 0:
            t1 = torch.sigmoid(torch.randn(n1, device=self.device) - 0.4)
            r1 = torch.sigmoid(torch.randn(n1, device=self.device) - 0.4)
            t[n0:n0+n1], h[n0:n0+n1] = torch.maximum(t1, r1), torch.abs(t1 - r1)
        t[n0+n1:], h[n0+n1:] = 1.0, 1.0
        perm = torch.randperm(bs, device=self.device)
        return t[perm], h[perm]
    
    def train_step(self, batch):
        self.model.train()
        hr, lr = batch['hr'].to(self.device), batch['lr'].to(self.device)
        if self.mode == 'latent':
            with torch.no_grad():
                hr, lr = self.vae.encode(hr), self.vae.encode(lr)
        
        t, h = self._sample_time(hr.shape[0])
        z_t = (1 - t[:,None,None,None]) * hr + t[:,None,None,None] * lr
        
        v_pred = self.model(z_t, lr, t, h)
        loss = F.mse_loss(v_pred, lr - hr)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._ema_update()
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def inference(self, lr, num_steps=1, use_ema=True):
        self.model.eval()
        if use_ema:
            orig = {n: p.clone() for n, p in self.model.named_parameters()}
            for n, p in self.model.named_parameters(): p.data.copy_(self.ema[n])
        
        lr_in = self.vae.encode(lr) if self.mode == 'latent' else lr
        x, dt = lr_in.clone(), 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((x.shape[0],), 1.0 - i*dt, device=self.device)
            hv = torch.full((x.shape[0],), dt, device=self.device)
            x = x - dt * self.model(x, lr_in, t, hv)
        
        if self.mode == 'latent': x = self.vae.decode(x)
        if use_ema:
            for n, p in self.model.named_parameters(): p.data.copy_(orig[n])
        return x
    
    @torch.no_grad()
    def validate(self, loader, num_steps=1):
        psnrs = []
        for batch in loader:
            hr, lr = batch['hr'].to(self.device), batch['lr'].to(self.device)
            pred = self.inference(lr, num_steps).clamp(-1, 1)
            mse = (((hr+1)/2 - (pred+1)/2)**2).mean(dim=[1,2,3])
            psnrs.extend((10 * torch.log10(1/(mse+1e-8))).cpu().tolist())
        return np.mean(psnrs)
    
    def save(self, path, epoch, loss, p1=0, p10=0):
        torch.save({'epoch': epoch, 'model': self.model.state_dict(), 'ema': self.ema,
                    'optimizer': self.optimizer.state_dict(), 'mode': self.mode,
                    'psnr_1step': p1, 'psnr_10step': p10}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt: self.ema = ckpt['ema']
        if 'optimizer' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='pixel', choices=['pixel', 'latent'])
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--vae_type', type=str, default='flux', choices=['sd3', 'flux'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_base', type=str, default='./checkpoints_mmdit')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join(args.save_base, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.mode}_{args.model_size}_x{args.scale}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print(f"MMDiT | {args.mode} | {args.model_size} | x{args.scale}")
    print("="*60)
    
    vae = VAEWrapper(args.vae_type, device) if args.mode == 'latent' else None
    img_size = args.patch_size // 8 if args.mode == 'latent' else args.patch_size
    
    train_loader = DataLoader(SRDataset(args.hr_dir, args.lr_dir, args.patch_size, args.scale),
                              args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(SRDataset(args.val_hr_dir, args.val_lr_dir, args.patch_size, args.scale, False, 1),
                            4, shuffle=False, num_workers=2) if args.val_hr_dir else None
    
    model = create_model(args.model_size, args.mode, img_size)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}, img_size={img_size}")
    
    trainer = MMDiTTrainer(model, device, args.mode, vae, args.lr)
    if args.resume: trainer.load(args.resume)
    
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    best = 0
    
    for ep in range(args.epochs):
        losses = [trainer.train_step(b)['loss'] for b in tqdm(train_loader, desc=f"Epoch {ep+1}")]
        p1 = p10 = 0
        if val_loader:
            p1, p10 = trainer.validate(val_loader, 1), trainer.validate(val_loader, 10)
            print(f"Epoch {ep+1}: loss={np.mean(losses):.4f}, 1-step={p1:.2f}dB, 10-step={p10:.2f}dB")
        scheduler.step()
        if p1 > best:
            best = p1
            trainer.save(os.path.join(save_dir, 'best.pt'), ep, np.mean(losses), p1, p10)
    
    trainer.save(os.path.join(save_dir, 'final.pt'), args.epochs-1, np.mean(losses), p1, p10)
    print(f"Done! Best: {best:.2f} dB | {save_dir}")


if __name__ == '__main__':
    main()