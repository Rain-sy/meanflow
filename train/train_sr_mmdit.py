"""
MeanFlow-MMDiT: Dual-Stream with Joint Attention

Key difference from DiT:
    - Two streams: img (z_t) and cond (LR)
    - Joint attention: each stream attends to both
    - model(z_t, lr, t, h) instead of model(z_t, t, h)

Time sampling strategy (same as DiT V3):
    - 60%: h = 0, t ~ logit-normal (instantaneous velocity)
    - 20%: h > 0, t ~ logit-normal (average velocity)
    - 20%: t = 1, h = 1 (for one-step inference) ← CRITICAL

Usage:
    python meanflow/train_sr_mmdit.py \
        --hr_dir meanflow/Data/DIV2K/DIV2K_train_HR \
        --lr_dir meanflow/Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir meanflow/Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir meanflow/Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --epochs 100 \
        --batch_size 8 \
        --model_size small \
        --scale 4 \
        --patch_size 128 \
        --device cuda:0
"""

import os
import math
import random
import argparse

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
    def __init__(self, hr_dir, lr_dir, patch_size=128, scale=2, augment=True, repeat=5):
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
        
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR ({len(self.hr_files)}) and LR ({len(self.lr_files)}) count mismatch"
        
        print(f"[Dataset] Found {len(self.hr_files)} image pairs (repeat={repeat})")
    
    def __len__(self):
        return len(self.hr_files) * self.repeat
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.hr_files)
        
        hr_path = os.path.join(self.hr_dir, self.hr_files[real_idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[real_idx])
        
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        hr_img, lr_up = self._random_crop(hr_img, lr_up)
        
        if self.augment:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_up = lr_up.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
                lr_up = lr_up.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                hr_img = hr_img.rotate(90 * k)
                lr_up = lr_up.rotate(90 * k)
        
        hr_tensor = torch.from_numpy(np.array(hr_img)).float() / 127.5 - 1.0
        lr_tensor = torch.from_numpy(np.array(lr_up)).float() / 127.5 - 1.0
        
        return {'hr': hr_tensor.permute(2, 0, 1), 'lr': lr_tensor.permute(2, 0, 1)}
    
    def _random_crop(self, hr_img, lr_up):
        w, h = hr_img.size
        
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            new_size = (int(w * scale), int(h * scale))
            hr_img = hr_img.resize(new_size, Image.BICUBIC)
            lr_up = lr_up.resize(new_size, Image.BICUBIC)
            w, h = new_size
        
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        
        return (hr_img.crop((x, y, x + self.patch_size, y + self.patch_size)),
                lr_up.crop((x, y, x + self.patch_size, y + self.patch_size)))


# ============================================================================
# Model Components
# ============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MMDiTBlock(nn.Module):
    """MMDiT Block: Dual-stream with joint attention"""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Norms
        self.norm1_img = RMSNorm(hidden_size)
        self.norm1_cond = RMSNorm(hidden_size)
        self.norm2_img = RMSNorm(hidden_size)
        self.norm2_cond = RMSNorm(hidden_size)
        
        # QKV for both streams
        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3)
        self.qkv_cond = nn.Linear(hidden_size, hidden_size * 3)
        self.proj_img = nn.Linear(hidden_size, hidden_size)
        self.proj_cond = nn.Linear(hidden_size, hidden_size)
        
        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_img = Mlp(hidden_size, mlp_hidden)
        self.mlp_cond = Mlp(hidden_size, mlp_hidden)
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x_img, x_cond, c):
        B = x_img.shape[0]
        
        # AdaLN params (shared for both streams)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Modulate
        xi = self.norm1_img(x_img) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        xc = self.norm1_cond(x_cond) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # QKV
        qkv_i = self.qkv_img(xi).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_c = self.qkv_cond(xc).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_i, k_i, v_i = qkv_i.unbind(0)
        q_c, k_c, v_c = qkv_c.unbind(0)
        
        # Joint attention
        k_joint = torch.cat([k_i, k_c], dim=2)
        v_joint = torch.cat([v_i, v_c], dim=2)
        
        attn_i = F.scaled_dot_product_attention(q_i, k_joint, v_joint).transpose(1, 2).reshape(B, -1, self.hidden_size)
        attn_c = F.scaled_dot_product_attention(q_c, k_joint, v_joint).transpose(1, 2).reshape(B, -1, self.hidden_size)
        
        # Residual
        x_img = x_img + gate_msa.unsqueeze(1) * self.proj_img(attn_i)
        x_cond = x_cond + gate_msa.unsqueeze(1) * self.proj_cond(attn_c)
        
        # MLP
        x_img = x_img + gate_mlp.unsqueeze(1) * self.mlp_img(
            self.norm2_img(x_img) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        x_cond = x_cond + gate_mlp.unsqueeze(1) * self.mlp_cond(
            self.norm2_cond(x_cond) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        
        return x_img, x_cond


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class MeanFlowMMDiT(nn.Module):
    """MMDiT for Super-Resolution with dual-stream joint attention"""
    
    def __init__(self, img_size=128, patch_size=4, in_channels=3, hidden_size=512, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Separate embeddings for each stream
        self.patch_embed_img = PatchEmbed(patch_size, in_channels, hidden_size)
        self.patch_embed_cond = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        self.blocks = nn.ModuleList([MMDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed_img.proj.weight.view([self.hidden_size, -1]))
        nn.init.zeros_(self.patch_embed_img.proj.bias)
        nn.init.xavier_uniform_(self.patch_embed_cond.proj.weight.view([self.hidden_size, -1]))
        nn.init.zeros_(self.patch_embed_cond.proj.bias)
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        return torch.einsum('nhwpqc->nchpwq', x).reshape(x.shape[0], self.in_channels, h * p, w * p)
    
    def forward(self, z_t, lr_cond, t, h):
        """
        Args:
            z_t: interpolated image (B, 3, H, W)
            lr_cond: LR condition (B, 3, H, W)
            t, h: time and interval (B,)
        """
        x_img = self.patch_embed_img(z_t) + self.pos_embed
        x_cond = self.patch_embed_cond(lr_cond) + self.pos_embed
        c = self.t_embedder(t) + self.h_embedder(h)
        
        for block in self.blocks:
            x_img, x_cond = block(x_img, x_cond, c)
        
        return self.unpatchify(self.final_layer(x_img, c))


# Model configs
def MeanFlowMMDiT_XS(img_size=128, **kwargs):
    return MeanFlowMMDiT(img_size, patch_size=4, hidden_size=256, depth=8, num_heads=4, **kwargs)

def MeanFlowMMDiT_S(img_size=128, **kwargs):
    return MeanFlowMMDiT(img_size, patch_size=4, hidden_size=384, depth=12, num_heads=6, **kwargs)

def MeanFlowMMDiT_B(img_size=128, **kwargs):
    return MeanFlowMMDiT(img_size, patch_size=4, hidden_size=512, depth=12, num_heads=8, **kwargs)

def MeanFlowMMDiT_L(img_size=128, **kwargs):
    return MeanFlowMMDiT(img_size, patch_size=4, hidden_size=768, depth=24, num_heads=12, **kwargs)

MODEL_CONFIGS = {'xs': MeanFlowMMDiT_XS, 'small': MeanFlowMMDiT_S, 'base': MeanFlowMMDiT_B, 'large': MeanFlowMMDiT_L}


# ============================================================================
# Trainer
# ============================================================================

class MeanFlowMMDiTTrainer:
    def __init__(self, model, device, lr=1e-4, weight_decay=0.01,
                 P_mean=-0.4, P_std=1.0,
                 prop_h0=0.6, prop_havg=0.2, prop_onestep=0.2):
        self.model = model.to(device)
        self.device = device
        self.P_mean = P_mean
        self.P_std = P_std
        
        assert abs(prop_h0 + prop_havg + prop_onestep - 1.0) < 1e-6
        self.prop_h0 = prop_h0
        self.prop_havg = prop_havg
        self.prop_onestep = prop_onestep
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        self.ema_decay = 0.9999
        self.ema_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    def update_ema(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def sample_time(self, batch_size):
        n_h0 = int(batch_size * self.prop_h0)
        n_havg = int(batch_size * self.prop_havg)
        n_onestep = batch_size - n_h0 - n_havg
        
        t = torch.zeros(batch_size, device=self.device)
        h = torch.zeros(batch_size, device=self.device)
        
        if n_h0 > 0:
            rnd = torch.randn(n_h0, device=self.device)
            t[:n_h0] = torch.sigmoid(rnd * self.P_std + self.P_mean)
        
        if n_havg > 0:
            idx = n_h0
            rnd_t = torch.randn(n_havg, device=self.device)
            rnd_r = torch.randn(n_havg, device=self.device)
            t_tmp = torch.sigmoid(rnd_t * self.P_std + self.P_mean)
            r_tmp = torch.sigmoid(rnd_r * self.P_std + self.P_mean)
            t_tmp, r_tmp = torch.maximum(t_tmp, r_tmp), torch.minimum(t_tmp, r_tmp)
            t[idx:idx+n_havg] = t_tmp
            h[idx:idx+n_havg] = t_tmp - r_tmp
        
        if n_onestep > 0:
            t[n_h0 + n_havg:] = 1.0
            h[n_h0 + n_havg:] = 1.0
        
        perm = torch.randperm(batch_size, device=self.device)
        return t[perm], h[perm]
    
    def train_step(self, batch):
        self.model.train()
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        batch_size = hr.shape[0]
        
        t, h = self.sample_time(batch_size)
        t_exp = t[:, None, None, None]
        
        z_t = (1 - t_exp) * hr + t_exp * lr
        v = lr - hr
        
        u = self.model(z_t, lr, t, h)
        loss = F.mse_loss(u, v)
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.update_ema()
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            't_mean': t.mean().item(),
            'h_mean': h.mean().item(),
        }
    
    @torch.no_grad()
    def inference(self, lr_images, num_steps=1, use_ema=True):
        self.model.eval()
        
        if use_ema:
            orig_params = {name: param.clone() for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                param.data.copy_(self.ema_params[name])
        
        batch_size = lr_images.shape[0]
        x = lr_images.clone()
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), 1.0 - i * dt, device=self.device)
            h = torch.full((batch_size,), dt, device=self.device)
            x = x - dt * self.model(x, lr_images, t, h)
        
        if use_ema:
            for name, param in self.model.named_parameters():
                param.data.copy_(orig_params[name])
        
        return x
    
    @torch.no_grad()
    def validate(self, val_loader, num_steps=1):
        psnr_list = []
        for batch in val_loader:
            hr = batch['hr'].to(self.device)
            lr = batch['lr'].to(self.device)
            hr_pred = self.inference(lr, num_steps, use_ema=True)
            
            hr_01 = (hr + 1) / 2
            hr_pred_01 = ((hr_pred + 1) / 2).clamp(0, 1)
            mse = ((hr_01 - hr_pred_01) ** 2).mean(dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            psnr_list.extend(psnr.cpu().tolist())
        return np.mean(psnr_list)
    
    def save_checkpoint(self, path, epoch, loss, psnr_1step=0, psnr_10step=0):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema': self.ema_params,
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'psnr_1step': psnr_1step,
            'psnr_10step': psnr_10step,
        }, path)
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            self.ema_params = ckpt['ema']
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow-MMDiT')
    
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=2)
    
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--patch_size', type=int, default=128)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    parser.add_argument('--prop_h0', type=float, default=0.6)
    parser.add_argument('--prop_havg', type=float, default=0.2)
    parser.add_argument('--prop_onestep', type=float, default=0.2)
    
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_base', type=str, default='./checkpoints_mmdit')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_every_steps', type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.save_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(
            args.save_base,
            f"{timestamp}_{args.model_size}_x{args.scale}_bs{args.batch_size}_ps{args.patch_size}"
        )
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("MeanFlow-MMDiT (Dual-Stream Joint Attention)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model_size}, Patch: {args.patch_size}")
    print(f"Scale: {args.scale}x")
    print(f"Batch size: {args.batch_size}")
    print(f"Time sampling: {args.prop_h0*100:.0f}% h=0, {args.prop_havg*100:.0f}% h>0, {args.prop_onestep*100:.0f}% t=1,h=1")
    print(f"Save directory: {args.save_dir}")
    print("="*70)
    
    train_dataset = SRDataset(args.hr_dir, args.lr_dir, args.patch_size, args.scale, True, 5)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(args.val_hr_dir, args.val_lr_dir, args.patch_size, args.scale, False, 1)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
        print(f"Validation: {len(val_dataset)} samples")
    
    model = MODEL_CONFIGS[args.model_size](img_size=args.patch_size)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = MeanFlowMMDiTTrainer(model, device, args.lr, args.weight_decay,
                                    prop_h0=args.prop_h0, prop_havg=args.prop_havg, 
                                    prop_onestep=args.prop_onestep)
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_psnr = 0
    global_step = start_epoch * len(train_loader)
    
    for epoch in range(start_epoch, args.epochs):
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            global_step += 1
            
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 't': f"{metrics['t_mean']:.2f}", 'h': f"{metrics['h_mean']:.2f}"})
            
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                trainer.save_checkpoint(os.path.join(args.save_dir, 'checkpoint_latest.pt'), epoch, metrics['loss'])
        
        avg_loss = np.mean(losses)
        
        val_psnr_1step, val_psnr_10step = 0, 0
        if val_loader:
            val_psnr_1step = trainer.validate(val_loader, num_steps=1)
            val_psnr_10step = trainer.validate(val_loader, num_steps=10)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR 1-step={val_psnr_1step:.2f}dB, 10-step={val_psnr_10step:.2f}dB")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}")
        
        scheduler.step()
        
        if val_loader and val_psnr_1step > best_psnr:
            best_psnr = val_psnr_1step
            trainer.save_checkpoint(os.path.join(args.save_dir, 'best_model.pt'), epoch, avg_loss, val_psnr_1step, val_psnr_10step)
            print("  Saved best!")
        
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt'), epoch, avg_loss, val_psnr_1step, val_psnr_10step)
    
    trainer.save_checkpoint(os.path.join(args.save_dir, 'final_model.pt'), args.epochs - 1, avg_loss, val_psnr_1step, val_psnr_10step)
    print(f"\nDone! Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()