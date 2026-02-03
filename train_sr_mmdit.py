"""
Latent MeanFlow-MMDiT for Super-Resolution (train_sr_mmdit.py)

Key features:
    - MMDiT architecture (dual-stream like SD3/FLUX)
    - Works in VAE latent space (8x compression)
    - Supports SD/SDXL/SD3/FLUX VAE
    - Auto-save experiment config and best metrics

Requirements:
    pip install diffusers transformers accelerate

Usage:
    python train_sr_mmdit.py \
        --hr_dir meanflow/Data/DIV2K/DIV2K_train_HR \
        --lr_dir meanflow/Data/DIV2K/DIV2K_train_LR_bicubic_X8 \
        --val_hr_dir meanflow/Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir meanflow/Data/DIV2K/DIV2K_valid_LR_bicubic_X8 \
        --vae_type sd \
        --scale 8 \
        --model_size small \
        --epochs 100
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
# VAE Wrapper
# ============================================================================

class VAEWrapper:
    """Wrapper for SD/SDXL/SD3/FLUX VAE"""
    
    def __init__(self, vae_type="sd", device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.vae_type = vae_type
        
        print(f"Loading VAE: {vae_type}...")
        
        if vae_type == "sd3":
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                subfolder="vae",
                torch_dtype=dtype
            ).to(device)
            self.scale_factor = 8
            self.latent_channels = 16
            
        elif vae_type == "flux":
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="vae",
                torch_dtype=dtype
            ).to(device)
            self.scale_factor = 8
            self.latent_channels = 16
            
        elif vae_type == "sdxl":
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae",
                torch_dtype=dtype
            ).to(device)
            self.scale_factor = 8
            self.latent_channels = 4
            
        else:  # sd
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=dtype
            ).to(device)
            self.scale_factor = 8
            self.latent_channels = 4
        
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        print(f"  Scale factor: {self.scale_factor}x")
        print(f"  Latent channels: {self.latent_channels}")
    
    @torch.no_grad()
    def encode(self, x):
        x = x.to(self.device, self.dtype)
        latent = self.vae.encode(x).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent
    
    @torch.no_grad()
    def decode(self, z):
        z = z.to(self.device, self.dtype)
        z = z / self.vae.config.scaling_factor
        image = self.vae.decode(z).sample
        return image


# ============================================================================
# Dataset
# ============================================================================

class LatentSRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=256, scale=8, augment=True, repeat=5):
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
        
        hr_tensor = torch.from_numpy(np.array(hr_img)).float() / 127.5 - 1.0
        lr_tensor = torch.from_numpy(np.array(lr_up)).float() / 127.5 - 1.0
        
        return {'hr': hr_tensor.permute(2, 0, 1), 'lr': lr_tensor.permute(2, 0, 1)}
    
    def _random_crop(self, hr_img, lr_up):
        w, h = hr_img.size
        ps = self.patch_size
        
        if w < ps or h < ps:
            scale = max(ps / w, ps / h) * 1.1
            new_size = (int(w * scale), int(h * scale))
            hr_img = hr_img.resize(new_size, Image.BICUBIC)
            lr_up = lr_up.resize(new_size, Image.BICUBIC)
            w, h = new_size
        
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        
        return (hr_img.crop((x, y, x + ps, y + ps)),
                lr_up.crop((x, y, x + ps, y + ps)))


# ============================================================================
# MMDiT Model Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MMDiTBlock(nn.Module):
    """MMDiT Block with dual-stream design"""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm1_img = RMSNorm(hidden_size)
        self.norm2_img = RMSNorm(hidden_size)
        self.norm1_cond = RMSNorm(hidden_size)
        self.norm2_cond = RMSNorm(hidden_size)
        
        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3)
        self.qkv_cond = nn.Linear(hidden_size, hidden_size * 3)
        self.proj_img = nn.Linear(hidden_size, hidden_size)
        self.proj_cond = nn.Linear(hidden_size, hidden_size)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_img = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden_size))
        self.mlp_cond = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden_size))
        
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
    
    def forward(self, x_img, x_cond, c):
        B = x_img.shape[0]
        
        shift_img, scale_img, gate_img, shift_cond, scale_cond, gate_cond = self.adaLN(c).chunk(6, dim=-1)
        
        x_img_norm = self.norm1_img(x_img) * (1 + scale_img.unsqueeze(1)) + shift_img.unsqueeze(1)
        x_cond_norm = self.norm1_cond(x_cond) * (1 + scale_cond.unsqueeze(1)) + shift_cond.unsqueeze(1)
        
        qkv_img = self.qkv_img(x_img_norm).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv_cond = self.qkv_cond(x_cond_norm).reshape(B, -1, 3, self.num_heads, self.head_dim)
        
        q_img, k_img, v_img = qkv_img.permute(2, 0, 3, 1, 4)
        q_cond, k_cond, v_cond = qkv_cond.permute(2, 0, 3, 1, 4)
        
        k_joint = torch.cat([k_img, k_cond], dim=2)
        v_joint = torch.cat([v_img, v_cond], dim=2)
        
        attn_img = F.scaled_dot_product_attention(q_img, k_joint, v_joint)
        attn_img = attn_img.transpose(1, 2).reshape(B, -1, self.hidden_size)
        
        attn_cond = F.scaled_dot_product_attention(q_cond, k_joint, v_joint)
        attn_cond = attn_cond.transpose(1, 2).reshape(B, -1, self.hidden_size)
        
        x_img = x_img + gate_img.unsqueeze(1) * self.proj_img(attn_img)
        x_cond = x_cond + gate_cond.unsqueeze(1) * self.proj_cond(attn_cond)
        
        x_img = x_img + gate_img.unsqueeze(1) * self.mlp_img(self.norm2_img(x_img))
        x_cond = x_cond + gate_cond.unsqueeze(1) * self.mlp_cond(self.norm2_cond(x_cond))
        
        return x_img, x_cond


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.freq_dim = freq_dim
    
    @staticmethod
    def timestep_embedding(t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


class LatentMMDiT(nn.Module):
    """MMDiT for Latent Space SR"""
    
    def __init__(self, latent_size=32, patch_size=2, in_channels=16, hidden_size=512, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        self.num_patches = (latent_size // patch_size) ** 2
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
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return x
    
    def unpatchify(self, x):
        B, N, _ = x.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        x = x.reshape(B, h, w, self.in_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, self.in_channels, h * p, w * p)
        return x
    
    def forward(self, z_t, z_cond, t, h):
        x_img = self.patch_embed_img(self.patchify(z_t)) + self.pos_embed
        x_cond = self.patch_embed_cond(self.patchify(z_cond)) + self.pos_embed
        
        c = self.t_embedder(t) + self.h_embedder(h)
        
        for block in self.blocks:
            x_img, x_cond = block(x_img, x_cond, c)
        
        x_img = self.norm_out(x_img)
        x_img = self.proj_out(x_img)
        
        return self.unpatchify(x_img)


# Model configs
def LatentMMDiT_S(latent_size=32, in_channels=16, **kwargs):
    return LatentMMDiT(latent_size, patch_size=2, in_channels=in_channels, hidden_size=384, depth=12, num_heads=6, **kwargs)

def LatentMMDiT_B(latent_size=32, in_channels=16, **kwargs):
    return LatentMMDiT(latent_size, patch_size=2, in_channels=in_channels, hidden_size=512, depth=12, num_heads=8, **kwargs)

def LatentMMDiT_L(latent_size=32, in_channels=16, **kwargs):
    return LatentMMDiT(latent_size, patch_size=2, in_channels=in_channels, hidden_size=768, depth=24, num_heads=12, **kwargs)

MODEL_CONFIGS = {'small': LatentMMDiT_S, 'base': LatentMMDiT_B, 'large': LatentMMDiT_L}


# ============================================================================
# Trainer
# ============================================================================

class LatentMeanFlowTrainer:
    def __init__(self, model, vae, device, lr=1e-4, weight_decay=0.01,
                 prop_h0=0.6, prop_havg=0.2, prop_onestep=0.2):
        self.model = model.to(device)
        self.vae = vae
        self.device = device
        
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
            t[:n_h0] = torch.sigmoid(rnd * 1.0 - 0.4)
            h[:n_h0] = 0.0
        
        if n_havg > 0:
            idx = n_h0
            rnd_t = torch.randn(n_havg, device=self.device)
            rnd_r = torch.randn(n_havg, device=self.device)
            t_tmp = torch.sigmoid(rnd_t * 1.0 - 0.4)
            r_tmp = torch.sigmoid(rnd_r * 1.0 - 0.4)
            t_tmp, r_tmp = torch.maximum(t_tmp, r_tmp), torch.minimum(t_tmp, r_tmp)
            t[idx:idx+n_havg] = t_tmp
            h[idx:idx+n_havg] = t_tmp - r_tmp
        
        if n_onestep > 0:
            idx = n_h0 + n_havg
            t[idx:] = 1.0
            h[idx:] = 1.0
        
        perm = torch.randperm(batch_size, device=self.device)
        return t[perm], h[perm]
    
    def train_step(self, batch):
        self.model.train()
        
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        batch_size = hr.shape[0]
        
        with torch.no_grad():
            hr_latent = self.vae.encode(hr).float()
            lr_latent = self.vae.encode(lr).float()
        
        t, h = self.sample_time(batch_size)
        t_exp = t[:, None, None, None]
        
        z_t = (1 - t_exp) * hr_latent + t_exp * lr_latent
        v = lr_latent - hr_latent
        
        u = self.model(z_t, lr_latent, t, h)
        loss = F.mse_loss(u, v)
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.update_ema()
        
        return {'loss': loss.item(), 'grad_norm': grad_norm.item()}
    
    @torch.no_grad()
    def inference(self, lr_images, num_steps=1, use_ema=True):
        self.model.eval()
        
        if use_ema:
            orig_params = {name: param.clone() for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                param.data.copy_(self.ema_params[name])
        
        lr_latent = self.vae.encode(lr_images).float()
        
        batch_size = lr_latent.shape[0]
        x = lr_latent
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), 1.0 - i * dt, device=self.device)
            h = torch.full((batch_size,), dt, device=self.device)
            u = self.model(x, lr_latent, t, h)
            x = x - dt * u
        
        hr_pred = self.vae.decode(x)
        
        if use_ema:
            for name, param in self.model.named_parameters():
                param.data.copy_(orig_params[name])
        
        return hr_pred
    
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
    
    def save_checkpoint(self, path, epoch, loss, psnr=0):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema': self.ema_params,
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'psnr': psnr,
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
# Experiment Logger
# ============================================================================

def save_experiment_config(save_dir, args, model, vae, best_psnr, best_loss, best_epoch, final_psnr, final_loss):
    """Save experiment configuration and results to txt file"""
    config_path = os.path.join(save_dir, 'experiment_config.txt')
    
    with open(config_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Latent MeanFlow-MMDiT Super-Resolution Experiment\n")
        f.write("="*70 + "\n\n")
        
        f.write("[Timestamp]\n")
        f.write(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("[Data]\n")
        f.write(f"  HR directory: {args.hr_dir}\n")
        f.write(f"  LR directory: {args.lr_dir}\n")
        if args.val_hr_dir:
            f.write(f"  Val HR directory: {args.val_hr_dir}\n")
            f.write(f"  Val LR directory: {args.val_lr_dir}\n")
        f.write(f"  Scale: {args.scale}x\n\n")
        
        f.write("[VAE]\n")
        f.write(f"  Type: {args.vae_type}\n")
        f.write(f"  Scale factor: {vae.scale_factor}x\n")
        f.write(f"  Latent channels: {vae.latent_channels}\n\n")
        
        f.write("[Model]\n")
        f.write(f"  Architecture: MMDiT (dual-stream)\n")
        f.write(f"  Size: {args.model_size}\n")
        f.write(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"  Image patch size: {args.patch_size}\n")
        f.write(f"  Latent patch size: {args.patch_size // vae.scale_factor}\n\n")
        
        f.write("[Training]\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Time sampling: 60% h=0, 20% h>0, 20% t=1,h=1\n\n")
        
        f.write("[Results]\n")
        f.write(f"  Best PSNR: {best_psnr:.4f} dB (epoch {best_epoch+1})\n")
        f.write(f"  Best Loss: {best_loss:.6f}\n")
        f.write(f"  Final PSNR: {final_psnr:.4f} dB\n")
        f.write(f"  Final Loss: {final_loss:.6f}\n\n")
        
        f.write("[Save Directory]\n")
        f.write(f"  {save_dir}\n")
        f.write("="*70 + "\n")
    
    print(f"Experiment config saved to: {config_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Latent MeanFlow-MMDiT SR Training')
    
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=8)
    
    parser.add_argument('--vae_type', type=str, default='sd', choices=['sd', 'sdxl', 'sd3', 'flux'])
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'])
    parser.add_argument('--patch_size', type=int, default=256)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_base', type=str, default='./checkpoints_mmdit')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-generate save_dir (without year)
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.save_dir = os.path.join(
            args.save_base,
            f"{timestamp}_{args.model_size}_x{args.scale}_{args.vae_type}_bs{args.batch_size}_ps{args.patch_size}"
        )
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("Latent MeanFlow-MMDiT SR Training")
    print("="*70)
    print(f"Device: {device}")
    print(f"VAE: {args.vae_type}")
    print(f"Model: {args.model_size}")
    print(f"Scale: {args.scale}x")
    print(f"Image patch: {args.patch_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save directory: {args.save_dir}")
    print("="*70)
    
    # Load VAE
    vae = VAEWrapper(args.vae_type, device)
    latent_size = args.patch_size // vae.scale_factor
    
    # Dataset
    train_dataset = LatentSRDataset(args.hr_dir, args.lr_dir, args.patch_size, args.scale, True, 5)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = LatentSRDataset(args.val_hr_dir, args.val_lr_dir, args.patch_size, args.scale, False, 1)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
        print(f"Validation: {len(val_dataset)} samples")
    
    # Model
    model_fn = MODEL_CONFIGS[args.model_size]
    model = model_fn(latent_size=latent_size, in_channels=vae.latent_channels)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = LatentMeanFlowTrainer(model, vae, device, args.lr)
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_psnr = 0
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(start_epoch, args.epochs):
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
        avg_loss = np.mean(losses)
        
        val_psnr = 0
        if val_loader:
            val_psnr = trainer.validate(val_loader, num_steps=1)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}dB")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}")
        
        scheduler.step()
        
        # Track best
        if val_loader and val_psnr > best_psnr:
            best_psnr = val_psnr
            best_loss = avg_loss
            best_epoch = epoch
            trainer.save_checkpoint(os.path.join(args.save_dir, 'best_model.pt'), epoch, avg_loss, val_psnr)
            print("  Saved best!")
        elif not val_loader and avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            trainer.save_checkpoint(os.path.join(args.save_dir, 'best_model.pt'), epoch, avg_loss)
        
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt'), epoch, avg_loss, val_psnr)
    
    # Save final
    trainer.save_checkpoint(os.path.join(args.save_dir, 'final_model.pt'), args.epochs - 1, avg_loss, val_psnr)
    
    # Save experiment config
    save_experiment_config(args.save_dir, args, model, vae, best_psnr, best_loss, best_epoch, val_psnr, avg_loss)
    
    print(f"\nDone! Best PSNR: {best_psnr:.2f} dB at epoch {best_epoch+1}")


if __name__ == '__main__':
    main()