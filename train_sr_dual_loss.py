"""
MeanFlow SR with FLUX Feature Loss (V1 Base + FLUX as Judge)

Architecture: Simple DiT (V1, 3ch input) - proven to work (28.51 dB)
Loss: MSE + FLUX Feature Loss (FLUX as judge, not generator)

The idea:
    - Model structure stays simple (V1)
    - FLUX VAE encoder extracts features from both x_pred and HR
    - Loss encourages x_pred to have similar FLUX features as HR
    - This teaches the model to produce FLUX-like textures

Loss = MSE(velocity, target) + λ * ||FLUX_Encode(x_pred) - FLUX_Encode(HR)||²

Usage:
    # Without FLUX loss (baseline)
    python train_sr_dual_loss.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --model_size small  

    # With FLUX loss
    python train_sr_dual_loss.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --model_size small \
        --use_flux_loss --flux_loss_weight 0.1   --device cuda:7
"""

import os
import math
import random
import argparse
import time
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
        self.patch_size, self.scale, self.augment, self.repeat = patch_size, scale, augment, repeat
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR/LR mismatch: {len(self.hr_files)} HR vs {len(self.lr_files)} LR"
        print(f"[Dataset] {len(self.hr_files)} pairs, repeat={repeat}")
    
    def __len__(self):
        return len(self.hr_files) * self.repeat
    
    def __getitem__(self, idx):
        idx = idx % len(self.hr_files)
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        w, h = hr_img.size
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            hr_img = hr_img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            lr_up = lr_up.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = hr_img.size
        
        x, y = random.randint(0, w - self.patch_size), random.randint(0, h - self.patch_size)
        hr_img = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        lr_up = lr_up.crop((x, y, x + self.patch_size, y + self.patch_size))
        
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
        
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 127.5 - 1.0
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1.0
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Model Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.freq_dim = freq_dim
    
    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = torch.cat([torch.cos(t[:, None] * freqs), torch.sin(t[:, None] * freqs)], dim=-1)
        return self.mlp(emb)


# ============================================================================
# DiT Block
# ============================================================================

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, c):
        B, N, C = x.shape
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = self.proj(attn.transpose(1, 2).reshape(B, N, C))
        
        x = x + gate_msa.unsqueeze(1) * attn
        
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


# ============================================================================
# Main Model (V1 - Simple, 3ch input)
# ============================================================================

class MeanFlowDiT(nn.Module):
    """
    Simple DiT model for SR (V1 architecture)
    Input: 3 channels (z_t only)
    Proven to work: 28.51 dB on DIV2K
    """
    def __init__(
        self,
        img_size=128,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # V1: 3 channel input only
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads)
            for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        return torch.einsum('nhwpqc->nchpwq', x).reshape(x.shape[0], self.in_channels, h * p, w * p)
    
    def forward(self, z_t, lr_cond, t, h):
        # V1: only use z_t, ignore lr_cond in forward
        # (lr_cond is kept in signature for compatibility)
        x = self.patch_embed(z_t) + self.pos_embed
        c = self.t_embedder(t) + self.h_embedder(h)
        
        for block in self.blocks:
            x = block(x, c)
        
        x = self.final_layer(x, c)
        return self.unpatchify(x)


# ============================================================================
# Model Configurations
# ============================================================================

def DiT_XS(img_size=128):
    return MeanFlowDiT(img_size, hidden_size=256, depth=8, num_heads=4)

def DiT_S(img_size=128):
    return MeanFlowDiT(img_size, hidden_size=384, depth=12, num_heads=6)

def DiT_B(img_size=128):
    return MeanFlowDiT(img_size, hidden_size=512, depth=12, num_heads=8)

def DiT_L(img_size=128):
    return MeanFlowDiT(img_size, hidden_size=768, depth=24, num_heads=12)

MODEL_CONFIGS = {
    'xs': DiT_XS,
    'small': DiT_S,
    'base': DiT_B,
    'large': DiT_L,
}


# ============================================================================
# FLUX Feature Loss (FLUX as Judge)
# ============================================================================

class FLUXFeatureLoss(nn.Module):
    """
    Use FLUX VAE Encoder to compute feature-level loss.
    
    FLUX is completely frozen - only used to extract features.
    This encourages the model to produce images that look good
    in FLUX's latent space (i.e., have FLUX-like textures).
    
    Loss = ||FLUX_Encode(x_pred) - FLUX_Encode(HR)||²
    """
    def __init__(self):
        super().__init__()
        self.vae = None
        self.flux_loaded = False
        self.scaling_factor = 1.0
    
    def load_flux(self, device='cuda'):
        """Load FLUX VAE encoder"""
        if self.flux_loaded:
            return True
        
        try:
            from diffusers import AutoencoderKL
            
            print("[FLUX] Loading FLUX VAE for feature loss...")
            self.vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="vae",
                torch_dtype=torch.float16
            ).to(device)
            
            # Completely freeze
            self.vae.requires_grad_(False)
            self.vae.eval()
            
            # Get scaling factor
            if hasattr(self.vae.config, 'scaling_factor'):
                self.scaling_factor = self.vae.config.scaling_factor
            
            self.flux_loaded = True
            print(f"[FLUX] VAE loaded successfully (scaling_factor={self.scaling_factor})")
            return True
            
        except Exception as e:
            print(f"[FLUX] Failed to load: {e}")
            print("[FLUX] Feature loss will be disabled")
            self.flux_loaded = False
            return False
    
    @torch.no_grad()
    def encode(self, x):
        """
        Encode image to FLUX latent space.
        
        Args:
            x: [B, 3, H, W] in [-1, 1]
        Returns:
            latent: [B, 16, H/8, W/8]
        """
        if self.vae is None:
            return None
        
        B, C, H, W = x.shape
        
        # Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Encode (use mean, not sample, for deterministic features)
        x_half = x.half()
        latent = self.vae.encode(x_half).latent_dist.mean
        
        # Remove padding in latent space
        if pad_h > 0 or pad_w > 0:
            lat_h = H // 8
            lat_w = W // 8
            latent = latent[:, :, :lat_h, :lat_w]
        
        return latent.float()
    
    def forward(self, x_pred, hr):
        """
        Compute FLUX feature loss.
        
        Args:
            x_pred: [B, 3, H, W] predicted SR image
            hr: [B, 3, H, W] ground truth HR image
        
        Returns:
            loss: scalar
        """
        if not self.flux_loaded or self.vae is None:
            return torch.tensor(0.0, device=x_pred.device)
        
        # Extract features
        F_pred = self.encode(x_pred)
        F_hr = self.encode(hr)
        
        if F_pred is None or F_hr is None:
            return torch.tensor(0.0, device=x_pred.device)
        
        # L2 loss in FLUX latent space
        loss = F.mse_loss(F_pred, F_hr)
        
        return loss


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    def __init__(
        self,
        model,
        device,
        lr=1e-4,
        weight_decay=0.01,
        P_mean=-0.4,
        P_std=1.0,
        prop_h0=0.6,
        prop_havg=0.2,
        prop_onestep=0.2,
        use_flux_loss=False,
        flux_loss_weight=0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.P_mean = P_mean
        self.P_std = P_std
        self.prop_h0 = prop_h0
        self.prop_havg = prop_havg
        self.prop_onestep = prop_onestep
        self.flux_loss_weight = flux_loss_weight
        
        # FLUX Feature Loss
        self.use_flux_loss = use_flux_loss
        self.flux_loss_fn = None
        if use_flux_loss:
            self.flux_loss_fn = FLUXFeatureLoss()
            if self.flux_loss_fn.load_flux(device):
                print(f"[Trainer] FLUX feature loss enabled (weight={flux_loss_weight})")
            else:
                self.use_flux_loss = False
                print("[Trainer] FLUX feature loss disabled (failed to load)")
        
        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        
        # EMA
        self.ema_decay = 0.9999
        self.ema_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    def update_ema(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.ema_params[n].mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def sample_time(self, batch_size):
        """MeanFlow 60/20/20 time sampling"""
        n0 = int(batch_size * self.prop_h0)
        na = int(batch_size * self.prop_havg)
        n1 = batch_size - n0 - na
        
        t = torch.zeros(batch_size, device=self.device)
        h = torch.zeros(batch_size, device=self.device)
        
        if n0 > 0:
            t[:n0] = torch.sigmoid(torch.randn(n0, device=self.device) * self.P_std + self.P_mean)
        
        if na > 0:
            t1 = torch.sigmoid(torch.randn(na, device=self.device) * self.P_std + self.P_mean)
            t2 = torch.sigmoid(torch.randn(na, device=self.device) * self.P_std + self.P_mean)
            t[n0:n0+na] = torch.maximum(t1, t2)
            h[n0:n0+na] = torch.abs(t1 - t2)
        
        if n1 > 0:
            t[-n1:] = 1.0
            h[-n1:] = 1.0
        
        perm = torch.randperm(batch_size, device=self.device)
        return t[perm], h[perm]
    
    def train_step(self, batch):
        self.model.train()
        
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        B = hr.shape[0]
        
        # Sample time
        t, h = self.sample_time(B)
        
        # Construct z_t
        t_exp = t[:, None, None, None]
        z_t = (1 - t_exp) * hr + t_exp * lr
        
        # Target velocity
        v_target = lr - hr
        
        # Forward
        u = self.model(z_t, lr, t, h)
        
        # MSE Loss
        loss_mse = F.mse_loss(u, v_target)
        
        # FLUX Feature Loss
        loss_flux = torch.tensor(0.0, device=self.device)
        if self.use_flux_loss and self.flux_loss_fn is not None:
            # Reconstruct predicted image
            x_pred = z_t - t_exp * u
            x_pred = x_pred.clamp(-1, 1)  # Clamp for stability
            loss_flux = self.flux_loss_fn(x_pred, hr)
        
        # Total loss
        loss = loss_mse + self.flux_loss_weight * loss_flux
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.update_ema()
        
        return {
            'loss': loss.item(),
            'loss_mse': loss_mse.item(),
            'loss_flux': loss_flux.item() if self.use_flux_loss else 0.0,
            'grad_norm': grad_norm.item(),
        }
    
    @torch.no_grad()
    def inference(self, lr_images, num_steps=1, use_ema=True):
        self.model.eval()
        
        if use_ema:
            orig = {n: p.clone() for n, p in self.model.named_parameters()}
            for n, p in self.model.named_parameters():
                p.data.copy_(self.ema_params[n])
        
        x = lr_images.clone()
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((x.shape[0],), 1.0 - i * dt, device=self.device)
            h = torch.full((x.shape[0],), dt, device=self.device)
            x = x - dt * self.model(x, lr_images, t, h)
        
        if use_ema:
            for n, p in self.model.named_parameters():
                p.data.copy_(orig[n])
        
        return x
    
    @torch.no_grad()
    def validate(self, loader, num_steps=1):
        psnrs = []
        for batch in loader:
            hr = batch['hr'].to(self.device)
            lr = batch['lr'].to(self.device)
            
            pred = self.inference(lr, num_steps, use_ema=True)
            
            hr_01 = (hr + 1) / 2
            pred_01 = ((pred + 1) / 2).clamp(0, 1)
            
            mse = ((hr_01 - pred_01) ** 2).mean(dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            psnrs.extend(psnr.cpu().tolist())
        
        return np.mean(psnrs)
    
    def save(self, path, epoch, loss, psnr=0):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema': self.ema_params,
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'psnr': psnr,
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'], strict=False)
        if 'ema' in ckpt:
            self.ema_params = ckpt['ema']
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow SR with FLUX Feature Loss')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=4)
    
    # Model
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--patch_size', type=int, default=128)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # FLUX Loss
    parser.add_argument('--use_flux_loss', action='store_true', help='Use FLUX feature loss')
    parser.add_argument('--flux_loss_weight', type=float, default=0.1, help='Weight for FLUX loss')
    
    # MeanFlow
    parser.add_argument('--prop_h0', type=float, default=0.6)
    parser.add_argument('--prop_havg', type=float, default=0.2)
    parser.add_argument('--prop_onestep', type=float, default=0.2)
    
    # Save
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_base', type=str, default='./checkpoints/dual')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Save dir
    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        flux_str = f"fluxloss{args.flux_loss_weight}" if args.use_flux_loss else "baseline"
        args.save_dir = os.path.join(args.save_base, f"{ts}_{args.model_size}_x{args.scale}_{flux_str}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 70)
    print("MeanFlow SR with FLUX Feature Loss (V1 Base)")
    print("=" * 70)
    print(f"Model: {args.model_size} | Scale: {args.scale}x | Patch: {args.patch_size}")
    print(f"FLUX Loss: {args.use_flux_loss} (weight={args.flux_loss_weight})")
    print(f"MeanFlow: {args.prop_h0}/{args.prop_havg}/{args.prop_onestep}")
    print(f"Save: {args.save_dir}")
    print("=" * 70)
    
    # Data
    train_data = SRDataset(args.hr_dir, args.lr_dir, args.patch_size, args.scale, True, 5)
    train_loader = DataLoader(
        train_data, args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_data = SRDataset(args.val_hr_dir, args.val_lr_dir, args.patch_size, args.scale, False, 1)
        val_loader = DataLoader(val_data, 4, shuffle=False, num_workers=2)
    
    # Model
    model = MODEL_CONFIGS[args.model_size](args.patch_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Trainer
    trainer = Trainer(
        model, device, args.lr, args.weight_decay,
        prop_h0=args.prop_h0, prop_havg=args.prop_havg, prop_onestep=args.prop_onestep,
        use_flux_loss=args.use_flux_loss, flux_loss_weight=args.flux_loss_weight,
    )
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training
    best_psnr = 0
    train_start = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        losses, losses_mse, losses_flux = [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            losses_mse.append(metrics['loss_mse'])
            losses_flux.append(metrics['loss_flux'])
            
            if args.use_flux_loss:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'mse': f"{metrics['loss_mse']:.4f}",
                    'flux': f"{metrics['loss_flux']:.4f}"
                })
            else:
                pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
        scheduler.step()
        avg_loss = np.mean(losses)
        avg_mse = np.mean(losses_mse)
        avg_flux = np.mean(losses_flux)
        
        # Validation
        if val_loader:
            psnr = trainer.validate(val_loader)
            if args.use_flux_loss:
                print(f"Epoch {epoch+1}: loss={avg_loss:.6f} (mse={avg_mse:.6f}, flux={avg_flux:.6f}), PSNR={psnr:.2f}dB")
            else:
                print(f"Epoch {epoch+1}: loss={avg_loss:.6f}, PSNR={psnr:.2f}dB")
            
            if psnr > best_psnr:
                best_psnr = psnr
                trainer.save(os.path.join(args.save_dir, 'best_model.pt'), epoch, avg_loss, psnr)
                print("  -> Saved best!")
        else:
            print(f"Epoch {epoch+1}: loss={avg_loss:.6f}")
        
        # Periodic save
        if (epoch + 1) % 20 == 0:
            trainer.save(os.path.join(args.save_dir, f'epoch{epoch+1}.pt'), epoch, avg_loss, best_psnr)
    
    # Final save
    total_time = time.time() - train_start
    trainer.save(os.path.join(args.save_dir, 'final.pt'), args.epochs - 1, avg_loss, best_psnr)
    
    print(f"\nTraining complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Checkpoint: {args.save_dir}")


if __name__ == '__main__':
    main()