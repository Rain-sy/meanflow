"""
Dual-Pipeline Super-Resolution with Delta Learning

Architecture:
    Main Pipeline (DiT): Structure preservation [TRAINABLE]
    Detail Pipeline (FLUX): Texture delta extraction [FROZEN]
    
    SR = F_main + α ⊙ Δ
    where Δ = I_flux - LR_bicubic (high-frequency detail)
          α = learned gating weight

MeanFlow Training Strategy:
    60%: h=0, t~logit-normal (instantaneous velocity)
    20%: h>0 (average velocity)
    20%: t=1, h=1 (one-step inference)

Usage:
    # Without FLUX (for testing)
    python train_sr_dual.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --model_size small --scale 4

    # With FLUX
    python train_sr_dual.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --use_flux
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
        assert len(self.hr_files) == len(self.lr_files)
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
# Basic Components
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
# DiT Block (Single Stream, NOT MMDiT)
# ============================================================================

class DiTBlock(nn.Module):
    """
    Single-stream DiT block with Self-Attention + AdaLN
    Simpler than MMDiT, fewer parameters
    """
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
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, c):
        B, N, C = x.shape
        
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-Attention
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = self.proj(attn.transpose(1, 2).reshape(B, N, C))
        
        x = x + gate_msa.unsqueeze(1) * attn
        
        # MLP
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
# Main Pipeline (DiT-based)
# ============================================================================

class MainPipeline(nn.Module):
    """
    Main pipeline: DiT blocks for structure preservation
    Single stream (not dual-stream MMDiT) to reduce parameters
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
        
        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        # DiT blocks (single stream)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        return torch.einsum('nhwpqc->nchpwq', x).reshape(x.shape[0], self.in_channels, h * p, w * p)
    
    def forward(self, z_t, t, h):
        # Patch embed
        x = self.patch_embed(z_t) + self.pos_embed
        
        # Time conditioning
        c = self.t_embedder(t) + self.h_embedder(h)
        
        # DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer + unpatchify
        x = self.final_layer(x, c)
        return self.unpatchify(x)


# ============================================================================
# Detail Pipeline (FLUX-based, Frozen)
# ============================================================================

class DetailPipeline(nn.Module):
    """
    Detail pipeline: FLUX VAE + DiT for texture details
    All components are frozen, only used for inference
    """
    def __init__(self, use_flux=True):
        super().__init__()
        self.use_flux = use_flux
        self.flux_loaded = False
        self.vae = None
        self.transformer = None
    
    def load_flux(self, device='cuda'):
        """Load FLUX model (lazy loading)"""
        if self.flux_loaded or not self.use_flux:
            return
        
        try:
            from diffusers import FluxPipeline
            
            print("Loading FLUX model...")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16
            )
            
            self.vae = pipe.vae.to(device)
            self.vae.requires_grad_(False)
            self.vae.eval()
            
            # Note: For full FLUX, we'd also load transformer
            # For now, we just use VAE encode/decode
            
            self.flux_loaded = True
            print("FLUX VAE loaded and frozen")
            
            # Clean up
            del pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Warning: Could not load FLUX: {e}")
            print("Detail pipeline will return zeros (no texture enhancement)")
            self.use_flux = False
    
    @torch.no_grad()
    def forward(self, lr_bicubic):
        """
        Extract texture details from FLUX reconstruction
        
        Args:
            lr_bicubic: [B, 3, H, W] in range [-1, 1]
        
        Returns:
            I_flux: [B, 3, H, W] FLUX reconstruction
        """
        if not self.use_flux or self.vae is None:
            # Return input as-is (no detail enhancement)
            return lr_bicubic.clone()
        
        B, C, H, W = lr_bicubic.shape
        
        # Ensure dimensions are divisible by 8 for VAE
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            lr_padded = F.pad(lr_bicubic, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            lr_padded = lr_bicubic
        
        # VAE encode -> decode (adds FLUX texture characteristics)
        lr_half = lr_padded.half()
        latent = self.vae.encode(lr_half).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        I_flux = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            I_flux = I_flux[:, :, :H, :W]
        
        return I_flux.float()


# ============================================================================
# Detail Extractor (Delta Learning)
# ============================================================================

class DetailExtractor(nn.Module):
    """
    Extract and gate high-frequency details from FLUX
    
    Δ = I_flux - LR_bicubic (high-frequency residual)
    α = sigmoid(Conv(Δ)) (learned gate)
    Detail = α ⊙ Δ
    """
    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()
        
        # Learn to extract useful details and create gating mask
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
            nn.Sigmoid()  # Output gate α in [0, 1]
        )
        
        # Optional: learnable global scale
        self.detail_scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, I_flux, lr_bicubic):
        """
        Args:
            I_flux: [B, 3, H, W] FLUX reconstruction
            lr_bicubic: [B, 3, H, W] bicubic upsampled LR
        
        Returns:
            detail: [B, 3, H, W] gated high-frequency detail
        """
        # High-frequency residual (what FLUX wants to add)
        delta = I_flux - lr_bicubic  # [B, 3, H, W]
        
        # Learn gating weights
        alpha = self.gate_net(delta)  # [B, 3, H, W] in [0, 1]
        
        # Gated detail with learnable scale
        detail = self.detail_scale * alpha * delta
        
        return detail


# ============================================================================
# Full Dual-Pipeline Model
# ============================================================================

class DualPipelineSR(nn.Module):
    """
    Dual-Pipeline Super-Resolution with Delta Learning
    
    SR = F_main + Detail
    where:
        F_main = MainPipeline(z_t, t, h)  -- structure
        Detail = α ⊙ (I_flux - LR_bic)    -- texture delta from FLUX
    """
    def __init__(
        self,
        img_size=128,
        patch_size=4,
        hidden_size=384,
        depth=12,
        num_heads=6,
        use_flux=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.use_flux = use_flux
        
        # Main pipeline (trainable)
        self.main_pipeline = MainPipeline(
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
        )
        
        # Detail pipeline (frozen)
        self.detail_pipeline = DetailPipeline(use_flux=use_flux)
        
        # Detail extractor (trainable)
        self.detail_extractor = DetailExtractor()
    
    def load_flux(self, device='cuda'):
        """Load FLUX model"""
        self.detail_pipeline.load_flux(device)
    
    def forward(self, z_t, lr_cond, t, h):
        """
        Args:
            z_t: [B, 3, H, W] interpolated state (training) or LR (inference)
            lr_cond: [B, 3, H, W] LR bicubic upsampled
            t: [B] timestep
            h: [B] step size
        
        Returns:
            velocity: [B, 3, H, W] predicted velocity
        """
        # Main pipeline: structure + base reconstruction
        F_main = self.main_pipeline(z_t, t, h)  # [B, 3, H, W]
        
        # Detail pipeline: FLUX texture
        if self.use_flux and self.detail_pipeline.flux_loaded:
            with torch.no_grad():
                I_flux = self.detail_pipeline(lr_cond)
            detail = self.detail_extractor(I_flux, lr_cond)
        else:
            detail = torch.zeros_like(F_main)
        
        # Combine: velocity = main + detail
        velocity = F_main + detail
        
        return velocity


# ============================================================================
# Model Configurations
# ============================================================================

def DualPipeline_XS(img_size=128, use_flux=True):
    return DualPipelineSR(img_size, hidden_size=256, depth=8, num_heads=4, use_flux=use_flux)

def DualPipeline_S(img_size=128, use_flux=True):
    return DualPipelineSR(img_size, hidden_size=384, depth=12, num_heads=6, use_flux=use_flux)

def DualPipeline_B(img_size=128, use_flux=True):
    return DualPipelineSR(img_size, hidden_size=512, depth=12, num_heads=8, use_flux=use_flux)

def DualPipeline_L(img_size=128, use_flux=True):
    return DualPipelineSR(img_size, hidden_size=768, depth=24, num_heads=12, use_flux=use_flux)

MODEL_CONFIGS = {
    'xs': DualPipeline_XS,
    'small': DualPipeline_S,
    'base': DualPipeline_B,
    'large': DualPipeline_L,
}


# ============================================================================
# Trainer (MeanFlow Strategy)
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
    ):
        self.model = model.to(device)
        self.device = device
        self.P_mean = P_mean
        self.P_std = P_std
        self.prop_h0 = prop_h0
        self.prop_havg = prop_havg
        self.prop_onestep = prop_onestep
        
        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        
        # EMA
        self.ema_decay = 0.9999
        self.ema_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    
    def update_ema(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.ema_params:
                    self.ema_params[n].mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def sample_time(self, batch_size):
        """MeanFlow time sampling: 60/20/20 strategy"""
        n0 = int(batch_size * self.prop_h0)
        na = int(batch_size * self.prop_havg)
        n1 = batch_size - n0 - na
        
        t = torch.zeros(batch_size, device=self.device)
        h = torch.zeros(batch_size, device=self.device)
        
        # 60%: h=0, t~logit-normal
        if n0 > 0:
            t[:n0] = torch.sigmoid(torch.randn(n0, device=self.device) * self.P_std + self.P_mean)
        
        # 20%: h>0, average velocity
        if na > 0:
            t1 = torch.sigmoid(torch.randn(na, device=self.device) * self.P_std + self.P_mean)
            t2 = torch.sigmoid(torch.randn(na, device=self.device) * self.P_std + self.P_mean)
            t[n0:n0+na] = torch.maximum(t1, t2)
            h[n0:n0+na] = torch.abs(t1 - t2)
        
        # 20%: t=1, h=1 (one-step)
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
        
        # Loss
        loss = F.mse_loss(u, v_target)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        self.update_ema()
        
        return {'loss': loss.item(), 'grad_norm': grad_norm.item()}
    
    @torch.no_grad()
    def inference(self, lr_images, num_steps=1, use_ema=True):
        self.model.eval()
        
        if use_ema:
            orig = {n: p.clone() for n, p in self.model.named_parameters() if n in self.ema_params}
            for n, p in self.model.named_parameters():
                if n in self.ema_params:
                    p.data.copy_(self.ema_params[n])
        
        x = lr_images.clone()
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((x.shape[0],), 1.0 - i * dt, device=self.device)
            h = torch.full((x.shape[0],), dt, device=self.device)
            x = x - dt * self.model(x, lr_images, t, h)
        
        if use_ema:
            for n, p in self.model.named_parameters():
                if n in orig:
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
    parser = argparse.ArgumentParser(description='Dual-Pipeline SR with Delta Learning')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=4)
    
    # Model
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--use_flux', action='store_true', help='Use FLUX for detail enhancement')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
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
        flux_str = "flux" if args.use_flux else "noflux"
        args.save_dir = os.path.join(args.save_base, f"{ts}_{args.model_size}_x{args.scale}_{flux_str}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Dual-Pipeline SR with Delta Learning")
    print("=" * 70)
    print(f"Model: {args.model_size} | Scale: {args.scale}x | Patch: {args.patch_size}")
    print(f"Use FLUX: {args.use_flux}")
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
    model = MODEL_CONFIGS[args.model_size](args.patch_size, use_flux=args.use_flux)
    
    # Load FLUX if needed
    if args.use_flux:
        model.load_flux(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Trainer
    trainer = Trainer(
        model, device, args.lr, args.weight_decay,
        prop_h0=args.prop_h0, prop_havg=args.prop_havg, prop_onestep=args.prop_onestep
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
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
        scheduler.step()
        avg_loss = np.mean(losses)
        
        # Validation
        if val_loader:
            psnr = trainer.validate(val_loader)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={psnr:.2f}dB")
            
            if psnr > best_psnr:
                best_psnr = psnr
                trainer.save(os.path.join(args.save_dir, 'best_model.pt'), epoch, avg_loss, psnr)
                print("  -> Saved best!")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}")
        
        # Periodic save
        if (epoch + 1) % 20 == 0:
            trainer.save(os.path.join(args.save_dir, f'epoch{epoch+1}.pt'), epoch, avg_loss, best_psnr)
    
    # Final save
    total_time = time.time() - train_start
    trainer.save(os.path.join(args.save_dir, 'final.pt'), args.epochs - 1, avg_loss, best_psnr)
    
    print(f"\nTraining complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Total time: {total_time/3600:.2f} hours")


if __name__ == '__main__':
    main()