"""
MeanFlow-DiT: MMDiT-based Super-Resolution Training Script

Architecture: Diffusion Transformer (DiT) with AdaLN-Zero
Key Features:
    1. MMDiT architecture replacing UNet
    2. Global self-attention for long-range dependencies
    3. AdaLN-Zero conditioning on (t, h)
    4. Optional degradation-aware conditioning
    5. One-step inference capability

Usage:
    python train_meanflow_dit.py \
        --hr_dir Flow_Restore/Data/DIV2K/DIV2K_train_HR \
        --lr_dir Flow_Restore/Data/DIV2K/DIV2K_train_LR_bicubic/X2 \
        --epochs 100 \
        --batch_size 8 \
        --patch_size 128 \
        --model_size small
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
    """SR Dataset with augmentation"""
    
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
        
        # Store original LR for potential cycle loss
        lr_original = lr_img.copy()
        
        # Upsample LR to HR size
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        # Random crop
        hr_img, lr_up, lr_original = self._random_crop(hr_img, lr_up, lr_original)
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_up = lr_up.transpose(Image.FLIP_LEFT_RIGHT)
                lr_original = lr_original.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
                lr_up = lr_up.transpose(Image.FLIP_TOP_BOTTOM)
                lr_original = lr_original.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                hr_img = hr_img.rotate(90 * k)
                lr_up = lr_up.rotate(90 * k)
                lr_original = lr_original.rotate(90 * k)
        
        # To tensor [-1, 1]
        hr_tensor = torch.from_numpy(np.array(hr_img)).float() / 127.5 - 1.0
        lr_tensor = torch.from_numpy(np.array(lr_up)).float() / 127.5 - 1.0
        lr_orig_tensor = torch.from_numpy(np.array(lr_original)).float() / 127.5 - 1.0
        
        hr_tensor = hr_tensor.permute(2, 0, 1)
        lr_tensor = lr_tensor.permute(2, 0, 1)
        lr_orig_tensor = lr_orig_tensor.permute(2, 0, 1)
        
        return {
            'hr': hr_tensor, 
            'lr': lr_tensor,
            'lr_original': lr_orig_tensor
        }
    
    def _random_crop(self, hr_img, lr_up, lr_original):
        w, h = hr_img.size
        lr_w, lr_h = lr_original.size
        lr_patch = self.patch_size // self.scale
        
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            new_size = (int(w * scale), int(h * scale))
            hr_img = hr_img.resize(new_size, Image.BICUBIC)
            lr_up = lr_up.resize(new_size, Image.BICUBIC)
            new_lr_size = (int(lr_w * scale), int(lr_h * scale))
            lr_original = lr_original.resize(new_lr_size, Image.BICUBIC)
            w, h = new_size
            lr_w, lr_h = new_lr_size
        
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        
        lr_x = x // self.scale
        lr_y = y // self.scale
        
        hr_crop = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        lr_crop = lr_up.crop((x, y, x + self.patch_size, y + self.patch_size))
        lr_orig_crop = lr_original.crop((lr_x, lr_y, lr_x + lr_patch, lr_y + lr_patch))
        
        return hr_crop, lr_crop, lr_orig_crop


# ============================================================================
# Model Components
# ============================================================================

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class TimestepEmbedder(nn.Module):
    """Embed timesteps (t, h) using sinusoidal encoding + MLP"""
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
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class Mlp(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use scaled dot-product attention (flash attention if available)
        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT Block with AdaLN-Zero conditioning
    
    Structure:
        x -> LayerNorm -> Modulate -> Attention -> + -> LayerNorm -> Modulate -> MLP -> +
                 ↑                         |                 ↑                      |
                 |---- condition c --------|                 |---- condition c -----|
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, 
                              attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=drop)
        
        # AdaLN-Zero modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Zero initialization for residual
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, c):
        # c: conditioning embedding (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention block
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # MLP block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class FinalLayer(nn.Module):
    """Final layer with AdaLN and linear projection"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        # Zero initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================================
# MeanFlow-DiT Model
# ============================================================================

class MeanFlowDiT(nn.Module):
    """
    MeanFlow-DiT: Diffusion Transformer for Image Super-Resolution
    
    Key features:
        1. Patch-based processing with positional embeddings
        2. AdaLN-Zero conditioning on (t, h)
        3. Global self-attention in each block
        4. Zero-initialized output for stable training
    
    Args:
        img_size: Input image size (must be divisible by patch_size)
        patch_size: Size of each patch
        in_channels: Number of input channels
        hidden_size: Transformer hidden dimension
        depth: Number of DiT blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        drop_rate: Dropout rate
    """
    def __init__(
        self,
        img_size=128,
        patch_size=4,
        in_channels=3,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Time embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        
        # Output layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights"""
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.patch_embed.proj.bias)
        
        # Initialize timestep embeddings
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.h_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.h_embedder.mlp[2].weight, std=0.02)
        
        # Initialize DiT blocks
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
    
    def unpatchify(self, x):
        """Convert patch tokens back to image: (B, N, patch_size^2 * C) -> (B, C, H, W)"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], self.in_channels, h * p, w * p)
        return imgs
    
    def forward(self, x, t, h):
        """
        Forward pass
        
        Args:
            x: Input image (B, C, H, W) - LR upsampled to HR size
            t: Time step (B,) - typically 1.0 for one-step inference
            h: Time interval (B,) - typically 1.0 for one-step inference
        
        Returns:
            u: Predicted velocity (B, C, H, W)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed   # Add positional embedding
        
        # Condition embedding
        c = self.t_embedder(t) + self.h_embedder(h)  # (B, D)
        
        # DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)  # (B, N, patch_size^2 * C)
        
        # Unpatchify
        x = self.unpatchify(x)  # (B, C, H, W)
        
        return x
    
    def forward_with_cfg(self, x, t, h, cfg_scale=1.0):
        """Forward with classifier-free guidance (optional)"""
        # For now, just regular forward
        return self.forward(x, t, h)


# ============================================================================
# Model Configurations
# ============================================================================

def MeanFlowDiT_XS(img_size=128, **kwargs):
    """Extra Small: ~7M params"""
    return MeanFlowDiT(
        img_size=img_size,
        patch_size=4,
        hidden_size=256,
        depth=8,
        num_heads=4,
        **kwargs
    )

def MeanFlowDiT_S(img_size=128, **kwargs):
    """Small: ~30M params"""
    return MeanFlowDiT(
        img_size=img_size,
        patch_size=4,
        hidden_size=384,
        depth=12,
        num_heads=6,
        **kwargs
    )

def MeanFlowDiT_B(img_size=128, **kwargs):
    """Base: ~80M params"""
    return MeanFlowDiT(
        img_size=img_size,
        patch_size=4,
        hidden_size=512,
        depth=12,
        num_heads=8,
        **kwargs
    )

def MeanFlowDiT_L(img_size=128, **kwargs):
    """Large: ~300M params"""
    return MeanFlowDiT(
        img_size=img_size,
        patch_size=4,
        hidden_size=768,
        depth=24,
        num_heads=12,
        **kwargs
    )


MODEL_CONFIGS = {
    'xs': MeanFlowDiT_XS,
    'small': MeanFlowDiT_S,
    'base': MeanFlowDiT_B,
    'large': MeanFlowDiT_L,
}


# ============================================================================
# Trainer
# ============================================================================

class MeanFlowDiTTrainer:
    """
    MeanFlow-DiT Trainer
    
    Training strategy:
        - 50% samples: h = t (learn full path)
        - 30% samples: h < t (learn partial path)
        - 20% samples: t = 1, h = 1 (one-step inference)
    """
    
    def __init__(
        self,
        model,
        device,
        lr=1e-4,
        weight_decay=0.01,
        lambda_cycle=0.0,
        scale=2,
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.scale = scale
        
        self.optimizer = AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # EMA
        self.ema_decay = 0.9999
        self.ema_params = {name: param.clone().detach() 
                          for name, param in model.named_parameters()}
    
    def update_ema(self):
        """Update EMA parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.ema_params[name].mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay)
    
    def sample_time(self, batch_size):
        """
        Sample time (t, h) with improved strategy
        
        Distribution:
            - 50%: h = t (full path from t to 0)
            - 30%: h < t (partial path)
            - 20%: t = 1, h = 1 (one-step)
        """
        n_full = int(batch_size * 0.5)
        n_partial = int(batch_size * 0.3)
        n_onestep = batch_size - n_full - n_partial
        
        # Sample t from logit-normal
        rnd = torch.randn(batch_size, device=self.device)
        t = torch.sigmoid(rnd * 1.0 + (-0.4))
        
        # Initialize h
        h = torch.zeros(batch_size, device=self.device)
        
        # Full path: h = t
        h[:n_full] = t[:n_full]
        
        # Partial path: h ~ uniform(0, t)
        h[n_full:n_full+n_partial] = torch.rand(n_partial, device=self.device) * t[n_full:n_full+n_partial]
        
        # One-step: t = 1, h = 1
        t[n_full+n_partial:] = 1.0
        h[n_full+n_partial:] = 1.0
        
        return t, h
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        batch_size = hr.shape[0]
        
        # Sample time
        t, h = self.sample_time(batch_size)
        
        # Interpolate: z_t = (1-t) * HR + t * LR
        t_exp = t[:, None, None, None]
        z_t = (1 - t_exp) * hr + t_exp * lr
        
        # Target velocity: v = LR - HR
        v = lr - hr
        
        # Forward pass
        u = self.model(z_t, t, h)
        
        # Main loss
        loss_main = F.mse_loss(u, v)
        
        # Optional cycle loss
        loss_cycle = torch.tensor(0.0, device=self.device)
        if self.lambda_cycle > 0:
            lr_original = batch['lr_original'].to(self.device)
            hr_pred = lr - u
            lr_recon = F.interpolate(hr_pred, scale_factor=1.0/self.scale, 
                                      mode='bicubic', align_corners=False)
            # Match sizes if needed
            if lr_recon.shape != lr_original.shape:
                lr_recon = F.interpolate(lr_recon, size=lr_original.shape[2:],
                                         mode='bicubic', align_corners=False)
            loss_cycle = F.mse_loss(lr_recon, lr_original)
        
        # Total loss
        loss = loss_main + self.lambda_cycle * loss_cycle
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.update_ema()
        
        return {
            'loss': loss.item(),
            'loss_main': loss_main.item(),
            'loss_cycle': loss_cycle.item() if self.lambda_cycle > 0 else 0,
            'grad_norm': grad_norm.item(),
        }
    
    @torch.no_grad()
    def inference(self, lr_images, num_steps=1, use_ema=True):
        """
        Inference
        
        Args:
            lr_images: LR images (B, C, H, W) - already upsampled to HR size
            num_steps: Number of sampling steps (1 for one-step)
            use_ema: Whether to use EMA parameters
        """
        self.model.eval()
        
        # Backup and load EMA if needed
        if use_ema:
            orig_params = {name: param.clone() 
                          for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                param.data.copy_(self.ema_params[name])
        
        batch_size = lr_images.shape[0]
        x = lr_images
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((batch_size,), t_val, device=self.device)
            h = torch.full((batch_size,), dt, device=self.device)
            
            u = self.model(x, t, h)
            x = x - dt * u
        
        # Restore original params
        if use_ema:
            for name, param in self.model.named_parameters():
                param.data.copy_(orig_params[name])
        
        return x
    
    @torch.no_grad()
    def validate(self, val_loader, num_steps=1):
        """Validate and compute PSNR"""
        self.model.eval()
        
        psnr_list = []
        
        for batch in val_loader:
            hr = batch['hr'].to(self.device)
            lr = batch['lr'].to(self.device)
            
            hr_pred = self.inference(lr, num_steps=num_steps, use_ema=True)
            
            # Compute PSNR in [0, 1] range
            hr_01 = (hr + 1) / 2
            hr_pred_01 = ((hr_pred + 1) / 2).clamp(0, 1)
            
            mse = ((hr_01 - hr_pred_01) ** 2).mean(dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            psnr_list.extend(psnr.cpu().tolist())
        
        return np.mean(psnr_list)
    
    def save_checkpoint(self, path, epoch, loss, psnr_1step=0, psnr_10step=0):
        """Save checkpoint"""
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
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if 'ema' in checkpoint:
            self.ema_params = checkpoint['ema']
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint.get('epoch', 0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow-DiT Training')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True, help='HR images directory')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR images directory')
    parser.add_argument('--val_hr_dir', type=str, default=None, help='Validation HR directory')
    parser.add_argument('--val_lr_dir', type=str, default=None, help='Validation LR directory')
    parser.add_argument('--scale', type=int, default=2, help='SR scale factor')
    
    # Model
    parser.add_argument('--model_size', type=str, default='small', 
                        choices=['xs', 'small', 'base', 'large'],
                        help='Model size')
    parser.add_argument('--patch_size', type=int, default=128, help='Training patch size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lambda_cycle', type=float, default=0.0, 
                        help='Cycle consistency loss weight')
    
    # Misc
    parser.add_argument('--save_dir', type=str, default='./checkpoints_dit')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("MeanFlow-DiT Training")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model_size}")
    print(f"Patch size: {args.patch_size}")
    print(f"Scale: {args.scale}x")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Cycle loss weight: {args.lambda_cycle}")
    print("="*70)
    
    # Dataset
    train_dataset = SRDataset(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        patch_size=args.patch_size,
        scale=args.scale,
        augment=True,
        repeat=5,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Validation
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(
            hr_dir=args.val_hr_dir,
            lr_dir=args.val_lr_dir,
            patch_size=args.patch_size,
            scale=args.scale,
            augment=False,
            repeat=1,
        )
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
        print(f"Validation: {len(val_dataset)} samples")
    
    # Model
    model_fn = MODEL_CONFIGS[args.model_size]
    model = model_fn(img_size=args.patch_size)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # Trainer
    trainer = MeanFlowDiTTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_cycle=args.lambda_cycle,
        scale=args.scale,
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Total steps: {len(train_loader) * args.epochs:,}")
    
    best_psnr = 0
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'grad': f"{metrics['grad_norm']:.2f}"
            })
        
        avg_loss = np.mean(losses)
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        val_psnr_1step = 0
        val_psnr_10step = 0
        if val_loader:
            val_psnr_1step = trainer.validate(val_loader, num_steps=1)
            val_psnr_10step = trainer.validate(val_loader, num_steps=10)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, LR={current_lr:.2e}, "
                  f"Val PSNR (1-step)={val_psnr_1step:.2f}dB, (10-step)={val_psnr_10step:.2f}dB")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, LR={current_lr:.2e}")
        
        # Update scheduler
        scheduler.step()
        
        # Save best
        save_best = False
        if val_loader and val_psnr_1step > best_psnr:
            best_psnr = val_psnr_1step
            save_best = True
        elif not val_loader and avg_loss < best_loss:
            best_loss = avg_loss
            save_best = True
        
        if save_best:
            trainer.save_checkpoint(
                os.path.join(args.save_dir, 'best_model.pt'),
                epoch, avg_loss, val_psnr_1step, val_psnr_10step
            )
            print(f"  Saved best model!")
        
        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(
                os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt'),
                epoch, avg_loss, val_psnr_1step, val_psnr_10step
            )
    
    # Save final
    trainer.save_checkpoint(
        os.path.join(args.save_dir, 'final_model.pt'),
        args.epochs - 1, avg_loss, val_psnr_1step, val_psnr_10step
    )
    
    print(f"\nTraining complete!")
    print(f"Best 1-step PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()