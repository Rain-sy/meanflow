"""
MeanFlow-DiT v2: Improved Training with Proper MeanFlow Loss

Key improvements over v1:
    1. JVP-based loss computation: u_tgt = v - h * du/dt
    2. Correct time sampling: 75% h=0 (instantaneous), 25% h>0 (average)
    3. Adaptive loss weighting: loss / (loss + eps)
    4. Logit-normal time distribution (matching original MeanFlow)

Reference: Original MeanFlow implementation (JAX/Flax)

Usage:
    python train_meanflow_dit_v2.py \
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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj_drop(self.proj(x))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), drop=drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class MeanFlowDiT(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_channels=3, hidden_size=512, depth=12, num_heads=8, mlp_ratio=4.0, drop_rate=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio, drop_rate) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.view([self.patch_embed.proj.weight.shape[0], -1]))
        nn.init.zeros_(self.patch_embed.proj.bias)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        return torch.einsum('nhwpqc->nchpwq', x).reshape(x.shape[0], self.in_channels, h * p, w * p)
    
    def forward(self, x, t, h):
        x = self.patch_embed(x) + self.pos_embed
        c = self.t_embedder(t) + self.h_embedder(h)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))


# Model configs
def MeanFlowDiT_XS(img_size=128, **kwargs):
    return MeanFlowDiT(img_size, patch_size=4, hidden_size=256, depth=8, num_heads=4, **kwargs)

def MeanFlowDiT_S(img_size=128, **kwargs):
    return MeanFlowDiT(img_size, patch_size=4, hidden_size=384, depth=12, num_heads=6, **kwargs)

def MeanFlowDiT_B(img_size=128, **kwargs):
    return MeanFlowDiT(img_size, patch_size=4, hidden_size=512, depth=12, num_heads=8, **kwargs)

def MeanFlowDiT_L(img_size=128, **kwargs):
    return MeanFlowDiT(img_size, patch_size=4, hidden_size=768, depth=24, num_heads=12, **kwargs)

MODEL_CONFIGS = {'xs': MeanFlowDiT_XS, 'small': MeanFlowDiT_S, 'base': MeanFlowDiT_B, 'large': MeanFlowDiT_L}


# ============================================================================
# MeanFlow Trainer V2 - with proper loss
# ============================================================================

class MeanFlowDiTTrainerV2:
    """
    MeanFlow-DiT Trainer V2 with proper MeanFlow loss
    
    Key improvements:
        1. JVP-based loss: u_tgt = v - h * du/dt
        2. Time sampling: 75% h=0, 25% h>0 (matching original MeanFlow)
        3. Adaptive loss weighting
        4. Logit-normal time distribution
    """
    
    def __init__(self, model, device, lr=1e-4, weight_decay=0.01,
                 data_proportion=0.75, P_mean=-0.4, P_std=1.0, norm_eps=0.01, use_jvp_loss=True):
        self.model = model.to(device)
        self.device = device
        self.data_proportion = data_proportion
        self.P_mean = P_mean
        self.P_std = P_std
        self.norm_eps = norm_eps
        self.use_jvp_loss = use_jvp_loss
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        self.ema_decay = 0.9999
        self.ema_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    def update_ema(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def sample_time(self, batch_size):
        """
        Sample (t, h) following original MeanFlow:
        - Logit-normal distribution for t and r
        - 75% samples: h = 0 (instantaneous velocity)
        - 25% samples: h > 0 (average velocity)
        """
        # Logit-normal sampling
        rnd_t = torch.randn(batch_size, device=self.device)
        t = torch.sigmoid(rnd_t * self.P_std + self.P_mean)
        
        rnd_r = torch.randn(batch_size, device=self.device)
        r = torch.sigmoid(rnd_r * self.P_std + self.P_mean)
        
        # Ensure t >= r
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        
        # 75% h=0, 25% h>0
        data_size = int(batch_size * self.data_proportion)
        mask = torch.arange(batch_size, device=self.device) < data_size
        r = torch.where(mask, t, r)
        
        h = t - r  # h = 0 for first 75%, h > 0 for rest
        return t, h
    
    def compute_loss(self, hr, lr):
        """
        Compute MeanFlow loss with JVP
        
        u_tgt = v - h * du/dt
        
        For h=0: u_tgt = v (instantaneous velocity)
        For h>0: u_tgt = v - h * du/dt (corrected average velocity)
        """
        batch_size = hr.shape[0]
        
        # Sample time
        t, h = self.sample_time(batch_size)
        
        # Expand for broadcasting
        t_exp = t[:, None, None, None]
        h_exp = h[:, None, None, None]
        
        # Interpolate: z_t = (1-t) * HR + t * LR
        z_t = (1 - t_exp) * hr + t_exp * lr
        
        # Target velocity: v = LR - HR
        v = lr - hr
        
        # Forward pass
        u = self.model(z_t, t, h)
        
        if self.use_jvp_loss and (h > 1e-6).any():
            # Compute du/dt using finite difference for h > 0 samples
            eps = 1e-4
            
            with torch.no_grad():
                # z_t changes with t: dz_t/dt = LR - HR = v
                t_plus = t + eps
                t_plus_exp = t_plus[:, None, None, None]
                z_t_plus = (1 - t_plus_exp) * hr + t_plus_exp * lr
                
                t_minus = t - eps
                t_minus_exp = t_minus[:, None, None, None]
                z_t_minus = (1 - t_minus_exp) * hr + t_minus_exp * lr
            
            # Forward at t+eps and t-eps
            u_plus = self.model(z_t_plus, t_plus, h)
            u_minus = self.model(z_t_minus, t_minus, h)
            
            # du/dt ≈ (u(t+eps) - u(t-eps)) / (2*eps)
            du_dt = (u_plus - u_minus) / (2 * eps)
            
            # Target: u_tgt = v - h * du/dt
            u_tgt = v - h_exp * du_dt
            u_tgt = u_tgt.detach()
        else:
            # Simple case: all h ≈ 0, target is just v
            u_tgt = v
        
        # Loss per sample (sum over pixels)
        loss_per_sample = ((u - u_tgt) ** 2).sum(dim=(1, 2, 3))
        
        # Adaptive weighting (matching original MeanFlow)
        with torch.no_grad():
            adp_wt = loss_per_sample.detach() + self.norm_eps
        loss_weighted = loss_per_sample / adp_wt
        
        # v_loss for monitoring (how close is u to v)
        v_loss = ((u - v) ** 2).sum(dim=(1, 2, 3)).mean()
        
        return loss_weighted.mean(), v_loss, h.mean(), (h > 1e-6).float().mean()
    
    def train_step(self, batch):
        self.model.train()
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        
        loss, v_loss, h_mean, h_nz_ratio = self.compute_loss(hr, lr)
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.update_ema()
        
        return {
            'loss': loss.item(),
            'v_loss': v_loss.item(),
            'h_mean': h_mean.item(),
            'h_nz_ratio': h_nz_ratio.item(),
            'grad_norm': grad_norm.item(),
        }
    
    @torch.no_grad()
    def inference(self, lr_images, num_steps=1, use_ema=True):
        self.model.eval()
        
        if use_ema:
            orig_params = {name: param.clone() for name, param in self.model.named_parameters()}
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
        
        if use_ema:
            for name, param in self.model.named_parameters():
                param.data.copy_(orig_params[name])
        
        return x
    
    @torch.no_grad()
    def validate(self, val_loader, num_steps=1):
        self.model.eval()
        psnr_list = []
        
        for batch in val_loader:
            hr = batch['hr'].to(self.device)
            lr = batch['lr'].to(self.device)
            hr_pred = self.inference(lr, num_steps=num_steps, use_ema=True)
            
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
    parser = argparse.ArgumentParser(description='MeanFlow-DiT V2 Training')
    
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
    
    # MeanFlow hyperparameters
    parser.add_argument('--data_proportion', type=float, default=0.75, help='Proportion of h=0 samples')
    parser.add_argument('--P_mean', type=float, default=-0.4, help='Logit-normal mean')
    parser.add_argument('--P_std', type=float, default=1.0, help='Logit-normal std')
    parser.add_argument('--norm_eps', type=float, default=0.01, help='Adaptive weighting epsilon')
    parser.add_argument('--no_jvp', action='store_true', help='Disable JVP loss')
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints_dit_v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_every_steps', type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("MeanFlow-DiT V2 Training")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model_size}, Patch: {args.patch_size}")
    print(f"Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Data proportion (h=0): {args.data_proportion}")
    print(f"Logit-normal: mean={args.P_mean}, std={args.P_std}")
    print(f"JVP loss: {'Disabled' if args.no_jvp else 'Enabled'}")
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
    
    trainer = MeanFlowDiTTrainerV2(
        model, device, args.lr, args.weight_decay,
        args.data_proportion, args.P_mean, args.P_std, args.norm_eps,
        use_jvp_loss=not args.no_jvp
    )
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_psnr = 0
    best_loss = float('inf')
    global_step = start_epoch * len(train_loader)
    
    for epoch in range(start_epoch, args.epochs):
        losses, v_losses = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            v_losses.append(metrics['v_loss'])
            global_step += 1
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'v_loss': f"{metrics['v_loss']:.4f}",
                'h_nz': f"{metrics['h_nz_ratio']:.2f}"
            })
            
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                trainer.save_checkpoint(os.path.join(args.save_dir, 'checkpoint_latest.pt'), epoch, metrics['loss'])
        
        avg_loss = np.mean(losses)
        avg_v_loss = np.mean(v_losses)
        
        val_psnr_1step, val_psnr_10step = 0, 0
        if val_loader:
            val_psnr_1step = trainer.validate(val_loader, num_steps=1)
            val_psnr_10step = trainer.validate(val_loader, num_steps=10)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, V_Loss={avg_v_loss:.4f}, "
                  f"PSNR 1-step={val_psnr_1step:.2f}dB, 10-step={val_psnr_10step:.2f}dB")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, V_Loss={avg_v_loss:.4f}")
        
        scheduler.step()
        
        if (val_loader and val_psnr_1step > best_psnr) or (not val_loader and avg_loss < best_loss):
            best_psnr = val_psnr_1step if val_loader else best_psnr
            best_loss = avg_loss if not val_loader else best_loss
            trainer.save_checkpoint(os.path.join(args.save_dir, 'best_model.pt'), epoch, avg_loss, val_psnr_1step, val_psnr_10step)
            print("  Saved best!")
        
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt'), epoch, avg_loss, val_psnr_1step, val_psnr_10step)
    
    trainer.save_checkpoint(os.path.join(args.save_dir, 'final_model.pt'), args.epochs - 1, avg_loss, val_psnr_1step, val_psnr_10step)
    print(f"\nDone! Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()