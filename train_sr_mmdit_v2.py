"""
MeanFlow-MMDiT V2: Window Attention + Auto Evaluation

Usage:
    python train_sr_mmdit_v2.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --model_size small --scale 4 \
        --use_window_attn \
        --auto_eval
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
        
        # Random crop
        w, h = hr_img.size
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            hr_img = hr_img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            lr_up = lr_up.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = hr_img.size
        x, y = random.randint(0, w - self.patch_size), random.randint(0, h - self.patch_size)
        hr_img = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        lr_up = lr_up.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5: hr_img, lr_up = hr_img.transpose(Image.FLIP_LEFT_RIGHT), lr_up.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5: hr_img, lr_up = hr_img.transpose(Image.FLIP_TOP_BOTTOM), lr_up.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                hr_img, lr_up = hr_img.rotate(90*k), lr_up.rotate(90*k)
        
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2,0,1) / 127.5 - 1.0
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2,0,1) / 127.5 - 1.0
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Model Components
# ============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

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

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ============================================================================
# Window Attention
# ============================================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid([torch.arange(window_size), torch.arange(window_size)], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = (attn + rel_bias.unsqueeze(0)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B_, N, C))


# ============================================================================
# MMDiT Block
# ============================================================================

class MMDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_window_attn=False, window_size=8):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        self.use_window_attn, self.window_size = use_window_attn, window_size
        
        self.norm1_img, self.norm1_cond = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.norm2_img, self.norm2_cond = RMSNorm(hidden_size), RMSNorm(hidden_size)
        
        if use_window_attn:
            self.attn_img = WindowAttention(hidden_size, window_size, num_heads)
            self.attn_cond = WindowAttention(hidden_size, window_size, num_heads)
        else:
            self.qkv_img = nn.Linear(hidden_size, hidden_size * 3)
            self.qkv_cond = nn.Linear(hidden_size, hidden_size * 3)
            self.proj_img = nn.Linear(hidden_size, hidden_size)
            self.proj_cond = nn.Linear(hidden_size, hidden_size)
        
        self.mlp_img = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.mlp_cond = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
    
    def forward(self, x_img, x_cond, c):
        B, N, C = x_img.shape
        H = W = int(math.sqrt(N))
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=1)
        xi = self.norm1_img(x_img) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        xc = self.norm1_cond(x_cond) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        if self.use_window_attn:
            xi = xi.view(B, H, W, C)
            xc = xc.view(B, H, W, C)
            xi_win = window_partition(xi, self.window_size).view(-1, self.window_size**2, C)
            xc_win = window_partition(xc, self.window_size).view(-1, self.window_size**2, C)
            attn_i = window_reverse(self.attn_img(xi_win).view(-1, self.window_size, self.window_size, C), self.window_size, H, W).view(B, N, C)
            attn_c = window_reverse(self.attn_cond(xc_win).view(-1, self.window_size, self.window_size, C), self.window_size, H, W).view(B, N, C)
        else:
            qkv_i = self.qkv_img(xi).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            qkv_c = self.qkv_cond(xc).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q_i, k_i, v_i = qkv_i.unbind(0)
            q_c, k_c, v_c = qkv_c.unbind(0)
            k_joint, v_joint = torch.cat([k_i, k_c], dim=2), torch.cat([v_i, v_c], dim=2)
            attn_i = self.proj_img(F.scaled_dot_product_attention(q_i, k_joint, v_joint).transpose(1, 2).reshape(B, -1, C))
            attn_c = self.proj_cond(F.scaled_dot_product_attention(q_c, k_joint, v_joint).transpose(1, 2).reshape(B, -1, C))
        
        x_img = x_img + gate_msa.unsqueeze(1) * attn_i
        x_cond = x_cond + gate_msa.unsqueeze(1) * attn_c
        x_img = x_img + gate_mlp.unsqueeze(1) * self.mlp_img(self.norm2_img(x_img) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        x_cond = x_cond + gate_mlp.unsqueeze(1) * self.mlp_cond(self.norm2_cond(x_cond) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        return x_img, x_cond


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class MeanFlowMMDiT(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_channels=3, hidden_size=512, 
                 depth=12, num_heads=8, mlp_ratio=4.0, use_window_attn=False, window_size=8):
        super().__init__()
        self.patch_size, self.in_channels, self.hidden_size = patch_size, in_channels, hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed_img = PatchEmbed(patch_size, in_channels, hidden_size)
        self.patch_embed_cond = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_size, num_heads, mlp_ratio, use_window_attn and (i % 2 == 1), window_size)
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        return torch.einsum('nhwpqc->nchpwq', x).reshape(x.shape[0], self.in_channels, h * p, w * p)
    
    def forward(self, z_t, lr_cond, t, h):
        x_img = self.patch_embed_img(z_t) + self.pos_embed
        x_cond = self.patch_embed_cond(lr_cond) + self.pos_embed
        c = self.t_embedder(t) + self.h_embedder(h)
        for block in self.blocks:
            x_img, x_cond = block(x_img, x_cond, c)
        return self.unpatchify(self.final_layer(x_img, c))


def MMDiT_XS(img_size=128, **kw): return MeanFlowMMDiT(img_size, hidden_size=256, depth=8, num_heads=4, **kw)
def MMDiT_S(img_size=128, **kw): return MeanFlowMMDiT(img_size, hidden_size=384, depth=12, num_heads=6, **kw)
def MMDiT_B(img_size=128, **kw): return MeanFlowMMDiT(img_size, hidden_size=512, depth=12, num_heads=8, **kw)
def MMDiT_L(img_size=128, **kw): return MeanFlowMMDiT(img_size, hidden_size=768, depth=24, num_heads=12, **kw)
MODEL_CONFIGS = {'xs': MMDiT_XS, 'small': MMDiT_S, 'base': MMDiT_B, 'large': MMDiT_L}


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    def __init__(self, model, device, lr=1e-4, weight_decay=0.01,
                 P_mean=-0.4, P_std=1.0, prop_h0=0.6, prop_havg=0.2, prop_onestep=0.2):
        self.model = model.to(device)
        self.device = device
        self.P_mean, self.P_std = P_mean, P_std
        self.prop_h0, self.prop_havg, self.prop_onestep = prop_h0, prop_havg, prop_onestep
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        self.ema_decay = 0.9999
        self.ema_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    def update_ema(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.ema_params[n].mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def sample_time(self, bs):
        n0, na = int(bs * self.prop_h0), int(bs * self.prop_havg)
        n1 = bs - n0 - na
        t, h = torch.zeros(bs, device=self.device), torch.zeros(bs, device=self.device)
        if n0 > 0:
            t[:n0] = torch.sigmoid(torch.randn(n0, device=self.device) * self.P_std + self.P_mean)
        if na > 0:
            t1 = torch.sigmoid(torch.randn(na, device=self.device) * self.P_std + self.P_mean)
            t2 = torch.sigmoid(torch.randn(na, device=self.device) * self.P_std + self.P_mean)
            t[n0:n0+na], h[n0:n0+na] = torch.maximum(t1, t2), torch.abs(t1 - t2)
        if n1 > 0:
            t[-n1:], h[-n1:] = 1.0, 1.0
        perm = torch.randperm(bs, device=self.device)
        return t[perm], h[perm]
    
    def train_step(self, batch):
        self.model.train()
        hr, lr = batch['hr'].to(self.device), batch['lr'].to(self.device)
        t, h = self.sample_time(hr.shape[0])
        t_exp = t[:, None, None, None]
        z_t = (1 - t_exp) * hr + t_exp * lr
        v_target = lr - hr
        
        u = self.model(z_t, lr, t, h)
        loss = F.mse_loss(u, v_target)
        
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
            hr, lr = batch['hr'].to(self.device), batch['lr'].to(self.device)
            pred = self.inference(lr, num_steps, use_ema=True)
            hr_01, pred_01 = (hr + 1) / 2, ((pred + 1) / 2).clamp(0, 1)
            mse = ((hr_01 - pred_01) ** 2).mean(dim=[1, 2, 3])
            psnrs.extend((10 * torch.log10(1.0 / (mse + 1e-8))).cpu().tolist())
        return np.mean(psnrs)
    
    def save(self, path, epoch, loss, psnr=0):
        torch.save({'epoch': epoch, 'model': self.model.state_dict(), 'ema': self.ema_params,
                    'optimizer': self.optimizer.state_dict(), 'loss': loss, 'psnr': psnr}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt: self.ema_params = ckpt['ema']
        if 'optimizer' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)


# ============================================================================
# Training Summary
# ============================================================================

def save_training_summary(save_dir, args, best_psnr, total_time, total_iters, iter_times):
    summary_path = os.path.join(save_dir, 'training_summary.txt')
    avg_iter_time = np.mean(iter_times) if iter_times else 0
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MeanFlow-MMDiT V2 Training Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Best PSNR: {best_psnr:.4f} dB\n\n")
        f.write(f"Training time: {total_time/3600:.2f} hours\n")
        f.write(f"Iterations: {total_iters}\n")
        f.write(f"Speed: {1/avg_iter_time:.2f} it/s ({avg_iter_time*1000:.1f} ms/it)\n\n")
        f.write(f"Model: {args.model_size}, Scale: {args.scale}x\n")
        f.write(f"Window Attention: {args.use_window_attn} (size={args.window_size})\n")
        f.write(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}\n")
        f.write("="*60 + "\n")
    print(f"Summary saved: {summary_path}")


# ============================================================================
# Auto Evaluation
# ============================================================================

def run_evaluation(checkpoint_path, hr_dir, lr_dir, model_size, scale, use_window_attn, window_size, 
                   device, output_base, tile_size=128, overlap=32):
    """Run evaluation on the trained model"""
    print("\n" + "="*60)
    print("Running Auto Evaluation...")
    print("="*60)
    
    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MODEL_CONFIGS[model_size](tile_size, use_window_attn=use_window_attn, window_size=window_size)
    
    state_dict = ckpt.get('ema', ckpt['model'])
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    
    # Get files
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Metrics
    psnr_list, psnr_bic_list = [], []
    
    for hf, lf in tqdm(zip(hr_files, lr_files), total=len(hr_files), desc="Evaluating"):
        hr_img = Image.open(os.path.join(hr_dir, hf)).convert('RGB')
        lr_img = Image.open(os.path.join(lr_dir, lf)).convert('RGB')
        hr_np = np.array(hr_img)
        H, W = hr_np.shape[:2]
        
        lr_bicubic_np = np.array(lr_img.resize((W, H), Image.BICUBIC))
        lr_up = lr_img.resize((W, H), Image.BICUBIC)
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device)
        
        # Tiled inference with Gaussian blending
        pred = tiled_inference(model, lr_t, device, tile_size, overlap)
        pred_np = ((pred[0].cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
        
        # PSNR
        psnr_val = 10 * np.log10(255**2 / np.mean((pred_np.astype(float) - hr_np.astype(float))**2))
        psnr_bic = 10 * np.log10(255**2 / np.mean((lr_bicubic_np.astype(float) - hr_np.astype(float))**2))
        psnr_list.append(psnr_val)
        psnr_bic_list.append(psnr_bic)
    
    avg_psnr = np.mean(psnr_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    
    print(f"\nResults on {len(psnr_list)} images:")
    print(f"  Bicubic: {avg_psnr_bic:.4f} dB")
    print(f"  Model:   {avg_psnr:.4f} dB (+{avg_psnr - avg_psnr_bic:.4f} dB)")
    
    return avg_psnr


@torch.no_grad()
def tiled_inference(model, lr_tensor, device, tile_size=128, overlap=32):
    """Tiled inference with Gaussian blending"""
    _, _, H, W = lr_tensor.shape
    
    if H <= tile_size and W <= tile_size:
        lr_padded = F.pad(lr_tensor, (0, (4 - W % 4) % 4, 0, (4 - H % 4) % 4), mode='reflect')
        x = lr_padded.clone()
        t = torch.ones(1, device=device)
        h = torch.ones(1, device=device)
        x = x - model(x, lr_padded, t, h)
        return x[:, :, :H, :W]
    
    output = torch.zeros_like(lr_tensor)
    weight = torch.zeros_like(lr_tensor)
    stride = tile_size - overlap
    
    # Gaussian weight
    sigma = tile_size / 6
    y = torch.arange(tile_size, device=device).float() - tile_size / 2
    x = torch.arange(tile_size, device=device).float() - tile_size / 2
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2)).unsqueeze(0).unsqueeze(0)
    
    h_starts = list(range(0, max(1, H - tile_size + 1), stride))
    if h_starts[-1] + tile_size < H: h_starts.append(max(0, H - tile_size))
    w_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if w_starts[-1] + tile_size < W: w_starts.append(max(0, W - tile_size))
    
    for h_start in h_starts:
        for w_start in w_starts:
            h_end, w_end = min(h_start + tile_size, H), min(w_start + tile_size, W)
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            th, tw = tile.shape[2], tile.shape[3]
            
            if th < tile_size or tw < tile_size:
                tile = F.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode='reflect')
            
            tile_padded = F.pad(tile, (0, (4 - tile.shape[3] % 4) % 4, 0, (4 - tile.shape[2] % 4) % 4), mode='reflect')
            x = tile_padded.clone()
            t = torch.ones(1, device=device)
            h_param = torch.ones(1, device=device)
            x = x - model(x, tile_padded, t, h_param)
            x = x[:, :, :tile_size, :tile_size]
            
            tile_out = x[:, :, :th, :tw]
            tile_weight = gaussian[:, :, :th, :tw]
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_out * tile_weight
            weight[:, :, h_start:h_end, w_start:w_end] += tile_weight
    
    return output / (weight + 1e-8)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow-MMDiT V2')
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=4)
    
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--use_window_attn', action='store_true')
    parser.add_argument('--window_size', type=int, default=8)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    parser.add_argument('--prop_h0', type=float, default=0.6)
    parser.add_argument('--prop_havg', type=float, default=0.2)
    parser.add_argument('--prop_onestep', type=float, default=0.2)
    
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_base', type=str, default='./checkpoints_mmdit_v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    
    # Auto evaluation
    parser.add_argument('--auto_eval', action='store_true', help='Run evaluation after training')
    parser.add_argument('--eval_overlap', type=int, default=32, help='Tile overlap for evaluation')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Save dir
    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        feat = f"win{args.window_size}" if args.use_window_attn else "baseline"
        args.save_dir = os.path.join(args.save_base, f"{ts}_{args.model_size}_x{args.scale}_{feat}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("MeanFlow-MMDiT V2")
    print("="*60)
    print(f"Model: {args.model_size} | Scale: {args.scale}x | Patch: {args.patch_size}")
    print(f"Window Attention: {args.use_window_attn} (size={args.window_size})")
    print(f"Auto Eval: {args.auto_eval}")
    print(f"Save: {args.save_dir}")
    print("="*60)
    
    # Data
    train_data = SRDataset(args.hr_dir, args.lr_dir, args.patch_size, args.scale, True, 5)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_data = SRDataset(args.val_hr_dir, args.val_lr_dir, args.patch_size, args.scale, False, 1)
        val_loader = DataLoader(val_data, 4, shuffle=False, num_workers=2)
    
    # Model & Trainer
    model = MODEL_CONFIGS[args.model_size](args.patch_size, use_window_attn=args.use_window_attn, window_size=args.window_size)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, device, args.lr, args.weight_decay, prop_h0=args.prop_h0, prop_havg=args.prop_havg, prop_onestep=args.prop_onestep)
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training
    best_psnr, total_iters, iter_times = 0, 0, []
    train_start = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            iter_start = time.time()
            m = trainer.train_step(batch)
            iter_times.append(time.time() - iter_start)
            total_iters += 1
            losses.append(m['loss'])
            pbar.set_postfix({'loss': f"{m['loss']:.4f}", 'it/s': f"{1/iter_times[-1]:.1f}"})
        
        scheduler.step()
        avg_loss = np.mean(losses)
        
        if val_loader:
            psnr = trainer.validate(val_loader)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={psnr:.2f}dB")
            if psnr > best_psnr:
                best_psnr = psnr
                trainer.save(os.path.join(args.save_dir, 'best_model.pt'), epoch, avg_loss, psnr)
                print("  -> Saved best!")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}")
        
        if (epoch + 1) % 20 == 0:
            trainer.save(os.path.join(args.save_dir, f'epoch{epoch+1}.pt'), epoch, avg_loss, best_psnr)
    
    total_time = time.time() - train_start
    trainer.save(os.path.join(args.save_dir, 'final.pt'), args.epochs - 1, avg_loss, best_psnr)
    save_training_summary(args.save_dir, args, best_psnr, total_time, total_iters, iter_times)
    
    print(f"\nTraining Done! Best PSNR: {best_psnr:.2f} dB, Time: {total_time/3600:.2f}h")
    
    # Auto evaluation
    if args.auto_eval and args.val_hr_dir and args.val_lr_dir:
        best_ckpt = os.path.join(args.save_dir, 'best_model.pt')
        if os.path.exists(best_ckpt):
            eval_psnr = run_evaluation(
                best_ckpt, args.val_hr_dir, args.val_lr_dir,
                args.model_size, args.scale, args.use_window_attn, args.window_size,
                device, args.save_dir, args.patch_size, args.eval_overlap
            )


if __name__ == '__main__':
    main()