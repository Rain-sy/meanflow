"""
MeanFlow-DiT (U-ViT Style) Evaluation Script [Fixed]

Fixes:
1. "Gray Border": Uses the robust tiling/cropping logic from evaluate_sr.py.
2. "Blurry": Ensures correct overlap blending and normalization.
3. Architecture: Matched exactly to train_meanflow_dit_uvit.py.

Usage:
    python meanflow/evaluate_sr_uvit_fixed.py \
        --checkpoint checkpoints_dit_uvit/best_model.pt \
        --hr_dir "meanflow/Data/DIV2K/DIV2K_valid_HR" \
        --lr_dir "meanflow/Data/DIV2K/DIV2K_valid_LR_bicubic_X2" \
        --model_size small \
        --tile_size 256 \
        --overlap 32
"""

import os
import re
import math
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. U-ViT Model Architecture (Exact Copy from Training)
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

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

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # Flash Attention Optimization
        if hasattr(F, 'scaled_dot_product_attention'):
            q, k, v = self.attn.in_proj_weight.split(self.attn.embed_dim, dim=0)
            q = F.linear(x_norm, q, self.attn.in_proj_bias[:self.attn.embed_dim])
            k = F.linear(x_norm, k, self.attn.in_proj_bias[self.attn.embed_dim:2*self.attn.embed_dim])
            v = F.linear(x_norm, v, self.attn.in_proj_bias[2*self.attn.embed_dim:])
            
            B, L, _ = x_norm.shape
            q = q.view(B, L, self.attn.num_heads, -1).transpose(1, 2)
            k = k.view(B, L, self.attn.num_heads, -1).transpose(1, 2)
            v = v.view(B, L, self.attn.num_heads, -1).transpose(1, 2)
            
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
            attn_out = self.attn.out_proj(attn_out)
        else:
             attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x

class MeanFlowDiT(nn.Module):
    def __init__(self, img_size=128, patch_size=2, in_channels=3, hidden_size=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # Dynamic Pos Embed logic
        pos_embed = get_2d_sincos_pos_embed(hidden_size, int(img_size//patch_size))
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.pred_head = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def unpatchify(self, x):
        p = self.patch_size
        c = self.in_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(self, x, t, h):
        x_in = self.x_embedder(x).flatten(2).transpose(1, 2)
        
        # Dynamic Pos Embed
        if x_in.shape[1] != self.pos_embed.shape[1]:
            grid_size = int(x_in.shape[1] ** 0.5)
            new_pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size)
            new_pos_embed = torch.from_numpy(new_pos_embed).float().unsqueeze(0).to(x.device)
            x = x_in + new_pos_embed
        else:
            x = x_in + self.pos_embed

        c = self.t_embedder(t) + self.h_embedder(h)
        
        skips = []
        half_depth = self.depth // 2
        
        for i, block in enumerate(self.blocks):
            if i < half_depth:
                x = block(x, c)
                skips.append(x)
            else:
                skip = skips.pop()
                x = x + skip
                x = block(x, c)
        
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.pred_head(x)
        return self.unpatchify(x)

# Model Configs
def MeanFlowDiT_XS(**kwargs): return MeanFlowDiT(hidden_size=256, depth=8, num_heads=4, **kwargs)
def MeanFlowDiT_S(**kwargs): return MeanFlowDiT(hidden_size=384, depth=12, num_heads=6, **kwargs)
def MeanFlowDiT_B(**kwargs): return MeanFlowDiT(hidden_size=512, depth=12, num_heads=8, **kwargs)
def MeanFlowDiT_L(**kwargs): return MeanFlowDiT(hidden_size=768, depth=24, num_heads=12, **kwargs)

MODEL_CONFIGS = {'xs': MeanFlowDiT_XS, 'small': MeanFlowDiT_S, 'base': MeanFlowDiT_B, 'large': MeanFlowDiT_L}

# ============================================================================
# 2. Metrics (From evaluate_sr.py)
# ============================================================================

def calculate_psnr(img1, img2, max_val=255.0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.ndim == 3:
        ssim_vals = [calculate_ssim(img1[:,:,c], img2[:,:,c]) for c in range(img1.shape[2])]
        return np.mean(ssim_vals)
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

def match_hr_lr_files(hr_dir, lr_dir):
    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*')))
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*')))
    # Simple matching assuming sorted order is correct. 
    # Can use complex ID matching if filenames differ significantly.
    min_len = min(len(hr_files), len(lr_files))
    return list(zip(hr_files[:min_len], lr_files[:min_len]))

# ============================================================================
# 3. Robust Inference (From evaluate_sr.py)
# ============================================================================

def pad_to_multiple(x, multiple=2):
    h, w = x.shape[2], x.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, (pad_h, pad_w)

@torch.no_grad()
def run_sr_direct(model, lr_tensor, device, num_steps=1, pad_multiple=2):
    lr_padded, (pad_h, pad_w) = pad_to_multiple(lr_tensor, multiple=pad_multiple)
    batch_size = lr_padded.shape[0]
    x = lr_padded
    
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = 1.0 - i * dt
        t = torch.full((batch_size,), t_val, device=device)
        h = torch.full((batch_size,), dt, device=device)
        u = model(x, t, h)
        x = x - dt * u
    
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :x.shape[2]-pad_h, :x.shape[3]-pad_w]
    return x

@torch.no_grad()
def run_sr_tiled(model, lr_tensor, device, num_steps=1, tile_size=256, overlap=32, pad_multiple=2):
    """
    Robust tiling logic from evaluate_sr.py
    Ensures no border artifacts and correct blending.
    """
    _, _, H, W = lr_tensor.shape
    
    # Direct inference if small
    if H <= tile_size and W <= tile_size:
        return run_sr_direct(model, lr_tensor, device, num_steps, pad_multiple)
    
    output = torch.zeros_like(lr_tensor)
    weight = torch.zeros_like(lr_tensor)
    
    stride = tile_size - overlap
    
    # Calculate tile starts
    h_starts = list(range(0, max(1, H - tile_size + 1), stride))
    if h_starts[-1] + tile_size < H: h_starts.append(max(0, H - tile_size))
    
    w_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if w_starts[-1] + tile_size < W: w_starts.append(max(0, W - tile_size))
    
    for h_start in h_starts:
        for w_start in w_starts:
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)
            
            # Extract tile
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            th, tw = tile.shape[2], tile.shape[3]
            
            # Pad tile to tile_size if edge is smaller
            if th < tile_size or tw < tile_size:
                tile = F.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode='reflect')
            
            # Pad for Model Patch (e.g. div by 2)
            tile_padded, (pad_h, pad_w) = pad_to_multiple(tile, multiple=pad_multiple)
            
            # Inference
            x = tile_padded
            batch_size = x.shape[0]
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t_val = 1.0 - i * dt
                t = torch.full((batch_size,), t_val, device=device)
                h_param = torch.full((batch_size,), dt, device=device)
                u = model(x, t, h_param)
                x = x - dt * u
            
            # Remove Model Padding
            if pad_h > 0 or pad_w > 0:
                x = x[:, :, :x.shape[2]-pad_h, :x.shape[3]-pad_w]
            
            # Crop back to tile size (removes the reflective padding)
            tile_output = x[:, :, :th, :tw]
            
            # Weight Mask (Only blend overlaps, preserve borders)
            tile_weight = torch.ones(1, 1, th, tw, device=device)
            
            if overlap > 0:
                if h_start > 0: # Top overlap
                    for i in range(min(overlap, th)): tile_weight[:,:,i,:] *= i/overlap
                if h_end < H:   # Bottom overlap
                    for i in range(min(overlap, th)): tile_weight[:,:,th-1-i,:] *= i/overlap
                if w_start > 0: # Left overlap
                    for i in range(min(overlap, tw)): tile_weight[:,:,:,i] *= i/overlap
                if w_end < W:   # Right overlap
                    for i in range(min(overlap, tw)): tile_weight[:,:,:,tw-1-i] *= i/overlap
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_output * tile_weight
            weight[:, :, h_start:h_end, w_start:w_end] += tile_weight
            
    return output / (weight + 1e-8)

# ============================================================================
# 4. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./meanflow/outputs')
    parser.add_argument('--num_steps', type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)
    
    # 1. Load Model
    print(f"Loading {args.model_size} model...")
    model = MODEL_CONFIGS[args.model_size](img_size=128, patch_size=2)
    
    # Weights_only=False to fix the warning/error
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    
    # Load EMA if exists (Important for quality)
    if 'ema' in ckpt:
        print("Using EMA weights (better quality)...")
        ema_params = ckpt['ema']
        for name, param in model.named_parameters():
            if name in ema_params: param.data.copy_(ema_params[name])
            
    model.to(device)
    model.eval()
    
    # 2. Pairs
    pairs = match_hr_lr_files(args.hr_dir, args.lr_dir)
    print(f"Evaluating on {len(pairs)} images...")
    
    psnrs, ssims = [], []
    psnrs_bic, ssims_bic = [], []
    
    # 3. Loop
    for hr_path, lr_path in tqdm(pairs):
        name = os.path.basename(hr_path)
        base_name = os.path.splitext(name)[0]
        
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Resize logic to match scale
        w, h = lr_img.size
        target_w, target_h = w * args.scale, h * args.scale
        if hr_img.size != (target_w, target_h):
            hr_img = hr_img.resize((target_w, target_h), Image.BICUBIC)
        
        # Bicubic Baseline
        lr_bic = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_bic_np = np.array(lr_bic)
        hr_np = np.array(hr_img)
        
        # Prepare Input (Norm: -1 to 1)
        lr_up = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_tensor = torch.from_numpy(np.array(lr_up)).float().permute(2,0,1).unsqueeze(0).to(device)
        lr_tensor = lr_tensor / 127.5 - 1.0
        
        # Inference
        try:
            pred_t = run_sr_tiled(model, lr_tensor, device, args.num_steps, 
                                  args.tile_size, args.overlap, pad_multiple=2)
            
            pred_np = ((pred_t.squeeze(0).permute(1,2,0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Metrics
            p = calculate_psnr(pred_np, hr_np)
            s = calculate_ssim(pred_np, hr_np)
            p_b = calculate_psnr(lr_bic_np, hr_np)
            s_b = calculate_ssim(lr_bic_np, hr_np)
            
            psnrs.append(p)
            ssims.append(s)
            psnrs_bic.append(p_b)
            ssims_bic.append(s_b)
            
            # Save
            Image.fromarray(pred_np).save(os.path.join(args.output_dir, 'predictions', name))
            
        except RuntimeError as e:
            print(f"Error on {name}: {e}")
            torch.cuda.empty_cache()
            
    print("="*60)
    print(f"MeanFlow-DiT Results (Scale x{args.scale})")
    print(f"Bicubic: PSNR={np.mean(psnrs_bic):.4f}, SSIM={np.mean(ssims_bic):.4f}")
    print(f"Ours:    PSNR={np.mean(psnrs):.4f}, SSIM={np.mean(ssims):.4f}")
    print(f"Gain:    {np.mean(psnrs)-np.mean(psnrs_bic):+.4f} dB")
    print("="*60)

if __name__ == '__main__':
    main()