"""
MeanFlow-MMDiT Evaluation Script

Output structure:
    meanflow/outputs/
    └── DIV2K/
        └── MMDiT/
            └── small_x4_1step_win8/
                ├── predictions/
                ├── comparisons/
                └── results.txt

Usage:
    python evaluate_sr_mmdit.py \
        --checkpoint checkpoints_mmdit_v2/.../best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --model_size small --scale 4
"""

import os
import math
import argparse
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
    
    def forward(self, x_img, x_cond, c):
        B, N, C = x_img.shape
        H = W = int(math.sqrt(N))
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
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
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class MeanFlowMMDiT(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_channels=3, hidden_size=512, 
                 depth=12, num_heads=8, mlp_ratio=4.0, use_window_attn=False, window_size=8):
        super().__init__()
        self.patch_size, self.in_channels, self.hidden_size = patch_size, in_channels, hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed_img = PatchEmbed(patch_size, in_channels, hidden_size)
        self.patch_embed_cond = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
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
# Metrics
# ============================================================================

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)

def calculate_ssim(img1, img2):
    C1, C2 = 6.5025, 58.5225
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    
    ssims = []
    for i in range(3):
        ch1, ch2 = img1[:,:,i], img2[:,:,i]
        mu1, mu2 = ch1.mean(), ch2.mean()
        sigma1_sq = ((ch1 - mu1) ** 2).mean()
        sigma2_sq = ((ch2 - mu2) ** 2).mean()
        sigma12 = ((ch1 - mu1) * (ch2 - mu2)).mean()
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        ssims.append(ssim)
    return np.mean(ssims)


# ============================================================================
# Tiled Inference
# ============================================================================

@torch.no_grad()
def run_sr_tiled(model, lr_tensor, device, num_steps=1, tile_size=128, overlap=32, blend_mode='gaussian'):
    """Tiled inference with Gaussian blending"""
    _, _, H, W = lr_tensor.shape
    
    # Small image: direct inference
    if H <= tile_size and W <= tile_size:
        pad_h, pad_w = (4 - H % 4) % 4, (4 - W % 4) % 4
        lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        x = lr_padded.clone()
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((1,), 1.0 - i * dt, device=device)
            h = torch.full((1,), dt, device=device)
            x = x - dt * model(x, lr_padded, t, h)
        return x[:, :, :H, :W]
    
    output = torch.zeros_like(lr_tensor)
    weight = torch.zeros_like(lr_tensor)
    stride = tile_size - overlap
    
    # Gaussian weight
    if blend_mode == 'gaussian':
        sigma = tile_size / 6
        y_coords = torch.arange(tile_size, device=device).float() - tile_size / 2
        x_coords = torch.arange(tile_size, device=device).float() - tile_size / 2
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
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
            
            pad_h, pad_w = (4 - tile.shape[2] % 4) % 4, (4 - tile.shape[3] % 4) % 4
            tile_padded = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
            
            x = tile_padded.clone()
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t_val = torch.full((1,), 1.0 - i * dt, device=device)
                h_val = torch.full((1,), dt, device=device)
                x = x - dt * model(x, tile_padded, t_val, h_val)
            
            x = x[:, :, :tile_size, :tile_size]
            tile_out = x[:, :, :th, :tw]
            
            if blend_mode == 'gaussian':
                tile_weight = gaussian[:, :, :th, :tw]
            else:
                tile_weight = torch.ones(1, 1, th, tw, device=device)
                if overlap > 0:
                    for idx in range(min(overlap, th)):
                        if h_start > 0: tile_weight[:, :, idx, :] *= idx / overlap
                        if h_end < H: tile_weight[:, :, th - 1 - idx, :] *= idx / overlap
                    for idx in range(min(overlap, tw)):
                        if w_start > 0: tile_weight[:, :, :, idx] *= idx / overlap
                        if w_end < W: tile_weight[:, :, :, tw - 1 - idx] *= idx / overlap
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_out * tile_weight
            weight[:, :, h_start:h_end, w_start:w_end] += tile_weight
    
    return output / (weight + 1e-8)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow-MMDiT Evaluation')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--use_window_attn', action='store_true')
    parser.add_argument('--window_size', type=int, default=8)
    
    parser.add_argument('--tile_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--blend_mode', type=str, default='gaussian', choices=['gaussian', 'linear'])
    parser.add_argument('--num_steps', type=int, default=1)
    
    # Output - 重要修改：输出到 meanflow/outputs/{dataset}/MMDiT/{exp_name}/
    parser.add_argument('--output_base', type=str, default='./meanflow/outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect dataset from hr_dir
    if args.dataset is None:
        hr_lower = args.hr_dir.lower()
        if 'urban' in hr_lower: args.dataset = 'Urban100'
        elif 'div2k' in hr_lower: args.dataset = 'DIV2K'
        elif 'set5' in hr_lower: args.dataset = 'Set5'
        elif 'set14' in hr_lower: args.dataset = 'Set14'
        elif 'bsd100' in hr_lower or 'b100' in hr_lower: args.dataset = 'BSD100'
        elif 'manga' in hr_lower: args.dataset = 'Manga109'
        else: args.dataset = 'Unknown'
    
    # Load checkpoint first to auto-detect window attention
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('ema', ckpt['model'])
    has_window_attn = any('attn_img.' in k for k in state_dict.keys())
    
    if has_window_attn and not args.use_window_attn:
        print("[Auto] Detected window attention in checkpoint, enabling it")
        args.use_window_attn = True
    elif not has_window_attn and args.use_window_attn:
        print("[Auto] Checkpoint doesn't have window attention, disabling it")
        args.use_window_attn = False
    
    # Build experiment name (after auto-detection)
    if args.exp_name:
        exp_name = args.exp_name
    else:
        parts = [args.model_size, f'x{args.scale}', f'{args.num_steps}step']
        if args.use_window_attn:
            parts.append(f'win{args.window_size}')
        exp_name = '_'.join(parts)
    
    # Output dir: meanflow/outputs/{dataset}/MMDiT/{exp_name}/
    output_dir = os.path.join(args.output_base, args.dataset, 'MMDiT', exp_name)
    
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print("="*70)
    print("MeanFlow-MMDiT Evaluation")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_size}")
    print(f"Window Attention: {args.use_window_attn} (size={args.window_size})")
    print(f"Dataset: {args.dataset}")
    print(f"Scale: {args.scale}x, Steps: {args.num_steps}")
    print(f"Tile: {args.tile_size}, Overlap: {args.overlap}, Blend: {args.blend_mode}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Build model
    model = MODEL_CONFIGS[args.model_size](
        img_size=args.tile_size,
        use_window_attn=args.use_window_attn,
        window_size=args.window_size
    )
    
    # Load with strict=False to handle buffer mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        missing_params = [k for k in missing if 'relative_position_index' not in k]
        if missing_params:
            print(f"  Warning: Missing keys: {missing_params}")
    
    model = model.to(device).eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if 'epoch' in ckpt:
        print(f"Trained for {ckpt['epoch']+1} epochs")
    
    # Get files
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nEvaluating {len(hr_files)} images...\n")
    
    psnr_list, ssim_list = [], []
    psnr_bic_list, ssim_bic_list = [], []
    
    for hf, lf in tqdm(zip(hr_files, lr_files), total=len(hr_files), desc="Evaluating"):
        base_name = os.path.splitext(hf)[0]
        
        hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lf)).convert('RGB')
        
        hr_np = np.array(hr_img)
        H, W = hr_np.shape[0], hr_np.shape[1]
        
        lr_bicubic = lr_img.resize((W, H), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        lr_t = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device)
        
        pred = run_sr_tiled(model, lr_t, device, args.num_steps, args.tile_size, args.overlap, args.blend_mode)
        pred_np = ((pred[0].cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
        
        psnr_val = calculate_psnr(pred_np, hr_np)
        ssim_val = calculate_ssim(pred_np, hr_np)
        psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        psnr_bic_list.append(psnr_bic)
        ssim_bic_list.append(ssim_bic)
        
        if args.save_images:
            Image.fromarray(pred_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))
        
        if args.save_comparisons:
            comp = Image.new('RGB', (W * 3, H))
            comp.paste(lr_bicubic, (0, 0))
            comp.paste(Image.fromarray(pred_np), (W, 0))
            comp.paste(hr_img, (W * 2, 0))
            comp.save(os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}")
    print(f"Model:    PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}")
    print(f"Gain:     +{avg_psnr - avg_psnr_bic:.4f} dB, +{avg_ssim - avg_ssim_bic:.4f}")
    print("="*70)
    
    # Save results
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("MeanFlow-MMDiT Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Model: {args.model_size}\n")
        f.write(f"Window Attention: {args.use_window_attn} (size={args.window_size})\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Images: {len(psnr_list)}\n\n")
        f.write(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}\n")
        f.write(f"Model:    PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}\n")
        f.write(f"Gain:     +{avg_psnr - avg_psnr_bic:.4f} dB, +{avg_ssim - avg_ssim_bic:.4f}\n")
    
    print(f"\nResults saved: {output_dir}")
    return avg_psnr


if __name__ == '__main__':
    main()