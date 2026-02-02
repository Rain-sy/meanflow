"""
MeanFlow-DiT (U-ViT Style) Evaluation Script

Matches the structure and output format of evaluate_sr.py.
Specifically designed for the Patch-2 U-ViT architecture.

Usage:
    python meanflow/evaluate_sr_uvit.py \
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
# 1. Model Architecture (Must match training script exactly)
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
        
        # Initialize pos_embed with default size, but allow dynamic resizing
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
        # 1. Embed Input
        x_in = self.x_embedder(x).flatten(2).transpose(1, 2)
        
        # 2. Dynamic Positional Embedding Handling
        # If input size doesn't match cached pos_embed, generate on the fly
        if x_in.shape[1] != self.pos_embed.shape[1]:
            # Recalculate based on current input spatial dimensions
            grid_size = int(x_in.shape[1] ** 0.5)
            new_pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size)
            new_pos_embed = torch.from_numpy(new_pos_embed).float().unsqueeze(0).to(x.device)
            x = x_in + new_pos_embed
        else:
            x = x_in + self.pos_embed

        # 3. Time Embeds
        c = self.t_embedder(t) + self.h_embedder(h)
        
        # 4. U-ViT Forward
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
# 2. Metrics & File Matching
# ============================================================================

def calculate_psnr(img1, img2, max_val=255.0):
    """Calculate PSNR"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate SSIM"""
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
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def extract_image_id(filename, is_lr=False):
    """Extract image ID from filename"""
    name = os.path.splitext(filename)[0]
    if name.endswith('_HR'): return name[:-3]
    if name.endswith('_LR'): return name[:-3]
    if is_lr:
        match = re.match(r'(.+)x\d+$', name)
        if match: return match.group(1)
    return name


def match_hr_lr_files(hr_dir, lr_dir):
    """Match HR and LR files"""
    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png'))) + \
               sorted(glob.glob(os.path.join(hr_dir, '*.jpg')))
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png'))) + \
               sorted(glob.glob(os.path.join(lr_dir, '*.jpg')))
    
    hr_dict = {extract_image_id(os.path.basename(f), is_lr=False): f for f in hr_files}
    lr_dict = {extract_image_id(os.path.basename(f), is_lr=True): f for f in lr_files}
    
    common_ids = sorted(set(hr_dict.keys()) & set(lr_dict.keys()))
    pairs = [(hr_dict[img_id], lr_dict[img_id]) for img_id in common_ids]
    
    print(f"Found {len(pairs)} matched image pairs")
    return pairs

# ============================================================================
# 3. Model Loading
# ============================================================================

def load_uvit_model(checkpoint_path, device, model_size='small'):
    """Load MeanFlow U-ViT model"""
    
    # Init model (Patch size fixed to 2 as per U-ViT training)
    model_fn = MODEL_CONFIGS[model_size]
    model = model_fn(img_size=128, patch_size=2)
    
    print(f"Loading MeanFlow U-ViT ({model_size}): {checkpoint_path}")
    
    # Fix: Added weights_only=False to prevent warning/error
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Load EMA if available
    if 'ema' in checkpoint:
        print("  Using EMA parameters")
        ema_params = checkpoint['ema']
        for name, param in model.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name])
    
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    
    return model

# ============================================================================
# 4. Inference
# ============================================================================

def pad_to_multiple(x, multiple=2):
    """Pad tensor to be divisible by multiple (Crucial for Patch=2)"""
    h, w = x.shape[2], x.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, (pad_h, pad_w)


@torch.no_grad()
def run_sr_direct(model, lr_tensor, device, num_steps=1, pad_multiple=2):
    """Run SR directly without tiling"""
    lr_padded, (pad_h, pad_w) = pad_to_multiple(lr_tensor, multiple=pad_multiple)
    
    batch_size = lr_padded.shape[0]
    x = lr_padded
    
    # One-step MeanFlow Inference: x_0 = x_1 - 1.0 * u
    # Or Multi-step
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = 1.0 - i * dt
        t = torch.full((batch_size,), t_val, device=device)
        h = torch.full((batch_size,), dt, device=device)
        u = model(x, t, h)
        x = x - dt * u
    
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :x.shape[2]-pad_h if pad_h > 0 else x.shape[2], 
                   :x.shape[3]-pad_w if pad_w > 0 else x.shape[3]]
    return x


@torch.no_grad()
def run_sr_tiled(model, lr_tensor, device, num_steps=1, tile_size=256, overlap=32, pad_multiple=2):
    """Run SR with tiling (Memory Efficient)"""
    _, _, H, W = lr_tensor.shape
    
    if H <= tile_size and W <= tile_size:
        return run_sr_direct(model, lr_tensor, device, num_steps, pad_multiple)
    
    output = torch.zeros_like(lr_tensor)
    weight = torch.zeros_like(lr_tensor)
    
    stride = tile_size - overlap
    h_starts = list(range(0, max(1, H - tile_size + 1), stride))
    if h_starts[-1] + tile_size < H: h_starts.append(max(0, H - tile_size))
    
    w_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if w_starts[-1] + tile_size < W: w_starts.append(max(0, W - tile_size))
    
    # Linear blending weights
    blend_weight = torch.ones(1, 1, tile_size, tile_size, device=device)
    if overlap > 0:
        for i in range(overlap):
            ratio = i / overlap
            blend_weight[:, :, i, :] *= ratio
            blend_weight[:, :, tile_size - 1 - i, :] *= ratio
            blend_weight[:, :, :, i] *= ratio
            blend_weight[:, :, :, tile_size - 1 - i] *= ratio
    
    for h_start in h_starts:
        for w_start in w_starts:
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)
            
            # Extract Tile
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            th, tw = tile.shape[2], tile.shape[3]
            
            # Pad Tile to tile_size (reflect) if edge case
            if th < tile_size or tw < tile_size:
                tile = F.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode='reflect')
            
            # Pad for Model Patch Alignment
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
            
            # Remove Tile Size Padding and Accumulate
            tile_output = x[:, :, :th, :tw]
            tile_weight = blend_weight[:, :, :th, :tw]
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_output * tile_weight
            weight[:, :, h_start:h_end, w_start:w_end] += tile_weight
            
    return output / (weight + 1e-8)

# ============================================================================
# 5. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow U-ViT Evaluation')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--hr_dir', type=str, required=True, help='HR directory')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR directory')
    
    # Model
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    
    # Output
    parser.add_argument('--output_base', type=str, default='./meanflow/outputs', help='Base output dir')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--model_name', type=str, default=None)
    
    # Settings
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1)
    
    # Tiling
    parser.add_argument('--no_tile', action='store_true', help='Disable tiling')
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--max_size', type=int, default=None)
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Auto-detect Dataset
    if args.dataset is None:
        path_lower = args.hr_dir.lower()
        if 'div2k' in path_lower: args.dataset = 'DIV2K'
        elif 'urban' in path_lower: args.dataset = 'Urban100'
        elif 'set5' in path_lower: args.dataset = 'Set5'
        elif 'set14' in path_lower: args.dataset = 'Set14'
        else: args.dataset = 'Custom'
        
    if args.model_name is None:
        args.model_name = f'MeanFlowUViT_{args.model_size}'
        
    # Output Paths
    tile_str = 'notile' if args.no_tile else f'tile{args.tile_size}'
    exp_name = f"{args.model_name}_{args.num_steps}step_x{args.scale}_{tile_str}"
    output_dir = os.path.join(args.output_base, args.dataset, exp_name)
    
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images: os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons: os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print(f"\nOutput Directory: {output_dir}")
    
    # Load Model
    model = load_uvit_model(args.checkpoint, device, args.model_size)
    
    # Match Files
    pairs = match_hr_lr_files(args.hr_dir, args.lr_dir)
    if not pairs:
        print("No pairs found!")
        return
        
    # Evaluation Loop
    psnrs, ssims = [], []
    psnrs_bic, ssims_bic = [], []
    results_list = []
    skipped = 0
    
    pad_multiple = 2 # Fixed for Patch-2
    
    print("\nStarting Evaluation...")
    for hr_path, lr_path in tqdm(pairs):
        name = os.path.basename(hr_path)
        base_name = os.path.splitext(name)[0]
        
        # Load Images
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        hr_np = np.array(hr_img)
        target_h, target_w = hr_np.shape[0], hr_np.shape[1]
        
        # Resize Constraint
        if args.max_size and (target_h > args.max_size or target_w > args.max_size):
            s = args.max_size / max(target_h, target_w)
            target_h, target_w = int(target_h*s), int(target_w*s)
            hr_img = hr_img.resize((target_w, target_h), Image.BICUBIC)
            lr_img = lr_img.resize((target_w//args.scale, target_h//args.scale), Image.BICUBIC)
            hr_np = np.array(hr_img)
            
        # Bicubic Baseline
        lr_bic = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_bic_np = np.array(lr_bic)
        
        # Prepare Input (Upsample LR -> Model)
        # MeanFlow expects input size = HR size
        lr_up_input = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_arr = np.array(lr_up_input).astype(np.float32) / 127.5 - 1.0
        lr_tensor = torch.from_numpy(lr_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        try:
            if args.no_tile:
                pred_t = run_sr_direct(model, lr_tensor, device, args.num_steps, pad_multiple)
            else:
                pred_t = run_sr_tiled(model, lr_tensor, device, args.num_steps, 
                                      args.tile_size, args.overlap, pad_multiple)
            
            pred_np = ((pred_t.squeeze(0).permute(1,2,0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Metrics
            p_val = calculate_psnr(pred_np, hr_np)
            s_val = calculate_ssim(pred_np, hr_np)
            p_bic = calculate_psnr(lr_bic_np, hr_np)
            s_bic = calculate_ssim(lr_bic_np, hr_np)
            
            psnrs.append(p_val)
            ssims.append(s_val)
            psnrs_bic.append(p_bic)
            ssims_bic.append(s_bic)
            
            results_list.append({
                'name': name,
                'psnr': p_val, 'ssim': s_val,
                'psnr_bic': p_bic, 'gain': p_val - p_bic
            })
            
            # Save
            if args.save_images:
                Image.fromarray(pred_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))
                
            if args.save_comparisons:
                lr_nn = lr_img.resize((target_w, target_h), Image.NEAREST)
                comp = np.concatenate([np.array(lr_nn), lr_bic_np, pred_np, hr_np], axis=1)
                Image.fromarray(comp).save(os.path.join(output_dir, 'comparisons', f'{base_name}_comp.png'))
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM on {name}, skipping.")
                torch.cuda.empty_cache()
                skipped += 1
            else:
                raise e
    
    if not psnrs:
        print("No results generated.")
        return

    # Averages
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    avg_p_bic = np.mean(psnrs_bic)
    avg_s_bic = np.mean(ssims_bic)
    
    # Print
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Bicubic: PSNR={avg_p_bic:.4f}, SSIM={avg_s_bic:.4f}")
    print(f"MeanFlow: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
    print(f"Gain: {avg_psnr - avg_p_bic:+.4f} dB")
    print("="*70)
    
    # Save results.txt
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Bicubic PSNR: {avg_p_bic:.4f}\n")
        f.write(f"Improvement: {avg_psnr - avg_p_bic:+.4f}\n\n")
        f.write(f"{'Image':<30} {'PSNR':<10} {'SSIM':<10} {'Gain':<10}\n")
        f.write("-"*60 + "\n")
        for r in sorted(results_list, key=lambda x: x['gain'], reverse=True):
            f.write(f"{r['name']:<30} {r['psnr']:<10.4f} {r['ssim']:<10.4f} {r['gain']:+.4f}\n")
            
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()