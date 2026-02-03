"""
Evaluation Script for Latent MeanFlow-MMDiT (evaluate_sr_mmdit.py)

Usage:
    python evaluate_sr_mmdit.py \
        --checkpoint checkpoints_mmdit/0131_220530_small_x8_sd_bs4_ps256/best_model.pt \
        --hr_dir meanflow/Data/DIV2K/DIV2K_valid_HR \
        --lr_dir meanflow/Data/DIV2K/DIV2K_valid_LR_bicubic_X8 \
        --vae_type sd \
        --model_size small \
        --scale 8 \
        --device cuda:0
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
# Metrics
# ============================================================================

def calculate_psnr(img1, img2, max_val=255.0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
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
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


# ============================================================================
# VAE Wrapper
# ============================================================================

class VAEWrapper:
    def __init__(self, vae_type="sd", device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.vae_type = vae_type
        
        print(f"Loading VAE: {vae_type}...")
        
        if vae_type == "sd3":
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                subfolder="vae", torch_dtype=dtype
            ).to(device)
            self.scale_factor = 8
            self.latent_channels = 16
            
        elif vae_type == "flux":
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="vae", torch_dtype=dtype
            ).to(device)
            self.scale_factor = 8
            self.latent_channels = 16
            
        elif vae_type == "sdxl":
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae", torch_dtype=dtype
            ).to(device)
            self.scale_factor = 8
            self.latent_channels = 4
            
        else:  # sd
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", torch_dtype=dtype
            ).to(device)
            self.scale_factor = 8
            self.latent_channels = 4
        
        self.vae.eval()
        self.vae.requires_grad_(False)
        print(f"  Latent channels: {self.latent_channels}")
    
    @torch.no_grad()
    def encode(self, x):
        x = x.to(self.device, self.dtype)
        latent = self.vae.encode(x).latent_dist.sample()
        return latent * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, z):
        z = z.to(self.device, self.dtype)
        z = z / self.vae.config.scaling_factor
        return self.vae.decode(z).sample


# ============================================================================
# MMDiT Model (same as train_sr_mmdit.py)
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
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        self.blocks = nn.ModuleList([MMDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        
        self.norm_out = RMSNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, patch_dim)
    
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


def LatentMMDiT_S(latent_size=32, in_channels=16, **kwargs):
    return LatentMMDiT(latent_size, patch_size=2, in_channels=in_channels, hidden_size=384, depth=12, num_heads=6, **kwargs)

def LatentMMDiT_B(latent_size=32, in_channels=16, **kwargs):
    return LatentMMDiT(latent_size, patch_size=2, in_channels=in_channels, hidden_size=512, depth=12, num_heads=8, **kwargs)

def LatentMMDiT_L(latent_size=32, in_channels=16, **kwargs):
    return LatentMMDiT(latent_size, patch_size=2, in_channels=in_channels, hidden_size=768, depth=24, num_heads=12, **kwargs)

MODEL_CONFIGS = {'small': LatentMMDiT_S, 'base': LatentMMDiT_B, 'large': LatentMMDiT_L}


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def run_sr(model, vae, lr_tensor, device, num_steps=1):
    """
    Run SR inference
    
    Args:
        model: LatentMMDiT model
        vae: VAE wrapper
        lr_tensor: LR image tensor (B, C, H, W) in [-1, 1]
        device: torch device
        num_steps: number of sampling steps
    
    Returns:
        HR image tensor (B, C, H, W) in [-1, 1]
    """
    model.eval()
    
    # Encode to latent
    lr_latent = vae.encode(lr_tensor).float()
    
    batch_size = lr_latent.shape[0]
    x = lr_latent
    dt = 1.0 / num_steps
    
    # Flow integration
    for i in range(num_steps):
        t = torch.full((batch_size,), 1.0 - i * dt, device=device)
        h = torch.full((batch_size,), dt, device=device)
        u = model(x, lr_latent, t, h)
        x = x - dt * u
    
    # Decode to image
    hr_pred = vae.decode(x)
    
    return hr_pred


@torch.no_grad()
def run_sr_tiled(model, vae, lr_tensor, device, num_steps=1, tile_size=256, overlap=64):
    """
    Run SR with tiling for large images
    
    Tiling is done in IMAGE space (before VAE encoding)
    """
    _, _, H, W = lr_tensor.shape
    
    # If small enough, process directly
    if H <= tile_size and W <= tile_size:
        return run_sr(model, vae, lr_tensor, device, num_steps)
    
    output = torch.zeros_like(lr_tensor)
    weight = torch.zeros_like(lr_tensor)
    
    stride = tile_size - overlap
    
    # Calculate tile positions
    h_starts = list(range(0, max(1, H - tile_size + 1), stride))
    if not h_starts:
        h_starts = [0]
    if h_starts[-1] + tile_size < H:
        h_starts.append(max(0, H - tile_size))
    
    w_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if not w_starts:
        w_starts = [0]
    if w_starts[-1] + tile_size < W:
        w_starts.append(max(0, W - tile_size))
    
    for h_start in h_starts:
        for w_start in w_starts:
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)
            
            th = h_end - h_start
            tw = w_end - w_start
            
            # Extract tile
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            
            # Pad to tile_size if needed (must be divisible by 8 for VAE)
            pad_h = (8 - th % 8) % 8
            pad_w = (8 - tw % 8) % 8
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
            
            # Process tile
            tile_output = run_sr(model, vae, tile, device, num_steps)
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                tile_output = tile_output[:, :, :th, :tw]
            
            # Create tile-specific weight
            tile_weight = torch.ones(1, 1, th, tw, device=device)
            
            if overlap > 0:
                if h_start > 0:
                    for i in range(min(overlap, th)):
                        tile_weight[:, :, i, :] *= i / overlap
                if h_end < H:
                    for i in range(min(overlap, th)):
                        tile_weight[:, :, th - 1 - i, :] *= i / overlap
                if w_start > 0:
                    for i in range(min(overlap, tw)):
                        tile_weight[:, :, :, i] *= i / overlap
                if w_end < W:
                    for i in range(min(overlap, tw)):
                        tile_weight[:, :, :, tw - 1 - i] *= i / overlap
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_output * tile_weight
            weight[:, :, h_start:h_end, w_start:w_end] += tile_weight
    
    return output / (weight + 1e-8)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Latent MeanFlow-MMDiT')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    
    parser.add_argument('--vae_type', type=str, default='sd', choices=['sd', 'sdxl', 'sd3', 'flux'])
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'])
    parser.add_argument('--scale', type=int, default=8)
    
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--no_tile', action='store_true')
    
    parser.add_argument('--output_dir', type=str, default='./meanflow/outputs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_images', action='store_true', default=True)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load VAE
    vae = VAEWrapper(args.vae_type, device)
    
    # Determine latent size from training config
    # Default: 256px image → 32px latent
    latent_size = args.tile_size // vae.scale_factor
    
    # Load model
    print(f"Loading model: {args.model_size}...")
    model_fn = MODEL_CONFIGS[args.model_size]
    model = model_fn(latent_size=latent_size, in_channels=vae.latent_channels).to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'ema' in ckpt:
        # Load EMA weights
        ema_params = ckpt['ema']
        model_state = {}
        for name, param in model.named_parameters():
            if name in ema_params:
                model_state[name] = ema_params[name]
        model.load_state_dict(model_state, strict=False)
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model'])
        print("Loaded model weights")
    
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get image files
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Match files
    hr_names = {os.path.splitext(f)[0]: f for f in hr_files}
    lr_names = {os.path.splitext(f)[0]: f for f in lr_files}
    common = sorted(set(hr_names.keys()) & set(lr_names.keys()))
    
    if not common:
        # Try matching by index
        common = list(range(min(len(hr_files), len(lr_files))))
        pairs = [(hr_files[i], lr_files[i]) for i in common]
    else:
        pairs = [(hr_names[name], lr_names[name]) for name in common]
    
    print(f"Found {len(pairs)} image pairs")
    
    # Setup output directory
    dataset_name = os.path.basename(args.hr_dir.rstrip('/'))
    if 'DIV2K' in args.hr_dir:
        dataset_name = 'DIV2K'
    
    model_name = f"MMDiT_{args.model_size}"
    tile_str = "notile" if args.no_tile else f"tile{args.tile_size}"
    output_subdir = f"{model_name}_{args.num_steps}step_x{args.scale}_{tile_str}"
    output_path = os.path.join(args.output_dir, dataset_name, output_subdir)
    
    os.makedirs(output_path, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_path, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'comparisons'), exist_ok=True)
    
    # Evaluate
    psnr_bicubic_list = []
    psnr_sr_list = []
    ssim_bicubic_list = []
    ssim_sr_list = []
    
    print(f"\nEvaluating...")
    print(f"  Tiling: {'disabled' if args.no_tile else f'{args.tile_size}x{args.tile_size}, overlap={args.overlap}'}")
    print(f"  Output: {output_path}")
    
    for hr_file, lr_file in tqdm(pairs, desc="Processing"):
        # Load images
        hr_img = Image.open(os.path.join(args.hr_dir, hr_file)).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lr_file)).convert('RGB')
        
        # Upsample LR to HR size
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        # To tensor
        hr_np = np.array(hr_img)
        lr_up_np = np.array(lr_up)
        
        lr_tensor = torch.from_numpy(lr_up_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_tensor = lr_tensor.to(device)
        
        # Run SR
        with torch.no_grad():
            if args.no_tile:
                hr_pred_tensor = run_sr(model, vae, lr_tensor, device, args.num_steps)
            else:
                hr_pred_tensor = run_sr_tiled(model, vae, lr_tensor, device, args.num_steps, 
                                              args.tile_size, args.overlap)
        
        # To numpy
        hr_pred_np = ((hr_pred_tensor[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Ensure same size
        if hr_pred_np.shape[:2] != hr_np.shape[:2]:
            hr_pred_img = Image.fromarray(hr_pred_np).resize((hr_np.shape[1], hr_np.shape[0]), Image.BICUBIC)
            hr_pred_np = np.array(hr_pred_img)
        
        # Calculate metrics
        psnr_bicubic = calculate_psnr(hr_np, lr_up_np)
        psnr_sr = calculate_psnr(hr_np, hr_pred_np)
        ssim_bicubic = calculate_ssim(hr_np, lr_up_np)
        ssim_sr = calculate_ssim(hr_np, hr_pred_np)
        
        psnr_bicubic_list.append(psnr_bicubic)
        psnr_sr_list.append(psnr_sr)
        ssim_bicubic_list.append(ssim_bicubic)
        ssim_sr_list.append(ssim_sr)
        
        # Save images
        if args.save_images:
            base_name = os.path.splitext(hr_file)[0]
            
            # Save prediction
            Image.fromarray(hr_pred_np).save(os.path.join(output_path, 'predictions', f'{base_name}.png'))
            
            # Save comparison (LR | Bicubic | SR | GT)
            h, w = hr_np.shape[:2]
            comparison = np.zeros((h, w * 4, 3), dtype=np.uint8)
            
            lr_resized = np.array(lr_img.resize((w, h), Image.NEAREST))
            comparison[:, 0:w] = lr_resized
            comparison[:, w:2*w] = lr_up_np
            comparison[:, 2*w:3*w] = hr_pred_np
            comparison[:, 3*w:4*w] = hr_np
            
            Image.fromarray(comparison).save(os.path.join(output_path, 'comparisons', f'{base_name}.png'))
    
    # Calculate averages
    avg_psnr_bicubic = np.mean(psnr_bicubic_list)
    avg_psnr_sr = np.mean(psnr_sr_list)
    avg_ssim_bicubic = np.mean(ssim_bicubic_list)
    avg_ssim_sr = np.mean(ssim_sr_list)
    
    # Print results
    print("\n" + "="*70)
    print("                        Evaluation Results")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name} ({args.vae_type} VAE)")
    print(f"Test images: {len(pairs)}")
    print(f"Scale: {args.scale}x")
    print(f"Sampling steps: {args.num_steps}")
    print(f"Tiling: {'disabled' if args.no_tile else f'tile_size={args.tile_size}, overlap={args.overlap}'}")
    print("-"*70)
    print(f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*70)
    print(f"{'Bicubic':<20} {avg_psnr_bicubic:<15.4f} {avg_ssim_bicubic:<15.4f}")
    print(f"{model_name:<20} {avg_psnr_sr:<15.4f} {avg_ssim_sr:<15.4f}")
    print("-"*70)
    print(f"{'Improvement':<20} {avg_psnr_sr - avg_psnr_bicubic:+.4f} dB        {avg_ssim_sr - avg_ssim_bicubic:+.4f}")
    print("="*70)
    print(f"Results saved to: {output_path}")
    
    # Save results to file
    results_path = os.path.join(output_path, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Latent MeanFlow-MMDiT Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"VAE: {args.vae_type}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Scale: {args.scale}x\n")
        f.write(f"Sampling steps: {args.num_steps}\n")
        f.write(f"Test images: {len(pairs)}\n\n")
        f.write("-"*70 + "\n")
        f.write("Average Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Bicubic:  PSNR = {avg_psnr_bicubic:.4f} dB,  SSIM = {avg_ssim_bicubic:.4f}\n")
        f.write(f"  {model_name}:  PSNR = {avg_psnr_sr:.4f} dB,  SSIM = {avg_ssim_sr:.4f}\n")
        f.write(f"  Improvement:  PSNR = {avg_psnr_sr - avg_psnr_bicubic:+.4f} dB,  SSIM = {avg_ssim_sr - avg_ssim_bicubic:+.4f}\n")
        f.write("="*70 + "\n")


if __name__ == '__main__':
    main()