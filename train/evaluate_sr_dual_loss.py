"""
Evaluate MeanFlow SR with FLUX Feature Loss (V1 Base)

This evaluates models trained with train_sr_dual_v1_flux_loss.py
Model architecture: Simple DiT (V1, 3ch input)

Usage:
    python evaluate_sr_dual_loss.py \
        --checkpoint checkpoints/dual/.../best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --model_size small  --device cuda:7   --eval_lpips 
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
        
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
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
        ch1, ch2 = img1[:, :, i], img2[:, :, i]
        mu1, mu2 = ch1.mean(), ch2.mean()
        sigma1_sq = ((ch1 - mu1) ** 2).mean()
        sigma2_sq = ((ch2 - mu2) ** 2).mean()
        sigma12 = ((ch1 - mu1) * (ch2 - mu2)).mean()
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        ssims.append(ssim)
    return np.mean(ssims)


# Optional: LPIPS metric
def calculate_lpips(img1_tensor, img2_tensor, lpips_fn):
    """Calculate LPIPS between two tensors"""
    if lpips_fn is None:
        return 0.0
    with torch.no_grad():
        return lpips_fn(img1_tensor, img2_tensor).item()


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
    if h_starts[-1] + tile_size < H:
        h_starts.append(max(0, H - tile_size))
    w_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if w_starts[-1] + tile_size < W:
        w_starts.append(max(0, W - tile_size))
    
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
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_out * tile_weight
            weight[:, :, h_start:h_end, w_start:w_end] += tile_weight
    
    return output / (weight + 1e-8)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate MeanFlow SR with FLUX Loss (V1)')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--scale', type=int, default=4)
    
    parser.add_argument('--tile_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--blend_mode', type=str, default='gaussian', choices=['gaussian', 'linear'])
    parser.add_argument('--num_steps', type=int, default=1)
    
    # Output
    parser.add_argument('--output_base', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    
    # Optional LPIPS evaluation
    parser.add_argument('--eval_lpips', action='store_true', help='Also compute LPIPS metric')
    
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect dataset
    if args.dataset is None:
        hr_lower = args.hr_dir.lower()
        if 'urban' in hr_lower:
            args.dataset = 'Urban100'
        elif 'div2k' in hr_lower:
            args.dataset = 'DIV2K'
        elif 'set5' in hr_lower:
            args.dataset = 'Set5'
        elif 'set14' in hr_lower:
            args.dataset = 'Set14'
        elif 'bsd100' in hr_lower or 'b100' in hr_lower:
            args.dataset = 'BSD100'
        elif 'manga' in hr_lower:
            args.dataset = 'Manga109'
        else:
            args.dataset = 'Unknown'
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('ema', ckpt.get('model', ckpt))
    
    # Detect if trained with FLUX loss from checkpoint path
    has_flux_loss = 'fluxloss' in args.checkpoint.lower()
    
    # Build experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [ts, args.model_size, f'x{args.scale}', f'{args.num_steps}step']
        if has_flux_loss:
            parts.append('fluxloss')
        exp_name = '_'.join(parts)
    
    # Output dir
    output_dir = os.path.join(args.output_base, args.dataset, 'MeanFlowDiT_FluxLoss', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print("=" * 70)
    print("MeanFlow SR Evaluation (V1 + FLUX Loss)")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_size}")
    print(f"Trained with FLUX Loss: {has_flux_loss}")
    print(f"Dataset: {args.dataset}")
    print(f"Scale: {args.scale}x, Steps: {args.num_steps}")
    print(f"Tile: {args.tile_size}, Overlap: {args.overlap}, Blend: {args.blend_mode}")
    print(f"Output: {output_dir}")
    
    if 'epoch' in ckpt:
        print(f"Trained for: {ckpt['epoch'] + 1} epochs")
    if 'psnr' in ckpt:
        print(f"Training best PSNR: {ckpt['psnr']:.2f} dB")
    if 'loss' in ckpt:
        print(f"Final loss: {ckpt['loss']:.6f}")
    print("=" * 70)
    
    # Build model
    model = MODEL_CONFIGS[args.model_size](img_size=args.tile_size)
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  [MISSING] {k}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    if unexpected:
        print(f"Warning: Unexpected keys: {len(unexpected)}")
        for k in unexpected[:20]:
            print(f"  [UNEXPECTED] {k}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")
    
    model = model.to(device).eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optional LPIPS
    lpips_fn = None
    if args.eval_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='vgg').to(device)
            lpips_fn.eval()
            print("LPIPS evaluation enabled")
        except ImportError:
            print("Warning: lpips not installed, skipping LPIPS evaluation")
    
    # Get files
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    assert len(hr_files) == len(lr_files), f"HR/LR mismatch: {len(hr_files)} vs {len(lr_files)}"
    print(f"\nEvaluating {len(hr_files)} images...\n")
    
    psnr_list, ssim_list, lpips_list = [], [], []
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
        
        # Metrics
        psnr_val = calculate_psnr(pred_np, hr_np)
        ssim_val = calculate_ssim(pred_np, hr_np)
        psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        psnr_bic_list.append(psnr_bic)
        ssim_bic_list.append(ssim_bic)
        
        # LPIPS (if enabled)
        if lpips_fn is not None:
            hr_t = torch.from_numpy(hr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            hr_t = hr_t.to(device)
            lpips_val = calculate_lpips(pred, hr_t, lpips_fn)
            lpips_list.append(lpips_val)
        
        if args.save_images:
            Image.fromarray(pred_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))
        
        if args.save_comparisons:
            # Comparison: LR (resized) | Bicubic | Ours | HR
            comp = Image.new('RGB', (W * 4, H))
            # LR: resize to HR size for display using nearest neighbor
            lr_display = lr_img.resize((W, H), Image.NEAREST)
            comp.paste(lr_display, (0, 0))
            comp.paste(lr_bicubic, (W, 0))
            comp.paste(Image.fromarray(pred_np), (W * 2, 0))
            comp.paste(hr_img, (W * 3, 0))
            comp.save(os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    avg_lpips = np.mean(lpips_list) if lpips_list else 0.0
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}")
    print(f"Model:    PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}", end="")
    if lpips_list:
        print(f", LPIPS={avg_lpips:.4f}")
    else:
        print()
    print(f"Gain:     +{avg_psnr - avg_psnr_bic:.4f} dB, +{avg_ssim - avg_ssim_bic:.4f}")
    print("=" * 70)
    
    # Convergence check
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    if avg_psnr > avg_psnr_bic + 0.5:
        print(f"✓ Model shows improvement over bicubic (+{avg_psnr - avg_psnr_bic:.2f} dB)")
    else:
        print("⚠ Model shows minimal improvement. May need more training.")
    
    if has_flux_loss:
        print(f"✓ Model trained with FLUX feature loss")
        if lpips_list:
            print(f"  LPIPS: {avg_lpips:.4f} (lower is better for perceptual quality)")
        print(f"  Note: FLUX loss adds ~0.1 to total loss, so higher loss value is expected")
    
    if 'loss' in ckpt:
        # For FLUX loss models, the loss will be higher due to FLUX component
        if has_flux_loss:
            if ckpt['loss'] < 0.2:
                print(f"✓ Loss ({ckpt['loss']:.6f}) is in normal range for FLUX loss training")
            else:
                print(f"⚠ Loss is {ckpt['loss']:.6f}, may need more training")
        else:
            if ckpt['loss'] < 0.01:
                print(f"✓ Loss is low ({ckpt['loss']:.6f}), model converged")
            else:
                print(f"⚠ Loss is {ckpt['loss']:.6f}, may need more training")
    
    print("=" * 70)
    
    # Save results
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("MeanFlow SR Evaluation Results (V1 + FLUX Loss)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Model: {args.model_size}\n")
        f.write(f"Trained with FLUX Loss: {has_flux_loss}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Images: {len(psnr_list)}\n")
        if 'epoch' in ckpt:
            f.write(f"Epochs trained: {ckpt['epoch'] + 1}\n")
        if 'loss' in ckpt:
            f.write(f"Final loss: {ckpt['loss']:.6f}\n")
        f.write("\n")
        f.write(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}\n")
        f.write(f"Model:    PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}")
        if lpips_list:
            f.write(f", LPIPS={avg_lpips:.4f}\n")
        else:
            f.write("\n")
        f.write(f"Gain:     +{avg_psnr - avg_psnr_bic:.4f} dB, +{avg_ssim - avg_ssim_bic:.4f}\n")
    
    print(f"\nResults saved: {output_dir}")
    return avg_psnr


if __name__ == '__main__':
    main()