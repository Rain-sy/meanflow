"""
MeanFlow-MMDiT Evaluation

Output structure:
    outputs/
    └── DIV2K/
        └── MMDiT_small_1step_x4/
            ├── predictions/
            ├── comparisons/
            └── results.txt

Usage:
    python evaluate_sr_mmdit.py \
        --checkpoint checkpoints_mmdit/xxx/best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --model_size small \
        --scale 4 \
        --num_steps 1
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
# Model (same as training)
# ============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.frequency_embedding_size = frequency_embedding_size
    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1, self.act, self.fc2 = nn.Linear(in_features, hidden_features), nn.GELU(), nn.Linear(hidden_features, in_features)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MMDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size, self.num_heads, self.head_dim = hidden_size, num_heads, hidden_size // num_heads
        self.norm1_img, self.norm1_cond = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.norm2_img, self.norm2_cond = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.qkv_img, self.qkv_cond = nn.Linear(hidden_size, hidden_size * 3), nn.Linear(hidden_size, hidden_size * 3)
        self.proj_img, self.proj_cond = nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_img, self.mlp_cond = Mlp(hidden_size, mlp_hidden), Mlp(hidden_size, mlp_hidden)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
    
    def forward(self, x_img, x_cond, c):
        B = x_img.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        xi = self.norm1_img(x_img) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        xc = self.norm1_cond(x_cond) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv_i = self.qkv_img(xi).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_c = self.qkv_cond(xc).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_i, k_i, v_i = qkv_i.unbind(0)
        q_c, k_c, v_c = qkv_c.unbind(0)
        k_joint, v_joint = torch.cat([k_i, k_c], dim=2), torch.cat([v_i, v_c], dim=2)
        attn_i = F.scaled_dot_product_attention(q_i, k_joint, v_joint).transpose(1, 2).reshape(B, -1, self.hidden_size)
        attn_c = F.scaled_dot_product_attention(q_c, k_joint, v_joint).transpose(1, 2).reshape(B, -1, self.hidden_size)
        x_img = x_img + gate_msa.unsqueeze(1) * self.proj_img(attn_i)
        x_cond = x_cond + gate_msa.unsqueeze(1) * self.proj_cond(attn_c)
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
    def __init__(self, img_size=128, patch_size=4, in_channels=3, hidden_size=512, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_size, self.in_channels, self.hidden_size = patch_size, in_channels, hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed_img = PatchEmbed(patch_size, in_channels, hidden_size)
        self.patch_embed_cond = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.t_embedder, self.h_embedder = TimestepEmbedder(hidden_size), TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([MMDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        return x.reshape(x.shape[0], h, w, p, p, self.in_channels).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], self.in_channels, h*p, w*p)
    
    def forward(self, z_t, lr_cond, t, h):
        x_img = self.patch_embed_img(z_t) + self.pos_embed
        x_cond = self.patch_embed_cond(lr_cond) + self.pos_embed
        c = self.t_embedder(t) + self.h_embedder(h)
        for block in self.blocks:
            x_img, x_cond = block(x_img, x_cond, c)
        return self.unpatchify(self.final_layer(x_img, c))

def MeanFlowMMDiT_XS(img_size=128): return MeanFlowMMDiT(img_size, hidden_size=256, depth=8, num_heads=4)
def MeanFlowMMDiT_S(img_size=128): return MeanFlowMMDiT(img_size, hidden_size=384, depth=12, num_heads=6)
def MeanFlowMMDiT_B(img_size=128): return MeanFlowMMDiT(img_size, hidden_size=512, depth=12, num_heads=8)
def MeanFlowMMDiT_L(img_size=128): return MeanFlowMMDiT(img_size, hidden_size=768, depth=24, num_heads=12)
MODEL_CONFIGS = {'xs': MeanFlowMMDiT_XS, 'small': MeanFlowMMDiT_S, 'base': MeanFlowMMDiT_B, 'large': MeanFlowMMDiT_L}


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def inference_tile(model, lr_img, device, tile_size, overlap, num_steps):
    _, C, H, W = lr_img.shape
    if H <= tile_size and W <= tile_size:
        return inference_single(model, lr_img, device, num_steps)
    
    stride = tile_size - overlap
    output = torch.zeros(1, 3, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)
    
    blend = torch.ones(1, 1, tile_size, tile_size, device=device)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap, device=device)
        blend[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
        blend[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
        blend[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
        blend[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            ye, xe = min(y + tile_size, H), min(x + tile_size, W)
            ys, xs = max(0, ye - tile_size), max(0, xe - tile_size)
            tile = lr_img[:, :, ys:ye, xs:xe]
            th, tw = tile.shape[2], tile.shape[3]
            if th < tile_size or tw < tile_size:
                tile = F.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode='reflect')
            out = inference_single(model, tile, device, num_steps)[:, :, :th, :tw]
            w = blend[:, :, :th, :tw]
            output[:, :, ys:ye, xs:xe] += out * w
            weight[:, :, ys:ye, xs:xe] += w
    
    return output / (weight + 1e-8)

@torch.no_grad()
def inference_single(model, lr_img, device, num_steps=1):
    x, dt = lr_img.clone(), 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((1,), 1.0 - i * dt, device=device)
        h = torch.full((1,), dt, device=device)
        x = x - dt * model(x, lr_img, t, h)
    return x


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow-MMDiT Evaluation')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    
    # Model
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--scale', type=int, default=4)
    
    # Inference
    parser.add_argument('--tile_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=1)
    
    # Output
    parser.add_argument('--output_base', type=str, default='./meanflow/outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect dataset
    if args.dataset is None:
        hr_lower = args.hr_dir.lower()
        if 'urban' in hr_lower: args.dataset = 'Urban100'
        elif 'div2k' in hr_lower: args.dataset = 'DIV2K'
        elif 'set5' in hr_lower: args.dataset = 'Set5'
        elif 'set14' in hr_lower: args.dataset = 'Set14'
        elif 'bsd100' in hr_lower or 'b100' in hr_lower: args.dataset = 'BSD100'
        elif 'manga' in hr_lower: args.dataset = 'Manga109'
        else: args.dataset = 'Unknown'
    
    # Build output dir
    model_name = f'MMDiT_{args.model_size}'
    exp_name = f"{model_name}_{args.num_steps}step_x{args.scale}"
    output_dir = os.path.join(args.output_base, args.dataset, exp_name)
    
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print("="*70)
    print("MeanFlow-MMDiT Evaluation")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Scale: {args.scale}x, Steps: {args.num_steps}")
    print(f"Tile: {args.tile_size}, Overlap: {args.overlap}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MODEL_CONFIGS[args.model_size](img_size=args.tile_size)
    model.load_state_dict(ckpt.get('ema', ckpt['model']))
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
    results_per_image = []
    
    for hf, lf in tqdm(zip(hr_files, lr_files), total=len(hr_files)):
        base_name = os.path.splitext(hf)[0]
        
        hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lf)).convert('RGB')
        
        hr_np = np.array(hr_img)
        H, W = hr_np.shape[0], hr_np.shape[1]
        
        # Bicubic baseline
        lr_bicubic = lr_img.resize((W, H), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        # Prepare input
        lr_up = lr_img.resize((W, H), Image.BICUBIC)
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device)
        
        # Inference
        pred = inference_tile(model, lr_t, device, args.tile_size, args.overlap, args.num_steps)
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
        
        results_per_image.append({
            'name': hf,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'psnr_bicubic': psnr_bic,
            'ssim_bicubic': ssim_bic,
            'psnr_gain': psnr_val - psnr_bic,
        })
        
        # Save prediction
        if args.save_images:
            Image.fromarray(pred_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))
        
        # Save comparison: LR | Bicubic | SR | GT
        if args.save_comparisons:
            lr_display = lr_img.resize((W, H), Image.NEAREST)
            comparison = np.concatenate([np.array(lr_display), lr_bicubic_np, pred_np, hr_np], axis=1)
            Image.fromarray(comparison).save(os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
    
    # Average metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    
    # Print results
    print("\n" + "="*70)
    print("                        Evaluation Results")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {model_name}")
    print(f"Test images: {len(psnr_list)}")
    print(f"Scale: {args.scale}x, Steps: {args.num_steps}")
    print("-"*70)
    print(f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*70)
    print(f"{'Bicubic':<20} {avg_psnr_bic:<15.4f} {avg_ssim_bic:<15.4f}")
    print(f"{model_name:<20} {avg_psnr:<15.4f} {avg_ssim:<15.4f}")
    print("-"*70)
    print(f"{'Improvement':<20} {avg_psnr - avg_psnr_bic:+.4f} dB        {avg_ssim - avg_ssim_bic:+.4f}")
    print("="*70)
    
    # Save results to file
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MeanFlow-MMDiT Evaluation Results\n")
        f.write("="*70 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"HR directory: {args.hr_dir}\n")
        f.write(f"LR directory: {args.lr_dir}\n")
        f.write(f"Scale: {args.scale}x\n")
        f.write(f"Sampling steps: {args.num_steps}\n")
        f.write(f"Tile size: {args.tile_size}, Overlap: {args.overlap}\n")
        f.write(f"Test images: {len(psnr_list)}\n")
        f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("Average Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Bicubic:      PSNR = {avg_psnr_bic:.4f} dB,  SSIM = {avg_ssim_bic:.4f}\n")
        f.write(f"  {model_name}:  PSNR = {avg_psnr:.4f} dB,  SSIM = {avg_ssim:.4f}\n")
        f.write(f"  Improvement:  PSNR = {avg_psnr - avg_psnr_bic:+.4f} dB,  SSIM = {avg_ssim - avg_ssim_bic:+.4f}\n")
        f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("Per-image Results (sorted by PSNR gain):\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Image':<35} {'PSNR':<10} {'SSIM':<10} {'Bicubic':<10} {'Gain':<10}\n")
        f.write("-"*70 + "\n")
        
        for r in sorted(results_per_image, key=lambda x: x['psnr_gain'], reverse=True):
            f.write(f"{r['name']:<35} {r['psnr']:<10.4f} {r['ssim']:<10.4f} "
                    f"{r['psnr_bicubic']:<10.4f} {r['psnr_gain']:+.4f}\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.txt: Detailed metrics")
    if args.save_images:
        print(f"  - predictions/: SR output images")
    if args.save_comparisons:
        print(f"  - comparisons/: LR | Bicubic | SR | GT")


if __name__ == '__main__':
    main()