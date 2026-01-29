"""
Unified MeanFlow SR Evaluation Script

Supports:
    - UNet model (train_sr_fixed.py)
    - DiT model (train_meanflow_dit.py)
    - Tiling for large images (default ON to avoid OOM)
    - No-tiling mode (for small images or large GPU memory)

Output structure:
    meanflow/outputs/
    ├── Urban100/
    │   └── MeanFlowSR_1step_x2/
    │       ├── predictions/
    │       ├── comparisons/
    │       └── results.txt
    └── DIV2K/
        └── MeanFlowDiT_small_1step_x2/
            └── ...

Usage:
    # UNet model with tiling (default)
    python evaluate_sr.py \\
        --checkpoint checkpoints_sr_fixed/best_model.pt \\
        --hr_dir "Flow_Restore/Data/Urban 100/X2 Urban100/X2/HIGH X2 Urban" \\
        --lr_dir "Flow_Restore/Data/Urban 100/X2 Urban100/X2/LOW X2 Urban" \\
        --model_type unet \\
        --dataset Urban100 \\
        --num_steps 1

    # DiT model with tiling
    python evaluate_sr.py \\
        --checkpoint checkpoints_dit/best_model.pt \\
        --hr_dir "..." \\
        --lr_dir "..." \\
        --model_type dit \\
        --model_size small \\
        --tile_size 128 \\
        --num_steps 1

    # No tiling (small images or large GPU)
    python evaluate_sr.py \\
        --checkpoint checkpoints_sr_fixed/best_model.pt \\
        --hr_dir "..." \\
        --lr_dir "..." \\
        --no_tile \\
        --max_size 512
"""

import os
import re
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
# Metrics
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


# ============================================================================
# File Matching
# ============================================================================

def extract_image_id(filename, is_lr=False):
    """Extract image ID from filename"""
    name = os.path.splitext(filename)[0]
    
    # Urban100 format: img_001_SRF_2_HR / img_001_SRF_2_LR
    if name.endswith('_HR'):
        return name[:-3]
    if name.endswith('_LR'):
        return name[:-3]
    
    # DIV2K format: 0801x2
    if is_lr:
        match = re.match(r'(.+)x\d+$', name)
        if match:
            return match.group(1)
    
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
# Model Loading
# ============================================================================

def load_unet_model(checkpoint_path, device):
    """Load UNet model from train_sr_fixed.py"""
    from train_sr import MeanFlowSRNet
    
    model = MeanFlowSRNet(
        in_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0
    )
    
    print(f"Loading UNet model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v 
                      for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Load EMA
    if 'ema' in checkpoint:
        print("  Using EMA parameters")
        ema_params = checkpoint['ema']
        for name, param in model.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name])
    
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    
    return model


def load_dit_model(checkpoint_path, device, img_size=128, model_size='small'):
    """Load DiT model from train_meanflow_dit.py"""
    from train_sr_dit import MODEL_CONFIGS
    
    model_fn = MODEL_CONFIGS[model_size]
    model = model_fn(img_size=img_size)
    
    print(f"Loading DiT model ({model_size}): {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v 
                      for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Load EMA
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
# Inference Functions
# ============================================================================

def pad_to_multiple(x, multiple=8):
    """Pad tensor to be divisible by multiple"""
    h, w = x.shape[2], x.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, (pad_h, pad_w)


@torch.no_grad()
def run_sr_direct(model, lr_tensor, device, num_steps=1, pad_multiple=8):
    """
    Run SR directly without tiling (for small images or large GPU memory)
    """
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
        x = x[:, :, :x.shape[2]-pad_h if pad_h > 0 else x.shape[2], 
                   :x.shape[3]-pad_w if pad_w > 0 else x.shape[3]]
    
    return x


@torch.no_grad()
def run_sr_tiled(model, lr_tensor, device, num_steps=1, tile_size=256, overlap=64, pad_multiple=8):
    """
    Run SR with tiling for large images
    """
    _, _, H, W = lr_tensor.shape
    
    # If small enough, process directly
    if H <= tile_size and W <= tile_size:
        return run_sr_direct(model, lr_tensor, device, num_steps, pad_multiple)
    
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
    
    # Create blending weights (linear ramp at edges)
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
            
            # Extract tile
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            th, tw = tile.shape[2], tile.shape[3]
            
            # Pad tile if needed
            if th < tile_size or tw < tile_size:
                tile = F.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode='reflect')
            
            # Process tile
            tile_padded, (tile_pad_h, tile_pad_w) = pad_to_multiple(tile, multiple=pad_multiple)
            
            batch_size = tile_padded.shape[0]
            x = tile_padded
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t_val = 1.0 - i * dt
                t = torch.full((batch_size,), t_val, device=device)
                h_param = torch.full((batch_size,), dt, device=device)
                u = model(x, t, h_param)
                x = x - dt * u
            
            # Remove model padding
            if tile_pad_h > 0 or tile_pad_w > 0:
                x = x[:, :, :x.shape[2]-tile_pad_h if tile_pad_h > 0 else x.shape[2],
                           :x.shape[3]-tile_pad_w if tile_pad_w > 0 else x.shape[3]]
            
            # Get actual tile output (remove size padding)
            tile_output = x[:, :, :th, :tw]
            tile_weight = blend_weight[:, :, :th, :tw]
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_output * tile_weight
            weight[:, :, h_start:h_end, w_start:w_end] += tile_weight
    
    output = output / (weight + 1e-8)
    return output


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified MeanFlow SR Evaluation')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--hr_dir', type=str, required=True, help='HR images directory')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR images directory')
    
    # Model settings
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'dit'],
                        help='Model type: unet or dit')
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['xs', 'small', 'base', 'large'],
                        help='DiT model size (ignored for unet)')
    
    # Output organization
    parser.add_argument('--output_base', type=str, default='./meanflow/outputs',
                        help='Base output directory')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (auto-detected if not specified)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name for output folder (auto-set if not specified)')
    
    # SR settings
    parser.add_argument('--scale', type=int, default=2, help='SR scale factor')
    parser.add_argument('--num_steps', type=int, default=1, help='Number of sampling steps')
    
    # Tiling settings (default ON to avoid OOM)
    parser.add_argument('--no_tile', action='store_true', 
                        help='Disable tiling (may cause OOM on large images)')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='Tile size for processing (default 256 for UNet, 128 for DiT)')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between tiles')
    parser.add_argument('--max_size', type=int, default=None,
                        help='Max image size (resize larger images)')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Auto-detect dataset name
    if args.dataset is None:
        hr_dir_lower = args.hr_dir.lower()
        if 'urban' in hr_dir_lower:
            args.dataset = 'Urban100'
        elif 'div2k' in hr_dir_lower:
            args.dataset = 'DIV2K'
        elif 'set5' in hr_dir_lower:
            args.dataset = 'Set5'
        elif 'set14' in hr_dir_lower:
            args.dataset = 'Set14'
        elif 'bsd100' in hr_dir_lower or 'b100' in hr_dir_lower:
            args.dataset = 'BSD100'
        elif 'manga' in hr_dir_lower:
            args.dataset = 'Manga109'
        else:
            args.dataset = 'Unknown'
    
    # Auto-set model name
    if args.model_name is None:
        if args.model_type == 'unet':
            args.model_name = 'MeanFlowSR'
        else:
            args.model_name = f'MeanFlowDiT_{args.model_size}'
    
    # Auto-adjust tile size for DiT
    if args.model_type == 'dit' and args.tile_size > 128:
        print(f"Note: DiT model, adjusting tile_size from {args.tile_size} to 128")
        args.tile_size = 128
        args.overlap = 32
    
    # Build output directory
    tile_str = 'notile' if args.no_tile else f'tile{args.tile_size}'
    exp_name = f"{args.model_name}_{args.num_steps}step_x{args.scale}_{tile_str}"
    output_dir = os.path.join(args.output_base, args.dataset, exp_name)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    # Load model
    if args.model_type == 'unet':
        model = load_unet_model(args.checkpoint, device)
        pad_multiple = 8
    else:
        model = load_dit_model(args.checkpoint, device, args.tile_size, args.model_size)
        pad_multiple = 4  # DiT uses patch_size=4
    
    # Match files
    pairs = match_hr_lr_files(args.hr_dir, args.lr_dir)
    
    if not pairs:
        print("Error: No matched image pairs found!")
        return
    
    # Evaluate
    psnr_list, ssim_list = [], []
    psnr_bicubic_list, ssim_bicubic_list = [], []
    results_per_image = []
    skipped = 0
    
    tile_mode = "disabled" if args.no_tile else f"tile_size={args.tile_size}, overlap={args.overlap}"
    print(f"\nStarting evaluation...")
    print(f"  Model: {args.model_name} ({args.model_type})")
    print(f"  Steps: {args.num_steps}")
    print(f"  Tiling: {tile_mode}")
    if args.max_size:
        print(f"  Max size: {args.max_size}")
    print()
    
    for hr_path, lr_path in tqdm(pairs):
        name = os.path.basename(hr_path)
        base_name = os.path.splitext(name)[0]
        
        # Load images
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        hr_np = np.array(hr_img)
        target_h, target_w = hr_np.shape[0], hr_np.shape[1]
        
        # Resize if max_size specified
        if args.max_size and (target_h > args.max_size or target_w > args.max_size):
            scale_factor = args.max_size / max(target_h, target_w)
            new_h, new_w = int(target_h * scale_factor), int(target_w * scale_factor)
            hr_img = hr_img.resize((new_w, new_h), Image.BICUBIC)
            lr_img = lr_img.resize((new_w // args.scale, new_h // args.scale), Image.BICUBIC)
            hr_np = np.array(hr_img)
            target_h, target_w = new_h, new_w
        
        # Bicubic baseline
        lr_bicubic = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        # Prepare input
        lr_arr = np.array(lr_img).astype(np.float32) / 127.5 - 1.0
        lr_tensor = torch.from_numpy(lr_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Upsample to target size
        lr_upsampled = F.interpolate(lr_tensor, size=(target_h, target_w),
                                      mode='bicubic', align_corners=False)
        
        # Run SR
        try:
            if args.no_tile:
                hr_pred_tensor = run_sr_direct(model, lr_upsampled, device, 
                                               args.num_steps, pad_multiple)
            else:
                hr_pred_tensor = run_sr_tiled(model, lr_upsampled, device,
                                              args.num_steps, args.tile_size, 
                                              args.overlap, pad_multiple)
            
            hr_pred_np = ((hr_pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
            hr_pred_np = hr_pred_np.clip(0, 255).astype(np.uint8)
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n  OOM on {name} ({target_h}x{target_w}), skipping...")
                print(f"  Hint: Try smaller --tile_size or use --max_size")
                torch.cuda.empty_cache()
                skipped += 1
                continue
            raise
        
        # Calculate metrics
        psnr_val = calculate_psnr(hr_pred_np, hr_np)
        ssim_val = calculate_ssim(hr_pred_np, hr_np)
        psnr_bicubic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bicubic = calculate_ssim(lr_bicubic_np, hr_np)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        psnr_bicubic_list.append(psnr_bicubic)
        ssim_bicubic_list.append(ssim_bicubic)
        
        results_per_image.append({
            'name': name,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'psnr_bicubic': psnr_bicubic,
            'ssim_bicubic': ssim_bicubic,
            'psnr_gain': psnr_val - psnr_bicubic,
        })
        
        # Save prediction
        if args.save_images:
            Image.fromarray(hr_pred_np).save(
                os.path.join(output_dir, 'predictions', f'{base_name}.png'))
        
        # Save comparison: LR | Bicubic | SR | GT
        if args.save_comparisons:
            lr_display = lr_img.resize((target_w, target_h), Image.NEAREST)
            comparison = np.concatenate([
                np.array(lr_display),
                lr_bicubic_np,
                hr_pred_np,
                hr_np
            ], axis=1)
            Image.fromarray(comparison).save(
                os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
    
    if not psnr_list:
        print("Error: All images were skipped due to OOM!")
        return
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bicubic_list)
    avg_ssim_bic = np.mean(ssim_bicubic_list)
    
    # Print results
    print("\n" + "="*70)
    print("                        Evaluation Results")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name} ({args.model_type})")
    print(f"Test images: {len(psnr_list)}" + (f" (skipped {skipped})" if skipped > 0 else ""))
    print(f"Scale: {args.scale}x")
    print(f"Sampling steps: {args.num_steps}")
    print(f"Tiling: {tile_mode}")
    if args.max_size:
        print(f"Max size: {args.max_size}")
    print("-"*70)
    print(f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*70)
    print(f"{'Bicubic':<20} {avg_psnr_bic:<15.4f} {avg_ssim_bic:<15.4f}")
    print(f"{args.model_name:<20} {avg_psnr:<15.4f} {avg_ssim:<15.4f}")
    print("-"*70)
    print(f"{'Improvement':<20} {avg_psnr - avg_psnr_bic:+.4f} dB        {avg_ssim - avg_ssim_bic:+.4f}")
    print("="*70)
    
    # Save results to file
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MeanFlow SR Evaluation Results\n")
        f.write("="*70 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model_name} ({args.model_type})\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"HR directory: {args.hr_dir}\n")
        f.write(f"LR directory: {args.lr_dir}\n")
        f.write(f"Scale: {args.scale}x\n")
        f.write(f"Sampling steps: {args.num_steps}\n")
        f.write(f"Tiling: {tile_mode}\n")
        f.write(f"Test images: {len(psnr_list)}\n")
        if skipped > 0:
            f.write(f"Skipped (OOM): {skipped}\n")
        if args.max_size:
            f.write(f"Max size: {args.max_size}\n")
        f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("Average Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Bicubic:         PSNR = {avg_psnr_bic:.4f} dB,  SSIM = {avg_ssim_bic:.4f}\n")
        f.write(f"  {args.model_name}:  PSNR = {avg_psnr:.4f} dB,  SSIM = {avg_ssim:.4f}\n")
        f.write(f"  Improvement:     PSNR = {avg_psnr - avg_psnr_bic:+.4f} dB,  SSIM = {avg_ssim - avg_ssim_bic:+.4f}\n")
        f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("Per-image Results (sorted by PSNR gain):\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Image':<30} {'PSNR':<10} {'SSIM':<10} {'Bicubic':<10} {'Gain':<10}\n")
        f.write("-"*70 + "\n")
        
        for r in sorted(results_per_image, key=lambda x: x['psnr_gain'], reverse=True):
            f.write(f"{r['name']:<30} {r['psnr']:<10.4f} {r['ssim']:<10.4f} "
                    f"{r['psnr_bicubic']:<10.4f} {r['psnr_gain']:+.4f}\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.txt: Detailed metrics")
    if args.save_images:
        print(f"  - predictions/: SR output images")
    if args.save_comparisons:
        print(f"  - comparisons/: Comparison images (LR | Bicubic | SR | GT)")


if __name__ == '__main__':
    main()