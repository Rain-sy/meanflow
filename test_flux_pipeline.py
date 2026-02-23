#!/usr/bin/env python
"""
Test jasperai/Flux.1-dev-Controlnet-Upscaler using official diffusers pipeline.
This is the correct way to use the pretrained ControlNet.

Usage:
    python test_flux_pipeline.py \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --num_samples 10 \
        --device cuda:0
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F


def calculate_psnr(img1, img2):
    """Calculate PSNR between two tensors in [0, 1] range"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=28)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--controlnet_scale', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./outputs/flux_pipeline_test')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.bfloat16
    
    print("=" * 70)
    print("Testing FLUX ControlNet with Official Diffusers Pipeline")
    print("=" * 70)
    print(f"Inference steps: {args.num_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"ControlNet scale: {args.controlnet_scale}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Load pipeline
    from diffusers import FluxControlNetPipeline, FluxControlNetModel
    
    print("\n[Loading] ControlNet...")
    controlnet = FluxControlNetModel.from_pretrained(
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
        torch_dtype=dtype
    )
    
    print("[Loading] FLUX Pipeline...")
    pipe = FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=dtype
    )
    pipe.to(device)
    
    # Optional: enable memory efficient attention
    # pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()
    
    print(f"[Ready] GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    
    # Get image files
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test
    print(f"\nTesting on {min(args.num_samples, len(hr_files))} images...")
    
    psnrs_sr = []
    psnrs_bicubic = []
    
    for i in tqdm(range(min(args.num_samples, len(hr_files)))):
        # Load images
        hr_img = Image.open(os.path.join(args.hr_dir, hr_files[i])).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lr_files[i])).convert('RGB')
        
        # Resize LR to target size (4x upscale)
        target_w, target_h = hr_img.size
        lr_up = lr_img.resize((target_w, target_h), Image.BICUBIC)
        
        # Center crop to 512x512 (or 1024x1024) for testing
        crop_size = 512
        w, h = hr_img.size
        if w >= crop_size and h >= crop_size:
            x, y = (w - crop_size) // 2, (h - crop_size) // 2
            hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
            lr_crop = lr_up.crop((x, y, x + crop_size, y + crop_size))
        else:
            hr_crop = hr_img.resize((crop_size, crop_size), Image.BICUBIC)
            lr_crop = lr_up.resize((crop_size, crop_size), Image.BICUBIC)
        
        # Run pipeline
        # The upscaler expects the low-res image as the control image
        result = pipe(
            prompt="",  # Empty prompt for SR task
            control_image=lr_crop,
            height=crop_size,
            width=crop_size,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images[0]
        
        # Convert to tensors for PSNR calculation
        hr_t = torch.from_numpy(np.array(hr_crop)).float() / 255.0
        sr_t = torch.from_numpy(np.array(result)).float() / 255.0
        lr_t = torch.from_numpy(np.array(lr_crop)).float() / 255.0
        
        # Calculate PSNR
        psnr_sr = calculate_psnr(sr_t, hr_t)
        psnr_bicubic = calculate_psnr(lr_t, hr_t)
        
        psnrs_sr.append(psnr_sr)
        psnrs_bicubic.append(psnr_bicubic)
        
        if args.save_images:
            result.save(os.path.join(args.output_dir, f'sr_{hr_files[i]}'))
            
            # Also save comparison
            comparison = Image.new('RGB', (crop_size * 3, crop_size))
            comparison.paste(lr_crop, (0, 0))
            comparison.paste(result, (crop_size, 0))
            comparison.paste(hr_crop, (crop_size * 2, 0))
            comparison.save(os.path.join(args.output_dir, f'cmp_{hr_files[i]}'))
    
    # Results
    print("\n" + "=" * 70)
    print("              FLUX PIPELINE BASELINE RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<25} {'PSNR (dB)':<15}")
    print("-" * 40)
    print(f"{'Bicubic':<25} {np.mean(psnrs_bicubic):.2f} ± {np.std(psnrs_bicubic):.2f}")
    print(f"{'FLUX ControlNet SR':<25} {np.mean(psnrs_sr):.2f} ± {np.std(psnrs_sr):.2f}")
    print("-" * 40)
    print(f"{'Improvement':<25} {np.mean(psnrs_sr) - np.mean(psnrs_bicubic):+.2f} dB")
    print("=" * 70)
    
    if np.mean(psnrs_sr) > np.mean(psnrs_bicubic) + 1:
        print("\n✅ Pipeline works correctly! SR is better than bicubic.")
    elif np.mean(psnrs_sr) > np.mean(psnrs_bicubic):
        print("\n⚠️ SR is slightly better than bicubic. Model may need finetuning.")
    else:
        print("\n❌ SR is worse than bicubic. There may be a configuration issue.")


if __name__ == '__main__':
    main()
