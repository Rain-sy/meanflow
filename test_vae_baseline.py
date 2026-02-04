"""
Test VAE reconstruction quality baseline

This helps understand the upper bound of latent-space methods.
If VAE(encode→decode) loses 2 dB, then latent-based SR cannot exceed GT - 2 dB.

Usage:
    python test_vae_baseline.py \
        --hr_dir meanflow/Data/DIV2K/DIV2K_valid_HR \
        --vae_type sd \
        --device cuda:0
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F


def calculate_psnr(img1, img2, max_val=255.0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


class VAEWrapper:
    def __init__(self, vae_type="sd3", device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        print(f"Loading VAE: {vae_type}...")
        
        from diffusers import AutoencoderKL
        
        if vae_type == "sd3":
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                subfolder="vae", torch_dtype=dtype
            ).to(device)
        elif vae_type == "flux":
            self.vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="vae", torch_dtype=dtype
            ).to(device)
        else:
            raise ValueError(f"Unsupported VAE type: {vae_type}. Use 'sd3' or 'flux'.")
        
        self.latent_channels = 16
        self.vae.eval()
        self.vae.requires_grad_(False)
    
    @torch.no_grad()
    def reconstruct(self, x):
        """Encode then decode"""
        x = x.to(self.device, self.dtype)
        latent = self.vae.encode(x).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        latent = latent / self.vae.config.scaling_factor
        recon = self.vae.decode(latent).sample
        return recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--vae_type', type=str, default='sd3', choices=['sd3', 'flux'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_images', type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    vae = VAEWrapper(args.vae_type, device)
    
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:args.max_images]
    
    psnr_list = []
    
    print(f"\nTesting VAE reconstruction on {len(hr_files)} images...")
    
    for hr_file in tqdm(hr_files):
        hr_img = Image.open(os.path.join(args.hr_dir, hr_file)).convert('RGB')
        
        # Ensure size is divisible by 8
        w, h = hr_img.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        hr_img = hr_img.crop((0, 0, w, h))
        
        hr_np = np.array(hr_img)
        hr_tensor = torch.from_numpy(hr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        hr_tensor = hr_tensor.to(device)
        
        # Reconstruct
        recon_tensor = vae.reconstruct(hr_tensor)
        recon_np = ((recon_tensor[0].cpu().float().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        psnr = calculate_psnr(hr_np, recon_np)
        psnr_list.append(psnr)
    
    avg_psnr = np.mean(psnr_list)
    
    print("\n" + "="*60)
    print(f"VAE Reconstruction Baseline ({args.vae_type})")
    print("="*60)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Min PSNR: {np.min(psnr_list):.2f} dB")
    print(f"Max PSNR: {np.max(psnr_list):.2f} dB")
    print("="*60)
    print(f"\n⚠️  This is the UPPER BOUND for latent-space SR methods!")
    print(f"    If bicubic baseline is 26.7 dB, and VAE recon is {avg_psnr:.1f} dB,")
    print(f"    then latent SR can at best reach ~{avg_psnr:.1f} dB (limited by VAE).")


if __name__ == '__main__':
    main()