#!/usr/bin/env python
"""
Test pretrained jasperai/Flux.1-dev-Controlnet-Upscaler directly (no finetuning)
This helps establish the baseline performance.

Usage:
    python test_pretrained_baseline.py \
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
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--controlnet', type=str, default='jasperai/Flux.1-dev-Controlnet-Upscaler')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./outputs/pretrained_baseline')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.bfloat16
    
    print("=" * 70)
    print("Testing Pretrained ControlNet Baseline (No Finetuning)")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"ControlNet: {args.controlnet}")
    print(f"Inference steps: {args.num_steps}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Load models
    from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    
    print("\n[Loading] VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model_name, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)
    
    print("[Loading] Text encoders (for caching empty prompt)...")
    text_enc = CLIPTextModel.from_pretrained(
        args.model_name, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_enc_2 = T5EncoderModel.from_pretrained(
        args.model_name, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    tok = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    tok_2 = T5TokenizerFast.from_pretrained(args.model_name, subfolder="tokenizer_2")
    
    with torch.no_grad():
        clip_out = text_enc(tok([""], padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt").input_ids.to(device))
        t5_out = text_enc_2(tok_2([""], padding="max_length", max_length=512,
                                  truncation=True, return_tensors="pt").input_ids.to(device))
        cached_embeds = {
            'pooled': clip_out.pooler_output.to(dtype),
            'prompt': t5_out[0].to(dtype),
            'text_ids': torch.zeros(t5_out[0].shape[1], 3, device=device, dtype=dtype),
        }
    
    del text_enc, text_enc_2
    torch.cuda.empty_cache()
    
    print("[Loading] Transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_name, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    transformer.requires_grad_(False)
    
    print("[Loading] Pretrained ControlNet...")
    controlnet = FluxControlNetModel.from_pretrained(
        args.controlnet, torch_dtype=dtype
    ).to(device)
    controlnet.requires_grad_(False)
    
    print(f"[Ready] GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    
    # Helper functions
    def pack(x):
        B, C, H, W = x.shape
        x = x.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H//2)*(W//2), C*4)
    
    def unpack(x, H, W):
        B, _, D = x.shape
        C = D // 4
        x = x.view(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, H, W)
    
    def img_ids(H, W, device, dtype):
        h, w = H//2, W//2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return ids.reshape(h*w, 3)
    
    @torch.no_grad()
    def forward(noisy, lr_lat, t, guidance=3.5):
        B, C, H, W = noisy.shape
        
        prompt = cached_embeds['prompt'].expand(B, -1, -1)
        pooled = cached_embeds['pooled'].expand(B, -1)
        txt_ids = cached_embeds['text_ids']
        img_id = img_ids(H, W, device, dtype)
        
        packed_noisy = pack(noisy)
        packed_cond = pack(lr_lat)
        
        # ControlNet
        ctrl_kwargs = {
            'hidden_states': packed_noisy,
            'controlnet_cond': packed_cond,
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_id,
            'return_dict': False,
        }
        if "Guidance" in type(controlnet.time_text_embed).__name__:
            ctrl_kwargs['guidance'] = torch.full((B,), guidance, device=device, dtype=dtype)
        
        ctrl_out = controlnet(**ctrl_kwargs)
        
        # Transformer
        trans_kwargs = {
            'hidden_states': packed_noisy,
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_id,
            'controlnet_block_samples': ctrl_out[0],
            'controlnet_single_block_samples': ctrl_out[1],
            'return_dict': False,
        }
        if "Guidance" in type(transformer.time_text_embed).__name__:
            trans_kwargs['guidance'] = torch.full((B,), guidance, device=device, dtype=dtype)
        
        pred = transformer(**trans_kwargs)[0]
        return unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(lr_lat, num_steps, guidance):
        lat = torch.randn_like(lr_lat)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((1,), t_val, device=device, dtype=dtype)
            v = forward(lat, lr_lat, t, guidance)
            lat = lat - dt * v
        
        return lat
    
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
        hr_img = Image.open(os.path.join(args.hr_dir, hr_files[i])).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lr_files[i])).convert('RGB')
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        # Center crop to 512x512 for testing
        w, h = hr_img.size
        x, y = (w - 512) // 2, (h - 512) // 2
        hr_crop = hr_img.crop((x, y, x + 512, y + 512))
        lr_crop = lr_up.crop((x, y, x + 512, y + 512))
        
        hr_t = torch.from_numpy(np.array(hr_crop)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_crop)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        
        hr_t = hr_t.to(device)
        lr_t = lr_t.to(device)
        
        # Encode
        hr_lat = vae.encode(hr_t.to(dtype)).latent_dist.sample() * vae.config.scaling_factor
        lr_lat = vae.encode(lr_t.to(dtype)).latent_dist.sample() * vae.config.scaling_factor
        
        # Generate SR
        sr_lat = inference(lr_lat, args.num_steps, args.guidance)
        sr = vae.decode(sr_lat / vae.config.scaling_factor).sample
        
        # Calculate PSNR
        hr_01 = (hr_t + 1) / 2
        sr_01 = ((sr + 1) / 2).clamp(0, 1)
        lr_01 = (lr_t + 1) / 2
        
        psnr_sr = calculate_psnr(sr_01, hr_01)
        psnr_bicubic = calculate_psnr(lr_01, hr_01)
        
        psnrs_sr.append(psnr_sr)
        psnrs_bicubic.append(psnr_bicubic)
        
        if args.save_images:
            sr_img = (sr_01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(sr_img).save(os.path.join(args.output_dir, f'sr_{hr_files[i]}'))
    
    # Results
    print("\n" + "=" * 70)
    print("                    PRETRAINED BASELINE RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<25} {'PSNR (dB)':<15}")
    print("-" * 40)
    print(f"{'Bicubic':<25} {np.mean(psnrs_bicubic):.2f} ± {np.std(psnrs_bicubic):.2f}")
    print(f"{'Pretrained ControlNet':<25} {np.mean(psnrs_sr):.2f} ± {np.std(psnrs_sr):.2f}")
    print("-" * 40)
    print(f"{'Improvement':<25} {np.mean(psnrs_sr) - np.mean(psnrs_bicubic):+.2f} dB")
    print("=" * 70)
    
    if np.mean(psnrs_sr) > 26:
        print("\n✅ Pretrained model works well! Finetuning may not be necessary.")
    elif np.mean(psnrs_sr) > np.mean(psnrs_bicubic):
        print("\n⚠️ Pretrained model is better than bicubic but has room for improvement.")
    else:
        print("\n❌ Pretrained model is underperforming. Check if the setup is correct.")


if __name__ == '__main__':
    main()
