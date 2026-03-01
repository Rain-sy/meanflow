#!/usr/bin/env python
"""
预计算 HR 和 LR 图像的 VAE Latent

这个脚本会：
1. 对 HR 图像进行 random crop 到 512x512，然后 encode
2. 对 LR 图像进行对应 crop (128x128) + bicubic 上采样到 512x512，然后 encode
3. 每张图片生成多个 crop（增加数据多样性）

输出结构：
    Data/Mix15K_Latents/
    ├── 0001_crop0_hr.pt    # [16, 64, 64]
    ├── 0001_crop0_lr.pt    # [16, 64, 64]
    ├── 0001_crop1_hr.pt
    ├── 0001_crop1_lr.pt
    └── ...

Usage:
    python cache_latents.py \
        --hr_dir Data/Mix15K_HR \
        --lr_dir Data/Mix15K_LR_bicubic_X4 \
        --output_dir Data/Mix15K_Latents \
        --crops_per_image 4 \
        --resolution 512
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True, help='HR 图像目录')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR 图像目录 (4x 下采样)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--resolution', type=int, default=512, help='Crop 分辨率')
    parser.add_argument('--crops_per_image', type=int, default=4, help='每张图生成几个 crop')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 设置随机种子保证可复现
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("预计算 VAE Latent")
    print("=" * 60)
    print(f"HR 目录: {args.hr_dir}")
    print(f"LR 目录: {args.lr_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"分辨率: {args.resolution}")
    print(f"每图 Crop 数: {args.crops_per_image}")
    print("=" * 60)
    
    # 加载 VAE
    print("\n🔄 正在加载 FLUX VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model_name,
        subfolder="vae",
        torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    # 获取 scaling_factor（与训练代码保持一致）
    scaling_factor = vae.config.scaling_factor
    print(f"VAE scaling_factor: {scaling_factor}")
    
    # 获取文件列表
    hr_files = sorted([f for f in os.listdir(args.hr_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\n找到 {len(hr_files)} 张 HR 图像, {len(lr_files)} 张 LR 图像")
    
    # 建立文件名映射（处理可能的命名差异）
    hr_basenames = {os.path.splitext(f)[0]: f for f in hr_files}
    lr_basenames = {os.path.splitext(f)[0]: f for f in lr_files}
    
    # 找到共同的文件
    common_names = set(hr_basenames.keys()) & set(lr_basenames.keys())
    print(f"匹配到 {len(common_names)} 对 HR/LR 图像")
    
    if len(common_names) == 0:
        # 尝试处理命名差异（如 0001.png vs 0001x4.png）
        print("⚠️ 尝试模糊匹配...")
        lr_to_hr = {}
        for lr_name in lr_basenames.keys():
            # 去掉可能的后缀如 'x4'
            clean_name = lr_name.replace('x4', '').replace('_x4', '').replace('X4', '').replace('_X4', '')
            if clean_name in hr_basenames:
                lr_to_hr[lr_name] = clean_name
        
        if lr_to_hr:
            print(f"模糊匹配到 {len(lr_to_hr)} 对")
            common_names = list(lr_to_hr.keys())
        else:
            print("❌ 无法匹配 HR/LR 文件，请检查文件命名")
            return
    else:
        lr_to_hr = {name: name for name in common_names}
        common_names = list(common_names)
    
    crop_size = args.resolution
    lr_crop_size = crop_size // 4  # LR 图像的 crop 大小
    
    total_crops = 0
    skipped = 0
    
    print(f"\n🚀 开始预计算 Latent (每图 {args.crops_per_image} 个 crop)...\n")
    
    for name in tqdm(common_names, desc="处理图像"):
        hr_filename = hr_basenames[lr_to_hr.get(name, name)]
        lr_filename = lr_basenames[name]
        
        try:
            # 加载图像
            hr_img = Image.open(os.path.join(args.hr_dir, hr_filename)).convert('RGB')
            lr_img = Image.open(os.path.join(args.lr_dir, lr_filename)).convert('RGB')
            
            hr_w, hr_h = hr_img.size
            lr_w, lr_h = lr_img.size
            
            # 检查尺寸是否足够
            if hr_w < crop_size or hr_h < crop_size:
                skipped += 1
                continue
            
            # 生成多个 random crop
            for crop_idx in range(args.crops_per_image):
                # 输出文件名
                base_name = os.path.splitext(hr_filename)[0]
                hr_latent_path = os.path.join(args.output_dir, f"{base_name}_crop{crop_idx}_hr.pt")
                lr_latent_path = os.path.join(args.output_dir, f"{base_name}_crop{crop_idx}_lr.pt")
                
                # 断点续传
                if os.path.exists(hr_latent_path) and os.path.exists(lr_latent_path):
                    total_crops += 1
                    continue
                
                # Random crop 位置（在 LR 空间计算，然后映射到 HR）
                if lr_w > lr_crop_size and lr_h > lr_crop_size:
                    x = np.random.randint(0, lr_w - lr_crop_size)
                    y = np.random.randint(0, lr_h - lr_crop_size)
                else:
                    x, y = 0, 0
                
                # Crop LR
                lr_crop = lr_img.crop((x, y, x + lr_crop_size, y + lr_crop_size))
                # Bicubic 上采样到目标分辨率
                lr_up = lr_crop.resize((crop_size, crop_size), Image.BICUBIC)
                
                # Crop HR（对应位置，4x）
                hr_crop = hr_img.crop((x * 4, y * 4, (x + lr_crop_size) * 4, (y + lr_crop_size) * 4))
                hr_crop = hr_crop.resize((crop_size, crop_size), Image.BICUBIC)
                
                # 转换为 tensor [-1, 1]
                hr_tensor = torch.from_numpy(np.array(hr_crop)).float().permute(2, 0, 1) / 127.5 - 1
                lr_tensor = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
                
                hr_tensor = hr_tensor.unsqueeze(0).to(device, dtype=dtype)
                lr_tensor = lr_tensor.unsqueeze(0).to(device, dtype=dtype)
                
                # VAE encode
                with torch.no_grad():
                    hr_latent = vae.encode(hr_tensor).latent_dist.sample() * scaling_factor
                    lr_latent = vae.encode(lr_tensor).latent_dist.sample() * scaling_factor
                
                # 保存（去掉 batch 维度，转到 CPU）
                torch.save(hr_latent.squeeze(0).cpu(), hr_latent_path)
                torch.save(lr_latent.squeeze(0).cpu(), lr_latent_path)
                
                total_crops += 1
                
        except Exception as e:
            print(f"❌ 处理 {hr_filename} 时出错: {e}")
            continue
    
    print(f"\n{'=' * 60}")
    print(f"✅ 预计算完成!")
    print(f"   总 Crop 数: {total_crops}")
    print(f"   跳过图像数: {skipped} (尺寸不足)")
    print(f"   输出目录: {args.output_dir}")
    print(f"{'=' * 60}")
    
    # 保存元信息
    meta_path = os.path.join(args.output_dir, "meta.txt")
    with open(meta_path, 'w') as f:
        f.write(f"hr_dir: {args.hr_dir}\n")
        f.write(f"lr_dir: {args.lr_dir}\n")
        f.write(f"resolution: {args.resolution}\n")
        f.write(f"crops_per_image: {args.crops_per_image}\n")
        f.write(f"scaling_factor: {scaling_factor}\n")
        f.write(f"total_crops: {total_crops}\n")
    
    print(f"\n💡 下一步: 使用预计算的 Latent 训练")
    print(f"   python train_dual_control_latent.py --latent_dir {args.output_dir} ...")


if __name__ == "__main__":
    main()