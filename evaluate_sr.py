"""
MeanFlow SR 评估脚本 (无 Tiling 版本)

直接处理整张图片，不分块。适合显存充足的情况。

用法：
    python evaluate_sr_notile.py \
        --checkpoint checkpoints_sr_fixed/best_model.pt \
        --hr_dir Flow_Restore/Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Flow_Restore/Data/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --num_steps 1
"""

import os
import argparse
import glob
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# 从修复版训练脚本导入模型
from train_sr_fixed import MeanFlowSRNet


def calculate_psnr(img1, img2, max_val=255.0):
    """计算 PSNR"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """计算 SSIM"""
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
    """提取图像 ID"""
    name = os.path.splitext(filename)[0]
    if is_lr:
        match = re.match(r'(.+)x\d+$', name)
        if match:
            return match.group(1)
    return name


def match_hr_lr_files(hr_dir, lr_dir):
    """匹配 HR 和 LR 文件"""
    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png'))) + \
               sorted(glob.glob(os.path.join(hr_dir, '*.jpg')))
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png'))) + \
               sorted(glob.glob(os.path.join(lr_dir, '*.jpg')))
    
    hr_dict = {extract_image_id(os.path.basename(f), is_lr=False): f for f in hr_files}
    lr_dict = {extract_image_id(os.path.basename(f), is_lr=True): f for f in lr_files}
    
    common_ids = sorted(set(hr_dict.keys()) & set(lr_dict.keys()))
    pairs = [(hr_dict[img_id], lr_dict[img_id]) for img_id in common_ids]
    
    print(f"找到 {len(pairs)} 组配对图像")
    return pairs


def load_model(checkpoint_path, device):
    """加载模型"""
    model = MeanFlowSRNet(
        in_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0
    )
    
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 加载 EMA
    ema_params = checkpoint.get('ema', None)
    if ema_params:
        print("  使用 EMA 参数")
        for name, param in model.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name])
    
    return model


def pad_to_multiple(x, multiple=8):
    """填充到 multiple 的整数倍"""
    h, w = x.shape[2], x.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, (pad_h, pad_w)


@torch.no_grad()
def run_sr(model, lr_tensor, device, num_steps=1):
    """
    运行 SR（不分块，直接处理整张图）
    """
    lr_padded, (pad_h, pad_w) = pad_to_multiple(lr_tensor, multiple=8)
    
    batch_size = lr_padded.shape[0]
    x = lr_padded
    
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t_current = 1.0 - i * dt
        t = torch.full((batch_size,), t_current, device=device)
        h = torch.full((batch_size,), dt, device=device)
        
        u = model(x, t, h)
        x = x - dt * u
    
    # 移除填充
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :x.shape[2]-pad_h if pad_h > 0 else x.shape[2], 
                   :x.shape[3]-pad_w if pad_w > 0 else x.shape[3]]
    
    return x


def main():
    parser = argparse.ArgumentParser(description='MeanFlow SR 评估 (无 Tiling)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval_results_notile')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_size', type=int, default=None, 
                        help='限制最大尺寸（避免 OOM），如 --max_size 512')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    model = load_model(args.checkpoint, device)
    pairs = match_hr_lr_files(args.hr_dir, args.lr_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)
    
    psnr_list, ssim_list = [], []
    psnr_bicubic_list, ssim_bicubic_list = [], []
    
    print(f"\n开始评估 (num_steps={args.num_steps}, max_size={args.max_size})...\n")
    
    for hr_path, lr_path in tqdm(pairs):
        name = os.path.basename(hr_path)
        
        # 加载图片
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        hr_np = np.array(hr_img)
        target_h, target_w = hr_np.shape[0], hr_np.shape[1]
        
        # 如果设置了 max_size，缩小图片
        if args.max_size and (target_h > args.max_size or target_w > args.max_size):
            scale_factor = args.max_size / max(target_h, target_w)
            new_h, new_w = int(target_h * scale_factor), int(target_w * scale_factor)
            hr_img = hr_img.resize((new_w, new_h), Image.BICUBIC)
            lr_img = lr_img.resize((new_w // args.scale, new_h // args.scale), Image.BICUBIC)
            hr_np = np.array(hr_img)
            target_h, target_w = new_h, new_w
        
        # Bicubic
        lr_bicubic = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        # 准备输入
        lr_arr = np.array(lr_img).astype(np.float32) / 127.5 - 1.0
        lr_tensor = torch.from_numpy(lr_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 上采样到目标尺寸
        lr_upsampled = F.interpolate(lr_tensor, size=(target_h, target_w), 
                                      mode='bicubic', align_corners=False)
        
        # SR
        try:
            hr_pred_tensor = run_sr(model, lr_upsampled, device, num_steps=args.num_steps)
            hr_pred_np = ((hr_pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n  OOM on {name} ({target_h}x{target_w}), skipping...")
                torch.cuda.empty_cache()
                continue
            raise
        
        # 计算指标
        psnr_val = calculate_psnr(hr_pred_np, hr_np)
        ssim_val = calculate_ssim(hr_pred_np, hr_np)
        psnr_bicubic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bicubic = calculate_ssim(lr_bicubic_np, hr_np)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        psnr_bicubic_list.append(psnr_bicubic)
        ssim_bicubic_list.append(ssim_bicubic)
        
        # 保存预测
        Image.fromarray(hr_pred_np).save(os.path.join(args.output_dir, 'predictions', name))
    
    # 结果
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bicubic_list)
    avg_ssim_bic = np.mean(ssim_bicubic_list)
    
    print("\n" + "="*60)
    print("                    评估结果")
    print("="*60)
    print(f"测试图片数: {len(psnr_list)}")
    print(f"采样步数: {args.num_steps}")
    if args.max_size:
        print(f"最大尺寸限制: {args.max_size}")
    print("-"*60)
    print(f"{'方法':<20} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*60)
    print(f"{'Bicubic':<20} {avg_psnr_bic:<15.4f} {avg_ssim_bic:<15.4f}")
    print(f"{'MeanFlow SR':<20} {avg_psnr:<15.4f} {avg_ssim:<15.4f}")
    print("-"*60)
    print(f"{'提升':<20} {avg_psnr - avg_psnr_bic:+.4f} dB        {avg_ssim - avg_ssim_bic:+.4f}")
    print("="*60)


if __name__ == '__main__':
    main()