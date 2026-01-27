"""
MeanFlow SR 验证脚本 (支持 DIV2K 命名格式)

DIV2K 命名格式:
    HR:  0801.png
    LR:  0801x2.png (2x) 或 0801x4.png (4x)

用法：
    python evaluate_sr.py \
        --checkpoint checkpoints_sr/best_model.pt \
        --hr_dir Flow_Restore/Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Flow_Restore/Data/DIV2K/DIV2K_valid_LR_bicubic/X2 \
        --output_dir ./eval_results \
        --scale 2
"""

import os
import argparse
import glob
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F

# 从训练脚本导入模型
from train_sr import MeanFlowSRNet


# =========================================================================
# 指标计算
# =========================================================================

def calculate_psnr(img1, img2, max_val=255.0):
    """计算 PSNR"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """计算 SSIM (简化版)"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    if img1.ndim == 3:
        ssim_vals = []
        for c in range(img1.shape[2]):
            ssim_vals.append(calculate_ssim(img1[:,:,c], img2[:,:,c]))
        return np.mean(ssim_vals)
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim


def calculate_ssim_windowed(img1, img2, window_size=11):
    """使用滑动窗口计算 SSIM（更准确）"""
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        return calculate_ssim(img1, img2)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    if img1.ndim == 3:
        ssim_vals = []
        for c in range(img1.shape[2]):
            ssim_vals.append(calculate_ssim_windowed(img1[:,:,c], img2[:,:,c], window_size))
        return np.mean(ssim_vals)
    
    mu1 = uniform_filter(img1, window_size)
    mu2 = uniform_filter(img2, window_size)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(img1 ** 2, window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, window_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, window_size) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


# =========================================================================
# 文件名匹配（DIV2K 格式）
# =========================================================================

def extract_image_id(filename, is_lr=False, scale=2):
    """
    从文件名中提取图像 ID
    
    Examples:
        HR:  "0801.png" -> "0801"
        LR:  "0801x2.png" -> "0801"
        LR:  "0801x4.png" -> "0801"
    """
    name = os.path.splitext(filename)[0]  # 去掉扩展名
    
    if is_lr:
        # 移除 x2, x4 等后缀
        # 匹配模式: 0801x2, 0801x4, img_001x2 等
        match = re.match(r'(.+)x\d+$', name)
        if match:
            return match.group(1)
    
    return name


def match_hr_lr_files(hr_dir, lr_dir, scale=2):
    """
    匹配 HR 和 LR 文件
    
    支持的命名格式:
        1. DIV2K: HR="0801.png", LR="0801x2.png"
        2. 相同文件名: HR="img001.png", LR="img001.png"
    """
    # 获取文件列表
    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png'))) + \
               sorted(glob.glob(os.path.join(hr_dir, '*.jpg')))
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png'))) + \
               sorted(glob.glob(os.path.join(lr_dir, '*.jpg')))
    
    print(f"找到 HR 文件: {len(hr_files)}")
    print(f"找到 LR 文件: {len(lr_files)}")
    
    # 构建 ID -> 文件路径 的映射
    hr_dict = {}
    for f in hr_files:
        img_id = extract_image_id(os.path.basename(f), is_lr=False)
        hr_dict[img_id] = f
    
    lr_dict = {}
    for f in lr_files:
        img_id = extract_image_id(os.path.basename(f), is_lr=True, scale=scale)
        lr_dict[img_id] = f
    
    # 找到共同的 ID
    common_ids = sorted(set(hr_dict.keys()) & set(lr_dict.keys()))
    
    if not common_ids:
        print("\n警告: 无法通过 ID 匹配文件，尝试按索引配对...")
        # Fallback: 按索引配对
        n = min(len(hr_files), len(lr_files))
        pairs = [(hr_files[i], lr_files[i]) for i in range(n)]
    else:
        pairs = [(hr_dict[img_id], lr_dict[img_id]) for img_id in common_ids]
    
    print(f"成功配对: {len(pairs)} 组图像\n")
    
    # 显示前几个配对示例
    print("配对示例:")
    for i, (hr, lr) in enumerate(pairs[:3]):
        print(f"  {i+1}. HR: {os.path.basename(hr)} <-> LR: {os.path.basename(lr)}")
    if len(pairs) > 3:
        print(f"  ... 共 {len(pairs)} 组")
    print()
    
    return pairs


# =========================================================================
# 模型加载和推理
# =========================================================================

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    
    model = MeanFlowSRNet(
        in_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0
    )
    
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 处理 'module.' 前缀
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 加载 EMA 参数
    ema_params = checkpoint.get('ema', None)
    if ema_params is not None:
        print("  使用 EMA 参数")
        for name, param in model.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name])
    
    if 'loss' in checkpoint:
        print(f"  训练 {checkpoint.get('epoch', '?')+1} epochs, Loss: {checkpoint['loss']:.6f}")
    
    return model


def pad_to_multiple(x, multiple=8):
    """填充到 multiple 的整数倍"""
    h, w = x.shape[2], x.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, (0, pad_w, 0, pad_h)


def preprocess_image(image_path, device):
    """加载并预处理图片"""
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.to(device), (w, h), img


def postprocess_tensor(tensor):
    """将 tensor 转换为 numpy 图像"""
    tensor = tensor.detach().cpu().squeeze(0).permute(1, 2, 0)
    arr = ((tensor.numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return arr


@torch.no_grad()
def run_sr_single(model, lr_tensor, device, num_steps=10):
    """
    运行 MeanFlow SR（支持多步采样）
    
    Args:
        num_steps: 采样步数
            - 1: 原始 one-step（需要模型学过 h=1）
            - 10+: 多步采样（更稳定，适合模型主要学了 h≈0 的情况）
    """
    
    lr_padded, pads = pad_to_multiple(lr_tensor, multiple=8)
    
    batch_size = lr_padded.shape[0]
    x = lr_padded  # 从 LR 开始
    
    # 多步采样: 从 t=1 逐步走到 t=0
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t_current = 1.0 - i * dt  # 从 1 递减到 dt
        
        t = torch.full((batch_size,), t_current, device=device)
        h = torch.full((batch_size,), dt, device=device)  # 每步走 dt
        
        u = model(x, t, h)
        x = x - dt * u  # 走一小步
    
    hr_padded = x
    
    # 移除填充
    _, pad_right, _, pad_bottom = pads
    h_end = hr_padded.shape[2] - pad_bottom if pad_bottom > 0 else hr_padded.shape[2]
    w_end = hr_padded.shape[3] - pad_right if pad_right > 0 else hr_padded.shape[3]
    
    hr_tensor = hr_padded[:, :, :h_end, :w_end]
    
    return hr_tensor


@torch.no_grad()
def run_sr(model, lr_tensor, device, tile_size=512, overlap=64, num_steps=10):
    """
    运行 MeanFlow SR（支持分块处理大图）
    
    Args:
        model: SR 模型
        lr_tensor: 输入 LR tensor [1, C, H, W]
        device: 设备
        tile_size: 分块大小（默认 512）
        overlap: 重叠区域（默认 64）
        num_steps: 采样步数（默认 10，多步更稳定）
    """
    _, _, H, W = lr_tensor.shape
    
    # 如果图片较小，直接处理
    if H <= tile_size and W <= tile_size:
        return run_sr_single(model, lr_tensor, device, num_steps=num_steps)
    
    # 分块处理大图（静默处理，不打印每张图的信息）
    
    # 创建输出 tensor
    output = torch.zeros_like(lr_tensor)
    weight = torch.zeros_like(lr_tensor)
    
    # 计算分块位置
    stride = tile_size - overlap
    
    h_starts = list(range(0, H - tile_size + 1, stride))
    if h_starts[-1] + tile_size < H:
        h_starts.append(H - tile_size)
    
    w_starts = list(range(0, W - tile_size + 1, stride))
    if w_starts[-1] + tile_size < W:
        w_starts.append(W - tile_size)
    
    # 创建权重窗口（用于融合重叠区域）
    # 使用线性渐变权重
    ramp = torch.linspace(0, 1, overlap, device=device)
    weight_1d = torch.ones(tile_size, device=device)
    weight_1d[:overlap] = ramp
    weight_1d[-overlap:] = ramp.flip(0)
    weight_2d = weight_1d.unsqueeze(0) * weight_1d.unsqueeze(1)
    weight_2d = weight_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 处理每个分块
    for h_start in h_starts:
        for w_start in w_starts:
            h_end = h_start + tile_size
            w_end = w_start + tile_size
            
            # 提取分块
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            
            # 处理分块
            tile_output = run_sr_single(model, tile, device, num_steps=num_steps)
            
            # 加权累加
            output[:, :, h_start:h_end, w_start:w_end] += tile_output * weight_2d
            weight[:, :, h_start:h_end, w_start:w_end] += weight_2d
    
    # 归一化
    output = output / (weight + 1e-8)
    
    return output


# =========================================================================
# 评估函数
# =========================================================================

def evaluate_with_gt(model, hr_dir, lr_dir, output_dir, scale, device, tile_size=512, num_steps=10):
    """在有 Ground Truth 的情况下评估"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    # 匹配文件
    pairs = match_hr_lr_files(hr_dir, lr_dir, scale)
    
    if not pairs:
        print("错误: 没有找到可配对的图像！")
        return
    
    psnr_list = []
    ssim_list = []
    psnr_bicubic_list = []
    ssim_bicubic_list = []
    results_log = []
    
    print(f"开始评估 {len(pairs)} 张图片 (采样步数: {num_steps})...\n")
    
    for hr_path, lr_path in tqdm(pairs):
        name = os.path.basename(hr_path)
        
        # 加载图片
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        hr_np = np.array(hr_img)
        target_h, target_w = hr_np.shape[0], hr_np.shape[1]
        
        # Bicubic 上采样
        lr_bicubic = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        # 准备输入
        lr_tensor, _, _ = preprocess_image(lr_path, device)
        
        # 上采样 LR 到目标尺寸
        lr_upsampled = F.interpolate(
            lr_tensor,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        
        # 运行 SR
        hr_pred_tensor = run_sr(model, lr_upsampled, device, tile_size=tile_size, num_steps=num_steps)
        hr_pred_np = postprocess_tensor(hr_pred_tensor)
        
        # 计算指标 - MeanFlow vs GT
        psnr_val = calculate_psnr(hr_pred_np, hr_np)
        ssim_val = calculate_ssim_windowed(hr_pred_np, hr_np)
        
        # 计算指标 - Bicubic vs GT
        psnr_bicubic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bicubic = calculate_ssim_windowed(lr_bicubic_np, hr_np)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        psnr_bicubic_list.append(psnr_bicubic)
        ssim_bicubic_list.append(ssim_bicubic)
        
        results_log.append({
            'name': name,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'psnr_bicubic': psnr_bicubic,
            'ssim_bicubic': ssim_bicubic,
            'psnr_gain': psnr_val - psnr_bicubic,
        })
        
        # 保存预测结果
        pred_img = Image.fromarray(hr_pred_np)
        pred_img.save(os.path.join(output_dir, 'predictions', name))
        
        # 保存对比图 (前 20 张)
        if len(psnr_list) <= 20:
            # LR (放大显示) | Bicubic | MeanFlow | GT
            lr_display = lr_img.resize((target_w, target_h), Image.NEAREST)
            
            # 添加标签文字
            comparison = np.concatenate([
                np.array(lr_display),
                lr_bicubic_np,
                hr_pred_np,
                hr_np
            ], axis=1)
            
            comp_img = Image.fromarray(comparison)
            base_name = os.path.splitext(name)[0]
            save_path = os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png')
            comp_img.save(save_path)
    
    # 计算平均指标
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bicubic = np.mean(psnr_bicubic_list)
    avg_ssim_bicubic = np.mean(ssim_bicubic_list)
    
    # 打印结果
    print("\n" + "="*70)
    print("                        评估结果")
    print("="*70)
    print(f"数据集: {os.path.basename(hr_dir)}")
    print(f"测试图片数: {len(pairs)}")
    print(f"放大倍数: {scale}x")
    print(f"采样步数: {num_steps}")
    print("-"*70)
    print(f"{'方法':<20} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*70)
    print(f"{'Bicubic':<20} {avg_psnr_bicubic:<15.4f} {avg_ssim_bicubic:<15.4f}")
    print(f"{'MeanFlow SR':<20} {avg_psnr:<15.4f} {avg_ssim:<15.4f}")
    print("-"*70)
    psnr_gain = avg_psnr - avg_psnr_bicubic
    ssim_gain = avg_ssim - avg_ssim_bicubic
    print(f"{'提升':<20} {psnr_gain:+.4f} dB        {ssim_gain:+.4f}")
    print("="*70)
    
    # 保存详细结果
    log_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("MeanFlow SR 评估结果\n")
        f.write("="*70 + "\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"HR 目录: {hr_dir}\n")
        f.write(f"LR 目录: {lr_dir}\n")
        f.write(f"Scale: {scale}x\n")
        f.write(f"测试图片数: {len(pairs)}\n\n")
        
        f.write("平均指标:\n")
        f.write("-"*50 + "\n")
        f.write(f"  Bicubic:     PSNR = {avg_psnr_bicubic:.4f} dB,  SSIM = {avg_ssim_bicubic:.4f}\n")
        f.write(f"  MeanFlow SR: PSNR = {avg_psnr:.4f} dB,  SSIM = {avg_ssim:.4f}\n")
        f.write(f"  提升:        PSNR = {psnr_gain:+.4f} dB,  SSIM = {ssim_gain:+.4f}\n\n")
        
        f.write("单图结果:\n")
        f.write("-"*90 + "\n")
        f.write(f"{'图片名':<25} {'PSNR':<12} {'SSIM':<12} {'Bicubic PSNR':<14} {'增益':<10}\n")
        f.write("-"*90 + "\n")
        
        # 按 PSNR 增益排序
        results_log_sorted = sorted(results_log, key=lambda x: x['psnr_gain'], reverse=True)
        
        for r in results_log_sorted:
            f.write(f"{r['name']:<25} {r['psnr']:<12.4f} {r['ssim']:<12.4f} "
                    f"{r['psnr_bicubic']:<14.4f} {r['psnr_gain']:+.4f}\n")
    
    print(f"\n详细结果已保存到: {log_path}")
    print(f"预测图像保存到: {os.path.join(output_dir, 'predictions')}")
    print(f"对比图保存到: {os.path.join(output_dir, 'comparisons')}")
    
    return avg_psnr, avg_ssim


def inference_only(model, input_path, output_dir, scale, device, tile_size=512, num_steps=10):
    """仅推理模式（无 GT）"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, '*.png'))) + \
                sorted(glob.glob(os.path.join(input_path, '*.jpg')))
    else:
        files = [input_path]
    
    print(f"\n处理 {len(files)} 张图片 ({scale}x 超分)...")
    
    for path in tqdm(files):
        name = os.path.basename(path)
        base_name = os.path.splitext(name)[0]
        
        lr_tensor, (w, h), lr_img = preprocess_image(path, device)
        
        target_h, target_w = h * scale, w * scale
        
        lr_upsampled = F.interpolate(
            lr_tensor,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        
        hr_tensor = run_sr(model, lr_upsampled, device, tile_size=tile_size, num_steps=num_steps)
        hr_np = postprocess_tensor(hr_tensor)
        
        # 保存 SR 结果
        hr_img = Image.fromarray(hr_np)
        hr_img.save(os.path.join(output_dir, f'{base_name}_sr.png'))
        
        # 保存对比图
        lr_display = lr_img.resize((target_w, target_h), Image.NEAREST)
        lr_bicubic = lr_img.resize((target_w, target_h), Image.BICUBIC)
        
        comparison = np.concatenate([
            np.array(lr_display),
            np.array(lr_bicubic),
            hr_np
        ], axis=1)
        
        Image.fromarray(comparison).save(os.path.join(output_dir, f'{base_name}_compare.png'))
    
    print(f"\n完成！结果已保存到: {output_dir}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow SR 评估脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--hr_dir', type=str, default=None, help='HR 图片目录')
    parser.add_argument('--lr_dir', type=str, default=None, help='LR 图片目录')
    parser.add_argument('--input', type=str, default=None, help='单张图片或目录 (无 GT 模式)')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='输出目录')
    parser.add_argument('--scale', type=int, default=2, help='放大倍数')
    parser.add_argument('--tile_size', type=int, default=512, help='分块大小（大图时使用，默认512）')
    parser.add_argument('--num_steps', type=int, default=10, help='采样步数（默认10，1=one-step但需要模型支持）')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 根据参数选择模式
    if args.hr_dir and args.lr_dir:
        evaluate_with_gt(model, args.hr_dir, args.lr_dir, args.output_dir, args.scale, device, args.tile_size, args.num_steps)
    elif args.input:
        inference_only(model, args.input, args.output_dir, args.scale, device, args.tile_size, args.num_steps)
    elif args.lr_dir:
        inference_only(model, args.lr_dir, args.output_dir, args.scale, device, args.tile_size, args.num_steps)
    else:
        print("请指定 --hr_dir 和 --lr_dir (评估模式) 或 --input (推理模式)")
        print("\n示例:")
        print("  评估模式:")
        print("    python evaluate_sr.py --checkpoint best_model.pt \\")
        print("        --hr_dir Flow_Restore/Data/DIV2K/DIV2K_valid_HR \\")
        print("        --lr_dir Flow_Restore/Data/DIV2K/DIV2K_valid_LR_bicubic/X2")
        print("\n  推理模式:")
        print("    python evaluate_sr.py --checkpoint best_model.pt --input ./test_images")


if __name__ == '__main__':
    main()