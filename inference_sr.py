"""
MeanFlow SR Inference Script (Pixel-Space Compatible)

Usage:
    python inference_sr.py --checkpoint checkpoints_sr/best_model.pt --input_dir ./test_lr --output_dir ./test_sr
"""

import os
import argparse
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Import the model class from your training script
# 确保 train_sr.py 和本脚本在同一目录下
from train_sr import MeanFlowSRNet

def load_model(checkpoint_path, device):
    """Load the trained Pixel-Space model"""
    
    # 必须与 train_sr.py 中的参数完全一致
    # 注意：如果你训练时修改了 channel_mult 或 hidden_channels，这里也必须改
    model = MeanFlowSRNet(
        in_channels=3,          # RGB 图片是 3 通道
        hidden_channels=128,    # 默认配置
        channel_mult=(1, 2, 4, 4), # 默认配置
        num_res_blocks=2,
        dropout=0.0
    )
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 加载权重
    # 有时候保存时会多一层 'module.' 前缀（如果是多卡训练），这里做个兼容处理
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    if 'loss' in checkpoint:
        print(f"  Model trained for {checkpoint['epoch']} epochs (Loss: {checkpoint['loss']:.6f})")
    
    return model

def pad_to_multiple(x, multiple=8):
    """
    将输入张量的 H, W 填充到 multiple 的整数倍
    """
    h, w = x.shape[2], x.shape[3]
    
    # 计算需要填充的量
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    # 如果不需要填充，直接返回
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    
    # 填充 (Left, Right, Top, Bottom)
    # 这里的顺序是 F.pad 的要求
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    return x_padded, (0, pad_w, 0, pad_h)

def preprocess_image(image_path, device):
    """Load and preprocess image to tensor [-1, 1]"""
    img = Image.open(image_path).convert('RGB')
    
    # 记录原始尺寸
    w, h = img.size
    
    # To tensor [-1, 1]
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.to(device), (w, h)

def postprocess_image(tensor):
    """Convert tensor [-1, 1] to PIL image"""
    tensor = tensor.detach().cpu().squeeze(0).permute(1, 2, 0) # CHW -> HWC
    arr = ((tensor.numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

@torch.no_grad()
def run_sr(model, lr_tensor, device):
    """
    Run One-step MeanFlow SR with Auto-Padding
    """
    # 1. 自动填充到 8 的倍数 (适配 UNet 3次下采样)
    lr_padded, pads = pad_to_multiple(lr_tensor, multiple=8)
    
    batch_size = lr_padded.shape[0]
    t = torch.ones(batch_size, device=device)
    h_time = torch.ones(batch_size, device=device)
    
    # 2. 推理
    u = model(lr_padded, t, h_time)
    hr_padded = lr_padded - u
    
    # 3. 移除填充 (Unpadding)
    # pads 格式: (left, right, top, bottom)
    # 我们只填了 right 和 bottom
    pad_left, pad_right, pad_top, pad_bottom = pads
    
    h_pad, w_pad = hr_padded.shape[2], hr_padded.shape[3]
    
    # 切片操作: [..., 0:H-pad_h, 0:W-pad_w]
    # 注意要防止 shape 变成 0
    h_end = h_pad - pad_bottom
    w_end = w_pad - pad_right
    
    hr_tensor = hr_padded[:, :, :h_end, :w_end]
    
    return hr_tensor

def main():
    parser = argparse.ArgumentParser(description='MeanFlow SR Inference (Pixel Space)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--input', type=str, default=None, help='Single input image')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory of input images')
    parser.add_argument('--output_dir', type=str, default='./sr_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    
    # 这个 Scale 参数很重要，决定你要放大几倍
    parser.add_argument('--scale', type=int, default=2, help='Upsampling scale (e.g. 2, 4)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    try:
        model = load_model(args.checkpoint, device)
    except Exception as e:
        print(f"\n[Error] 加载模型失败: {e}")
        print("请确保你的 train_sr.py 中的 MeanFlowSRNet 类定义与这里一致。")
        return

    # 2. 准备文件列表
    if args.input:
        files = [args.input]
    elif args.input_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, '*.png'))) + \
                sorted(glob.glob(os.path.join(args.input_dir, '*.jpg'))) + \
                sorted(glob.glob(os.path.join(args.input_dir, '*.jpeg')))
    else:
        print("请指定 --input 或 --input_dir")
        return
    
    if not files:
        print("未找到输入图片")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n开始推理: {len(files)} 张图片 | 放大倍率: {args.scale}x")
    
    # 3. 循环处理
    for path in tqdm(files):
        filename = os.path.basename(path)
        save_path = os.path.join(args.output_dir, filename)
        
        # 加载低清图
        lr_tensor, (w, h) = preprocess_image(path, device)
        
        # 计算目标尺寸
        target_h, target_w = h * args.scale, w * args.scale
        
        # [重要步骤]
        # MeanFlow 这种 Flow Matching 方法通常是在同一分辨率下优化残差
        # 所以我们需要先把 LR 图片插值放大到目标尺寸，作为 z_t (t=1)
        lr_upsampled = F.interpolate(
            lr_tensor, 
            size=(target_h, target_w), 
            mode='bicubic', 
            align_corners=False
        )
        
        # 推理
        hr_tensor = run_sr(model, lr_upsampled, device)
        
        # 保存
        res_img = postprocess_image(hr_tensor)
        res_img.save(save_path)
        
    print(f"\n完成！结果已保存在: {args.output_dir}")

if __name__ == '__main__':
    main()