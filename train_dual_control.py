#!/usr/bin/env python
"""
Dual-Stream FLUX SR ControlNet Training (Accelerate Version)

核心思想：突破 VAE 的"马赛克瓶颈"
- Latent Stream: FLUX ControlNet 负责结构、语义、细节生成
- Pixel Stream: 轻量 CNN 提取高频边缘特征，无损注入到 Latent Stream

关键技术：
1. Zero Conv 初始化 - 不破坏预训练权重
2. MeanFlow 条件流匹配 - 从 LR 出发而非纯噪声
3. GroupNorm 稳定训练

Usage:
    accelerate launch --num_processes=4 train_dual_control.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --epochs 100 --lr 1e-5
"""

import os
import gc
import math
import argparse
import numpy as np
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm


# ============================================================================
# Pixel Feature Extractor (with GroupNorm for stability)
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """
    从原始像素空间提取高频特征，并映射到 Latent 空间维度。
    使用 Zero Conv 确保初始化时不破坏预训练 ControlNet。
    使用 GroupNorm 稳定训练。
    
    Input: RGB image [B, 3, H, W] (H, W = 512)
    Output: Latent-space features [B, 16, H/8, W/8] (64x64)
    """
    def __init__(self, latent_channels=16):
        super().__init__()
        
        # 编码器：3 通道 RGB → 16 通道 Latent 维度，stride=8 匹配空间分辨率
        # 🌟 加入 GroupNorm 稳定训练
        self.encoder = nn.Sequential(
            # 512 → 256
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            # 256 → 128
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # 128 → 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
        )
        
        # 🌟 Zero Conv: 初始化权重为 0，确保初始时 Pixel 特征全为 0
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        """
        x: [B, 3, 512, 512] - Bicubic upscaled LR image
        return: [B, 16, 64, 64] - Latent-space pixel features
        """
        feat = self.encoder(x)
        return self.zero_conv(feat)


# ============================================================================
# Dataset
# ============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, resolution=512):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR/LR count mismatch: {len(self.hr_files)} vs {len(self.lr_files)}"
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        
        # Random crop
        lr_w, lr_h = lr_img.size
        crop_lr = self.resolution // 4
        
        if lr_w > crop_lr and lr_h > crop_lr:
            x = np.random.randint(0, lr_w - crop_lr)
            y = np.random.randint(0, lr_h - crop_lr)
            lr_img = lr_img.crop((x, y, x + crop_lr, y + crop_lr))
            hr_img = hr_img.crop((x * 4, y * 4, (x + crop_lr) * 4, (y + crop_lr) * 4))
        
        # Bicubic upscale LR to match HR size
        lr_up = lr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
        hr_img = hr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
        
        # To tensor [-1, 1]
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Dual-Stream FLUX SR System
# ============================================================================

class DualStreamFLUXSR(nn.Module):
    """
    Dual-Stream FLUX SR System
    
    - Latent Stream: FLUX ControlNet (frozen or trainable)
    - Pixel Stream: PixelFeatureExtractor (trainable)
    
    融合方式：lr_lat + pixel_feat → ControlNet
    """
    
    def __init__(self, model_name, device, pretrained_controlnet=None, train_controlnet=True):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self._cached_embeds = None
        
        self._load_models(pretrained_controlnet)
    
    def _load_models(self, pretrained_controlnet):
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        import time
        
        dtype = torch.bfloat16
        
        # 🌟 错峰加载：防止多卡同时加载导致内存爆炸
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank > 0:
            time.sleep(local_rank * 5)
        
        # Load VAE
        print(f"[Rank {local_rank}] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        
        # Cache text embeddings (empty prompt for SR)
        print(f"[Rank {local_rank}] Caching text embeddings...")
        text_enc = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        text_enc_2 = T5EncoderModel.from_pretrained(
            self.model_name, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(self.device)
        tok = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        tok_2 = T5TokenizerFast.from_pretrained(self.model_name, subfolder="tokenizer_2")
        
        with torch.no_grad():
            clip_out = text_enc(tok([""], padding="max_length", max_length=77,
                                   truncation=True, return_tensors="pt").input_ids.to(self.device))
            t5_out = text_enc_2(tok_2([""], padding="max_length", max_length=512,
                                     truncation=True, return_tensors="pt").input_ids.to(self.device))
            self._cached_embeds = {
                'pooled': clip_out.pooler_output.to(dtype),
                'prompt': t5_out[0].to(dtype),
                'text_ids': torch.zeros(t5_out[0].shape[1], 3, device=self.device, dtype=dtype),
            }
        
        del text_enc, text_enc_2
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load Transformer (always frozen)
        print(f"[Rank {local_rank}] Loading FLUX Transformer (frozen)...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # Load ControlNet
        print(f"[Rank {local_rank}] Loading ControlNet...")
        if pretrained_controlnet and pretrained_controlnet.lower() != 'none':
            self.controlnet = FluxControlNetModel.from_pretrained(
                pretrained_controlnet, torch_dtype=dtype
            ).to(self.device)
            print(f"[Rank {local_rank}] Loaded pretrained ControlNet: {pretrained_controlnet}")
        else:
            raise ValueError("pretrained_controlnet is required for Dual-Stream training")
        
        if self.train_controlnet:
            self.controlnet.requires_grad_(True)
            print(f"[Rank {local_rank}] ControlNet: trainable")
        else:
            self.controlnet.requires_grad_(False)
            print(f"[Rank {local_rank}] ControlNet: frozen")
        
        # 🌟 Initialize Pixel Feature Extractor
        print(f"[Rank {local_rank}] Initializing Pixel Feature Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(True)
        
        print(f"[Rank {local_rank}] Pixel Extractor params: {sum(p.numel() for p in self.pixel_extractor.parameters()):,}")
        print(f"[Rank {local_rank}] GPU memory: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB")
    
    def get_trainable_params(self):
        """Return all trainable parameters"""
        params = list(self.pixel_extractor.parameters())
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        return [p for p in params if p.requires_grad]
    
    def _pack(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H//2)*(W//2), C*4)
    
    def _unpack(self, x, H, W):
        B, _, D = x.shape
        C = D // 4
        x = x.view(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, H, W)
    
    def _img_ids(self, H, W, device, dtype):
        h, w = H//2, W//2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return ids.reshape(h*w, 3)
    
    @torch.no_grad()
    def encode(self, img):
        return self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, lat):
        return self.vae.decode(lat / self.vae.config.scaling_factor).sample
    
    def forward(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
        """
        Dual-Stream Forward Pass
        """
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16  # 🌟 强制使用 bfloat16
        
        # 🌟 确保所有输入都是 bfloat16
        noisy = noisy.to(dtype)
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        t = t.to(dtype)
        
        # Get cached embeddings
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        # 🌟 Dual-Stream: Extract pixel-level high-frequency features
        pixel_feat = self.pixel_extractor(lr_pixel).to(dtype)  # [B, 16, H, W] 🌟 确保 bfloat16
        
        # 🌟 Fusion: Add pixel features to latent (Zero Conv ensures safe initialization)
        fused_cond = (lr_lat + pixel_feat).to(dtype)
        
        # Pack for transformer
        packed_noisy = self._pack(noisy)
        packed_cond = self._pack(fused_cond)
        
        # ControlNet forward
        # 🌟 FLUX.1-dev ControlNet 必须传 guidance 参数
        # 🌟 确保所有输入都是 bfloat16
        ctrl_kwargs = {
            'hidden_states': packed_noisy.to(dtype),
            'controlnet_cond': packed_cond.to(dtype),
            'timestep': t.to(dtype),
            'encoder_hidden_states': prompt.to(dtype),
            'pooled_projections': pooled.to(dtype),
            'txt_ids': txt_ids.to(dtype),
            'img_ids': img_ids.to(dtype),
            'guidance': torch.full((B,), guidance, device=device, dtype=dtype),
            'return_dict': False,
        }
        
        ctrl_out = self.controlnet(**ctrl_kwargs)
        
        # 🌟 确保 ControlNet 输出是 bfloat16
        ctrl_block_samples = [x.to(dtype) for x in ctrl_out[0]] if ctrl_out[0] is not None else None
        ctrl_single_samples = [x.to(dtype) for x in ctrl_out[1]] if ctrl_out[1] is not None else None
        
        # Transformer forward
        trans_kwargs = {
            'hidden_states': packed_noisy.to(dtype),  # 🌟 确保 bfloat16
            'timestep': t.to(dtype),
            'encoder_hidden_states': prompt.to(dtype),
            'pooled_projections': pooled.to(dtype),
            'txt_ids': txt_ids.to(dtype),
            'img_ids': img_ids.to(dtype),
            'guidance': torch.full((B,), guidance, device=device, dtype=dtype),
            'controlnet_block_samples': ctrl_block_samples,
            'controlnet_single_block_samples': ctrl_single_samples,
            'return_dict': False,
        }
        
        pred = self.transformer(**trans_kwargs)[0]
        return self._unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5):
        """
        🌟 MeanFlow 推理：从 lr_lat + sigma*noise 出发（与训练一致）
        """
        B = lr_lat.shape[0]
        dtype = torch.bfloat16  # 🌟 强制使用 bfloat16
        
        # 确保输入是 bfloat16
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        # MeanFlow: 从 LR latent 出发，而非纯噪声
        sigma = 0.1
        noise = torch.randn_like(lr_lat)
        lat = lr_lat + sigma * noise
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=lr_lat.device, dtype=dtype)
            v = self.forward(lat, lr_lat, lr_pixel, t, guidance)
            lat = lat - dt * v
        
        return lat


# ============================================================================
# Training Functions
# ============================================================================

def compute_flow_matching_loss(system, hr_lat, lr_lat, lr_pixel):
    """
    🌟 MeanFlow 条件流匹配 Loss (修复版)
    确保训练与推理逻辑完全对齐：都从 lr_lat + sigma*noise 出发
    """
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16  
    
    t = torch.rand(B, device=device, dtype=dtype)
    noise = torch.randn_like(hr_lat)
    
    # 1. 构造与推理完全一致的起点 (MeanFlow)
    sigma = 0.1
    x_start = lr_lat + sigma * noise 
    
    # 2. 轨迹：从 x_start (起点) 流向 hr_lat (终点)
    t_expand = t.view(B, 1, 1, 1)
    z_t = (1 - t_expand) * hr_lat + t_expand * x_start
    
    # 3. 目标速度：指向起点方向
    v_target = x_start - hr_lat
    
    # 4. 前向预测
    v_pred = system(z_t, lr_lat, lr_pixel, t)
    
    # 计算 loss 时转回 float32 保证数值稳定
    return F.mse_loss(v_pred.float(), v_target.float())


def calculate_psnr(img1, img2):
    """Calculate PSNR between two tensors in [-1, 1]"""
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


@torch.no_grad()
def validate(system, val_loader, device, num_samples=5, num_steps=20):
    """Validate the model"""
    # 🌟 确保所有模块在 eval 模式
    if hasattr(system.pixel_extractor, 'module'):
        system.pixel_extractor.module.eval()
    else:
        system.pixel_extractor.eval()
    
    if hasattr(system.controlnet, 'module'):
        system.controlnet.module.eval()
    else:
        system.controlnet.eval()
    
    psnr_list = []
    dtype = torch.bfloat16  # 🌟 与训练一致
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        hr = batch['hr'].to(device).to(dtype)
        lr = batch['lr'].to(device).to(dtype)
        
        # Encode
        hr_lat = system.encode(hr)
        lr_lat = system.encode(lr)
        
        # Inference with Dual-Stream
        sr_lat = system.inference(lr_lat, lr, num_steps=num_steps)
        sr = system.decode(sr_lat)
        
        psnr = calculate_psnr(sr.float(), hr.float())
        psnr_list.append(psnr)
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, epoch, loss, psnr, path):
    # 🌟 强制脱壳获取原始模型
    unwrapped_sys = accelerator.unwrap_model(system)
    
    # 提取 state_dict 时，DeepSpeed 会自动帮你收集分片后的权重
    controlnet_state = unwrapped_sys.controlnet.state_dict()
    pixel_extractor_state = unwrapped_sys.pixel_extractor.state_dict()
    
    torch.save({
        'epoch': epoch,
        'controlnet': controlnet_state,
        'pixel_extractor': pixel_extractor_state,
        'loss': loss,
        'psnr': psnr,
    }, path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    
    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, 
                        default='jasperai/Flux.1-dev-Controlnet-Upscaler')
    parser.add_argument('--train_controlnet', action='store_true', default=True,
                        help='Train ControlNet along with Pixel Extractor')
    parser.add_argument('--freeze_controlnet', action='store_true', default=False,
                        help='Freeze ControlNet, only train Pixel Extractor')
    
    # Training
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    
    # 🌟 显存优化选项
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--low_memory', action='store_true', default=False,
                        help='Low memory mode: freeze ControlNet, only train PixelExtractor')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/dual_control')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 🌟 强制绑定物理卡号，防止多卡挤在 cuda:0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
    )
    
    device = accelerator.device
    is_main = accelerator.is_main_process  # 🌟 用于控制只在主进程打印/保存
    
    # Set seed
    set_seed(args.seed)
    
    # Determine if training ControlNet
    # 🌟 low_memory 模式下强制冻结 ControlNet
    train_controlnet = args.train_controlnet and not args.freeze_controlnet and not args.low_memory
    
    if args.low_memory and is_main:
        print("[Memory] Low memory mode: ControlNet frozen, only training PixelExtractor")
    
    # Create output directory (only main process)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "ctrl+pixel" if train_controlnet else "pixel_only"
    save_dir = os.path.join(args.output_dir, f"{timestamp}_{mode}")
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 70)
        print("Dual-Stream FLUX SR ControlNet Training (Accelerate)")
        print("=" * 70)
        print(f"Model: {args.model_name}")
        print(f"ControlNet: {args.pretrained_controlnet}")
        print(f"Mode: {'ControlNet + Pixel Extractor' if train_controlnet else 'Pixel Extractor only'}")
        print(f"Resolution: {args.resolution}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Output: {save_dir}")
        print(f"GPUs: {accelerator.num_processes}")
        print("=" * 70)
    
    # Create model
    system = DualStreamFLUXSR(
        args.model_name, 
        device, 
        args.pretrained_controlnet,
        train_controlnet=train_controlnet
    )
    
    # 🌟 Gradient Checkpointing 减少显存
    if args.gradient_checkpointing:
        if hasattr(system.transformer, 'enable_gradient_checkpointing'):
            system.transformer.enable_gradient_checkpointing()
        if hasattr(system.controlnet, 'enable_gradient_checkpointing'):
            system.controlnet.enable_gradient_checkpointing()
        if is_main:
            print("[Memory] Gradient checkpointing enabled")
    
    # Get trainable parameters
    trainable_params = system.get_trainable_params()
    total_params = sum(p.numel() for p in trainable_params)
    if is_main:
        print(f"[Training] Trainable parameters: {total_params:,}")
    
    # Create datasets
    train_dataset = SRDataset(args.hr_dir, args.lr_dir, args.resolution)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # 🌟 保持 Worker 存活
        prefetch_factor=2         # 🌟 提前预读数据
    )
    
    if is_main:
        print(f"[Data] Training samples: {len(train_dataset)}")
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(args.val_hr_dir, args.val_lr_dir, args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        if is_main:
            print(f"[Data] Validation samples: {len(val_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = args.warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare with Accelerator
    # 🌟 DeepSpeed ZeRO 终极修复：无论冻结与否，必须始终 prepare 整个 system！
    # DeepSpeed 会自动识别并忽略 requires_grad=False 的冻结参数，绝不会报错。
    system, optimizer, train_loader, scheduler = accelerator.prepare(
        system, optimizer, train_loader, scheduler
    )
    
    if not train_controlnet and is_main:
        print("[Memory] ControlNet frozen, but entire system is wrapped safely by DeepSpeed/DDP.")
    
    # Resume if specified
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        if is_main:
            print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Load Pixel Extractor
        pixel_ext = system.pixel_extractor.module if hasattr(system.pixel_extractor, 'module') else system.pixel_extractor
        raw_state = ckpt['pixel_extractor']
        clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
        pixel_ext.load_state_dict(clean_state)
        
        # Load ControlNet if training it
        if train_controlnet and 'controlnet' in ckpt:
            ctrl = system.controlnet.module if hasattr(system.controlnet, 'module') else system.controlnet
            raw_state = ckpt['controlnet']
            clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
            ctrl.load_state_dict(clean_state)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        if is_main:
            print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # Training history
    history = {'train_loss': [], 'val_psnr': []}
    
    # Training loop
    if is_main:
        print("\n[Training] Starting...\n")
    
    for epoch in range(start_epoch, args.epochs):
        # 🌟 确保所有可训练模块都在训练模式
        if hasattr(system.pixel_extractor, 'module'):
            system.pixel_extractor.module.train()
        else:
            system.pixel_extractor.train()
        
        if train_controlnet:
            if hasattr(system.controlnet, 'module'):
                system.controlnet.module.train()
            else:
                system.controlnet.train()
        
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        
        for batch in pbar:
            hr = batch['hr'].to(device).to(torch.bfloat16)  # 🌟 确保 bfloat16
            lr = batch['lr'].to(device).to(torch.bfloat16)  # 🌟 确保 bfloat16
            # 🌟 修复：获取脱壳后的原始系统，才能调用自定义的 encode 和 inference 方法
            unwrapped_sys = accelerator.unwrap_model(system)
            # Encode to latent space
            with torch.no_grad():
                hr_lat = unwrapped_sys.encode(hr)
                lr_lat = unwrapped_sys.encode(lr)
            
            # Compute loss with autocast
            # 🌟 关键修复：根据是否训练 ControlNet 选择 accumulate 的模块
            if train_controlnet:
                accumulate_modules = (system.controlnet, system.pixel_extractor)
            else:
                accumulate_modules = (system.pixel_extractor,)
            
            with accelerator.accumulate(*accumulate_modules):
                # 🌟 加上这层保护罩，让 bf16 运行得更顺滑
                with accelerator.autocast():
                    loss = compute_flow_matching_loss(system, hr_lat, lr_lat, lr)
                
                # Backward
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        
        # 🌟 Validation and saving only on main process
        if is_main:
            val_psnr = 0.0
            if val_loader and (epoch + 1) % args.val_interval == 0:
                val_psnr = validate(system, val_loader, device, num_samples=5)
                history['val_psnr'].append(val_psnr)
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
                
                # 🌟 验证结束后清理显存碎片
                torch.cuda.empty_cache()
            else:
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    system, epoch, avg_loss, val_psnr,
                    os.path.join(save_dir, f'epoch{epoch+1}.pt')
                )
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    system, epoch, avg_loss, val_psnr,
                    os.path.join(save_dir, 'best_model.pt')
                )
                print(f"[Epoch {epoch+1}] ✓ New best PSNR: {best_psnr:.2f} dB")
        
        # Synchronize
        accelerator.wait_for_everyone()
    
    # Save final model (main process only)
    if is_main:
        save_checkpoint(
            system, args.epochs - 1, avg_loss, best_psnr,
            os.path.join(save_dir, 'final_model.pt')
        )
        
        # Save training summary
        summary_path = os.path.join(save_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Dual-Stream FLUX SR ControlNet Training Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"ControlNet: {args.pretrained_controlnet}\n")
            f.write(f"Mode: {'ControlNet + Pixel Extractor' if train_controlnet else 'Pixel Extractor only'}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"GPUs: {accelerator.num_processes}\n")
            f.write(f"Best PSNR: {best_psnr:.2f} dB\n")
            f.write(f"Final loss: {avg_loss:.6f}\n")
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints saved to: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()