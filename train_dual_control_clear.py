#!/usr/bin/env python
"""
Dual-Stream FLUX SR ControlNet Training with CLEAR Acceleration

🚀 CLEAR 加速：将 O(N²) 注意力线性化为 O(N)

⚠️ 前置步骤：下载 CLEAR 文件和预训练权重
    
    # 1. 下载 CLEAR 核心文件 (放在同级目录)
    wget https://raw.githubusercontent.com/Huage001/CLEAR/main/transformer_flux.py
    wget https://raw.githubusercontent.com/Huage001/CLEAR/main/attention_processor.py
    
    # 2. 下载预训练权重
    mkdir ckpt
    wget https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_16_down_4.safetensors \
         -O ckpt/clear_local_16_down_4.safetensors

Usage:
    accelerate launch --num_processes=8 train_dual_control_clear.py \
        --hr_dir Data/Mix15K_HR \
        --lr_dir Data/Mix15K_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --epochs 20 --lr 1e-5 \
        --window_size 16 --down_factor 4 \
        --clear_ckpt ckpt/clear_local_16_down_4.safetensors
"""

import os
import gc
import math
import argparse
import numpy as np
from datetime import datetime
from PIL import Image

import torch
# 🌟 修复 FlexAttention 共享内存溢出的致命报错
import torch._inductor.config as inductor_config
inductor_config.max_autotune = True
inductor_config.coordinate_descent_tuning = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

# 🌟 从 CLEAR 仓库导入
from attention_processor import (
    LocalFlexAttnProcessor, 
    LocalDownsampleFlexAttnProcessor,
    init_local_mask_flex, 
    init_local_downsample_mask_flex
)


# ============================================================================
# Pixel Feature Extractor
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    def __init__(self, latent_channels=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
        )
        
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        return self.zero_conv(self.encoder(x))


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
        assert len(self.hr_files) == len(self.lr_files)
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        
        lr_w, lr_h = lr_img.size
        crop_lr = self.resolution // 4
        
        if lr_w > crop_lr and lr_h > crop_lr:
            x = np.random.randint(0, lr_w - crop_lr)
            y = np.random.randint(0, lr_h - crop_lr)
            lr_img = lr_img.crop((x, y, x + crop_lr, y + crop_lr))
            hr_img = hr_img.crop((x * 4, y * 4, (x + crop_lr) * 4, (y + crop_lr) * 4))
        
        lr_up = lr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
        hr_img = hr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
        
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Dual-Stream FLUX SR System with CLEAR
# ============================================================================

class DualStreamFLUXSR_CLEAR(nn.Module):
    """
    Dual-Stream FLUX SR with CLEAR Acceleration
    
    🌟 关键：加载 CLEAR 预训练权重到 Transformer
    """
    
    def __init__(self, model_name, device, pretrained_controlnet=None,
                 train_controlnet=True, pixel_weight=0.1,
                 window_size=16, down_factor=4, clear_ckpt=None, 
                 resolution=512): # 🌟 新增
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        self.window_size = window_size
        self.down_factor = down_factor
        self.clear_ckpt = clear_ckpt
        self.resolution = resolution
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self._cached_embeds = None
        
        self._load_models(pretrained_controlnet)
    
    def _load_models(self, pretrained_controlnet):
        from diffusers import AutoencoderKL, FluxControlNetModel
        from transformer_flux import FluxTransformer2DModel  # 🌟 使用 CLEAR 的 transformer
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        from safetensors.torch import load_file
        import time
        
        dtype = torch.bfloat16
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if local_rank > 0:
            time.sleep(local_rank * 5)
        
        # Load VAE
        print(f"[Rank {local_rank}] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        
        # Cache text embeddings
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
        
        # 🌟 Load Transformer (使用 CLEAR 的 FluxTransformer2DModel)
        print(f"[Rank {local_rank}] Loading FLUX Transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # 🌟 启用 CLEAR 并加载预训练权重
        if self.clear_ckpt:
            self._enable_clear(local_rank, dtype)
        
        # Load ControlNet
        print(f"[Rank {local_rank}] Loading ControlNet...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            pretrained_controlnet, torch_dtype=dtype
        ).to(self.device)
        
        if self.train_controlnet:
            self.controlnet.requires_grad_(True)
        else:
            self.controlnet.requires_grad_(False)
        
        # Initialize Pixel Feature Extractor
        print(f"[Rank {local_rank}] Initializing Pixel Feature Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(True)
        # 为 ControlNet 开启专属加速
        self._enable_flash_attention(local_rank)
        print(f"[Rank {local_rank}] GPU memory: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB")
    
    def _enable_clear(self, local_rank, dtype):
        """启用 CLEAR 并加载预训练权重"""
        from safetensors.torch import load_file
        
        # 🌟 极其致命的修复：CLEAR 需要的是 Patch Grid 的尺寸，而不是原图分辨率！
        # FLUX 的总空间降维率是 16 (VAE 缩小 8 倍，Patchify 再缩小 2 倍)
        patch_size = self.resolution // 16  # 对于 512 分辨率，这里自动等于 32
        text_length = 512
        
        print(f"[Rank {local_rank}] [CLEAR] Enabling CLEAR attention...")
        print(f"[Rank {local_rank}] [CLEAR] window_size={self.window_size}, down_factor={self.down_factor}, patch_size={patch_size}x{patch_size}")
        
        # 🌟 Step 1: 初始化 attention mask (严格传入 patch_size)
        if self.down_factor == 1:
            init_local_mask_flex(
                height=patch_size, width=patch_size,  
                text_length=text_length, window_size=self.window_size,
                device=self.device
            )
            attn_processors = {}
            for k in self.transformer.attn_processors.keys():
                attn_processors[k] = LocalFlexAttnProcessor()
        else:
            init_local_downsample_mask_flex(
                height=patch_size, width=patch_size,  
                text_length=text_length, window_size=self.window_size,
                down_factor=self.down_factor, device=self.device
            )
            attn_processors = {}
            for k in self.transformer.attn_processors.keys():
                attn_processors[k] = LocalDownsampleFlexAttnProcessor(
                    down_factor=self.down_factor
                ).to(self.device, dtype)
        
        # 设置 processors
        self.transformer.set_attn_processor(attn_processors)
        
        # 🌟 Step 2: 加载 CLEAR 预训练权重
        print(f"[Rank {local_rank}] [CLEAR] Loading pretrained weights: {self.clear_ckpt}")
        clear_state_dict = load_file(self.clear_ckpt)
        
        missing_keys, unexpected_keys = self.transformer.load_state_dict(clear_state_dict, strict=False)
        
        attn_keys = ['.attn.to_q.', '.attn.to_k.', '.attn.to_v.', '.attn.to_out.', 'spatial_weight']
        critical_missing = [k for k in missing_keys if any(x in k for x in attn_keys)]
        
        if critical_missing:
            print(f"[Rank {local_rank}] [CLEAR] ⚠ Missing {len(critical_missing)} attention keys")
        else:
            print(f"[Rank {local_rank}] [CLEAR] ✓ All attention weights loaded")
    
    def _enable_flash_attention(self, local_rank=0):
        """🌟 仅对 ControlNet 启用原生 Flash Attention"""
        try:
            import xformers
            self.controlnet.enable_xformers_memory_efficient_attention()
            if local_rank == 0:
                print(f"[Rank {local_rank}] [Flash] ✓ Enabled xformers for ControlNet")
        except ImportError:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                if local_rank == 0:
                    print(f"[Rank {local_rank}] [Flash] ✓ Enabled PyTorch SDPA for ControlNet")


    def get_trainable_params(self):
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
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        noisy = noisy.to(dtype)
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        t = t.to(dtype)
        
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        pixel_feat = self.pixel_extractor(lr_pixel).to(dtype)
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        
        packed_noisy = self._pack(noisy)
        packed_cond = self._pack(fused_cond)
        
        # ControlNet
        ctrl_out = self.controlnet(
            hidden_states=packed_noisy,
            controlnet_cond=packed_cond,
            timestep=t,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=torch.full((B,), guidance, device=device, dtype=dtype),
            return_dict=False,
        )
        
        ctrl_block_samples = [x.to(dtype) for x in ctrl_out[0]] if ctrl_out[0] else None
        ctrl_single_samples = [x.to(dtype) for x in ctrl_out[1]] if ctrl_out[1] else None
        
        # Transformer (with CLEAR attention)
        pred = self.transformer(
            hidden_states=packed_noisy,
            timestep=t,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=torch.full((B,), guidance, device=device, dtype=dtype),
            controlnet_block_samples=ctrl_block_samples,
            controlnet_single_block_samples=ctrl_single_samples,
            return_dict=False,
        )[0]
        
        return self._unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5):
        B = lr_lat.shape[0]
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        lat = torch.randn_like(lr_lat)
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
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16
    
    t = torch.rand(B, device=device, dtype=dtype)
    noise = torch.randn_like(hr_lat)
    
    t_expand = t.view(B, 1, 1, 1)
    z_t = (1 - t_expand) * hr_lat + t_expand * noise
    v_target = noise - hr_lat
    
    v_pred = system(z_t, lr_lat, lr_pixel, t)
    
    return F.mse_loss(v_pred.float(), v_target.float())


def calculate_psnr(img1, img2):
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


@torch.no_grad()
def validate(system, accelerator, val_loader, device, num_samples=5, num_steps=20):
    unwrapped = accelerator.unwrap_model(system)
    unwrapped.pixel_extractor.eval()
    unwrapped.controlnet.eval()
    
    psnr_list = []
    dtype = torch.bfloat16
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        hr = batch['hr'].to(device).to(dtype)
        lr = batch['lr'].to(device).to(dtype)
        
        hr_lat = unwrapped.encode(hr)
        lr_lat = unwrapped.encode(lr)
        
        sr_lat = unwrapped.inference(lr_lat, lr, num_steps=num_steps)
        sr = unwrapped.decode(sr_lat)
        
        psnr = calculate_psnr(sr.float(), hr.float())
        psnr_list.append(psnr)
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, accelerator, epoch, loss, psnr, args, path):
    unwrapped = accelerator.unwrap_model(system)
    torch.save({
        'epoch': epoch,
        'controlnet': unwrapped.controlnet.state_dict(),
        'pixel_extractor': unwrapped.pixel_extractor.state_dict(),
        'pixel_weight': args.pixel_weight,
        'window_size': args.window_size,
        'down_factor': args.down_factor,
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
    parser.add_argument('--train_controlnet', action='store_true', default=True)
    parser.add_argument('--freeze_controlnet', action='store_true', default=False)
    parser.add_argument('--pixel_weight', type=float, default=0.1)
    
    # 🌟 CLEAR 参数
    parser.add_argument('--window_size', type=int, default=16,
                        help='CLEAR window size. Options: 8, 16, 32')
    parser.add_argument('--down_factor', type=int, default=4,
                        help='Downsample factor. 1=no downsample, 4=recommended')
    parser.add_argument('--clear_ckpt', type=str, required=True,
                        help='Path to CLEAR pretrained weights (REQUIRED)')
    
    # Training
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=20)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/dual_clear')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 验证 CLEAR 权重文件存在
    if not os.path.exists(args.clear_ckpt):
        print("=" * 70)
        print("❌ ERROR: CLEAR checkpoint not found!")
        print(f"   Expected path: {args.clear_ckpt}")
        print("")
        print("📥 Please download CLEAR weights first:")
        print("   mkdir ckpt")
        print("   wget https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_16_down_4.safetensors \\")
        print("        -O ckpt/clear_local_16_down_4.safetensors")
        print("=" * 70)
        return
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
    )
    
    device = accelerator.device
    is_main = accelerator.is_main_process
    
    set_seed(args.seed)
    
    train_controlnet = args.train_controlnet and not args.freeze_controlnet
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "ctrl+pixel" if train_controlnet else "pixel_only"
    save_dir = os.path.join(args.output_dir, 
                            f"{timestamp}_{mode}_pw{args.pixel_weight}_clear{args.window_size}_d{args.down_factor}")
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 70)
        print("🚀 Dual-Stream FLUX SR with CLEAR Acceleration")
        print("=" * 70)
        print(f"CLEAR Configuration:")
        print(f"  - window_size: {args.window_size}")
        print(f"  - down_factor: {args.down_factor}")
        print(f"  - checkpoint: {args.clear_ckpt}")
        print(f"Pixel Weight: {args.pixel_weight}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"GPUs: {accelerator.num_processes}")
        print(f"Output: {save_dir}")
        print("=" * 70)
    
    # Create model with CLEAR
    system = DualStreamFLUXSR_CLEAR(
        args.model_name,
        device,
        args.pretrained_controlnet,
        train_controlnet=train_controlnet,
        pixel_weight=args.pixel_weight,
        window_size=args.window_size,
        down_factor=args.down_factor,
        clear_ckpt=args.clear_ckpt,
        resolution=args.resolution, # 🌟 传入当前设定的分辨率
    )
    
    trainable_params = system.get_trainable_params()
    total_params = sum(p.numel() for p in trainable_params)
    
    if train_controlnet:
        optimizer_grouped_parameters = [
            {"params": system.controlnet.parameters(), "lr": args.lr},
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    
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
        persistent_workers=True,
        prefetch_factor=2
    )
    
    if is_main:
        print(f"[Data] Training samples: {len(train_dataset)}")
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(args.val_hr_dir, args.val_lr_dir, args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        if is_main:
            print(f"[Data] Validation samples: {len(val_dataset)}")
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = args.warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    system, optimizer, train_loader, scheduler = accelerator.prepare(
        system, optimizer, train_loader, scheduler
    )
    
    # Resume
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        if is_main:
            print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        unwrapped = accelerator.unwrap_model(system)
        
        raw_state = ckpt['pixel_extractor']
        clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
        unwrapped.pixel_extractor.load_state_dict(clean_state)
        
        if train_controlnet and 'controlnet' in ckpt:
            raw_state = ckpt['controlnet']
            clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
            unwrapped.controlnet.load_state_dict(clean_state)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        if is_main:
            print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    history = {'train_loss': [], 'val_psnr': []}
    
    if is_main:
        print("\n[Training] Starting...\n")
    
    for epoch in range(start_epoch, args.epochs):
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.train()
        if train_controlnet:
            unwrapped.controlnet.train()
        
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        
        for batch in pbar:
            hr = batch['hr'].to(device).to(torch.bfloat16)
            lr = batch['lr'].to(device).to(torch.bfloat16)
            
            with torch.no_grad():
                hr_lat = unwrapped.encode(hr)
                lr_lat = unwrapped.encode(lr)
            
            with accelerator.accumulate(system):
                with accelerator.autocast():
                    loss = compute_flow_matching_loss(system, hr_lat, lr_lat, lr)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        
        if is_main:
            val_psnr = 0.0
            if val_loader and (epoch + 1) % args.val_interval == 0:
                val_psnr = validate(system, accelerator, val_loader, device, num_samples=5)
                history['val_psnr'].append(val_psnr)
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
                torch.cuda.empty_cache()
            else:
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr, args,
                               os.path.join(save_dir, f'epoch{epoch+1}.pt'))
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr, args,
                               os.path.join(save_dir, 'best_model.pt'))
                print(f"[Epoch {epoch+1}] ✓ New best PSNR: {best_psnr:.2f} dB")
        
        accelerator.wait_for_everyone()
    
    if is_main:
        save_checkpoint(system, accelerator, args.epochs - 1, avg_loss, best_psnr, args,
                       os.path.join(save_dir, 'final_model.pt'))
        
        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print(f"CLEAR: window_size={args.window_size}, down_factor={args.down_factor}")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()
