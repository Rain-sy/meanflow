#!/usr/bin/env python
"""
Dual-Stream FLUX SR ControlNet Training

核心思想：突破 VAE 的"马赛克瓶颈"
- Latent Stream: FLUX ControlNet 负责结构、语义、细节生成
- Pixel Stream: 轻量 CNN 提取高频边缘特征，无损注入到 Latent Stream

关键技术：Zero Conv 初始化
- 初始时 Pixel 特征全为 0，不破坏预训练 ControlNet
- 训练过程中逐渐学习有用的高频特征

Usage:
    deepspeed --num_gpus=4 train_dual_control.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
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

import deepspeed
from tqdm import tqdm


# ============================================================================
# Pixel Feature Extractor (轻量级高频特征提取器)
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """
    从原始像素空间提取高频特征，并映射到 Latent 空间维度。
    使用 Zero Conv 确保初始化时不破坏预训练 ControlNet。
    
    Input: RGB image [B, 3, H, W] (H, W = 512)
    Output: Latent-space features [B, 16, H/8, W/8] (64x64)
    """
    def __init__(self, latent_channels=16):
        super().__init__()
        
        # 编码器：3 通道 RGB → 16 通道 Latent 维度，stride=8 匹配空间分辨率
        self.encoder = nn.Sequential(
            # 512 → 256
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            
            # 256 → 128
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            
            # 128 → 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
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
        
        dtype = torch.bfloat16
        
        # Load VAE
        print("[System] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        
        # Cache text embeddings (empty prompt for SR)
        print("[System] Caching text embeddings...")
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
        print("[System] Loading FLUX Transformer (frozen)...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        # Load ControlNet
        print("[System] Loading ControlNet...")
        if pretrained_controlnet and pretrained_controlnet.lower() != 'none':
            self.controlnet = FluxControlNetModel.from_pretrained(
                pretrained_controlnet, torch_dtype=dtype
            ).to(self.device)
            print(f"[System] Loaded pretrained ControlNet: {pretrained_controlnet}")
        else:
            raise ValueError("pretrained_controlnet is required for Dual-Stream training")
        
        if self.train_controlnet:
            self.controlnet.requires_grad_(True)
            print("[System] ControlNet: trainable")
        else:
            self.controlnet.requires_grad_(False)
            print("[System] ControlNet: frozen")
        
        # 🌟 Initialize Pixel Feature Extractor
        print("[System] Initializing Pixel Feature Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(True)
        
        print(f"[System] Pixel Extractor params: {sum(p.numel() for p in self.pixel_extractor.parameters()):,}")
        print(f"[System] GPU memory: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB")
    
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
        
        Args:
            noisy: [B, 16, H, W] - Noisy latent
            lr_lat: [B, 16, H, W] - LR latent from VAE
            lr_pixel: [B, 3, H*8, W*8] - Original LR pixel image (bicubic upscaled)
            t: [B] - Timesteps
            guidance: float - Guidance scale
        
        Returns:
            v_pred: [B, 16, H, W] - Predicted velocity
        """
        B, C, H, W = noisy.shape
        device, dtype = noisy.device, noisy.dtype
        
        # Get cached embeddings
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        txt_ids = self._cached_embeds['text_ids']
        img_ids = self._img_ids(H, W, device, dtype)
        
        # 🌟 Dual-Stream: Extract pixel-level high-frequency features
        pixel_feat = self.pixel_extractor(lr_pixel)  # [B, 16, H, W]
        
        # 🌟 Fusion: Add pixel features to latent (Zero Conv ensures safe initialization)
        fused_cond = lr_lat.to(dtype) + pixel_feat.to(dtype)
        
        # Pack for transformer
        packed_noisy = self._pack(noisy)
        packed_cond = self._pack(fused_cond)
        
        # ControlNet forward
        ctrl_kwargs = {
            'hidden_states': packed_noisy,
            'controlnet_cond': packed_cond,  # 使用融合后的条件
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_ids,
            'return_dict': False,
        }
        if hasattr(self.controlnet, 'time_text_embed') and \
           "Guidance" in type(self.controlnet.time_text_embed).__name__:
            ctrl_kwargs['guidance'] = torch.full((B,), guidance, device=device, dtype=dtype)
        
        ctrl_out = self.controlnet(**ctrl_kwargs)
        
        # Transformer forward
        trans_kwargs = {
            'hidden_states': packed_noisy,
            'timestep': t,
            'encoder_hidden_states': prompt,
            'pooled_projections': pooled,
            'txt_ids': txt_ids,
            'img_ids': img_ids,
            'controlnet_block_samples': ctrl_out[0],
            'controlnet_single_block_samples': ctrl_out[1],
            'return_dict': False,
        }
        if hasattr(self.transformer, 'time_text_embed') and \
           "Guidance" in type(self.transformer.time_text_embed).__name__:
            trans_kwargs['guidance'] = torch.full((B,), guidance, device=device, dtype=dtype)
        
        pred = self.transformer(**trans_kwargs)[0]
        return self._unpack(pred, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5):
        """
        Dual-Stream Inference
        
        Args:
            lr_lat: [B, 16, H, W] - LR latent
            lr_pixel: [B, 3, H*8, W*8] - LR pixel image
            num_steps: int - Number of denoising steps
            guidance: float - Guidance scale
        
        Returns:
            sr_lat: [B, 16, H, W] - SR latent
        """
        B = lr_lat.shape[0]
        
        # Start from random noise
        lat = torch.randn_like(lr_lat)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=lr_lat.device, dtype=lr_lat.dtype)
            v = self.forward(lat, lr_lat, lr_pixel, t, guidance)
            lat = lat - dt * v
        
        return lat


# ============================================================================
# Training Functions
# ============================================================================

def compute_flow_matching_loss(system, hr_lat, lr_lat, lr_pixel):
    """
    Compute Flow Matching loss for Dual-Stream system
    
    Args:
        system: DualStreamFLUXSR model
        hr_lat: [B, 16, H, W] - HR latent (target)
        lr_lat: [B, 16, H, W] - LR latent (condition)
        lr_pixel: [B, 3, H*8, W*8] - LR pixel image
    
    Returns:
        loss: scalar tensor
    """
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = hr_lat.dtype
    
    # Sample timestep
    t = torch.rand(B, device=device, dtype=dtype)
    
    # Sample noise
    noise = torch.randn_like(hr_lat)
    
    # Create noisy latent (linear interpolation)
    # z_t = (1 - t) * x_0 + t * noise
    t_expand = t.view(B, 1, 1, 1)
    z_t = (1 - t_expand) * hr_lat + t_expand * noise
    
    # Target velocity: noise - x_0
    v_target = noise - hr_lat
    
    # Predict velocity (with Dual-Stream conditioning)
    v_pred = system(z_t, lr_lat, lr_pixel, t)
    
    # MSE Loss
    loss = F.mse_loss(v_pred.float(), v_target.float())
    
    return loss


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
    system.eval()
    psnr_list = []
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        hr = batch['hr'].to(device)
        lr = batch['lr'].to(device)
        
        # Encode
        hr_lat = system.encode(hr)
        lr_lat = system.encode(lr)
        
        # Inference with Dual-Stream
        sr_lat = system.inference(lr_lat, lr, num_steps=num_steps)
        sr = system.decode(sr_lat)
        
        psnr = calculate_psnr(sr, hr)
        psnr_list.append(psnr)
    
    system.train()
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, optimizer, epoch, loss, psnr, path):
    """Save checkpoint with module prefix handling"""
    # Handle DeepSpeed wrapped model
    if hasattr(system, 'module'):
        controlnet_state = system.module.controlnet.state_dict()
        pixel_extractor_state = system.module.pixel_extractor.state_dict()
    else:
        controlnet_state = system.controlnet.state_dict()
        pixel_extractor_state = system.pixel_extractor.state_dict()
    
    torch.save({
        'epoch': epoch,
        'controlnet': controlnet_state,
        'pixel_extractor': pixel_extractor_state,
        'optimizer': optimizer.state_dict() if optimizer else None,
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
    
    # DeepSpeed
    parser.add_argument('--local_rank', type=int, default=0)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/dual_control')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(device)
    
    # Determine if training ControlNet
    train_controlnet = args.train_controlnet and not args.freeze_controlnet
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "ctrl+pixel" if train_controlnet else "pixel_only"
    save_dir = os.path.join(args.output_dir, f"{timestamp}_{mode}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Dual-Stream FLUX SR ControlNet Training")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"ControlNet: {args.pretrained_controlnet}")
    print(f"Mode: {'ControlNet + Pixel Extractor' if train_controlnet else 'Pixel Extractor only'}")
    print(f"Resolution: {args.resolution}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {save_dir}")
    print("=" * 70)
    
    # Create model
    system = DualStreamFLUXSR(
        args.model_name, 
        device, 
        args.pretrained_controlnet,
        train_controlnet=train_controlnet
    )
    
    # Get trainable parameters
    trainable_params = system.get_trainable_params()
    total_params = sum(p.numel() for p in trainable_params)
    print(f"[Training] Trainable parameters: {total_params:,}")
    
    # Create datasets
    train_dataset = SRDataset(args.hr_dir, args.lr_dir, args.resolution)
    print(f"[Data] Training samples: {len(train_dataset)}")
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(args.val_hr_dir, args.val_lr_dir, args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        print(f"[Data] Validation samples: {len(val_dataset)}")
    
    # DeepSpeed config
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_epochs * len(train_dataset),
                "total_num_steps": args.epochs * len(train_dataset)
            }
        },
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "contiguous_gradients": True,
            "overlap_comm": True
        },
        "gradient_clipping": 1.0
    }
    
    # Initialize DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=system,
        model_parameters=trainable_params,
        training_data=train_dataset,
        config=ds_config
    )
    
    # Resume if specified
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Load Pixel Extractor
        system.pixel_extractor.load_state_dict(ckpt['pixel_extractor'])
        
        # Load ControlNet if training it
        if train_controlnet and 'controlnet' in ckpt:
            raw_state = ckpt['controlnet']
            clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
            system.controlnet.load_state_dict(clean_state)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # Training history
    history = {'train_loss': [], 'val_psnr': []}
    
    # Training loop
    print("\n[Training] Starting...\n")
    
    for epoch in range(start_epoch, args.epochs):
        model_engine.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            hr = batch['hr'].to(device)
            lr = batch['lr'].to(device)
            
            # Encode to latent space
            with torch.no_grad():
                hr_lat = system.encode(hr)
                lr_lat = system.encode(lr)
            
            # Compute loss (lr is also the pixel input)
            loss = compute_flow_matching_loss(model_engine, hr_lat, lr_lat, lr)
            
            # Backward
            model_engine.backward(loss)
            model_engine.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_psnr = 0.0
        if val_loader and (epoch + 1) % args.val_interval == 0:
            val_psnr = validate(system, val_loader, device, num_samples=5)
            history['val_psnr'].append(val_psnr)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
        else:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model_engine, optimizer, epoch, avg_loss, val_psnr,
                os.path.join(save_dir, f'epoch{epoch+1}.pt')
            )
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                model_engine, optimizer, epoch, avg_loss, val_psnr,
                os.path.join(save_dir, 'best_model.pt')
            )
            print(f"[Epoch {epoch+1}] ✓ New best PSNR: {best_psnr:.2f} dB")
    
    # Save final model
    save_checkpoint(
        model_engine, optimizer, args.epochs - 1, avg_loss, val_psnr,
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
        f.write(f"Best PSNR: {best_psnr:.2f} dB\n")
        f.write(f"Final loss: {avg_loss:.6f}\n")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()