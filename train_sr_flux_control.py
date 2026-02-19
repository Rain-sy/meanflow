#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Super-Resolution with Frozen FLUX DiT + Trainable ControlNet

Architecture:
    - FLUX DiT: Frozen (12B parameters)
    - ControlNet: Trainable (~3B parameters)
    - Training: Flow matching loss

Memory Analysis:
    FLUX DiT (12B bf16):     ~24 GB
    ControlNet (~3B bf16):   ~6 GB  
    VAE + Text Encoders:     ~5 GB
    Activations + Gradients: ~10 GB
    Total:                   ~45 GB (fits in 48GB!)

    ⚠️  DDP adds ~6-12GB gradient buffer per GPU, causing OOM!
    ✓  Solution: Single GPU or manual sharding

Usage:
    # Single GPU (recommended for 48GB GPU)
    python train_sr_flux_control.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --batch_size 1

    # Multiple GPUs - run on each GPU separately with different shards
    CUDA_VISIBLE_DEVICES=0 python train_sr_flux_control.py --shard 0 --num_shards 8 &
    CUDA_VISIBLE_DEVICES=1 python train_sr_flux_control.py --shard 1 --num_shards 8 &
    ...
"""

import os
import sys
import math
import random
import argparse
from datetime import datetime
from typing import Optional, Dict, List
from contextlib import nullcontext

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from accelerate import Accelerator

# ============================================================================
# Dataset
# ============================================================================

class SRDataset(Dataset):
    """SR Dataset: HR and LR image pairs"""
    
    def __init__(
        self, 
        hr_dir: str, 
        lr_dir: str, 
        resolution: int = 512, 
        scale: int = 4, 
        augment: bool = True,
    ):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        self.scale = scale
        self.augment = augment
        
        # Get file lists
        self.hr_files = sorted([
            f for f in os.listdir(hr_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        self.lr_files = sorted([
            f for f in os.listdir(lr_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR/LR count mismatch: {len(self.hr_files)} vs {len(self.lr_files)}"
        
        print(f"[Dataset] Loaded {len(self.hr_files)} image pairs")
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # Load images
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Bicubic upsample LR to HR size
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        # Random crop to resolution
        w, h = hr_img.size
        
        if w < self.resolution or h < self.resolution:
            scale_factor = max(self.resolution / w, self.resolution / h) * 1.1
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            hr_img = hr_img.resize((new_w, new_h), Image.BICUBIC)
            lr_up = lr_up.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h
        
        x = random.randint(0, w - self.resolution)
        y = random.randint(0, h - self.resolution)
        hr_img = hr_img.crop((x, y, x + self.resolution, y + self.resolution))
        lr_up = lr_up.crop((x, y, x + self.resolution, y + self.resolution))
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_up = lr_up.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
                lr_up = lr_up.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Convert to tensor: [0, 255] -> [-1, 1]
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 127.5 - 1.0
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1.0
        
        return {'hr': hr_t, 'lr': lr_t, 'filename': self.hr_files[idx]}


# ============================================================================
# Condition Encoder
# ============================================================================

class ControlNetConditionEncoder(nn.Module):
    """Encodes LR latent into condition for ControlNet"""
    
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_channels: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.GroupNorm(32, hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                nn.GroupNorm(32, hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            ))
        
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.blocks:
            h = h + block(h)
        return self.conv_out(h)


# ============================================================================
# FLUX SR ControlNet Model
# ============================================================================

class FLUXSRControlNet(nn.Module):
    """SR using Frozen FLUX + Trainable ControlNet"""
    
    def __init__(
        self,
        flux_model_name: str = "black-forest-labs/FLUX.1-schnell",
        controlnet_num_layers: int = 4,
        controlnet_num_single_layers: int = 10,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.flux_model_name = flux_model_name
        self.controlnet_num_layers = controlnet_num_layers
        self.controlnet_num_single_layers = controlnet_num_single_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.transformer = None
        self.vae = None
        self.controlnet = None
        self.condition_encoder = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        
        self.flux_loaded = False
        
        # Cached text embeddings (computed once)
        self._cached_prompt_embeds = None
        self._cached_prompt_embeds_2 = None
        self._cached_pooled_embeds = None
    
    def load_flux(self, device: str = 'cuda'):
        """Load FLUX model and create ControlNet"""
        
        if self.flux_loaded:
            return True
        
        try:
            import diffusers
            from diffusers import FluxTransformer2DModel, AutoencoderKL
            
            # Import FluxControlNetModel
            try:
                from diffusers import FluxControlNetModel
            except ImportError:
                try:
                    from diffusers.models import FluxControlNetModel
                except ImportError:
                    print("[FLUX] FluxControlNetModel not available!")
                    print("[FLUX] Please: pip install --upgrade diffusers>=0.30.0")
                    return False
            
            from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
            
            dtype = torch.bfloat16
            
            # Load transformer
            print("[FLUX] Loading transformer...")
            self.transformer = FluxTransformer2DModel.from_pretrained(
                self.flux_model_name,
                subfolder="transformer",
                torch_dtype=dtype,
            ).to(device)
            
            # Load VAE
            print("[FLUX] Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(
                self.flux_model_name,
                subfolder="vae",
                torch_dtype=dtype,
            ).to(device)
            
            # Load text encoders (we'll cache embeddings and offload to CPU)
            print("[FLUX] Loading text encoders...")
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.flux_model_name,
                subfolder="text_encoder",
                torch_dtype=dtype,
            )
            self.text_encoder_2 = T5EncoderModel.from_pretrained(
                self.flux_model_name,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
            )
            
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.flux_model_name, subfolder="tokenizer"
            )
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(
                self.flux_model_name, subfolder="tokenizer_2"
            )
            
            # Get config
            hidden_dim = self.transformer.config.joint_attention_dim
            latent_channels = self.vae.config.latent_channels
            num_blocks = len(self.transformer.transformer_blocks)
            num_single = len(self.transformer.single_transformer_blocks)
            
            print(f"[FLUX] Hidden dim: {hidden_dim}")
            print(f"[FLUX] Transformer blocks: {num_blocks} + {num_single} single")
            
            # Freeze FLUX
            self.transformer.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            
            # Enable gradient checkpointing
            if self.use_gradient_checkpointing:
                self.transformer.enable_gradient_checkpointing()
            
            # Create ControlNet
            print("[FLUX] Creating ControlNet...")
            self.controlnet = FluxControlNetModel.from_transformer(
                self.transformer,
                num_layers=min(self.controlnet_num_layers, num_blocks),
                num_single_layers=min(self.controlnet_num_single_layers, num_single),
                attention_head_dim=self.transformer.config.attention_head_dim,
                num_attention_heads=self.transformer.config.num_attention_heads,
            ).to(device)
            
            # Enable gradient checkpointing for ControlNet too
            if self.use_gradient_checkpointing and hasattr(self.controlnet, 'enable_gradient_checkpointing'):
                self.controlnet.enable_gradient_checkpointing()
            
            # Create condition encoder
            self.condition_encoder = ControlNetConditionEncoder(
                in_channels=latent_channels,
                out_channels=latent_channels,
                hidden_channels=128,
                num_blocks=4,
            ).to(device)
            
            # Pre-compute empty text embeddings and cache them
            print("[FLUX] Caching empty text embeddings...")
            self._cache_text_embeddings(device, dtype)
            
            # Now offload text encoders to CPU to save GPU memory
            self.text_encoder = self.text_encoder.to('cpu')
            self.text_encoder_2 = self.text_encoder_2.to('cpu')
            torch.cuda.empty_cache()
            
            self.flux_loaded = True
            self._device = device
            self._dtype = dtype
            
            # Memory stats
            flux_params = sum(p.numel() for p in self.transformer.parameters())
            ctrl_params = sum(p.numel() for p in self.controlnet.parameters())
            enc_params = sum(p.numel() for p in self.condition_encoder.parameters())
            
            print(f"[FLUX] Transformer: {flux_params/1e9:.2f}B params (frozen)")
            print(f"[FLUX] ControlNet: {ctrl_params/1e9:.2f}B params (trainable)")
            print(f"[FLUX] Encoder: {enc_params/1e6:.2f}M params (trainable)")
            
            # Print GPU memory
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                print(f"[FLUX] GPU memory used: {mem_gb:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"[FLUX] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cache_text_embeddings(self, device, dtype):
        """Cache empty text embeddings (we use empty prompts for SR)"""
        with torch.no_grad():
            # Move text encoders to GPU temporarily
            self.text_encoder = self.text_encoder.to(device)
            self.text_encoder_2 = self.text_encoder_2.to(device)
            
            # Empty prompt
            prompt = [""]
            
            # CLIP
            text_inputs = self.tokenizer(
                prompt, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            )
            text_ids = text_inputs.input_ids.to(device)
            outputs = self.text_encoder(text_ids, output_hidden_states=False)
            self._cached_prompt_embeds = outputs[0].to(dtype)
            self._cached_pooled_embeds = outputs.pooler_output.to(dtype)
            
            # T5
            text_inputs_2 = self.tokenizer_2(
                prompt, padding="max_length", max_length=512,
                truncation=True, return_tensors="pt"
            )
            text_ids_2 = text_inputs_2.input_ids.to(device)
            self._cached_prompt_embeds_2 = self.text_encoder_2(
                text_ids_2, output_hidden_states=False
            )[0].to(dtype)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get only trainable parameters"""
        params = []
        if self.controlnet is not None:
            params.extend(self.controlnet.parameters())
        if self.condition_encoder is not None:
            params.extend(self.condition_encoder.parameters())
        return params
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent"""
        image = image.to(dtype=self.vae.dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        return latent * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        latent = latent / self.vae.config.scaling_factor
        return self.vae.decode(latent).sample
    
    def get_text_embeds(self, batch_size: int, device, dtype):
        """Get cached text embeddings expanded to batch size"""
        prompt_embeds = self._cached_prompt_embeds.expand(batch_size, -1, -1).to(device)
        prompt_embeds_2 = self._cached_prompt_embeds_2.expand(batch_size, -1, -1).to(device)
        pooled = self._cached_pooled_embeds.expand(batch_size, -1).to(device)
        text_ids = torch.zeros(batch_size, prompt_embeds_2.shape[1], 3, device=device, dtype=dtype)
        return prompt_embeds, prompt_embeds_2, pooled, text_ids
    
    def _prepare_latent_image_ids(self, batch_size: int, height: int, width: int, device, dtype):
        """Prepare image IDs for RoPE (2x2 packing)"""
        h, w = height // 2, width // 2
        latent_image_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        latent_image_ids = latent_image_ids.reshape(h * w, 3)
        return latent_image_ids.unsqueeze(0).expand(batch_size, -1, -1)
    
    def _pack_latents(self, latents: torch.Tensor):
        """Pack latents: [B, 16, H, W] -> [B, (H/2)*(W/2), 64]"""
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(B, (H // 2) * (W // 2), C * 4)
    
    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int):
        """Unpack latents: [B, (H/2)*(W/2), 64] -> [B, 16, H, W]"""
        B = latents.shape[0]
        C = latents.shape[-1] // 4
        latents = latents.view(B, height // 2, width // 2, C, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(B, C, height, width)
    
    def forward(
        self,
        noisy_latent: torch.Tensor,
        lr_latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass"""
        B, C, H, W = noisy_latent.shape
        device = noisy_latent.device
        dtype = noisy_latent.dtype
        
        # Encode LR condition
        lr_condition = self.condition_encoder(lr_latent.float()).to(dtype)
        
        # Get cached text embeddings
        prompt_embeds, prompt_embeds_2, pooled, text_ids = self.get_text_embeds(B, device, dtype)
        encoder_hidden_states = prompt_embeds_2
        
        # Prepare image IDs
        latent_image_ids = self._prepare_latent_image_ids(B, H, W, device, dtype)
        
        # Pack latents
        packed_noisy = self._pack_latents(noisy_latent)
        packed_lr_cond = self._pack_latents(lr_condition)

        # 去掉 Batch 维度 (只取第 0 个 sample 的 ids 即可)
        if text_ids.dim() == 3:
            text_ids = text_ids[0]
        if latent_image_ids.dim() == 3:
            latent_image_ids = latent_image_ids[0]
        # ControlNet forward
        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=packed_noisy,
            controlnet_cond=packed_lr_cond,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )
        
        # FLUX transformer forward (frozen)
        noise_pred = self.transformer(
            hidden_states=packed_noisy,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=False,
        )[0]
        
        # Unpack
        return self._unpack_latents(noise_pred, H, W)


# ============================================================================
# Trainer (Single GPU)
# ============================================================================

class Trainer:
    """Single GPU Trainer"""
    
    def __init__(
        self,
        model: FLUXSRControlNet,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
    ):
        self.model = model
        self.device = device
        self.grad_accum_steps = gradient_accumulation_steps
        self.accum_count = 0
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with gradient accumulation"""
        self.model.controlnet.train()
        self.model.condition_encoder.train()
        
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        B = hr.shape[0]
        
        # Encode to latent
        with torch.no_grad():
            hr_latent = self.model.encode_image(hr)
            lr_latent = self.model.encode_image(lr)
        
        # Sample timestep [0, 1]
        timestep = torch.rand(B, device=self.device, dtype=torch.bfloat16)
        
        # Add noise (flow matching)
        noise = torch.randn_like(hr_latent)
        t_expand = timestep[:, None, None, None]
        noisy_latent = (1 - t_expand) * hr_latent + t_expand * noise
        
        # Target velocity
        v_target = noise - hr_latent
        
        # Forward with autocast
        v_pred = self.model(noisy_latent, lr_latent, timestep)
        loss = F.mse_loss(v_pred.float(), v_target.float())
        loss = loss / self.grad_accum_steps
        
        # Backward 
        from accelerate import Accelerator
        accelerator = Accelerator()
        accelerator.backward(loss)
        
        self.accum_count += 1
        
        # Optimizer step 
        if self.accum_count >= self.grad_accum_steps:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.model.get_trainable_params(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accum_count = 0
        
        return {'loss': loss.item() * self.grad_accum_steps}
    
    def save(self, path: str, epoch: int, loss: float):
        """Save checkpoint"""
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'controlnet': self.model.controlnet.state_dict(),
            'condition_encoder': self.model.condition_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Saved: {path}")
    
    def load(self, path: str) -> int:
        """Load checkpoint"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.controlnet.load_state_dict(ckpt['controlnet'])
        self.model.condition_encoder.load_state_dict(ckpt['condition_encoder'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FLUX SR ControlNet Training')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--scale', type=int, default=4)
    
    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-schnell')
    parser.add_argument('--controlnet_layers', type=int, default=4)
    parser.add_argument('--controlnet_single_layers', type=int, default=10)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Multi-GPU sharding (run separate processes)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    
    # Save
    parser.add_argument('--save_dir', type=str, default='./checkpoints/flux_control')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed + args.shard)
    random.seed(args.seed + args.shard)
    np.random.seed(args.seed + args.shard)
    
    # Device
    accelerator = Accelerator()
    device = accelerator.device
    
    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("FLUX SR ControlNet Training (Single GPU)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Resolution: {args.resolution}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} accum")
    print(f"Shard: {args.shard + 1}/{args.num_shards}")
    print("=" * 60)
    
    # Dataset
    dataset = SRDataset(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        resolution=args.resolution,
        scale=args.scale,
        augment=True,
    )
    
    # Shard dataset if multi-GPU
    if args.num_shards > 1:
        indices = list(range(args.shard, len(dataset), args.num_shards))
        dataset = Subset(dataset, indices)
        print(f"[Shard {args.shard}] Using {len(dataset)} samples")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # Model
    model = FLUXSRControlNet(
        flux_model_name=args.model_name,
        controlnet_num_layers=args.controlnet_layers,
        controlnet_num_single_layers=args.controlnet_single_layers,
        use_gradient_checkpointing=True,
    )
    
    if not model.load_flux(device):
        print("Failed to load FLUX!")
        return
    
    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    model, trainer.optimizer, loader = accelerator.prepare(
        model, trainer.optimizer, loader
    )
    
    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training
    print("\nStarting training...")
    
    for epoch in range(start_epoch, args.epochs):
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'avg': f"{np.mean(losses[-50:]):.4f}"})
        
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.6f}")
        
        if (epoch + 1) % args.save_every == 0:
            trainer.save(
                os.path.join(args.save_dir, f'epoch{epoch+1}_shard{args.shard}.pt'),
                epoch, avg_loss
            )
    
    trainer.save(os.path.join(args.save_dir, f'final_shard{args.shard}.pt'), args.epochs-1, avg_loss)
    print("Training complete!")


if __name__ == '__main__':
    main()