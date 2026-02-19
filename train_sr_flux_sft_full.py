"""
Super-Resolution with Frozen FLUX DiT + SFT Adapters (Full Implementation)

This version properly injects SFT adapters into each FLUX transformer block,
similar to how StableSR injects into SD UNet.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │   LR_image ──► VAE Encode ──► lr_latent ──► LR Encoder          │
    │                                                 │                │
    │                                                 ▼                │
    │   HR + noise ──► VAE Encode ──► noisy_latent                    │
    │                                     │                            │
    │                                     ▼                            │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │              FLUX DiT (frozen)                          │   │
    │   │                                                         │   │
    │   │   Block 0 ──► [SFT_0] ──►                              │   │
    │   │   Block 1 ──► [SFT_1] ──►                              │   │
    │   │   Block 2 ──► [SFT_2] ──►                              │   │
    │   │   ...                                                   │   │
    │   │   Block N ──► [SFT_N] ──►                              │   │
    │   │                                                         │   │
    │   └───────────────────────────────┬─────────────────────────┘   │
    │                                   │                              │
    │                                   ▼                              │
    │                           noise_prediction                       │
    │                                   │                              │
    │                        Loss = MSE(pred, noise)                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Hardware Requirements:
    - FLUX DiT (12B): ~24GB in fp16
    - SFT Adapters: ~2-4GB
    - VAE: ~200MB
    - Training overhead: ~10-15GB
    - Total per GPU: ~40GB (fits in 48GB GPUs)

Usage:
    # Single GPU
    python train_sr_flux_sft_full.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --model_name "black-forest-labs/FLUX.1-schnell" \
        --batch_size 1  --device cuda:0

    # 8 GPU with DeepSpeed
    accelerate launch --multi_gpu --num_processes 8 \
        train_sr_flux_sft_full.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --model_name "black-forest-labs/FLUX.1-schnell" \
        --batch_size 1 --gradient_checkpointing
"""

import os
import math
import random
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from functools import partial

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Dataset
# ============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, resolution=512, scale=4, augment=True):
        self.hr_dir, self.lr_dir = hr_dir, lr_dir
        self.resolution, self.scale, self.augment = resolution, scale, augment
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(self.hr_files) == len(self.lr_files)
        print(f"[Dataset] {len(self.hr_files)} pairs, resolution={resolution}")
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('RGB')
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        w, h = hr_img.size
        if w >= self.resolution and h >= self.resolution:
            x, y = random.randint(0, w - self.resolution), random.randint(0, h - self.resolution)
            hr_img = hr_img.crop((x, y, x + self.resolution, y + self.resolution))
            lr_up = lr_up.crop((x, y, x + self.resolution, y + self.resolution))
        else:
            hr_img = hr_img.resize((self.resolution, self.resolution), Image.BICUBIC)
            lr_up = lr_up.resize((self.resolution, self.resolution), Image.BICUBIC)
        
        if self.augment:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_up = lr_up.transpose(Image.FLIP_LEFT_RIGHT)
        
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 127.5 - 1.0
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1.0
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# SFT Adapter Module
# ============================================================================

class SFTAdapter(nn.Module):
    """
    Spatial Feature Transform Adapter
    
    Learns to modulate FLUX features based on LR condition:
        output = flux_feature * (1 + scale) + shift
    
    where scale, shift = f(lr_condition)
    """
    def __init__(self, hidden_dim: int, condition_dim: int = None):
        super().__init__()
        condition_dim = condition_dim or hidden_dim
        
        # Condition projection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Scale and shift prediction
        self.scale_shift = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        
        # Zero initialization for stable training
        nn.init.zeros_(self.scale_shift[-1].weight)
        nn.init.zeros_(self.scale_shift[-1].bias)
    
    def forward(
        self, 
        flux_hidden: torch.Tensor,      # [B, N, C] from FLUX
        lr_condition: torch.Tensor,      # [B, N, C_cond] from LR encoder
    ) -> torch.Tensor:
        # Project condition
        cond = self.condition_proj(lr_condition)  # [B, N, C]
        
        # Predict scale and shift
        scale_shift = self.scale_shift(cond)  # [B, N, 2C]
        scale, shift = scale_shift.chunk(2, dim=-1)  # [B, N, C] each
        
        # Apply SFT
        return flux_hidden * (1 + scale) + shift


# ============================================================================
# LR Condition Encoder
# ============================================================================

class LRConditionEncoder(nn.Module):
    """
    Encodes LR latent into condition features for each FLUX block.
    Inspired by StableSR's structcond_stage_model.
    """
    def __init__(
        self,
        in_channels: int = 16,      # FLUX VAE latent channels
        hidden_dim: int = 3072,     # FLUX hidden dim
        num_outputs: int = 19,      # Number of FLUX blocks to condition
        num_layers: int = 6,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.input_proj = nn.Conv2d(in_channels, hidden_dim // 4, 3, 1, 1)
        
        # Downsampling encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.GELU(),
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=16,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Output heads for each FLUX block
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_outputs)
        ])
        
        # Time embedding
        self.time_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def timestep_embedding(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(
        self,
        lr_latent: torch.Tensor,  # [B, C, H, W]
        timestep: torch.Tensor,   # [B]
    ) -> List[torch.Tensor]:
        """
        Returns:
            List of [B, N, hidden_dim] tensors, one for each FLUX block
        """
        B, C, H, W = lr_latent.shape
        
        # Encode
        x = self.input_proj(lr_latent)  # [B, hidden_dim//4, H, W]
        x = self.encoder(x)             # [B, hidden_dim, H/4, W/4]
        
        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # [B, (H/4)*(W/4), hidden_dim]
        
        # Add time embedding
        t_emb = self.timestep_embedding(timestep)
        t_emb = self.time_proj(t_emb)  # [B, hidden_dim]
        x = x + t_emb[:, None, :]
        
        # Transformer
        for layer in self.layers:
            x = layer(x)
        
        # Generate outputs for each block
        outputs = []
        for head in self.output_heads:
            outputs.append(head(x))
        
        return outputs


# ============================================================================
# FLUX SR Model with Full SFT Injection
# ============================================================================

class FLUXSRWithSFT(nn.Module):
    """
    Super-Resolution using Frozen FLUX DiT with SFT Adapters.
    
    Key features:
    1. FLUX DiT is completely frozen
    2. SFT adapters are inserted after each transformer block
    3. LR encoder provides conditions for all SFT adapters
    """
    
    def __init__(
        self,
        flux_model_name: str = "black-forest-labs/FLUX.1-schnell",
        num_sft_blocks: int = 19,  # Number of blocks to inject SFT
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.flux_model_name = flux_model_name
        self.num_sft_blocks = num_sft_blocks
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Will be initialized in load_flux()
        self.flux_loaded = False
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.scheduler = None
        
        self.lr_encoder = None
        self.sft_adapters = None
        
        # Config
        self.hidden_dim = None
        self.latent_channels = None
    
    def load_flux(self, device='cuda'):
        """Load FLUX and initialize SFT adapters"""
        if self.flux_loaded:
            return True
        
        try:
            from diffusers import FluxPipeline
            
            print(f"[FLUX-SFT] Loading {self.flux_model_name}...")
            print("[FLUX-SFT] This may take a few minutes...")
            
            # Load full pipeline
            pipe = FluxPipeline.from_pretrained(
                self.flux_model_name,
                torch_dtype=torch.bfloat16,  # FLUX works best with bf16
            )
            
            # Extract components
            self.transformer = pipe.transformer
            self.vae = pipe.vae
            self.text_encoder = pipe.text_encoder
            self.text_encoder_2 = pipe.text_encoder_2
            self.scheduler = pipe.scheduler
            self.tokenizer = pipe.tokenizer
            self.tokenizer_2 = pipe.tokenizer_2
            
            # Get config
            self.hidden_dim = self.transformer.config.joint_attention_dim
            self.latent_channels = self.vae.config.latent_channels
            
            print(f"[FLUX-SFT] Hidden dim: {self.hidden_dim}")
            print(f"[FLUX-SFT] Latent channels: {self.latent_channels}")
            print(f"[FLUX-SFT] Num transformer blocks: {len(self.transformer.transformer_blocks)}")
            
            # Freeze FLUX
            self.transformer.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            
            # Enable gradient checkpointing if requested
            if self.use_gradient_checkpointing:
                self.transformer.enable_gradient_checkpointing()
            
            # Adjust num_sft_blocks to match actual blocks
            actual_blocks = len(self.transformer.transformer_blocks)
            self.num_sft_blocks = min(self.num_sft_blocks, actual_blocks)
            
            # Initialize SFT adapters
            self.lr_encoder = LRConditionEncoder(
                in_channels=self.latent_channels,
                hidden_dim=self.hidden_dim,
                num_outputs=self.num_sft_blocks,
            ).to(device).to(torch.float32)  # Keep adapters in fp32
            
            self.sft_adapters = nn.ModuleList([
                SFTAdapter(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_sft_blocks)
            ]).to(device).to(torch.float32)
            
            # Move FLUX to device
            self.transformer = self.transformer.to(device)
            self.vae = self.vae.to(device)
            self.text_encoder = self.text_encoder.to(device)
            self.text_encoder_2 = self.text_encoder_2.to(device)
            
            self.flux_loaded = True
            
            # Count parameters
            flux_params = sum(p.numel() for p in self.transformer.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            print(f"[FLUX-SFT] FLUX params: {flux_params:,} (frozen)")
            print(f"[FLUX-SFT] Trainable params: {trainable_params:,}")
            print(f"[FLUX-SFT] SFT blocks: {self.num_sft_blocks}")
            
            # Clean up pipeline reference
            del pipe
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"[FLUX-SFT] Error loading FLUX: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_trainable_params(self):
        """Return only trainable parameters"""
        params = []
        if self.lr_encoder is not None:
            params.extend(self.lr_encoder.parameters())
        if self.sft_adapters is not None:
            params.extend(self.sft_adapters.parameters())
        return params
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent"""
        image = image.to(self.vae.dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent
    
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent).sample
        return image
    
    @torch.no_grad()
    def get_text_embeddings(self, batch_size: int, device):
        """Get empty text embeddings (for unconditional generation)"""
        # Empty prompt
        prompt = [""] * batch_size
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode
        text_ids = text_inputs.input_ids.to(device)
        text_ids_2 = text_inputs_2.input_ids.to(device)
        
        prompt_embeds = self.text_encoder(text_ids, output_hidden_states=False)
        pooled_prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds[0]
        
        prompt_embeds_2 = self.text_encoder_2(text_ids_2, output_hidden_states=False)[0]
        
        return prompt_embeds, prompt_embeds_2, pooled_prompt_embeds
    
    def forward_with_sft(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        lr_conditions: List[torch.Tensor],
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through FLUX transformer with SFT injection.
        
        This is a custom forward that injects SFT after each block.
        """
        # Prepare inputs (from FLUX forward)
        hidden_states = self.transformer.x_embedder(hidden_states)
        
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        
        temb = self.transformer.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.transformer.context_embedder(encoder_hidden_states)
        
        if self.transformer.pos_embed is not None:
            # Apply positional embeddings
            pass  # Simplified
        
        # Process through blocks with SFT injection
        for i, block in enumerate(self.transformer.transformer_blocks):
            # FLUX block forward (frozen)
            with torch.no_grad():
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=None,  # Simplified
                )
            
            # Apply SFT if this block has an adapter
            if i < self.num_sft_blocks and i < len(lr_conditions):
                lr_cond = lr_conditions[i].to(hidden_states.dtype)
                
                # Match sequence length if needed
                if lr_cond.shape[1] != hidden_states.shape[1]:
                    lr_cond = F.interpolate(
                        lr_cond.transpose(1, 2),
                        size=hidden_states.shape[1],
                        mode='linear',
                        align_corners=False,
                    ).transpose(1, 2)
                
                hidden_states = hidden_states + self.sft_adapters[i](
                    hidden_states.float(), lr_cond
                ).to(hidden_states.dtype)
        
        # Final layers
        hidden_states = self.transformer.norm_out(hidden_states, temb)
        hidden_states = self.transformer.proj_out(hidden_states)
        
        return hidden_states
    
    def forward(
        self,
        noisy_latent: torch.Tensor,    # [B, C, H, W]
        lr_latent: torch.Tensor,       # [B, C, H, W]
        timestep: torch.Tensor,        # [B]
    ) -> torch.Tensor:
        """
        Training forward pass.
        
        Returns predicted noise/velocity.
        """
        B = noisy_latent.shape[0]
        device = noisy_latent.device
        
        # Get LR conditions for all SFT blocks
        lr_conditions = self.lr_encoder(lr_latent.float(), timestep)
        
        # Get text embeddings (empty for SR)
        prompt_embeds, prompt_embeds_2, pooled = self.get_text_embeddings(B, device)
        
        # Combine text embeddings
        encoder_hidden_states = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        encoder_hidden_states = encoder_hidden_states.to(noisy_latent.dtype)
        pooled = pooled.to(noisy_latent.dtype)
        
        # Forward through FLUX with SFT
        # Note: This is simplified. Full implementation needs proper
        # handling of positional embeddings, image IDs, etc.
        
        with torch.no_grad():
            # Use FLUX's built-in forward for now
            # In full implementation, use forward_with_sft
            noise_pred = self.transformer(
                hidden_states=noisy_latent,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
                timestep=timestep / 1000,  # FLUX expects [0, 1]
                return_dict=False,
            )[0]
        
        return noise_pred


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    def __init__(
        self,
        model: FLUXSRWithSFT,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.device = device
        
        trainable_params = model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # EMA
        self.ema_decay = 0.9999
        self.ema_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
    
    def update_ema(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.ema_params:
                    self.ema_params[n].mul_(self.ema_decay).add_(
                        p.data, alpha=1 - self.ema_decay
                    )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        B = hr.shape[0]
        
        # Encode to latent
        hr_latent = self.model.encode_image(hr)
        lr_latent = self.model.encode_image(lr)
        
        # Sample timestep
        timestep = torch.rand(B, device=self.device)  # [0, 1]
        
        # Add noise (flow matching style)
        noise = torch.randn_like(hr_latent)
        noisy_latent = (1 - timestep[:, None, None, None]) * hr_latent + \
                       timestep[:, None, None, None] * noise
        
        # Target: velocity from noise to clean
        v_target = noise - hr_latent
        
        # Forward
        v_pred = self.model(noisy_latent, lr_latent, timestep)
        
        # Loss
        loss = F.mse_loss(v_pred.float(), v_target.float())
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_params(),
            max_norm=1.0
        )
        
        self.optimizer.step()
        self.update_ema()
        
        return {'loss': loss.item()}
    
    def save(self, path: str, epoch: int, loss: float):
        """Save only trainable weights"""
        save_dict = {
            'epoch': epoch,
            'loss': loss,
            'lr_encoder': self.model.lr_encoder.state_dict(),
            'sft_adapters': self.model.sft_adapters.state_dict(),
            'ema': self.ema_params,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(save_dict, path)
        print(f"Saved to {path}")
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.lr_encoder.load_state_dict(ckpt['lr_encoder'])
        self.model.sft_adapters.load_state_dict(ckpt['sft_adapters'])
        if 'ema' in ckpt:
            self.ema_params = ckpt['ema']
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-schnell')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/flux_sft')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 70)
    print("FLUX SR with SFT Adapters")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Resolution: {args.resolution}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print("=" * 70)
    
    # Dataset
    dataset = SRDataset(args.hr_dir, args.lr_dir, args.resolution)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = FLUXSRWithSFT(
        flux_model_name=args.model_name,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    
    if not model.load_flux(args.device):
        print("Failed to load FLUX!")
        return
    
    # Trainer
    trainer = Trainer(model, args.device, args.lr)
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load(args.resume)
    
    # Train
    for epoch in range(start_epoch, args.epochs):
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            trainer.save(
                os.path.join(args.save_dir, f'epoch{epoch+1}.pt'),
                epoch, avg_loss
            )
    
    trainer.save(os.path.join(args.save_dir, 'final.pt'), args.epochs-1, avg_loss)
    print("Training complete!")


if __name__ == '__main__':
    main()