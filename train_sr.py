"""
MeanFlow Super-Resolution - PyTorch Version

This is a standalone PyTorch implementation that's easier to run
than the JAX version.

Usage:
    python train_sr_torch.py \
        --hr_dir "Flow_Restore/Data/Urban 100/X2 Urban100/X2/HIGH X2 Urban" \
        --lr_dir "Flow_Restore/Data/Urban 100/X2 Urban100/X2/LOW X2 Urban" \
        --epochs 500 \
        --batch_size 4

Key Formula:
    z_t = (1-t) * HR + t * LR     # Interpolation
    v = LR - HR                    # Instantaneous velocity
    u = network(z_t, t, h=t-r)    # Predicted average velocity
    u_target = v - (t-r) * du/dt  # MeanFlow target

One-step inference:
    HR = LR - u(LR, t=1, h=1)
"""

import os
import math
import random
import argparse
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# =========================================================================
# Dataset
# =========================================================================

class Urban100Dataset(Dataset):
    """Urban100 SR Dataset"""
    
    def __init__(self, hr_dir, lr_dir, patch_size=128, augment=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.augment = augment
        
        # Get files
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR ({len(self.hr_files)}) and LR ({len(self.lr_files)}) count mismatch"
        
        print(f"[Dataset] Found {len(self.hr_files)} image pairs")
    
    def __len__(self):
        # Repeat dataset for more iterations per epoch
        return len(self.hr_files) * 10
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.hr_files)
        
        # Load images
        hr_path = os.path.join(self.hr_dir, self.hr_files[real_idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[real_idx])
        
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Upsample LR to HR size
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        # Random crop
        hr_img, lr_up = self._random_crop(hr_img, lr_up)
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_up = lr_up.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
                lr_up = lr_up.transpose(Image.FLIP_TOP_BOTTOM)
        
        # To tensor [-1, 1]
        hr_tensor = torch.from_numpy(np.array(hr_img)).float() / 127.5 - 1.0
        lr_tensor = torch.from_numpy(np.array(lr_up)).float() / 127.5 - 1.0
        
        # HWC -> CHW
        hr_tensor = hr_tensor.permute(2, 0, 1)
        lr_tensor = lr_tensor.permute(2, 0, 1)
        
        return {'hr': hr_tensor, 'lr': lr_tensor}
    
    def _random_crop(self, hr_img, lr_img):
        w, h = hr_img.size
        
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            new_size = (int(w * scale), int(h * scale))
            hr_img = hr_img.resize(new_size, Image.BICUBIC)
            lr_img = lr_img.resize(new_size, Image.BICUBIC)
            w, h = new_size
        
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        
        hr_crop = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        lr_crop = lr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        return hr_crop, lr_crop


# =========================================================================
# Model Components
# =========================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning and channel projection"""
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 确保时间嵌入映射到正确的输出维度
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        )
        self.dropout = nn.Dropout(dropout)
        
        # 关键修复：如果输入输出通道不一致，使用 1x1 卷积调整 Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Time conditioning
        t_proj = self.time_mlp(t_emb)
        scale, shift = t_proj.chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.shortcut(x) + h


class Attention(nn.Module):
    """Self-attention block"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        
        return x + out


class MeanFlowSRNet(nn.Module):
    """
    MeanFlow SR Network - Robust Channel Matching
    """
    
    def __init__(
        self,
        in_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 4, 8), # 假设你的配置可能用到了8，或者是(1,2,4,4)
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
    ):
        super().__init__()
        
        # 自动计算时间嵌入维度
        time_dim = hidden_channels * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.h_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Input
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # ----------------------------------------------------------------
        # 关键修复：预计算 Skip Connection 的通道数栈
        # ----------------------------------------------------------------
        # 我们模拟一遍 Encoder 的构建过程，记录下每一次 skips.append() 时张量的通道数
        skip_ch_stack = [hidden_channels] # 对应 forward 中的 skips = [x]
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        ch = hidden_channels
        current_ch = ch
        
        for i, mult in enumerate(channel_mult):
            out_ch = hidden_channels * mult
            
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                # Encoder Block
                blocks.append(ResBlock(current_ch, out_ch, time_dim, dropout))
                current_ch = out_ch
                # 记录：每次 Block 输出都会被 append 到 skips
                skip_ch_stack.append(current_ch)
            
            self.encoder.append(blocks)
            
            if i < len(channel_mult) - 1:
                # Downsample
                self.downsample.append(nn.Conv2d(current_ch, current_ch, 3, stride=2, padding=1))
                # 记录：Downsample 输出也会被 append 到 skips
                skip_ch_stack.append(current_ch)
        
        # Middle
        self.mid_block1 = ResBlock(current_ch, current_ch, time_dim, dropout)
        self.mid_attn = Attention(current_ch)
        self.mid_block2 = ResBlock(current_ch, current_ch, time_dim, dropout)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = hidden_channels * mult
            
            blocks = nn.ModuleList()
            
            # Decoder 每个 Stage 有 num_res_blocks + 1 个块
            # 这里的 +1 是为了消化掉 Encoder 对应层级产生的额外 skip connection (如 downsample 的输出)
            for j in range(num_res_blocks + 1):
                # 关键修复：从栈顶弹出一个通道数，这正是当前层将要拼接的 skip connection 的真实大小
                skip_ch = skip_ch_stack.pop()
                
                # 输入维度 = 上一层输出 (current_ch) + 跳跃连接 (skip_ch)
                # 我们要求 ResBlock 将其融合并输出为该层级的目标维度 (out_ch)
                blocks.append(ResBlock(current_ch + skip_ch, out_ch, time_dim, dropout))
                current_ch = out_ch
            
            self.decoder.append(blocks)
            
            if i > 0:
                self.upsample.append(nn.ConvTranspose2d(current_ch, current_ch, 4, stride=2, padding=1))
        
        # Output
        self.out_norm = nn.GroupNorm(8, current_ch)
        self.out_conv = nn.Conv2d(current_ch, in_channels, 3, padding=1)
        
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, t, h):
        t_emb = self.time_embed(t) + self.h_embed(h)
        
        x = self.in_conv(x)
        
        # Encoder
        skips = [x]
        for blocks, down in zip(self.encoder, self.downsample + [None]):
            for block in blocks:
                x = block(x, t_emb)
                skips.append(x)
            if down is not None:
                x = down(x)
                skips.append(x)
        
        # Middle
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # Decoder
        for blocks, up in zip(self.decoder, [None] + list(self.upsample)):
            if up is not None:
                x = up(x)
            for block in blocks:
                # 此时 skips.pop() 拿到的 tensor 维度，必然等于我们在 __init__ 中
                # skip_ch_stack.pop() 记录的数值，因为顺序完全一致。
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
        
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        
        return x

# =========================================================================
# MeanFlow SR Trainer
# =========================================================================

class MeanFlowSRTrainer:
    """Trainer for MeanFlow SR"""
    
    def __init__(
        self,
        model,
        device,
        lr=1e-4,
        data_proportion=0.75,
        P_mean=-0.4,
        P_std=1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.data_proportion = data_proportion
        self.P_mean = P_mean
        self.P_std = P_std
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
        
        # EMA
        self.ema_decay = 0.9999
        self.ema_params = {name: param.clone() for name, param in model.named_parameters()}
    
    def update_ema(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            self.ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def sample_time(self, batch_size):
        """Sample t and r with logit-normal distribution"""
        # Logit-normal
        rnd = torch.randn(batch_size, device=self.device)
        t = torch.sigmoid(rnd * self.P_std + self.P_mean)
        
        rnd_r = torch.randn(batch_size, device=self.device)
        r = torch.sigmoid(rnd_r * self.P_std + self.P_mean)
        
        # Ensure t >= r
        t, r = torch.max(t, r), torch.min(t, r)
        
        # 75% of samples: r = t
        data_size = int(batch_size * self.data_proportion)
        mask = torch.arange(batch_size, device=self.device) < data_size
        r = torch.where(mask, t, r)
        
        return t, r
    
    def train_step(self, batch):
        """
        MeanFlow training step
        
        z_t = (1-t)*HR + t*LR
        v = LR - HR
        u_target ≈ v (simplified for training stability)
        """
        self.model.train()
        
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        batch_size = hr.shape[0]
        
        # Sample time
        t, r = self.sample_time(batch_size)
        h = t - r
        
        # Interpolate
        t_exp = t[:, None, None, None]
        z_t = (1 - t_exp) * hr + t_exp * lr
        
        # Ground truth velocity
        v = lr - hr
        
        # Forward
        u = self.model(z_t, t, h)
        
        # Simplified loss (works well in practice)
        # Weight by 1/(h+eps) to focus on learning v first
        h_exp = h[:, None, None, None]
        weight = 1.0 / (h_exp + 0.01)
        
        loss = ((u - v) ** 2 * weight).mean()
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        self.update_ema()
        
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def inference(self, lr_images, use_ema=True):
        """
        One-step SR inference
        
        HR = LR - u(LR, t=1, h=1)
        """
        self.model.eval()
        
        # Load EMA params if requested
        if use_ema:
            orig_params = {name: param.clone() for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                param.data.copy_(self.ema_params[name])
        
        batch_size = lr_images.shape[0]
        t = torch.ones(batch_size, device=self.device)
        h = torch.ones(batch_size, device=self.device)
        
        u = self.model(lr_images, t, h)
        hr = lr_images - u
        
        # Restore original params
        if use_ema:
            for name, param in self.model.named_parameters():
                param.data.copy_(orig_params[name])
        
        return hr


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow SR Training (PyTorch)')
    parser.add_argument('--hr_dir', type=str, required=True, help='HR images directory')
    parser.add_argument('--lr_dir', type=str, required=True, help='LR images directory')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_sr')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("MeanFlow Super-Resolution Training (PyTorch)")
    print("="*60)
    print(f"Device: {device}")
    print(f"HR dir: {args.hr_dir}")
    print(f"LR dir: {args.lr_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patch size: {args.patch_size}")
    print("="*60)
    
    # Create dataset
    dataset = Urban100Dataset(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        patch_size=args.patch_size,
        augment=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    model = MeanFlowSRNet(
        in_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.1,
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Create trainer
    trainer = MeanFlowSRTrainer(
        model=model,
        device=device,
        lr=args.lr,
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': trainer.ema_params,
                'optimizer': trainer.optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  Saved best model (loss={avg_loss:.6f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': trainer.ema_params,
                'optimizer': trainer.optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt'))
        
        # Visualization every 20 epochs
        if (epoch + 1) % 20 == 0:
            # Get a batch
            test_batch = next(iter(loader))
            lr_test = test_batch['lr'].to(device)
            hr_test = test_batch['hr'].to(device)
            
            hr_pred = trainer.inference(lr_test, use_ema=True)
            
            # Compute PSNR
            mse = ((hr_pred - hr_test) ** 2).mean()
            psnr = 10 * torch.log10(4.0 / (mse + 1e-8))  # range is [-1,1] so max diff is 2
            print(f"  Sample PSNR: {psnr.item():.2f} dB")
            
            # Save sample images
            sample_dir = os.path.join(args.save_dir, 'samples')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save first image
            lr_img = ((lr_test[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            hr_img = ((hr_test[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            pred_img = ((hr_pred[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Concatenate: LR | Pred | HR
            combined = np.concatenate([lr_img, pred_img, hr_img], axis=1)
            Image.fromarray(combined).save(os.path.join(sample_dir, f'epoch{epoch+1}.png'))
    
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()