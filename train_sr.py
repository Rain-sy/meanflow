"""
MeanFlow SR 训练脚本 (修复版)

修复内容：
1. 移除不合理的 weight = 1/(h+0.01) 加权
2. 增加 h=1 样本的比例，确保模型能学会 one-step 推理
3. 添加验证时的 PSNR 监控

用法：
    python train_sr_fixed.py \
        --hr_dir "Flow_Restore/Data/DIV2K/DIV2K_train_HR" \
        --lr_dir "Flow_Restore/Data/DIV2K/DIV2K_train_LR_bicubic/X2" \
        --epochs 100 \
        --batch_size 8
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


# =========================================================================
# Dataset
# =========================================================================

class SRDataset(Dataset):
    """SR Dataset with proper augmentation"""
    
    def __init__(self, hr_dir, lr_dir, patch_size=128, augment=True, repeat=5):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.augment = augment
        self.repeat = repeat
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR ({len(self.hr_files)}) and LR ({len(self.lr_files)}) count mismatch"
        
        print(f"[Dataset] Found {len(self.hr_files)} image pairs (repeat={repeat})")
    
    def __len__(self):
        return len(self.hr_files) * self.repeat
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.hr_files)
        
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
            # Random 90 degree rotation
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                hr_img = hr_img.rotate(90 * k)
                lr_up = lr_up.rotate(90 * k)
        
        # To tensor [-1, 1]
        hr_tensor = torch.from_numpy(np.array(hr_img)).float() / 127.5 - 1.0
        lr_tensor = torch.from_numpy(np.array(lr_up)).float() / 127.5 - 1.0
        
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
# Model (same as before)
# =========================================================================

class SinusoidalPosEmb(nn.Module):
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
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        )
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        t_proj = self.time_mlp(t_emb)
        scale, shift = t_proj.chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.shortcut(x) + h


class Attention(nn.Module):
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
        
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        
        return x + out


class MeanFlowSRNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
    ):
        super().__init__()
        
        time_dim = hidden_channels * 4
        
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
        
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Build skip channel stack
        skip_ch_stack = [hidden_channels]
        
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        current_ch = hidden_channels
        
        for i, mult in enumerate(channel_mult):
            out_ch = hidden_channels * mult
            
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(current_ch, out_ch, time_dim, dropout))
                current_ch = out_ch
                skip_ch_stack.append(current_ch)
            
            self.encoder.append(blocks)
            
            if i < len(channel_mult) - 1:
                self.downsample.append(nn.Conv2d(current_ch, current_ch, 3, stride=2, padding=1))
                skip_ch_stack.append(current_ch)
        
        self.mid_block1 = ResBlock(current_ch, current_ch, time_dim, dropout)
        self.mid_attn = Attention(current_ch)
        self.mid_block2 = ResBlock(current_ch, current_ch, time_dim, dropout)
        
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = hidden_channels * mult
            
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                skip_ch = skip_ch_stack.pop()
                blocks.append(ResBlock(current_ch + skip_ch, out_ch, time_dim, dropout))
                current_ch = out_ch
            
            self.decoder.append(blocks)
            
            if i > 0:
                self.upsample.append(nn.ConvTranspose2d(current_ch, current_ch, 4, stride=2, padding=1))
        
        self.out_norm = nn.GroupNorm(8, current_ch)
        self.out_conv = nn.Conv2d(current_ch, in_channels, 3, padding=1)
        
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, t, h):
        t_emb = self.time_embed(t) + self.h_embed(h)
        
        x = self.in_conv(x)
        
        skips = [x]
        for blocks, down in zip(self.encoder, self.downsample + [None]):
            for block in blocks:
                x = block(x, t_emb)
                skips.append(x)
            if down is not None:
                x = down(x)
                skips.append(x)
        
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        for blocks, up in zip(self.decoder, [None] + list(self.upsample)):
            if up is not None:
                x = up(x)
            for block in blocks:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
        
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        
        return x


# =========================================================================
# Trainer (修复版)
# =========================================================================

class MeanFlowSRTrainer:
    """
    修复版 Trainer
    
    关键修改：
    1. 移除 weight = 1/(h+0.01) 的不合理加权
    2. 增加 h 接近 1 的样本比例
    3. 使用更合理的时间采样策略
    """
    
    def __init__(
        self,
        model,
        device,
        lr=1e-4,
        P_mean=-0.4,
        P_std=1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.P_mean = P_mean
        self.P_std = P_std
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
        
        # EMA
        self.ema_decay = 0.9999
        self.ema_params = {name: param.clone() for name, param in model.named_parameters()}
    
    def update_ema(self):
        for name, param in self.model.named_parameters():
            self.ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def sample_time(self, batch_size):
        """
        改进的时间采样策略
        
        策略：
        - 50% 样本: t 从 logit-normal 采样, h = t (从 t 直接走到 0，学习完整路径)
        - 30% 样本: t 从 logit-normal 采样, h 从 uniform(0, t) 采样 (学习部分路径)
        - 20% 样本: t = 1, h = 1 (专门学习 one-step 推理！)
        """
        # 分配比例
        n_full = int(batch_size * 0.5)      # h = t
        n_partial = int(batch_size * 0.3)    # h < t
        n_onestep = batch_size - n_full - n_partial  # t=1, h=1
        
        # 采样 t
        rnd = torch.randn(batch_size, device=self.device)
        t = torch.sigmoid(rnd * self.P_std + self.P_mean)
        
        # 采样 h
        h = torch.zeros(batch_size, device=self.device)
        
        # 50%: h = t (完整路径)
        h[:n_full] = t[:n_full]
        
        # 30%: h ~ uniform(0, t)
        h[n_full:n_full+n_partial] = torch.rand(n_partial, device=self.device) * t[n_full:n_full+n_partial]
        
        # 20%: t = 1, h = 1 (one-step)
        t[n_full+n_partial:] = 1.0
        h[n_full+n_partial:] = 1.0
        
        return t, h
    
    def train_step(self, batch):
        """
        修复版训练步骤
        
        关键修改：
        1. 移除 weight = 1/(h+0.01) 加权
        2. 使用改进的时间采样
        """
        self.model.train()
        
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        batch_size = hr.shape[0]
        
        # 采样时间
        t, h = self.sample_time(batch_size)
        
        # 计算 r = t - h
        r = t - h
        
        # 插值得到 z_t
        t_exp = t[:, None, None, None]
        z_t = (1 - t_exp) * hr + t_exp * lr
        
        # 目标速度 v = LR - HR
        v = lr - hr
        
        # 网络预测
        u = self.model(z_t, t, h)
        
        # Loss: MSE without weird weighting
        loss = ((u - v) ** 2).mean()
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_ema()
        
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def inference(self, lr_images, num_steps=1, use_ema=True):
        """
        SR 推理
        
        Args:
            lr_images: LR 图像 tensor
            num_steps: 采样步数 (1 = one-step)
            use_ema: 是否使用 EMA 参数
        """
        self.model.eval()
        
        if use_ema:
            orig_params = {name: param.clone() for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                param.data.copy_(self.ema_params[name])
        
        batch_size = lr_images.shape[0]
        x = lr_images
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_current = 1.0 - i * dt
            
            t = torch.full((batch_size,), t_current, device=self.device)
            h = torch.full((batch_size,), dt, device=self.device)
            
            u = self.model(x, t, h)
            x = x - dt * u
        
        if use_ema:
            for name, param in self.model.named_parameters():
                param.data.copy_(orig_params[name])
        
        return x
    
    @torch.no_grad()
    def validate(self, val_loader, num_steps=1):
        """验证并计算 PSNR"""
        self.model.eval()
        
        psnr_list = []
        
        for batch in val_loader:
            hr = batch['hr'].to(self.device)
            lr = batch['lr'].to(self.device)
            
            hr_pred = self.inference(lr, num_steps=num_steps, use_ema=True)
            
            # 计算 PSNR (在 [0, 1] 范围)
            hr_01 = (hr + 1) / 2
            hr_pred_01 = (hr_pred + 1) / 2
            hr_pred_01 = hr_pred_01.clamp(0, 1)
            
            mse = ((hr_01 - hr_pred_01) ** 2).mean(dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            psnr_list.extend(psnr.cpu().tolist())
        
        return np.mean(psnr_list)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow SR Training (Fixed)')
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None, help='Validation HR dir')
    parser.add_argument('--val_lr_dir', type=str, default=None, help='Validation LR dir')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_sr_fixed')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    #device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device = torch.device(2)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("MeanFlow SR Training (Fixed Version)")
    print("="*60)
    print(f"Device: {device}")
    print(f"HR dir: {args.hr_dir}")
    print(f"LR dir: {args.lr_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patch size: {args.patch_size}")
    print("="*60)
    
    # Dataset
    dataset = SRDataset(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        patch_size=args.patch_size,
        augment=True,
        repeat=5,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Validation dataset (optional)
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(
            hr_dir=args.val_hr_dir,
            lr_dir=args.val_lr_dir,
            patch_size=args.patch_size,
            augment=False,
            repeat=1,
        )
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
        print(f"Validation set: {len(val_dataset)} samples")
    
    # Model
    model = MeanFlowSRNet(
        in_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.1,
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Trainer
    trainer = MeanFlowSRTrainer(model=model, device=device, lr=args.lr)
    
    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
    
    best_psnr = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            losses.append(metrics['loss'])
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
        avg_loss = np.mean(losses)
        
        # Validation
        val_psnr_1step = 0
        val_psnr_10step = 0
        if val_loader:
            val_psnr_1step = trainer.validate(val_loader, num_steps=1)
            val_psnr_10step = trainer.validate(val_loader, num_steps=10)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, Val PSNR (1-step)={val_psnr_1step:.2f}dB, (10-step)={val_psnr_10step:.2f}dB")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}")
        
        # Save best (by loss or PSNR)
        save_best = False
        if val_loader and val_psnr_1step > best_psnr:
            best_psnr = val_psnr_1step
            save_best = True
        elif not val_loader and avg_loss < best_loss:
            best_loss = avg_loss
            save_best = True
        
        if save_best:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': trainer.ema_params,
                'optimizer': trainer.optimizer.state_dict(),
                'loss': avg_loss,
                'psnr_1step': val_psnr_1step,
                'psnr_10step': val_psnr_10step,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  Saved best model!")
        
        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': trainer.ema_params,
                'optimizer': trainer.optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt'))
    
    print(f"\nTraining complete!")
    print(f"Best 1-step PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()