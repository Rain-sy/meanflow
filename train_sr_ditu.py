"""
MeanFlow-DiT (U-ViT Style) Training Script for Super-Resolution
Compatible with previous arguments interface.

Key Features:
1. U-ViT Style Architecture: Long Skip Connections for better detail recovery.
2. Correct Time Sampling: 50% h=t, 30% h<t, 20% t=1,h=1 (Critical for one-step).
3. 2D Sin-Cos Positional Embedding: Better generalization for different resolutions.
4. PixelShuffle Upsampling: Reduced checkerboard artifacts.

Usage Example:
    python meanflow/train_sr_ditu.py \
        --hr_dir meanflow/Data/DIV2K/DIV2K_train_HR \
        --lr_dir meanflow/Data/DIV2K/DIV2K_train_LR_bicubic_X2 \
        --val_hr_dir meanflow/Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir meanflow/Data/DIV2K/DIV2K_valid_LR_bicubic_X2 \
        --model_size small \
        --batch_size 8 \
        --epochs 100 \
        --device cuda:0
"""

import os
import math
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================================
# 1. Dataset
# ============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=128, scale=2, augment=True, repeat=5):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.repeat = repeat
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.hr_files) == len(self.lr_files), f"HR ({len(self.hr_files)}) and LR ({len(self.lr_files)}) count mismatch"
        print(f"[Dataset] Found {len(self.hr_files)} pairs (repeat={repeat})")
    
    def __len__(self):
        return len(self.hr_files) * self.repeat
    
    def __getitem__(self, idx):
        idx = idx % len(self.hr_files)
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Upsample LR to HR size immediately (MeanFlow Input Requirement)
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        hr_img, lr_up = self._random_crop(hr_img, lr_up)
        
        if self.augment:
            if random.random() < 0.5:
                hr_img, lr_up = hr_img.transpose(Image.FLIP_LEFT_RIGHT), lr_up.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                hr_img, lr_up = hr_img.transpose(Image.FLIP_TOP_BOTTOM), lr_up.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                hr_img, lr_up = hr_img.rotate(90*k), lr_up.rotate(90*k)
        
        # Normalize to [-1, 1]
        hr_t = torch.from_numpy(np.array(hr_img)).float().permute(2,0,1) / 127.5 - 1.0
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2,0,1) / 127.5 - 1.0
        return {'hr': hr_t, 'lr': lr_t}
    
    def _random_crop(self, hr, lr):
        w, h = hr.size
        if w < self.patch_size or h < self.patch_size:
            s = max(self.patch_size/w, self.patch_size/h) * 1.1
            new_size = (int(w*s), int(h*s))
            hr = hr.resize(new_size, Image.BICUBIC)
            lr = lr.resize(new_size, Image.BICUBIC)
            w, h = new_size
            
        x, y = random.randint(0, w - self.patch_size), random.randint(0, h - self.patch_size)
        return hr.crop((x, y, x+self.patch_size, y+self.patch_size)), lr.crop((x, y, x+self.patch_size, y+self.patch_size))

# ============================================================================
# 2. U-ViT / MMDiT-Style Architecture components
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-Attention
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # --- FIX: Use PyTorch Scaled Dot Product Attention (FlashAttention) ---
        # Instead of self.attn(x, x, x), use the functional API which is memory efficient
        q, k, v = self.attn.in_proj_weight.split(self.attn.embed_dim, dim=0)
        q = F.linear(x_norm, q, self.attn.in_proj_bias[:self.attn.embed_dim])
        k = F.linear(x_norm, k, self.attn.in_proj_bias[self.attn.embed_dim:2*self.attn.embed_dim])
        v = F.linear(x_norm, v, self.attn.in_proj_bias[2*self.attn.embed_dim:])
        
        # Reshape for multi-head: (B, L, H, D) -> (B, H, L, D)
        B, L, _ = x_norm.shape
        q = q.view(B, L, self.attn.num_heads, -1).transpose(1, 2)
        k = k.view(B, L, self.attn.num_heads, -1).transpose(1, 2)
        v = v.view(B, L, self.attn.num_heads, -1).transpose(1, 2)
        
        # This function automatically selects FlashAttention if available
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        
        # Reshape back: (B, H, L, D) -> (B, L, H*D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        attn_out = self.attn.out_proj(attn_out)
        # ----------------------------------------------------------------------

        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x

class MeanFlowDiT(nn.Module):
    """
    U-ViT style DiT: Adds skip connections between encoder and decoder blocks.
    """
    def __init__(self, img_size=128, patch_size=2, in_channels=3, hidden_size=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        
        # Patch Embed (Smaller patch size = better details)
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # Fixed 2D Pos Embed
        pos_embed = get_2d_sincos_pos_embed(hidden_size, int(img_size//patch_size))
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        
        # Time Embeds
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        
        # Blocks
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        
        # Final Layer
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.pred_head = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        
        # Init
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

    def unpatchify(self, x):
        p = self.patch_size
        c = self.in_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(self, x, t, h):
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, :x.shape[1], :]
        c = self.t_embedder(t) + self.h_embedder(h)
        
        skips = []
        half_depth = self.depth // 2
        
        # U-ViT Forward
        for i, block in enumerate(self.blocks):
            if i < half_depth:
                x = block(x, c)
                skips.append(x)
            else:
                skip = skips.pop()
                x = x + skip  # Skip Connection
                x = block(x, c)
        
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.pred_head(x)
        return self.unpatchify(x)

# ============================================================================
# 3. Model Configurations & Helper Functions
# ============================================================================

def MeanFlowDiT_XS(img_size=128, **kwargs):
    return MeanFlowDiT(img_size, patch_size=2, hidden_size=256, depth=8, num_heads=4)

def MeanFlowDiT_S(img_size=128, **kwargs):
    return MeanFlowDiT(img_size, patch_size=2, hidden_size=384, depth=12, num_heads=6)

def MeanFlowDiT_B(img_size=128, **kwargs):
    return MeanFlowDiT(img_size, patch_size=2, hidden_size=512, depth=12, num_heads=8)

def MeanFlowDiT_L(img_size=128, **kwargs):
    return MeanFlowDiT(img_size, patch_size=2, hidden_size=768, depth=24, num_heads=12)

MODEL_CONFIGS = {
    'xs': MeanFlowDiT_XS,
    'small': MeanFlowDiT_S,
    'base': MeanFlowDiT_B,
    'large': MeanFlowDiT_L,
}

# ============================================================================
# 4. Trainer with Correct Sampling
# ============================================================================

class MeanFlowDiTTrainer:
    def __init__(self, model, device, lr=1e-4, weight_decay=0.01):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_decay = 0.9999
        self.ema_params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.P_mean = -0.4
        self.P_std = 1.0

    def update_ema(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.ema_params[n].mul_(self.ema_decay).add_(p.data, alpha=1-self.ema_decay)

    def sample_time(self, batch_size):
        """
        Critical Fix: 50% h=t, 30% h<t, 20% one-step (t=1,h=1)
        """
        n_full = int(batch_size * 0.5)
        n_partial = int(batch_size * 0.3)
        
        rnd = torch.randn(batch_size, device=self.device)
        t = torch.sigmoid(rnd * self.P_std + self.P_mean)
        h = torch.zeros_like(t)
        
        # 1. Full Path (h=t)
        h[:n_full] = t[:n_full]
        
        # 2. Partial Path (h ~ U[0, t])
        h[n_full:n_full+n_partial] = torch.rand(n_partial, device=self.device) * t[n_full:n_full+n_partial]
        
        # 3. One-step Anchor (t=1, h=1)
        t[n_full+n_partial:] = 1.0
        h[n_full+n_partial:] = 1.0
        
        return t, h

    def train_step(self, batch):
        self.model.train()
        hr = batch['hr'].to(self.device)
        lr = batch['lr'].to(self.device)
        
        t, h = self.sample_time(hr.shape[0])
        
        t_exp = t[:, None, None, None]
        z_t = (1 - t_exp) * hr + t_exp * lr
        v = lr - hr
        
        u = self.model(z_t, t, h)
        loss = F.mse_loss(u, v)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.update_ema()
        
        return loss.item()

    @torch.no_grad()
    def validate(self, val_loader, num_steps=1):
        self.model.eval()
        psnrs = []
        
        # Swap EMA params
        orig_params = {n: p.clone() for n, p in self.model.named_parameters()}
        for n, p in self.model.named_parameters():
            p.data.copy_(self.ema_params[n])
            
        for batch in val_loader:
            hr, lr = batch['hr'].to(self.device), batch['lr'].to(self.device)
            b = lr.shape[0]
            
            # One-step Inference
            t = torch.ones(b, device=self.device)
            h = torch.ones(b, device=self.device)
            u = self.model(lr, t, h)
            hr_pred = lr - u # z_0 = z_1 - 1.0 * u
            
            # PSNR
            hr_pred = ((hr_pred + 1) / 2).clamp(0, 1)
            hr_gt = (hr + 1) / 2
            mse = ((hr_pred - hr_gt)**2).mean(dim=[1,2,3])
            psnr = -10 * torch.log10(mse + 1e-8)
            psnrs.extend(psnr.cpu().tolist())
            
        # Restore params
        for n, p in self.model.named_parameters():
            p.data.copy_(orig_params[n])
            
        return np.mean(psnrs)

    def save_checkpoint(self, path, epoch, loss, psnr=0):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema': self.ema_params,
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'psnr': psnr
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt: self.ema_params = ckpt['ema']
        if 'optimizer' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0)

# ============================================================================
# 5. Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MeanFlow-DiT (U-ViT Style) Training')
    
    # Dataset
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=2)
    
    # Model
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--patch_size', type=int, default=128)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # Checkpointing & Device
    parser.add_argument('--save_dir', type=str, default='./checkpoints_dit_uvit')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_every_steps', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup
    if 'cuda' in args.device and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print(f"MeanFlow-DiT (U-ViT Style) Training")
    print(f"Device: {device} | Model: {args.model_size}")
    print("="*60)
    
    # Dataset
    train_dataset = SRDataset(args.hr_dir, args.lr_dir, patch_size=args.patch_size, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(args.val_hr_dir, args.val_lr_dir, patch_size=args.patch_size, augment=False, repeat=1)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
        print(f"Validation Set: {len(val_dataset)} images")
    
    # Initialize Model
    model_fn = MODEL_CONFIGS[args.model_size]
    # Note: We use patch_size=2 internally for better SR quality, regardless of args.patch_size (training crop size)
    model = model_fn(img_size=args.patch_size, patch_size=4) 
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Trainer
    trainer = MeanFlowDiTTrainer(model, device, lr=args.lr, weight_decay=args.weight_decay)
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training Loop
    global_step = 0
    best_psnr = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            loss = trainer.train_step(batch)
            losses.append(loss)
            pbar.set_postfix({'loss': f"{loss:.4f}"})
            global_step += 1
            
            # Step-based saving
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pt'), epoch, loss)
        
        avg_loss = np.mean(losses)
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        val_info = ""
        if val_loader:
            val_psnr = trainer.validate(val_loader)
            val_info = f" | Val PSNR: {val_psnr:.2f}dB"
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                trainer.save_checkpoint(os.path.join(args.save_dir, 'best_model.pt'), epoch, avg_loss, val_psnr)
                print(f"  New Best PSNR: {best_psnr:.2f} dB")
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.5f} | LR={current_lr:.2e}{val_info}")
        
        # Save Epoch Checkpoint
        if (epoch + 1) % 10 == 0:
             trainer.save_checkpoint(os.path.join(args.save_dir, f'ckpt_epoch_{epoch+1}.pt'), epoch, avg_loss)
             
        scheduler.step()

    # Save Final
    trainer.save_checkpoint(os.path.join(args.save_dir, 'final.pt'), args.epochs, avg_loss)
    print("Training Complete.")

if __name__ == '__main__':
    main()