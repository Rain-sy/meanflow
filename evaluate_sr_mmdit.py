"""
MeanFlow-MMDiT Evaluation (Unified: Pixel + Latent)

Auto-detects mode from checkpoint or uses --mode flag.

Usage:
    python evaluate_sr_mmdit.py \
        --checkpoint checkpoints_mmdit/xxx/best.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --model_size small \
        --scale 4
        python evaluate_sr_mmdit.py --mode pixel ...
python evaluate_sr_mmdit.py --mode latent --vae_type flux ...
"""

import os, math, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# VAE Wrapper
# ============================================================================

class VAEWrapper:
    def __init__(self, vae_type='flux', device='cuda'):
        self.vae_type, self.device = vae_type, device
        from diffusers import AutoencoderKL
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers" if vae_type == 'sd3' else "black-forest-labs/FLUX.1-dev"
        print(f"Loading {vae_type.upper()} VAE...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        for p in self.vae.parameters(): p.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, z):
        return self.vae.decode(z / self.vae.config.scaling_factor).sample


# ============================================================================
# MMDiT Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MMDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm1_img, self.norm2_img = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.norm1_cond, self.norm2_cond = RMSNorm(hidden_size), RMSNorm(hidden_size)
        
        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3)
        self.qkv_cond = nn.Linear(hidden_size, hidden_size * 3)
        self.proj_img = nn.Linear(hidden_size, hidden_size)
        self.proj_cond = nn.Linear(hidden_size, hidden_size)
        
        mlp_h = int(hidden_size * mlp_ratio)
        self.mlp_img = nn.Sequential(nn.Linear(hidden_size, mlp_h), nn.GELU(), nn.Linear(mlp_h, hidden_size))
        self.mlp_cond = nn.Sequential(nn.Linear(hidden_size, mlp_h), nn.GELU(), nn.Linear(mlp_h, hidden_size))
        
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
    
    def forward(self, x_img, x_cond, c):
        B = x_img.shape[0]
        shift_i, scale_i, gate_i, shift_c, scale_c, gate_c = self.adaLN(c).chunk(6, dim=-1)
        
        xi = self.norm1_img(x_img) * (1 + scale_i.unsqueeze(1)) + shift_i.unsqueeze(1)
        xc = self.norm1_cond(x_cond) * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)
        
        qkv_i = self.qkv_img(xi).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_c = self.qkv_cond(xc).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_i, k_i, v_i = qkv_i[0], qkv_i[1], qkv_i[2]
        q_c, k_c, v_c = qkv_c[0], qkv_c[1], qkv_c[2]
        
        k_j, v_j = torch.cat([k_i, k_c], dim=2), torch.cat([v_i, v_c], dim=2)
        attn_i = F.scaled_dot_product_attention(q_i, k_j, v_j).transpose(1,2).reshape(B, -1, self.hidden_size)
        attn_c = F.scaled_dot_product_attention(q_c, k_j, v_j).transpose(1,2).reshape(B, -1, self.hidden_size)
        
        x_img = x_img + gate_i.unsqueeze(1) * self.proj_img(attn_i)
        x_cond = x_cond + gate_c.unsqueeze(1) * self.proj_cond(attn_c)
        x_img = x_img + gate_i.unsqueeze(1) * self.mlp_img(self.norm2_img(x_img))
        x_cond = x_cond + gate_c.unsqueeze(1) * self.mlp_cond(self.norm2_cond(x_cond))
        return x_img, x_cond


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.freq_dim = freq_dim
    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = torch.cat([torch.cos(t[:,None] * freqs), torch.sin(t[:,None] * freqs)], dim=-1)
        return self.mlp(emb)


class MMDiT(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_channels=3, hidden_size=512, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_size, self.in_channels, self.hidden_size = patch_size, in_channels, hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        
        self.patch_embed_img = nn.Linear(patch_dim, hidden_size)
        self.patch_embed_cond = nn.Linear(patch_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([MMDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        
        self.norm_out = RMSNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, patch_dim)
    
    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        return x.reshape(B, C, H//p, p, W//p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*p*p)
    
    def unpatchify(self, x, h, w):
        B, N, _ = x.shape
        p = self.patch_size
        return x.reshape(B, h//p, w//p, self.in_channels, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, self.in_channels, h, w)
    
    def forward(self, z_t, z_cond, t, h):
        B, C, H, W = z_t.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        pos = self._interp_pos(H, W) if num_patches != self.num_patches else self.pos_embed
        
        x_img = self.patch_embed_img(self.patchify(z_t)) + pos
        x_cond = self.patch_embed_cond(self.patchify(z_cond)) + pos
        c = self.t_embedder(t) + self.h_embedder(h)
        
        for blk in self.blocks:
            x_img, x_cond = blk(x_img, x_cond, c)
        return self.unpatchify(self.proj_out(self.norm_out(x_img)), H, W)
    
    def _interp_pos(self, h, w):
        old = int(self.num_patches ** 0.5)
        nh, nw = h // self.patch_size, w // self.patch_size
        pos = self.pos_embed.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(nh, nw), mode='bilinear', align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, nh * nw, -1)


# ============================================================================
# Model Configs
# ============================================================================

CONFIGS = {
    'xs': (256, 8, 4), 'small': (384, 12, 6), 'base': (512, 12, 8), 'large': (768, 24, 12)
}

def create_model(model_size, mode, img_size):
    h, d, n = CONFIGS[model_size]
    ps, ch = (4, 3) if mode == 'pixel' else (2, 16)
    return MMDiT(img_size=img_size, patch_size=ps, in_channels=ch, hidden_size=h, depth=d, num_heads=n)


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def inference_tile(model, lr_img, device, tile_size, overlap, num_steps, mode, vae):
    """Tiled inference for large images"""
    _, C, H, W = lr_img.shape
    
    # Small image: direct process
    if H <= tile_size and W <= tile_size:
        return inference_single(model, lr_img, device, num_steps, mode, vae)
    
    # Tiled processing
    stride = tile_size - overlap
    output = torch.zeros(1, 3, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)
    blend = _create_blend_weight(tile_size, overlap, device)
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            ye, xe = min(y + tile_size, H), min(x + tile_size, W)
            ys, xs = max(0, ye - tile_size), max(0, xe - tile_size)
            
            tile = lr_img[:, :, ys:ye, xs:xe]
            th, tw = tile.shape[2], tile.shape[3]
            
            if th < tile_size or tw < tile_size:
                tile = F.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode='reflect')
            
            out = inference_single(model, tile, device, num_steps, mode, vae)
            out = out[:, :, :th, :tw]
            
            w = blend[:, :, :th, :tw]
            output[:, :, ys:ye, xs:xe] += out * w
            weight[:, :, ys:ye, xs:xe] += w
    
    return output / (weight + 1e-8)


@torch.no_grad()
def inference_single(model, lr_img, device, num_steps, mode, vae):
    """Single tile inference"""
    lr_in = vae.encode(lr_img) if mode == 'latent' else lr_img
    x, dt = lr_in.clone(), 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.full((1,), 1.0 - i * dt, device=device)
        h = torch.full((1,), dt, device=device)
        x = x - dt * model(x, lr_in, t, h)
    
    return vae.decode(x) if mode == 'latent' else x


def _create_blend_weight(size, overlap, device):
    w = torch.ones(1, 1, size, size, device=device)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap, device=device)
        w[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
        w[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
        w[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
        w[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)
    return w


# ============================================================================
# Metrics
# ============================================================================

def calc_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return 10 * np.log10(255**2 / (mse + 1e-8)) if mse > 0 else float('inf')


def calc_ssim(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, channel_axis=2, data_range=255)
    except:
        return 0.0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_size', type=str, default='small', choices=['xs', 'small', 'base', 'large'])
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--mode', type=str, default=None, choices=['pixel', 'latent'], help='Auto-detect if not set')
    parser.add_argument('--vae_type', type=str, default='flux', choices=['sd3', 'flux'])
    parser.add_argument('--tile_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Auto-detect mode
    mode = args.mode or ckpt.get('mode', 'pixel')
    print(f"Mode: {mode}")
    
    # Load VAE if latent mode
    vae = VAEWrapper(args.vae_type, device) if mode == 'latent' else None
    
    # Determine img_size from checkpoint
    pos_shape = ckpt.get('ema', ckpt['model'])['pos_embed'].shape
    num_patches = pos_shape[1]
    grid_size = int(num_patches ** 0.5)
    
    # Detect patch_size from model weights
    patch_embed_shape = ckpt.get('ema', ckpt['model'])['patch_embed_img.weight'].shape
    patch_dim = patch_embed_shape[1]
    in_channels = 3 if mode == 'pixel' else 16
    patch_size = int((patch_dim / in_channels) ** 0.5)
    
    img_size = grid_size * patch_size
    print(f"Detected: img_size={img_size}, patch_size={patch_size}, in_channels={in_channels}")
    
    # Create model
    model = create_model(args.model_size, mode, img_size)
    
    # Load weights (use EMA if available)
    weights = ckpt.get('ema', ckpt['model'])
    model.load_state_dict(weights)
    model = model.to(device).eval()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get files
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    assert len(hr_files) == len(lr_files)
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nEvaluating {len(hr_files)} images...")
    print(f"Tile: {args.tile_size}, Overlap: {args.overlap}, Steps: {args.num_steps}")
    print("="*60)
    
    psnrs, ssims, bicubic_psnrs = [], [], []
    
    for hf, lf in tqdm(zip(hr_files, lr_files), total=len(hr_files)):
        hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lf)).convert('RGB')
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        # To tensor
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device)
        
        # Inference
        pred = inference_tile(model, lr_t, device, args.tile_size, args.overlap, args.num_steps, mode, vae)
        
        # To numpy
        pred_np = ((pred[0].cpu().clamp(-1,1) + 1) * 127.5).permute(1,2,0).numpy().astype(np.uint8)
        hr_np = np.array(hr_img)
        lr_up_np = np.array(lr_up)
        
        # Metrics
        psnrs.append(calc_psnr(pred_np, hr_np))
        ssims.append(calc_ssim(pred_np, hr_np))
        bicubic_psnrs.append(calc_psnr(lr_up_np, hr_np))
        
        if args.output_dir:
            Image.fromarray(pred_np).save(os.path.join(args.output_dir, hf))
    
    print("="*60)
    print(f"Results ({len(hr_files)} images):")
    print(f"  PSNR:    {np.mean(psnrs):.4f} dB (±{np.std(psnrs):.4f})")
    print(f"  SSIM:    {np.mean(ssims):.4f} (±{np.std(ssims):.4f})")
    print(f"  Bicubic: {np.mean(bicubic_psnrs):.4f} dB")
    print(f"  Gain:    +{np.mean(psnrs) - np.mean(bicubic_psnrs):.4f} dB")
    print("="*60)


if __name__ == '__main__':
    main()