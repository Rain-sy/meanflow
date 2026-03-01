#!/usr/bin/env python
"""
Dual-Stream FLUX SR Evaluation with CLEAR Acceleration

支持高分辨率推理，使用 CLEAR 局部注意力加速

Usage:
    python evaluate_dual_control_clear.py \
        --checkpoint checkpoints/best_model.pt \
        --input_dir test_images/LR \
        --output_dir results/SR \
        --window_size 16
"""

import os
import argparse
import time
import gc
from PIL import Image
from functools import partial, lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 🌟 必须加上：防止 Triton 编译时共享内存溢出
import torch._inductor.config as inductor_config
inductor_config.max_autotune = True
inductor_config.coordinate_descent_tuning = True


# ============================================================================
# CLEAR: Local Attention Mask & Processor
# ============================================================================

BLOCK_MASK = None
LATENT_HEIGHT = None
LATENT_WIDTH = None


@lru_cache
def init_local_mask_flex(height, width, text_length, window_size, device):
    """初始化 CLEAR 的局部注意力 mask"""
    from torch.nn.attention.flex_attention import create_block_mask
    create_block_mask_compiled = torch.compile(create_block_mask)
    
    def local_mask(b, h, q_idx, kv_idx):
        q_y = (q_idx - text_length) // width
        q_x = (q_idx - text_length) % width
        kv_y = (kv_idx - text_length) // width
        kv_x = (kv_idx - text_length) % width
        
        return torch.logical_or(
            torch.logical_or(q_idx < text_length, kv_idx < text_length),
            (q_y - kv_y) ** 2 + (q_x - kv_x) ** 2 < window_size ** 2
        )
    
    global BLOCK_MASK, LATENT_HEIGHT, LATENT_WIDTH
    
    total_len = text_length + height * width
    BLOCK_MASK = create_block_mask_compiled(
        local_mask, B=None, H=None, device=device,
        Q_LEN=total_len, KV_LEN=total_len, _compile=True
    )
    LATENT_HEIGHT = height
    LATENT_WIDTH = width
    
    return BLOCK_MASK


class LocalFlexAttnProcessor:
    """CLEAR 的局部注意力处理器"""
    
    def __init__(self):
        import math
        from torch.nn.attention.flex_attention import flex_attention
        
        assert BLOCK_MASK is not None
        self.flex_attn = partial(flex_attention, block_mask=BLOCK_MASK)
        self.flex_attn = torch.compile(self.flex_attn, dynamic=False)
    
    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs
    ) -> torch.FloatTensor:
        import math
        from diffusers.models.embeddings import apply_rotary_emb
        
        batch_size = hidden_states.shape[0]
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        attention_scale = math.sqrt(1 / head_dim)
        
        hidden_states = self.flex_attn(query, key, value, scale=attention_scale)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


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
        feat = self.encoder(x)
        return self.zero_conv(feat)


# ============================================================================
# Dual-Stream FLUX SR System with CLEAR
# ============================================================================

class DualStreamFLUXSR_CLEAR:
    """Dual-Stream FLUX SR with CLEAR for inference"""
    
    def __init__(self, model_name, device, pretrained_controlnet,
                 checkpoint_path, window_size=16, down_factor=4, use_clear=True):
        self.device = device
        self.window_size = window_size
        self.down_factor = down_factor # 🌟 必须知道训练时的 down_factor
        self.use_clear = use_clear
        
        self._load_models(model_name, pretrained_controlnet, checkpoint_path)
    
    def _load_models(self, model_name, pretrained_controlnet, checkpoint_path):
        # 🌟 直接从旁边的文件导入
        from transformer_flux import FluxTransformer2DModel 
        from diffusers import AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.enable_tiling()
        
        # ... (保留你原有的 Text Encoder 加载和 Cache 代码) ...
        # (这部分写得很好，无需改动)
        
        print("Loading Transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        
        print("Loading ControlNet...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            pretrained_controlnet, torch_dtype=dtype
        ).to(self.device)
        self.controlnet.requires_grad_(False)
        
        print("Loading Pixel Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(False)
        
        # 🌟 必须在 Load Checkpoint 之前，先把网络结构调整为 CLEAR 模式！
        # 否则 checkpoint 里的 spatial_weight 找不到对应的层
        if self.use_clear:
            from attention_processor import LocalDownsampleFlexAttnProcessor, LocalFlexAttnProcessor
            attn_processors = {}
            for k in self.transformer.attn_processors.keys():
                if self.down_factor > 1:
                    attn_processors[k] = LocalDownsampleFlexAttnProcessor(down_factor=self.down_factor).to(self.device, dtype)
                else:
                    attn_processors[k] = LocalFlexAttnProcessor()
            self.transformer.set_attn_processor(attn_processors)
            print(f"[CLEAR] Initialized Transformer Attention Processors (down_factor={self.down_factor})")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        ctrl_state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
        self.controlnet.load_state_dict(ctrl_state)
        
        pixel_state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
        self.pixel_extractor.load_state_dict(pixel_state)
        
        self.pixel_weight = ckpt.get('pixel_weight', 0.1)
        print(f"Pixel weight: {self.pixel_weight}")
        
        self.controlnet.eval()
        self.pixel_extractor.eval()

    def enable_clear(self, height, width):
        """为指定分辨率启用 CLEAR 的 Mask"""
        if not self.use_clear:
            return
            
        from attention_processor import init_local_downsample_mask_flex, init_local_mask_flex
        
        # 🌟 修复降维陷阱：这里必须是 // 16
        patch_h = height // 16
        patch_w = width // 16
        text_length = 512
        
        print(f"[CLEAR] Initializing mask for {height}x{width} (patch grid: {patch_h}x{patch_w})...")
        
        if self.down_factor > 1:
            init_local_downsample_mask_flex(
                height=patch_h,
                width=patch_w,
                text_length=text_length,
                window_size=self.window_size,
                down_factor=self.down_factor,
                device=self.device
            )
        else:
            init_local_mask_flex(
                height=patch_h,
                width=patch_w,
                text_length=text_length,
                window_size=self.window_size,
                device=self.device
            )
        print(f"[CLEAR] ✓ Mask Generated (window_size={self.window_size})")
    
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
    
    def forward_step(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
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
    def inference(self, lr_img, target_size=None, num_steps=20, guidance=3.5):
        """
        SR inference with CLEAR acceleration
        
        Args:
            lr_img: PIL Image (LR input)
            target_size: (height, width) of output, None means 4x upscale
            num_steps: number of denoising steps
            guidance: classifier-free guidance scale
        """
        dtype = torch.bfloat16
        
        # Prepare input
        lr_w, lr_h = lr_img.size
        if target_size is None:
            target_h, target_w = lr_h * 4, lr_w * 4
        else:
            target_h, target_w = target_size
        
        # Ensure divisible by 32 (for VAE)
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32
        
        # Enable CLEAR for this resolution
        if self.use_clear:
            self.enable_clear(target_h, target_w)
        
        # Upscale LR to target size
        lr_up = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
        lr_t = lr_t.unsqueeze(0).to(self.device, dtype=dtype)
        
        # Encode
        lr_lat = self.encode(lr_t)
        
        # Flow matching inference
        lat = torch.randn_like(lr_lat)
        dt = 1.0 / num_steps
        
        for i in tqdm(range(num_steps), desc="Denoising"):
            t_val = 1.0 - i * dt
            t = torch.full((1,), t_val, device=self.device, dtype=dtype)
            v = self.forward_step(lat, lr_lat, lr_t, t, guidance)
            lat = lat - dt * v
        
        # Decode
        sr = self.decode(lat)
        
        # To PIL
        sr_np = ((sr[0].float().cpu().clamp(-1, 1) + 1) * 127.5).numpy().transpose(1, 2, 0).astype(np.uint8)
        
        return Image.fromarray(sr_np)


# ============================================================================
# Metrics
# ============================================================================

def calculate_psnr(img1, img2):
    """Calculate PSNR between two PIL Images"""
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate SSIM between two PIL Images"""
    try:
        from skimage.metrics import structural_similarity as ssim
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        return ssim(arr1, arr2, channel_axis=2, data_range=255)
    except:
        return 0.0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, default=None, help='Ground truth for metrics')
    
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, 
                        default='jasperai/Flux.1-dev-Controlnet-Upscaler')
    
    # CLEAR params
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--no_clear', action='store_true')
    
    # Inference params
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--scale', type=int, default=4, help='Upscale factor')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Dual-Stream FLUX SR Evaluation with CLEAR")
    print("=" * 60)
    print(f"CLEAR: {'Disabled' if args.no_clear else f'Enabled (r={args.window_size})'}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Load model
    system = DualStreamFLUXSR_CLEAR(
        model_name=args.model_name,
        device=device,
        pretrained_controlnet=args.pretrained_controlnet,
        checkpoint_path=args.checkpoint,
        window_size=args.window_size,
        use_clear=not args.no_clear,
        down_factor=4, # 🌟 填入你训练时使用的参数
    )
    
    # Get input images
    image_files = sorted([f for f in os.listdir(args.input_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nProcessing {len(image_files)} images...\n")
    
    psnr_list = []
    ssim_list = []
    time_list = []
    
    for img_file in tqdm(image_files):
        lr_path = os.path.join(args.input_dir, img_file)
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Calculate target size
        lr_w, lr_h = lr_img.size
        target_h = (lr_h * args.scale // 32) * 32
        target_w = (lr_w * args.scale // 32) * 32
        
        # Inference
        start_time = time.time()
        sr_img = system.inference(
            lr_img, 
            target_size=(target_h, target_w),
            num_steps=args.num_steps,
            guidance=args.guidance
        )
        elapsed = time.time() - start_time
        time_list.append(elapsed)
        
        # Save
        output_path = os.path.join(args.output_dir, img_file)
        sr_img.save(output_path)
        
        # Metrics if GT available
        if args.gt_dir:
            gt_path = os.path.join(args.gt_dir, img_file)
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert('RGB')
                gt_img = gt_img.resize((target_w, target_h), Image.BICUBIC)
                
                psnr = calculate_psnr(sr_img, gt_img)
                ssim = calculate_ssim(sr_img, gt_img)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
    
    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Avg time per image: {np.mean(time_list):.2f}s")
    
    if psnr_list:
        print(f"Avg PSNR: {np.mean(psnr_list):.2f} dB")
        print(f"Avg SSIM: {np.mean(ssim_list):.4f}")
    
    print(f"Output saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
