#!/usr/bin/env python
"""
Dual-Stream FLUX SR Evaluation with CLEAR Acceleration

python evaluate_dual_control_clear.py \
    --checkpoint checkpoints/xxx/best_model.pt \
    --hr_dir Data/DIV2K/DIV2K_valid_HR \
    --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4

"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 🌟 必须在导入 torch 之前设置这些环境变量
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCH_COMPILE_DEBUG"] = "0"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_cache"

import gc
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

# 🌟 CLEAR 需要的 inductor 配置
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.verbose_progress = False
    inductor_config.benchmark_kernel = False
    inductor_config.autotune_in_subproc = False
    inductor_config.trace.enabled = False
except:
    pass

# 禁用所有 torch 相关日志
import logging
for name in ['torch', 'torch._dynamo', 'torch._inductor', 'torch._functorch', 'torch.fx', 'triton']:
    logging.getLogger(name).setLevel(logging.CRITICAL)

try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune = True
    inductor_config.coordinate_descent_tuning = True
except:
    pass

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Note: lpips not installed. Run: pip install lpips")


# ============================================================================
# Pixel Feature Extractor (与训练代码完全一致)
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    def __init__(self, latent_channels=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels), nn.SiLU(),
        )
        
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        return self.zero_conv(self.encoder(x))


# ============================================================================
# Metrics
# ============================================================================

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq, sigma2_sq = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))


# ============================================================================
# Evaluator (完全匹配训练代码)
# ============================================================================

class DualStreamEvaluator:
    def __init__(self, model_name, device, checkpoint_path=None, pixel_weight=0.1,
                 window_size=16, down_factor=4, clear_ckpt="ckpt/clear_local_16_down_4.safetensors"):
        self.model_name = model_name
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.pixel_weight = pixel_weight
        self.window_size = window_size
        self.down_factor = down_factor
        self.clear_ckpt = clear_ckpt
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self._cached_embeds = None
        
        # 记录当前 mask 对应的分辨率，避免重复初始化
        self._current_mask_size = None
    
    def load(self):
        from transformer_flux import FluxTransformer2DModel  # 🌟 使用 CLEAR 版本
        from diffusers import AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        from safetensors.torch import load_file
        from attention_processor import (
            LocalDownsampleFlexAttnProcessor,
            LocalFlexAttnProcessor,
            init_local_downsample_mask_flex,
            init_local_mask_flex
        )
        
        dtype = torch.bfloat16
        print(f"[Eval] Loading models (CLEAR: w={self.window_size}, d={self.down_factor})...", end=" ", flush=True)
        
        # Load VAE (先加载到 CPU，用时再搬到 GPU，节省显存)
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        )
        self.vae.requires_grad_(False)
        self.vae.enable_tiling()  # 🌟 极大降低高分辨率 Decode 时的显存峰值
        self.vae.to("cpu")  # 🌟 平时放 CPU，用时再搬
        print("VAE(tiled)", end=" ", flush=True)
        
        # Cache text embeddings (与训练完全一致)
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
        print("Text", end=" ", flush=True)

        # Load CLEAR Transformer
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        print("Transformer", end=" ", flush=True)

        # Load ControlNet
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=dtype
        ).to(self.device)
        self.controlnet.requires_grad_(False)
        
        # 为 ControlNet 启用 Flash Attention
        try:
            import xformers
            self.controlnet.enable_xformers_memory_efficient_attention()
        except:
            pass
        print("ControlNet", end=" ", flush=True)
        
        # Load Pixel Extractor
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(False)
        print("PixelNet", end=" ", flush=True)
        
        # 🌟 初始化 CLEAR
        dummy_patch_size = 32  # 对应 512x512
        device_str = str(self.device)
        
        # 🌟 初始化 CLEAR mask
        dummy_patch_size = 32
        device_str = str(self.device)
        
        if self.down_factor > 1:
            init_local_downsample_mask_flex(
                height=dummy_patch_size, width=dummy_patch_size, text_length=512,
                window_size=self.window_size, down_factor=self.down_factor, device=device_str
            )
        else:
            init_local_mask_flex(
                height=dummy_patch_size, width=dummy_patch_size, text_length=512,
                window_size=self.window_size, device=device_str
            )
        
        # 🌟 修复：精准投递 CLEAR，只对双流模块使用，单流模块保留原生！
        dtype = torch.bfloat16
        original_processors = self.transformer.attn_processors
        trans_attn_processors = {}
        
        for k, proc in original_processors.items():
            if k.startswith("transformer_blocks"):
                # 只有双流模块换上 CLEAR（这些有预训练权重）
                if self.down_factor > 1:
                    trans_attn_processors[k] = LocalDownsampleFlexAttnProcessor(
                        down_factor=self.down_factor
                    ).to(self.device, dtype)
                else:
                    trans_attn_processors[k] = LocalFlexAttnProcessor()
            else:
                # 🌟 极其重要：single_transformer_blocks 必须保留原生处理器！
                trans_attn_processors[k] = proc
        
        self.transformer.set_attn_processor(trans_attn_processors)
        
        # 统计
        clear_count = sum(1 for k in trans_attn_processors if k.startswith("transformer_blocks"))
        native_count = len(trans_attn_processors) - clear_count
        print(f"CLEAR({clear_count}) Native({native_count})", end=" ", flush=True)
        
        # 🌟 加载 CLEAR 预训练权重
        if os.path.exists(self.clear_ckpt):
            clear_state_dict = load_file(self.clear_ckpt)
            self.transformer.load_state_dict(clear_state_dict, strict=False)
        else:
            raise FileNotFoundError(f"CLEAR checkpoint not found: {self.clear_ckpt}")
        
        # 🌟 加载你训练的 checkpoint
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # 读取 pixel_weight
            if 'pixel_weight' in ckpt:
                self.pixel_weight = ckpt['pixel_weight']
            
            # 加载 Pixel Extractor
            if 'pixel_extractor' in ckpt:
                state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
                self.pixel_extractor.load_state_dict(state)
            
            # 加载 ControlNet
            if 'controlnet' in ckpt:
                state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
                self.controlnet.load_state_dict(state)
            
            print(f"Checkpoint(epoch={ckpt.get('epoch', '?')}, psnr={ckpt.get('psnr', 0):.2f})")
        else:
            print("Done")
        
        self.controlnet.eval()
        self.pixel_extractor.eval()
        self.transformer.eval()
        
        # 设置初始 mask 尺寸（对应 512x512）
        self._current_mask_size = (32, 32)
        
        print(f"[Eval] Ready. pixel_weight={self.pixel_weight}, GPU={torch.cuda.memory_allocated(self.device)/1e9:.1f}GB")
    
    def update_mask_for_size(self, height, width):
        """根据图像尺寸更新 CLEAR mask"""
        import attention_processor
        from attention_processor import init_local_downsample_mask_flex, init_local_mask_flex
        
        patch_h = height // 16
        patch_w = width // 16
        
        # 如果尺寸没变，不需要更新
        if self._current_mask_size == (patch_h, patch_w):
            return

        # 🌟 致命修复：彻底清空 PyTorch Dynamo 的编译内核缓存！
        import torch._dynamo
        torch._dynamo.reset()
        
        # 释放上一张图的旧 Mask 物理内存
        attention_processor.BLOCK_MASK = None
        torch.cuda.empty_cache()
        gc.collect()
        
        # 🌟 清除 lru_cache 强制重新生成
        if hasattr(init_local_downsample_mask_flex, 'cache_clear'):
            init_local_downsample_mask_flex.cache_clear()
        if hasattr(init_local_mask_flex, 'cache_clear'):
            init_local_mask_flex.cache_clear()
        
        # 🌟 重新初始化 mask（使用 "cuda"）
        if self.down_factor > 1:
            init_local_downsample_mask_flex(
                height=patch_h, width=patch_w, text_length=512,
                window_size=self.window_size, down_factor=self.down_factor, 
                device="cuda"
            )
        else:
            init_local_mask_flex(
                height=patch_h, width=patch_w, text_length=512,
                window_size=self.window_size, device="cuda"
            )
        
        # 更新全局变量
        attention_processor.HEIGHT = patch_h
        attention_processor.WIDTH = patch_w
        
        self._current_mask_size = (patch_h, patch_w)
    
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
    
    # 🌟 与训练代码完全一致的 encode/decode
    @torch.no_grad()
    def encode(self, img):
        # 🌟 VAE CPU-GPU 调度：用时搬到 GPU，用完踢回 CPU
        self.vae.to(self.device)
        img_tensor = img.to(self.vae.dtype).to(self.device)
        lat = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        return lat
    
    @torch.no_grad()
    def decode(self, lat):
        # 🌟 VAE CPU-GPU 调度：用时搬到 GPU，用完踢回 CPU
        self.vae.to(self.device)
        lat = lat.to(self.device)
        sample = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        return sample
    
    # 🌟 与训练代码完全一致的 forward
    @torch.no_grad()
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
    
    # 🌟 与训练代码完全一致的 inference
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
            # 🌟 关键：t 在 [0, 1] 范围，不乘 1000！
            t = torch.full((B,), t_val, device=lr_lat.device, dtype=dtype)
            v = self.forward(lat, lr_lat, lr_pixel, t, guidance)
            lat = lat - dt * v
        
        return lat


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--pixel_weight', type=float, default=None)
    
    # 🌟 添加最大尺寸限制
    parser.add_argument('--max_size', type=int, default=1536,
                        help='Maximum image dimension (default: 1536). Larger images will be resized.')
    
    # CLEAR 参数
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--down_factor', type=int, default=4)
    parser.add_argument('--clear_ckpt', type=str, default='ckpt/clear_local_16_down_4.safetensors')
    
    parser.add_argument('--output_base', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    
    parser.add_argument('--device', type=str, default='cuda')
    
    # 🌟 新增：是否强制使用 512x512 (匹配训练)
    parser.add_argument('--force_512', action='store_true', default=False,
                        help='Force 512x512 evaluation to match training resolution')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect dataset
    if args.dataset is None:
        hr_lower = args.hr_dir.lower()
        if 'urban' in hr_lower:
            args.dataset = 'Urban100'
        elif 'div2k' in hr_lower:
            args.dataset = 'DIV2K'
        elif 'set5' in hr_lower:
            args.dataset = 'Set5'
        elif 'set14' in hr_lower:
            args.dataset = 'Set14'
        elif 'bsd100' in hr_lower:
            args.dataset = 'BSD100'
        elif 'manga' in hr_lower:
            args.dataset = 'Manga109'
        else:
            args.dataset = 'Unknown'
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"{ts}_clear_w{args.window_size}_d{args.down_factor}_{args.num_steps}step"
    
    output_dir = os.path.join(args.output_base, args.dataset, 'CLEARControl', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print("=" * 70)
    print("Dual-Stream FLUX SR Evaluation (CLEAR)")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"CLEAR: window={args.window_size}, down_factor={args.down_factor}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance}")
    print(f"Max size: {args.max_size} (images larger than this will be resized)")
    if args.force_512:
        print("⚠️  Force 512x512 mode enabled")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    evaluator = DualStreamEvaluator(
        args.model_name, device, args.checkpoint,
        pixel_weight=args.pixel_weight or 0.1,
        window_size=args.window_size,
        down_factor=args.down_factor,
        clear_ckpt=args.clear_ckpt
    )
    evaluator.load()
    
    if args.pixel_weight is not None:
        evaluator.pixel_weight = args.pixel_weight
        print(f"[Eval] Override pixel_weight: {args.pixel_weight}")
    
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()
    
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    assert len(hr_files) == len(lr_files)
    
    print(f"\nEvaluating {len(hr_files)} images...\n")
    
    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_bic_list, ssim_bic_list, lpips_bic_list = [], [], []
    filenames = []
    
    for hf, lf in tqdm(zip(hr_files, lr_files), total=len(hr_files), desc="Evaluating"):
        base_name = os.path.splitext(hf)[0]
        filenames.append(base_name)
        
        hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lf)).convert('RGB')
        
        if args.force_512:
            W, H = 512, 512
            hr_img = hr_img.resize((W, H), Image.BICUBIC)
            lr_img = lr_img.resize((W // 4, H // 4), Image.BICUBIC)
        else:
            orig_W, orig_H = hr_img.size
            
            # 🌟 限制最大尺寸，避免 OOM
            max_dim = max(orig_W, orig_H)
            if max_dim > args.max_size:
                scale = args.max_size / max_dim
                orig_W = int(orig_W * scale)
                orig_H = int(orig_H * scale)
                hr_img = hr_img.resize((orig_W, orig_H), Image.BICUBIC)
                lr_img = lr_img.resize((orig_W // 4, orig_H // 4), Image.BICUBIC)
            
            # 对齐到 64 的倍数
            W = orig_W - (orig_W % 64)
            H = orig_H - (orig_H % 64)
            hr_img = hr_img.crop((0, 0, W, H))
            lr_img = lr_img.crop((0, 0, W // 4, H // 4))
        
        hr_np = np.array(hr_img)
        
        lr_bicubic = lr_img.resize((W, H), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        lr_t = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device).to(torch.bfloat16)
        
        # 更新 CLEAR mask
        evaluator.update_mask_for_size(H, W)
        
        # 推理
        lr_lat = evaluator.encode(lr_t)
        sr_lat = evaluator.inference(lr_lat, lr_t, num_steps=args.num_steps, guidance=args.guidance)
        sr_t = evaluator.decode(sr_lat)
        
        # 🌟 立即释放不需要的 tensor
        del lr_lat, sr_lat, lr_t
        torch.cuda.empty_cache()
        
        sr_np = ((sr_t[0].float().cpu().clamp(-1, 1) + 1) * 127.5).numpy().transpose(1, 2, 0).astype(np.uint8)
        
        # 🌟 立即释放 sr_t
        del sr_t
        torch.cuda.empty_cache()
        
        # 计算指标
        psnr_val = calculate_psnr(sr_np, hr_np)
        ssim_val = calculate_ssim(sr_np, hr_np)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
        psnr_bic_list.append(psnr_bic)
        ssim_bic_list.append(ssim_bic)
        
        if lpips_fn:
            hr_t = torch.from_numpy(hr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            sr_t_lpips = torch.from_numpy(sr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            lr_t_lpips = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            lpips_val = lpips_fn(sr_t_lpips.to(device), hr_t.to(device)).item()
            lpips_bic = lpips_fn(lr_t_lpips.to(device), hr_t.to(device)).item()
            lpips_list.append(lpips_val)
            lpips_bic_list.append(lpips_bic)
        
        if args.save_images:
            Image.fromarray(sr_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))
        
        if args.save_comparisons:
            comp = Image.new('RGB', (W * 4, H))
            comp.paste(lr_img.resize((W, H), Image.NEAREST), (0, 0))
            comp.paste(lr_bicubic, (W, 0))
            comp.paste(Image.fromarray(sr_np), (W * 2, 0))
            comp.paste(hr_img, (W * 3, 0))
            comp.save(os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
        
        # 🌟 更积极地清理内存
        del hr_np, sr_np, lr_bicubic_np
        if lpips_fn:
            del hr_t, sr_t_lpips, lr_t_lpips
        torch.cuda.empty_cache()
        gc.collect()
    
    # 汇总
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    avg_lpips = np.mean(lpips_list) if lpips_list else 0
    avg_lpips_bic = np.mean(lpips_bic_list) if lpips_bic_list else 0
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"CLEAR: window={args.window_size}, down_factor={args.down_factor}")
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    if LPIPS_AVAILABLE:
        print(f"Bicubic:      PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}")
        print(f"CLEARControl: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        print(f"Δ:            {avg_psnr - avg_psnr_bic:+.4f}, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}")
    else:
        print(f"Bicubic:      PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}")
        print(f"CLEARControl: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
    print("=" * 70)
    
    # 保存结果
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write("Dual-Stream FLUX SR (CLEARControl)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"CLEAR: window={args.window_size}, down_factor={args.down_factor}\n")
        f.write(f"Pixel Weight: {evaluator.pixel_weight}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Steps: {args.num_steps}, Guidance: {args.guidance}\n")
        f.write(f"Force 512: {args.force_512}\n")
        f.write("\nSummary:\n")
        if LPIPS_AVAILABLE:
            f.write(f"Bicubic:      PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}\n")
            f.write(f"CLEARControl: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}\n")
        else:
            f.write(f"Bicubic:      PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}\n")
            f.write(f"CLEARControl: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}\n")
        f.write("\nPer-image:\n")
        for i, fname in enumerate(filenames):
            f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (Δ{psnr_list[i]-psnr_bic_list[i]:+.2f})\n")
    
    print(f"\n✅ Results saved: {output_dir}")


if __name__ == '__main__':
    main()