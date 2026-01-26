"""
Configuration for MeanFlow Super-Resolution on Urban100

Usage:
    python main_sr.py --config configs/sr_urban100.py --workdir ./workdir_sr
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    
    # ============================================================
    # Dataset Configuration
    # ============================================================
    config.dataset = dataset = ml_collections.ConfigDict()
    
    # Urban100 paths - MODIFY THESE TO YOUR ACTUAL PATHS
    dataset.hr_dir = 'Flow_Restore/Data/Urban 100/X2 Urban100/X2/HIGH X2 Urban'
    dataset.lr_dir = 'Flow_Restore/Data/Urban 100/X2 Urban100/X2/LOW X2 Urban'
    
    dataset.image_size = 256      # HR image size (latent will be 32x32)
    dataset.image_channels = 4    # Latent channels
    dataset.scale = 2             # SR scale factor
    dataset.num_classes = 1       # Not used for SR
    
    dataset.num_workers = 4
    dataset.prefetch_factor = 2
    dataset.pin_memory = True
    
    # ============================================================
    # Training Configuration  
    # ============================================================
    config.training = training = ml_collections.ConfigDict()
    
    training.seed = 42
    training.batch_size = 8       # Small batch for Urban100 (100 images)
    training.learning_rate = 1e-4
    training.num_epochs = 500     # More epochs for small dataset
    
    training.adam_b2 = 0.95
    training.weight_decay = 0.0
    
    # Logging
    training.log_per_step = 10
    training.sample_per_epoch = 20
    training.checkpoint_per_epoch = 50
    
    # EMA
    training.ema_val = 0.9999
    
    # ============================================================
    # MeanFlow Method Configuration
    # ============================================================
    config.method = method = ml_collections.ConfigDict()
    
    # Time sampling distribution
    method.noise_dist = 'logit_normal'
    method.P_mean = -0.4          # Logit-normal mean
    method.P_std = 1.0            # Logit-normal std
    
    # Training proportion
    method.data_proportion = 0.75  # 75% learn v, 25% learn u
    
    # No guidance for SR
    method.guidance_eq = None
    method.omega = 1.0
    method.kappa = 0.0
    method.class_dropout_prob = 0.0
    
    method.t_start = 0.0
    method.t_end = 1.0
    
    # Loss normalization
    method.norm_p = 1.0
    method.norm_eps = 0.01
    
    # ============================================================
    # Model Configuration
    # ============================================================
    config.model = model = ml_collections.ConfigDict()
    
    # DiT variants: DiT_S_4, DiT_B_4, DiT_L_4, DiT_XL_4
    # S = Small (384 hidden, 12 layers)
    # B = Base (768 hidden, 12 layers)  
    # L = Large (1024 hidden, 24 layers)
    # _4 means patch_size=4 (for 32x32 latent -> 8x8 tokens)
    model.cls = 'DiT_B_4'  # Good balance of speed and quality
    
    # ============================================================
    # Sampling Configuration (Inference)
    # ============================================================
    config.sampling = sampling = ml_collections.ConfigDict()
    
    sampling.seed = 0
    sampling.num_steps = 1        # One-step inference!
    sampling.schedule = 'default'  # [1.0, 0.0]
    sampling.sampling_timesteps = None
    sampling.num_classes = 1
    
    # ============================================================
    # Other
    # ============================================================
    config.load_from = None       # Path to resume from
    config.eval_only = False
    
    return config


def get_config_small():
    """Smaller model for faster training"""
    config = get_config()
    config.model.cls = 'DiT_S_4'  # 384 hidden, 12 layers
    config.training.batch_size = 16
    return config


def get_config_large():
    """Larger model for better quality"""
    config = get_config()
    config.model.cls = 'DiT_L_4'  # 1024 hidden, 24 layers
    config.training.batch_size = 4  # Smaller batch due to memory
    return config


def get_config_x4():
    """4x Super-Resolution"""
    config = get_config()
    config.dataset.scale = 4
    # Update LR directory path for x4
    config.dataset.lr_dir = 'Flow_Restore/Data/Urban 100/X4 Urban100/X4/LOW X4 Urban'
    return config