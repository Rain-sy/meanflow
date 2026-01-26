"""
MeanFlow for Super-Resolution

Extends the original MeanFlow to handle SR tasks.

Key Changes:
1. Input: LR-HR image pairs instead of noise-clean
2. Flow: z_t = (1-t)*HR + t*LR
3. No class conditioning (or use degradation type as class)

Usage:
    from meanflow_sr import MeanFlowSR, generate_sr
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

# Import DiT models from original repo
from models import models_dit


def generate_sr(variable, model, x_lr, config):
    """
    Generate HR from LR using MeanFlow SR
    
    Args:
        variable: Model parameters
        model: MeanFlowSR model
        x_lr: LR image latents [B, H, W, C]
        config: Configuration
    
    Returns:
        HR image latents [B, H, W, C]
    """
    num_steps = config.sampling.num_steps
    
    # Get time schedule
    t_steps = model.apply({}, method=model.sampling_schedule())
    
    # Dummy labels (not used in SR)
    bz = x_lr.shape[0]
    labels = jnp.zeros((bz,), dtype=jnp.int32)
    
    def step_fn(i, inputs):
        x_i, _ = inputs
        x_i = model.apply(
            variable, x_i, labels, i, t_steps,
            method=model.sample_one_step,
        )
        return (x_i, None)
    
    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_lr, None))
    return outputs[0]


class MeanFlowSR(nn.Module):
    """
    MeanFlow for Super-Resolution
    
    Core difference from original MeanFlow:
    - Input flow: z_t = (1-t)*HR + t*LR (not noise)
    - Output: HR estimation from LR
    - No class conditioning needed (optional degradation type)
    """
    
    # Model settings
    model_str:              str
    model_config:           dict
    dtype =                 jnp.float32
    
    # SR settings
    num_classes:            int = 1    # Not used, but kept for compatibility
    
    # Noise distribution for t sampling
    noise_dist:             str   = 'logit_normal'
    P_mean:                 float = -0.4
    P_std:                  float = 1.0
    
    # Loss settings
    data_proportion:        float = 0.75
    
    # No guidance for SR (set to identity)
    guidance_eq:            str   = None
    omega:                  float = 1.0
    kappa:                  float = 0.0
    class_dropout_prob:     float = 0.0
    
    t_start:                float = 0.0
    t_end:                  float = 1.0
    
    # Training dynamics
    norm_p:                 float = 1.0
    norm_eps:               float = 0.01
    
    # Inference
    seed:                   int = 0
    num_steps:              int = 1
    schedule:               str = 'default'
    sampling_timesteps:     jnp.ndarray = None
    
    def setup(self):
        model_str = self.model_str
        net_fn = getattr(models_dit, model_str)
        # No class dropout for SR
        self.net = net_fn(name="net", class_dropout_prob=0.0, num_classes=self.num_classes)
    
    #######################################################
    # Solver (Inference)
    #######################################################
    
    def sample_one_step(self, z_t, labels, i, t_steps):
        """One step of SR sampling"""
        t = t_steps[i]
        r = t_steps[i + 1]
        
        t = jnp.repeat(t, z_t.shape[0])
        r = jnp.repeat(r, z_t.shape[0])
        
        return self.solver_step(z_t, t, r, labels)
    
    def solver_step(self, z_t, t, r, labels):
        """
        SR Solver Step
        
        For SR: z_r = z_t - (t-r) * u(z_t, t, h)
        At t=1, r=0: HR = LR - 1.0 * u(LR, 1, 1)
        """
        u = self.u_fn(z_t, t=t, h=(t - r), y=labels, train=False)
        return z_t - jnp.einsum('n,n...->n...', t - r, u)
    
    def sampling_schedule(self):
        if self.schedule == 'default':
            return self._default_schedule
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def _default_schedule(self):
        """Default: one-step from t=1 (LR) to t=0 (HR)"""
        if self.sampling_timesteps is None:
            return jnp.array([1.0, 0.0])
        return self.sampling_timesteps
    
    #######################################################
    # Time Sampling
    #######################################################
    
    def noise_distribution(self):
        if self.noise_dist == 'logit_normal':
            return self._logit_normal_dist
        elif self.noise_dist == 'uniform':
            return self._uniform_dist
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")
    
    def _logit_normal_dist(self, bz):
        rnd_normal = jax.random.normal(self.make_rng('gen'), [bz, 1, 1, 1], dtype=self.dtype)
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)
    
    def _uniform_dist(self, bz):
        return jax.random.uniform(self.make_rng('gen'), [bz, 1, 1, 1], dtype=self.dtype)
    
    def sample_tr(self, bz):
        """Sample time points t >= r"""
        t = self.noise_distribution()(bz)
        r = self.noise_distribution()(bz)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)
        
        # 75% of samples: r = t (learns instantaneous velocity)
        data_size = int(bz * self.data_proportion)
        zero_mask = jnp.arange(bz) < data_size
        zero_mask = zero_mask.reshape(bz, 1, 1, 1)
        r = jnp.where(zero_mask, t, r)
        
        return t, r
    
    #######################################################
    # Network Wrapper
    #######################################################
    
    def u_fn(self, x, t, h, y, train=True):
        """Network forward: predict average velocity u"""
        bz = x.shape[0]
        return self.net(x, t.reshape(bz), h.reshape(bz), y, train=train, key=self.make_rng('gen'))
    
    def v_fn(self, x, t, y, train=False):
        """Instantaneous velocity (h=0)"""
        h = jnp.zeros_like(t)
        return self.u_fn(x, t, h, y=y, train=train)
    
    #######################################################
    # Forward Pass and Loss (SR Version)
    #######################################################
    
    def forward(self, imgs_hr, imgs_lr, labels=None, train=True):
        """
        SR Training Forward Pass
        
        Args:
            imgs_hr: High-resolution images [B, H, W, C]
            imgs_lr: Low-resolution images (upsampled to HR size) [B, H, W, C]
            labels: Optional labels (not used for basic SR)
        
        Key Formula:
            z_t = (1 - t) * HR + t * LR
            v = LR - HR  (degradation direction)
            u_target = v - (t - r) * du/dt
        """
        x_hr = imgs_hr.astype(self.dtype)
        x_lr = imgs_lr.astype(self.dtype)
        bz = imgs_hr.shape[0]
        
        # Dummy labels if not provided
        if labels is None:
            labels = jnp.zeros((bz,), dtype=jnp.int32)
        
        # -----------------------------------------------------------------
        # Sample time points
        t, r = self.sample_tr(bz)
        
        # -----------------------------------------------------------------
        # Construct z_t via interpolation
        # z_t = (1 - t) * HR + t * LR
        # At t=0: z_0 = HR (clean/target)
        # At t=1: z_1 = LR (degraded/input)
        z_t = (1 - t) * x_hr + t * x_lr
        
        # Instantaneous velocity (degradation direction)
        # v = dz/dt = LR - HR
        v = x_lr - x_hr
        
        # -----------------------------------------------------------------
        # Compute u and du/dt using JVP
        def u_fn_inner(z_t, t, r):
            return self.u_fn(z_t, t, t - r, y=labels, train=train)
        
        dt_dt = jnp.ones_like(t)
        dr_dt = jnp.zeros_like(t)
        
        u, du_dt = jax.jvp(u_fn_inner, (z_t, t, r), (v, dt_dt, dr_dt))
        
        # -----------------------------------------------------------------
        # Target for u
        u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt
        u_tgt = jax.lax.stop_gradient(u_tgt)
        
        # -----------------------------------------------------------------
        # Loss computation
        loss = (u - u_tgt) ** 2
        loss = jnp.sum(loss, axis=(1, 2, 3))  # Sum over spatial dims
        
        # Adaptive weighting
        adp_wt = (loss + self.norm_eps) ** self.norm_p
        loss = loss / jax.lax.stop_gradient(adp_wt)
        
        loss = loss.mean()  # Mean over batch
        
        # -----------------------------------------------------------------
        # Monitoring losses
        v_loss = (u - v) ** 2
        v_loss = jnp.sum(v_loss, axis=(1, 2, 3)).mean()
        
        # Reconstruction loss at t=1
        pred_hr = x_lr - u
        recon_loss = jnp.sum((pred_hr - x_hr) ** 2, axis=(1, 2, 3)).mean()
        
        dict_losses = {
            'loss': loss,
            'v_loss': v_loss,
            'recon_loss': recon_loss,
        }
        return loss, dict_losses
    
    def __call__(self, x, t, y, train=False, key=None):
        """For initialization only"""
        return self.net(x, t, t, y, key=key, train=train)