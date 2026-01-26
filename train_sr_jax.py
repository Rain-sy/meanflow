"""
Training script for MeanFlow Super-Resolution

Based on the original MeanFlow train.py, adapted for SR tasks.

Usage:
    python train_sr.py --config configs/sr_config.py --workdir ./workdir_sr
"""

from copy import deepcopy
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from clu import metric_writers
from flax import jax_utils
from flax.training import common_utils, train_state
from jax import lax, random

from meanflow_sr import MeanFlowSR, generate_sr
from utils.ckpt_util import restore_checkpoint, save_checkpoint
from utils.ema_util import ema_schedules, update_ema
from utils.info_util import print_params
from utils.logging_util import Timer, log_for_0
from utils.vis_util import make_grid_visualization
import utils.input_pipeline_sr as input_pipeline_sr


#######################################################
# Initialize
#######################################################

def initialized(key, image_size, model):
    """Initialize model parameters"""
    input_shape = (1, image_size, image_size, 4)  # Latent space
    x = jnp.ones(input_shape)
    t = jnp.ones((1,), dtype=jnp.float32)
    y = jnp.ones((1,), dtype=jnp.int32)
    
    @jax.jit
    def init(*args):
        return model.init(*args)
    
    log_for_0('Initializing params...')
    variables = init({'params': key}, x, t, y)
    log_for_0('Initializing params done.')
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
    log_for_0(f"Total trainable parameters: {param_count}")
    return variables, variables['params']


class TrainState(train_state.TrainState):
    ema_params: Any


def create_train_state(rng, config, model, image_size, lr_value):
    """Create initial training state"""
    rng, rng_init = random.split(rng)
    
    _, params = initialized(rng_init, image_size, model)
    ema_params = deepcopy(params)
    ema_params = update_ema(ema_params, params, 0)
    
    if 'net' in params:
        print_params(params['net'])
    
    tx = optax.adamw(
        learning_rate=lr_value,
        weight_decay=config.training.get('weight_decay', 0),
        b2=config.training.adam_b2,
    )
    
    state = TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        tx=tx,
    )
    return state


#######################################################
# Train Step
#######################################################

def compute_metrics(dict_losses):
    metrics = dict_losses.copy()
    metrics = lax.all_gather(metrics, axis_name='batch')
    metrics = jax.tree_util.tree_map(lambda x: x.flatten(), metrics)
    return metrics


def train_step_sr(state, batch, rng_init, config, lr, ema_fn):
    """
    Single training step for SR
    
    batch contains:
        - 'image_hr': high-resolution images [devices, batch/devices, H, W, C]
        - 'image_lr': low-resolution images (upsampled) [devices, batch/devices, H, W, C]
    """
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
    
    imgs_hr = batch['image_hr']
    imgs_lr = batch['image_lr']
    
    def loss_fn(params):
        variables = {"params": params}
        outputs = state.apply_fn(
            variables,
            imgs_hr=imgs_hr,
            imgs_lr=imgs_lr,
            labels=None,
            rngs=dict(gen=rng_base),
        )
        loss, dict_losses = outputs
        return loss, (dict_losses,)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')
    
    dict_losses, = aux[1]
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr
    
    new_state = state.apply_gradients(grads=grads)
    
    ema_value = ema_fn(state.step)
    new_ema = update_ema(new_state.ema_params, new_state.params, ema_value)
    new_state = new_state.replace(ema_params=new_ema)
    
    return new_state, metrics


#######################################################
# Sampling (SR Inference)
#######################################################

def sample_step_sr(variable, lr_images, model, rng_init):
    """
    SR sampling step
    
    Args:
        variable: model parameters
        lr_images: LR images [B, H, W, C]
        model: MeanFlowSR model
        rng_init: random key
    
    Returns:
        Predicted HR images [B, H, W, C]
    """
    bz = lr_images.shape[0]
    
    # Get time schedule [1.0, 0.0] for one-step
    t_steps = model.apply({}, method=model.sampling_schedule())
    
    # Dummy labels
    labels = jnp.zeros((bz,), dtype=jnp.int32)
    
    def step_fn(i, inputs):
        x_i, rng = inputs
        rng_step = jax.random.fold_in(rng, i)
        
        x_i = model.apply(
            variable, x_i, labels, i, t_steps,
            method=model.sample_one_step,
            rngs=dict(gen=rng_step),
        )
        return (x_i, rng)
    
    num_steps = len(t_steps) - 1
    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (lr_images, rng_init))
    
    return outputs[0]


def run_sr_inference(p_sample_step, state, batch, ema=True):
    """Run SR inference on a batch"""
    variable = {"params": state.ema_params if ema else state.params}
    
    lr_images = batch['image_lr']  # Already on device
    hr_pred = p_sample_step(variable, lr_images)
    
    # Reshape from pmap format
    hr_pred = hr_pred.reshape(-1, *hr_pred.shape[2:])
    
    # To uint8
    hr_pred = (hr_pred + 1.0) * 127.5
    hr_pred = jnp.clip(hr_pred, 0, 255).astype(jnp.uint8)
    
    jax.random.normal(random.key(0), ()).block_until_ready()
    return hr_pred


#######################################################
# Metrics
#######################################################

def compute_psnr(pred, target, max_val=255.0):
    """Compute PSNR between prediction and target"""
    pred = pred.astype(jnp.float32)
    target = target.astype(jnp.float32)
    mse = jnp.mean((pred - target) ** 2)
    psnr = 20 * jnp.log10(max_val / jnp.sqrt(mse + 1e-8))
    return psnr


#######################################################
# Main Training Loop
#######################################################

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Main training function for SR"""
    
    # Initialize writer
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )
    
    rng = random.key(config.training.seed)
    image_size = config.dataset.image_size
    
    log_for_0(f'Config batch_size: {config.training.batch_size}')
    
    # Batch size setup
    if config.training.batch_size % jax.process_count() > 0:
        raise ValueError('Batch size must be divisible by process count')
    local_batch_size = config.training.batch_size // jax.process_count()
    log_for_0(f'Local batch_size: {local_batch_size}')
    log_for_0(f'JAX local devices: {jax.local_device_count()}')
    
    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError('Local batch size must be divisible by local device count')
    
    # Create data loaders
    train_loader, steps_per_epoch = input_pipeline_sr.create_sr_split(
        config.dataset,
        local_batch_size,
        split='train',
    )
    log_for_0(f'Steps per epoch: {steps_per_epoch}')
    
    # Create model
    model_config = config.model.to_dict()
    model_str = model_config.pop('cls')
    
    model = MeanFlowSR(
        model_str=model_str,
        model_config=model_config,
        **config.method,
    )
    
    # Create train state
    base_lr = config.training.learning_rate
    state = create_train_state(rng, config, model, image_size, lr_value=base_lr)
    
    if config.get('load_from') is not None:
        state = restore_checkpoint(state, config.load_from)
    
    step_offset = int(state.step)
    epoch_offset = step_offset // steps_per_epoch
    
    state = jax_utils.replicate(state)
    ema_fn = ema_schedules(config)
    
    # Compile training step
    p_train_step = jax.pmap(
        partial(
            train_step_sr,
            rng_init=rng,
            config=config,
            lr=base_lr,
            ema_fn=ema_fn,
        ),
        axis_name='batch',
    )
    
    # Compile sampling step
    p_sample_step = jax.pmap(
        partial(
            sample_step_sr,
            model=model,
            rng_init=random.PRNGKey(config.sampling.seed),
        ),
        axis_name='batch',
    )
    
    train_metrics = []
    log_for_0('Initial compilation...')
    
    # Training loop
    for epoch in range(epoch_offset, config.training.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        
        log_for_0(f'Epoch {epoch}...')
        timer = Timer()
        timer.reset()
        
        for n_batch, batch in enumerate(train_loader):
            step = epoch * steps_per_epoch + n_batch
            
            batch = input_pipeline_sr.prepare_sr_batch_data(batch)
            state, metrics = p_train_step(state, batch)
            
            if epoch == epoch_offset and n_batch == 0:
                log_for_0('Initial compilation completed.')
                timer.reset()
            
            train_metrics.append(metrics)
            
            # Logging
            if (step + 1) % config.training.log_per_step == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = jax.tree_util.tree_map(lambda x: float(x.mean()), train_metrics)
                summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset()
                summary['epoch'] = epoch
                
                writer.write_scalars(step + 1, summary)
                log_for_0(
                    f'Epoch {epoch}, Step {step}, Loss: {summary["loss"]:.6f}, '
                    f'V_loss: {summary.get("v_loss", 0):.6f}'
                )
                
                train_metrics = []
        
        # Visualization
        if (epoch + 1) % config.training.get('sample_per_epoch', 10) == 0:
            log_for_0(f'Generating samples at epoch {epoch}...')
            
            # Get a batch
            vis_batch = next(iter(train_loader))
            vis_batch = input_pipeline_sr.prepare_sr_batch_data(vis_batch)
            
            # Run inference
            hr_pred = run_sr_inference(p_sample_step, state, vis_batch, ema=True)
            
            # Get GT for comparison
            hr_gt = vis_batch['image_hr']
            hr_gt = hr_gt.reshape(-1, *hr_gt.shape[2:])
            hr_gt = ((hr_gt + 1.0) * 127.5).clip(0, 255).astype(jnp.uint8)
            
            lr_vis = vis_batch['image_lr']
            lr_vis = lr_vis.reshape(-1, *lr_vis.shape[2:])
            lr_vis = ((lr_vis + 1.0) * 127.5).clip(0, 255).astype(jnp.uint8)
            
            # Stack: LR | Pred | GT (first 4 images)
            n_vis = min(4, hr_pred.shape[0])
            vis_images = jnp.concatenate([
                lr_vis[:n_vis],
                hr_pred[:n_vis],
                hr_gt[:n_vis],
            ], axis=2)  # Concatenate along width
            
            vis_grid = make_grid_visualization(vis_images, grid=2)
            writer.write_images(epoch + 1, {'sr_samples': vis_grid})
            
            # Compute PSNR
            psnr = compute_psnr(hr_pred[:n_vis], hr_gt[:n_vis])
            log_for_0(f'Sample PSNR: {psnr:.2f} dB')
            writer.write_scalars(epoch + 1, {'sample_psnr': psnr})
            writer.flush()
        
        # Save checkpoint
        if (epoch + 1) % config.training.checkpoint_per_epoch == 0:
            save_checkpoint(state, workdir)
    
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state