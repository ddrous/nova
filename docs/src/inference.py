# src/inference.py
"""
Contains the exact JAX-accelerated inference signatures from main.py
for python-side evaluation. 
(Note: Browser ONNX utilizes exported sub-blocks rather than these monolithic scans)
"""

import jax
import jax.numpy as jnp
import equinox as eqx

@eqx.filter_jit
def inference_rollout(model, ref_video, coords_grid, context_ratio=0.0):
    T = ref_video.shape[0]
    init_frame = ref_video[0]
    
    z_init = model.encoder(jnp.transpose(init_frame, (2, 0, 1)))
    m_init = model.action_model.reset_memory(T)

    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        z_t, m_t = carry
        o_tp1, step_idx = scan_inputs

        time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
        coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
        pred_out = model.render_frame(z_t, coords_grid_t)

        is_context = (step_idx / T) < context_ratio

        a_t = jax.lax.cond(
            is_context,
            lambda: model.action_model.inverse_dynamics(
                z_t, 
                model.encoder(jnp.transpose(o_tp1, (2, 0, 1)))
            ),
            lambda: model.action_model.decode_memory(m_t, step_idx, z_t)
        )

        m_tp1 = model.action_model.encode_memory(m_t, step_idx, z_t, a_t)
        z_tp1 = model.forward_dyn(z_t, a_t)

        return (z_tp1, m_tp1), (a_t, z_t, pred_out)

    scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
    _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init), scan_inputs)
    
    return actions, pred_latents, pred_video

@eqx.filter_jit
def inference_rollout_morph(model, ref_video, coords_grid, context_ratio=0.0):
    T = ref_video.shape[0]
    init_frame = ref_video[0]
    
    z_init = model.encoder(jnp.transpose(init_frame, (2, 0, 1)))
    m_init = model.action_model.reset_memory(T)
    
    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        z_t, m_t, a_tm1 = carry
        o_tp1, step_idx = scan_inputs

        time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
        coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
        pred_out = model.render_frame(z_t, coords_grid_t)

        is_context = (step_idx / T) < context_ratio

        a_t = jax.lax.cond(
            is_context,
            lambda: model.action_model.inverse_dynamics(
                z_t, 
                model.encoder(jnp.transpose(o_tp1, (2, 0, 1)))
            ),
            lambda: model.action_model.decode_memory(m_t, step_idx, z_t)
        )

        a_zeros = jnp.zeros((model.lam_dim,), dtype=z_init.dtype)
        a_ones = 1*jnp.ones((model.lam_dim,), dtype=z_init.dtype)
        
        ratio = jnp.clip((step_idx+1) / T, 0, 1)
        a_t = ratio * a_zeros + (1-ratio) * a_t

        m_tp1 = model.action_model.encode_memory(m_t, step_idx, z_t, a_t)
        z_tp1 = model.forward_dyn(z_t, a_t)

        return (z_tp1, m_tp1, a_t), (a_t, z_t, pred_out)

    scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
    a_init = jnp.zeros((model.lam_dim,), dtype=z_init.dtype)
    _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init, a_init), scan_inputs) 
    
    return pred_video