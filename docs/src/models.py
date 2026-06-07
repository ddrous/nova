# src/models.py
import jax
import jax.numpy as jnp
import numpy as np       # <--- ADD THIS IMPORT
import equinox as eqx
from typing import Optional
from jax.flatten_util import ravel_pytree

from .config import CONFIG

def fourier_encode(x, num_freqs):
    # Ensure float32 to avoid JAX x64 warnings
    freqs = (2.0 ** np.arange(num_freqs)).astype(np.float32) 
    
    # Broadcast across arbitrary leading dimensions natively (no vmap needed)
    angles = x[..., None] * freqs * jnp.pi
    angles = angles.reshape(*x.shape[:-1], -1)
    
    return jnp.concatenate([x, jnp.sin(angles), jnp.cos(angles)], axis=-1)

class RootMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, width, depth, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = [eqx.nn.Linear(in_size, width, key=keys[0])]
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

class CNNEncoder(eqx.Module):
    layers: list

    def __init__(self, in_channels, out_dim, spatial_shape, key, hidden_width=8, depth=4):
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 1)
        
        conv_layers = []
        current_in = in_channels
        current_out = hidden_width
        
        for i in range(depth):
            conv_layers.append(
                eqx.nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1, key=keys[i])
            )
            current_in = current_out
            current_out *= 2
            
        dummy_x = jnp.zeros((in_channels, H, W))
        for layer in conv_layers:
            dummy_x = layer(dummy_x)

        flat_dim = dummy_x.reshape(-1).shape[0]
        self.layers = conv_layers + [eqx.nn.Linear(flat_dim, out_dim, key=keys[depth])]
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = x.reshape(-1)
        x = self.layers[-1](x)
        return x

class ForwardDynamics(eqx.Module):
    mlp_A: Optional[eqx.nn.MLP]
    mlp_B: Optional[eqx.nn.MLP]
    giant_mlp: Optional[eqx.nn.MLP]
    split_forward: bool = eqx.field(static=True)

    def __init__(self, dyn_dim, lam_dim, split_forward, key):
        self.split_forward = split_forward
        k1, k2, k3 = jax.random.split(key, 3)
        if split_forward:
            self.mlp_A = eqx.nn.MLP(dyn_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k1)
            self.mlp_B = eqx.nn.MLP(lam_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k2)
            self.giant_mlp = None
        else:
            self.mlp_A = None
            self.mlp_B = None
            self.giant_mlp = eqx.nn.MLP(dyn_dim + lam_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k3)

    def __call__(self, z_prev, a):
        if self.split_forward:
            return self.mlp_A(z_prev) + self.mlp_B(a)
        else:
            return self.giant_mlp(jnp.concatenate([z_prev, a], axis=-1))

class TransformerBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, d_model, num_heads, key):
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=d_model,
            use_query_bias=True, use_key_bias=True,
            use_value_bias=True, use_output_bias=True, key=k1
        )
        self.mlp = eqx.nn.MLP(d_model, d_model, width_size=d_model * 4, depth=1, key=k2)
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask):
        x_norm = jax.vmap(self.ln1)(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_out
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
        return x

class InverseDynamics(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, dyn_dim, lam_dim, key, num_actions=None):
        if num_actions:
            self.mlp = eqx.nn.MLP(dyn_dim * 2, num_actions, width_size=dyn_dim*1, depth=2, key=key)
        else:
            self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim*1, depth=2, key=key)
        
    def __call__(self, z_prev, z_target):
        return self.mlp(jnp.concatenate([z_prev, z_target], axis=-1))

class MemoryModuleAtt(eqx.Module):
    d_model: int
    max_len: int
    pos_emb: jax.Array
    blocks: tuple
    proj_in: eqx.nn.Linear
    lam_dim: int = eqx.field(static=True)
    icl_decoding: bool = eqx.field(static=True)
    action_mlp: Optional[eqx.nn.MLP]
    output_proj: Optional[eqx.nn.Linear]

    def __init__(self, lam_dim, mem_dim, latent_dim, key, max_len=20, num_heads=4, num_blocks=4, num_actions=4):
        self.max_len = max_len
        self.icl_decoding = CONFIG["icl_decoding"]
        self.lam_dim = lam_dim
        self.d_model = mem_dim
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        
        self.proj_in = eqx.nn.Linear(latent_dim + lam_dim, self.d_model, key=k1)
        self.pos_emb = jax.random.normal(k2, (max_len, self.d_model)) * 0.02
        block_keys = jax.random.split(k3, num_blocks)
        self.blocks = tuple(TransformerBlock(self.d_model, num_heads, bk) for bk in block_keys)

        if self.icl_decoding:
            self.action_mlp = None
            self.output_proj = eqx.nn.Linear(self.d_model, num_actions if num_actions else lam_dim, key=k6)
        else:
            self.action_mlp = eqx.nn.MLP(self.d_model + latent_dim, lam_dim, width_size=self.d_model * 2, depth=3, key=k4)
            self.output_proj = None

    def reset(self, T):
        return jnp.zeros((T, self.d_model))

    def encode(self, buffer, step_idx, z, a):
        token = self.proj_in(jnp.concatenate([z, a], axis=-1))
        return buffer.at[step_idx - 1].set(token)

    def decode(self, buffer, step_idx, z_current):
        T = buffer.shape[0]
        if self.icl_decoding:
            zero_action = jnp.zeros((self.lam_dim,), dtype=z_current.dtype)
            query_token = self.proj_in(jnp.concatenate([z_current, zero_action], axis=-1))
            temp_buffer = buffer.at[step_idx - 1].set(query_token)
            
            x = temp_buffer + self.pos_emb[:T]
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            
            for block in self.blocks:
                x = block(x, mask)
            context = x[step_idx - 1]
            return self.output_proj(context)
        else:
            def compute_context():
                x = buffer + self.pos_emb[:T]
                mask = jnp.tril(jnp.ones((T, T), dtype=bool))
                for block in self.blocks:
                    x = block(x, mask)
                return x[step_idx - 2]
                
            context = jax.lax.cond(step_idx > 1, compute_context, lambda: jnp.zeros(self.d_model))
            return self.action_mlp(jnp.concatenate([context, z_current], axis=-1))

class VanillaRNNCell(eqx.Module):
    weight_ih: eqx.nn.Linear
    weight_hh: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.weight_ih = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=k1)
        self.weight_hh = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k2)

    def __call__(self, input: jax.Array, hidden: jax.Array) -> jax.Array:
        return jax.nn.tanh(self.weight_ih(input) + self.weight_hh(hidden))

class MemoryModule(eqx.Module):
    d_model: int
    rnn_type: str = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    num_actions: Optional[int] = eqx.field(static=True)
    rnn_cell: eqx.Module
    action_decoder: eqx.nn.MLP

    def __init__(self, lam_dim, mem_dim, latent_dim, key, rnn_type="GRU", num_actions=None, **kwargs):
        self.lam_dim = lam_dim
        self.d_model = mem_dim
        self.rnn_type = rnn_type.upper()
        self.num_actions = num_actions
        k1, k2 = jax.random.split(key, 2)
        
        input_dim = latent_dim + lam_dim
        if self.rnn_type == "LSTM":
            self.rnn_cell = eqx.nn.LSTMCell(input_dim, self.d_model, key=k1)
        elif self.rnn_type == "GRU":
            self.rnn_cell = eqx.nn.GRUCell(input_dim, self.d_model, key=k1)
        elif self.rnn_type == "RNN":
            self.rnn_cell = VanillaRNNCell(input_dim, self.d_model, key=k1)
        
        out_dim = num_actions if num_actions is not None else lam_dim
        self.action_decoder = eqx.nn.MLP(
            in_size=self.d_model + latent_dim, out_size=out_dim, 
            width_size=self.d_model * 1, depth=1, key=k2
        )

    def reset(self, T):
        if self.rnn_type == "LSTM":
            return (jnp.zeros((self.d_model,)), jnp.zeros((self.d_model,)))
        return jnp.zeros((self.d_model,))

    def encode(self, state, step_idx, z, a):
        rnn_input = jnp.concatenate([z, a], axis=-1)
        return self.rnn_cell(rnn_input, state)

    def decode(self, state, step_idx, z_current):
        h = state[0] if self.rnn_type == "LSTM" else state
        decode_input = jnp.concatenate([h, z_current], axis=-1)
        return self.action_decoder(decode_input)

class LAM(eqx.Module):
    idm: InverseDynamics
    gcm: Optional[MemoryModule]
    discrete_actions: bool = eqx.field(static=True)
    action_embedding: Optional[eqx.nn.Embedding]

    def __init__(self, dyn_dim, lam_dim, mem_dim, max_len, num_heads, num_blocks, num_actions, key, phase=1):
        k1, k2 = jax.random.split(key)
        self.discrete_actions = num_actions is not None

        self.idm = InverseDynamics(dyn_dim, lam_dim, key=k1, num_actions=num_actions if self.discrete_actions else None)
        if phase == 2:
            self.gcm = MemoryModule(lam_dim, mem_dim, dyn_dim, key=k2, rnn_type="GRU", num_actions=num_actions if self.discrete_actions else None)
        else:
            self.gcm = None

        if self.discrete_actions:
            self.action_embedding = eqx.nn.Embedding(num_actions, lam_dim, key=k2)
        else:
            self.action_embedding = None

    def discretise_action(self, logits):
        soft_probs = jax.nn.softmax(logits, axis=-1)
        hard_idx = jnp.argmax(logits, axis=-1)
        hard_probs = jax.nn.one_hot(hard_idx, num_classes=logits.shape[-1])
        ste_probs = soft_probs + jax.lax.stop_gradient(hard_probs - soft_probs)
        return jnp.dot(ste_probs, self.action_embedding.weight)

    def inverse_dynamics(self, z_prev, z_target):
        if not self.discrete_actions:
            return self.idm(z_prev, z_target)
        logits = self.idm(z_prev, z_target)
        return self.discretise_action(logits)

    def decode_memory(self, buffer, step_idx, z_current):
        if not self.discrete_actions:
            return self.gcm.decode(buffer, step_idx, z_current)
        logits = self.gcm.decode(buffer, step_idx, z_current)
        return self.discretise_action(logits)

    def encode_memory(self, buffer, step_idx, z_current, a):
        return self.gcm.encode(buffer, step_idx, z_current, a)
    
    def reset_memory(self, T):
        return self.gcm.reset(T)

class WARP(eqx.Module):
    encoder: CNNEncoder
    forward_dyn: ForwardDynamics
    theta_base: jax.Array
    action_model: LAM

    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)
    split_forward: bool = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    mem_dim: int = eqx.field(static=True)
    phase: int = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, lam_dim, mem_dim, split_forward, key, phase=1):
        k_root, k_enc, k_lam, k_fwd, k_mem = jax.random.split(key, 5)
        self.frame_shape = frame_shape
        self.num_freqs = num_freqs
        self.lam_dim = lam_dim
        self.split_forward = split_forward
        self.phase = phase
        H, W, C = frame_shape

        coord_dim = 2 + 2 * 2 * num_freqs 
        root_out_dim = C * 2 if CONFIG["use_nll_loss"] else C
        add_time = 1 if CONFIG["use_time_in_root"] else 0
        template_root = RootMLP(coord_dim+add_time, root_out_dim, root_width, root_depth, k_root)
        
        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.theta_base = flat_params

        self.encoder = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_enc, hidden_width=64, depth=4)
        self.forward_dyn = ForwardDynamics(self.d_theta, lam_dim, split_forward, key=k_fwd)
        self.mem_dim = mem_dim

        num_actions = 4 if CONFIG["discrete_actions"] else None
        self.action_model = LAM(self.d_theta, lam_dim, mem_dim, max_len=20, num_heads=4, num_blocks=4, num_actions=num_actions, key=k_lam, phase=self.phase)

    def render_pixels(self, theta, coords):
        # 1. Apply coordinate encoding to the ENTIRE batched array at once
        if CONFIG["use_time_in_root"]:
            spatial_encoded = fourier_encode(coords[..., 1:], self.num_freqs)
            encoded_coords = jnp.concatenate([coords[..., :1], spatial_encoded], axis=-1)
        else:
            encoded_coords = fourier_encode(coords[..., 1:], self.num_freqs)
            
        # 2. Reconstruct the Equinox MLP
        root = self.unravel_fn(theta)
        
        # 3. Apply vmap ONLY to the pure MLP neural network call!
        # jax2onnx handles vmapped MLPs perfectly.
        out = jax.vmap(root)(encoded_coords)
        
        if CONFIG["use_nll_loss"]:
            C = self.frame_shape[2]
            mean, std = out[..., :C], out[..., C:]
            std = jax.nn.softplus(std) + 1e-4
            out = jnp.concatenate([mean, std], axis=-1)
            
        return out

    def render_frame(self, theta_offset, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 3)
        
        if not CONFIG["pretrain_encoder"]:
            theta = theta_offset + self.theta_base
        else:
            theta = theta_offset + jax.lax.stop_gradient(self.theta_base)

        pred_flat = self.render_pixels(theta, flat_coords)
        return pred_flat.reshape(H, W, -1)