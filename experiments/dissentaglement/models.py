#%% Imports
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


#%% Weight-space renderer

def fourier_encode(x, num_freqs):
    freqs = 2.0 ** jnp.arange(num_freqs)
    angles = x[..., None] * freqs * jnp.pi
    angles = angles.reshape(*x.shape[:-1], -1)
    return jnp.concatenate([x, jnp.sin(angles), jnp.cos(angles)], axis=-1)


def get_activation(name):
    if name == "sin":
        return jnp.sin
    if name == "gelu":
        return jax.nn.gelu
    return jax.nn.relu


class RootMLP(eqx.Module):
    layers: list
    activation: callable = eqx.field(static=True)

    def __init__(self, in_size, out_size, width, depth, activation_name, key):
        self.activation = get_activation(activation_name)
        keys = jax.random.split(key, depth + 1)
        self.layers = [eqx.nn.Linear(in_size, width, key=keys[0])]
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i + 1]))
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class WeightCNN(eqx.Module):
    layers: list
    theta_base: jax.Array

    def __init__(self, in_channels, out_dim, spatial_shape, theta_base, key, hidden_width=32, depth=4):
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 1)
        convs = []
        current_in, current_out = in_channels, hidden_width
        for i in range(depth):
            convs.append(eqx.nn.Conv2d(current_in, current_out, 3, stride=2, padding=1, key=keys[i]))
            current_in, current_out = current_out, current_out * 2

        dummy = jnp.zeros((in_channels, H, W))
        for layer in convs:
            dummy = layer(dummy)
        flat_dim = dummy.size

        self.layers = convs + [eqx.nn.Linear(flat_dim, out_dim, key=keys[-1])]
        self.theta_base = theta_base

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x.reshape(-1))


#%% Standard pixel autoencoder baseline
class ConvEncoder(eqx.Module):
    convs: list
    linear: eqx.nn.Linear

    def __init__(self, in_channels, latent_dim, spatial_shape, width, key):
        H, W = spatial_shape
        keys = jax.random.split(key, 5)
        channels = [width, 2 * width, 4 * width, 8 * width]
        convs = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            convs.append(eqx.nn.Conv2d(c_in, c_out, 4, stride=2, padding=1, key=keys[i]))
            c_in = c_out
        dummy = jnp.zeros((in_channels, H, W))
        for layer in convs:
            dummy = layer(dummy)
        self.convs = convs
        self.linear = eqx.nn.Linear(dummy.size, latent_dim, key=keys[-1])

    def __call__(self, x):
        for layer in self.convs:
            x = jax.nn.relu(layer(x))
        return self.linear(x.reshape(-1))


class ConvDecoder(eqx.Module):
    linear: eqx.nn.Linear
    deconvs: list
    base_shape: tuple = eqx.field(static=True)

    def __init__(self, out_channels, latent_dim, spatial_shape, width, key):
        H, W = spatial_shape
        if H % 16 or W % 16:
            raise ValueError("Standard decoder expects H and W divisible by 16")
        h0, w0 = H // 16, W // 16
        c0 = 8 * width
        keys = jax.random.split(key, 5)
        self.base_shape = (c0, h0, w0)
        self.linear = eqx.nn.Linear(latent_dim, c0 * h0 * w0, key=keys[0])
        self.deconvs = [
            eqx.nn.ConvTranspose2d(8 * width, 4 * width, 4, stride=2, padding=1, key=keys[1]),
            eqx.nn.ConvTranspose2d(4 * width, 2 * width, 4, stride=2, padding=1, key=keys[2]),
            eqx.nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1, key=keys[3]),
            eqx.nn.ConvTranspose2d(width, out_channels, 4, stride=2, padding=1, key=keys[4]),
        ]

    def __call__(self, z):
        x = jax.nn.relu(self.linear(z)).reshape(self.base_shape)
        for layer in self.deconvs[:-1]:
            x = jax.nn.relu(layer(x))
        return self.deconvs[-1](x)


#%% Dynamics
class ForwardDynamicsModule(eqx.Module):
    mlp_A: Optional[eqx.nn.MLP]
    mlp_B: Optional[eqx.nn.MLP]
    giant_mlp: Optional[eqx.nn.MLP]
    split_forward: bool = eqx.field(static=True)

    def __init__(self, dyn_dim, action_dim, split_forward, key, width=None):
        self.split_forward = split_forward
        k1, k2, k3 = jax.random.split(key, 3)
        width = dyn_dim * 2 if width is None else int(width)
        depth = 3
        if split_forward:
            self.mlp_A = eqx.nn.MLP(dyn_dim, dyn_dim, width_size=width, depth=depth, key=k1)
            self.mlp_B = eqx.nn.MLP(action_dim, dyn_dim, width_size=width, depth=depth, key=k2)
            self.giant_mlp = None
        else:
            self.mlp_A = None
            self.mlp_B = None
            self.giant_mlp = eqx.nn.MLP(dyn_dim + action_dim, dyn_dim, width_size=width, depth=depth, key=k3)

    def __call__(self, z_prev, action):
        if self.split_forward:
            z_a = self.mlp_A(z_prev)
            z_b = self.mlp_B(action)
            return (z_a, z_b), z_a + z_b
        out = self.giant_mlp(jnp.concatenate([z_prev, action]))
        return None, out


class InverseDynamicsModule(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, dyn_dim, action_dim, key):
        self.mlp = eqx.nn.MLP(2 * dyn_dim, action_dim, width_size=dyn_dim, depth=3, key=key)

    def __call__(self, z_prev, z_next):
        return self.mlp(jnp.concatenate([z_prev, z_next]))


class LatentActionModule(eqx.Module):
    idm: InverseDynamicsModule
    embeddings: Optional[eqx.nn.Embedding]
    discrete: bool = eqx.field(static=True)

    def __init__(self, dyn_dim, action_dim, discrete, num_actions, key):
        k_idm, k_emb = jax.random.split(key)
        self.idm = InverseDynamicsModule(dyn_dim, action_dim, k_idm)
        self.discrete = bool(discrete)
        if self.discrete:
            weights = 0.05 * jax.random.normal(k_emb, (int(num_actions), action_dim))
            self.embeddings = eqx.nn.Embedding(weight=weights, key=k_emb)
        else:
            self.embeddings = None

    def decode(self, z_prev, z_next):
        raw = self.idm(z_prev, z_next)
        if not self.discrete:
            return raw, raw, jnp.asarray(-1, dtype=jnp.int32)
        dists = jnp.sum((self.embeddings.weight - raw) ** 2, axis=-1)
        idx = jnp.argmin(dists)
        return raw, self.embeddings(idx), idx


#%% Unified four-mode world model
class WorldModel(eqx.Module):
    encoder: eqx.Module
    decoder: Optional[ConvDecoder]
    transition_model: ForwardDynamicsModule
    action_model: LatentActionModule

    unravel_fn: Optional[callable] = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    use_time_in_root: bool = eqx.field(static=True)

    def __init__(self, config, frame_shape, key):
        requested_mode = config["model"]["mode"]
        # Backwards compatibility for v1 runs: standard meant standard + joint FDM.
        self.mode = "standard_joint" if requested_mode == "standard" else requested_mode
        valid_modes = ("weight_ab", "weight_joint", "standard_ab", "standard_joint")
        if self.mode not in valid_modes:
            raise ValueError(f"model.mode must be one of {valid_modes}; got {requested_mode!r}")

        self.frame_shape = tuple(frame_shape)
        self.num_freqs = int(config["model"]["num_fourier_freqs"])
        self.use_time_in_root = bool(config["model"].get("use_time_in_root", False))
        self.action_dim = int(config["model"]["action_dim"])

        H, W, C = self.frame_shape
        k_root, k_enc, k_dec, k_fwd, k_act = jax.random.split(key, 5)

        coord_dim = 2 + 4 * self.num_freqs + int(self.use_time_in_root)
        root = RootMLP(
            coord_dim, C,
            int(config["model"]["root_width"]),
            int(config["model"]["root_depth"]),
            config["model"]["root_activation"],
            k_root,
        )
        flat_root, unravel_fn = ravel_pytree(root)
        self.latent_dim = int(flat_root.shape[0])

        if self.mode.startswith("weight_"):
            self.encoder = WeightCNN(
                C, self.latent_dim, (H, W), flat_root, k_enc,
                hidden_width=int(config["model"]["weight_cnn_width"]),
                depth=int(config["model"]["weight_cnn_depth"]),
            )
            self.decoder = None
            self.unravel_fn = unravel_fn
        else:
            width = int(config["model"]["standard_cnn_width"])
            self.encoder = ConvEncoder(C, self.latent_dim, (H, W), width, k_enc)
            self.decoder = ConvDecoder(C, self.latent_dim, (H, W), width, k_dec)
            self.unravel_fn = None

        split_forward = self.mode.endswith("_ab")
        width_key = "fdm_width_multiplier_ab" if split_forward else "fdm_width_multiplier_joint"
        fdm_width = round(self.latent_dim * float(config["model"].get(width_key, 2.0)))
        self.transition_model = ForwardDynamicsModule(
            self.latent_dim,
            self.action_dim,
            split_forward=split_forward,
            key=k_fwd,
            width=fdm_width,
        )
        self.action_model = LatentActionModule(
            self.latent_dim,
            self.action_dim,
            discrete=bool(config["model"].get("discrete_actions", False)),
            num_actions=int(config["model"].get("num_actions", 9)),
            key=k_act,
        )

    def encode_frame(self, frame):
        return self.encoder(jnp.transpose(frame, (2, 0, 1)))

    def decode_frame(self, z, coords_grid, time_value=0.0):
        H, W, _ = self.frame_shape
        if self.mode.startswith("standard_"):
            return jnp.transpose(self.decoder(z), (1, 2, 0))

        theta = z + self.encoder.theta_base
        root = self.unravel_fn(theta)
        flat_xy = coords_grid.reshape(-1, 2)

        def render_point(xy):
            encoded = fourier_encode(xy, self.num_freqs)
            if self.use_time_in_root:
                encoded = jnp.concatenate([jnp.asarray([time_value], dtype=z.dtype), encoded])
            return root(encoded)

        return jax.vmap(render_point)(flat_xy).reshape(H, W, -1)

    def infer_action(self, z_prev, z_next):
        raw, quant, idx = self.action_model.decode(z_prev, z_next)
        if self.action_model.discrete:
            action = raw + jax.lax.stop_gradient(quant - raw)
        else:
            action = raw
        return raw, quant, action, idx

    def video_representation(self, video):
        latents = jax.vmap(self.encode_frame)(video)
        _, quant_actions, _, _ = jax.vmap(self.infer_action)(latents[:-1], latents[1:])
        return jnp.concatenate([latents[0], quant_actions.reshape(-1)])
