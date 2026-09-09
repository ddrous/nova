#%% Imports
import numpy as np
from functools import lru_cache


#%% Ground-truth factors
PARAMETER_NAMES = (
    "shape", "rotation", "x0", "y0",
    "vx0", "vy0", "vx1", "vy1",
)


def make_sprite(shape, rotation, shape_size=9.0, supersample=12):
    """Rasterise one binary sprite: 0=heart, 1=oval, 2=square."""
    if int(shape) not in (0, 1, 2):
        raise ValueError("shape must be 0 (heart), 1 (oval), or 2 (square)")

    half = int(np.ceil(np.sqrt(2) * shape_size))
    p = np.arange(-half, half + 1, dtype=float)
    q = (np.arange(supersample) + 0.5) / supersample - 0.5
    oy, ox = np.meshgrid(q, q, indexing="ij")

    x = p[None, :, None, None] + ox[None, None, :, :]
    y = -(p[:, None, None, None] + oy[None, None, :, :])

    c, s = np.cos(rotation), np.sin(rotation)
    xr = c * x + s * y
    yr = -s * x + c * y

    if int(shape) == 0:
        u, v = 1.10 * xr / shape_size, 1.10 * yr / shape_size
        inside = (u*u + v*v - 1)**3 - u*u*v**3 <= 0
    elif int(shape) == 1:
        inside = (xr / shape_size)**2 + (yr / (0.72 * shape_size))**2 <= 1
    else:
        inside = (np.abs(xr) <= 0.85 * shape_size) & (np.abs(yr) <= 0.85 * shape_size)

    return (inside.mean(axis=(-1, -2)) >= 0.5).astype(np.uint8)


def make_trajectory(x0, y0, velocities):
    """Return integer [x, y] centres for frame 0 and all later frames."""
    velocities = np.asarray(velocities, dtype=float).reshape(-1, 2)
    positions = np.vstack(([x0, y0], [x0, y0] + np.cumsum(velocities, axis=0)))
    if not np.allclose(positions, np.rint(positions)):
        raise ValueError("x0, y0 and velocities must give integer pixel positions")
    return np.rint(positions).astype(int)


#%% Simulator

def simulate_video(parameters, image_size=64, shape_size=9.0):
    """Generate [T,H,W] binary video from [shape,rotation,x0,y0,vx0,vy0,...]."""
    p = np.asarray(parameters, dtype=float).ravel()
    if len(p) < 4 or (len(p) - 4) % 2:
        raise ValueError("parameters must be [shape, rotation, x0, y0, vx0, vy0, ...]")

    shape = int(p[0])
    rotation = p[1]
    positions = make_trajectory(p[2], p[3], p[4:])
    sprite = make_sprite(shape, rotation, shape_size)

    half = sprite.shape[0] // 2
    dy, dx = np.where(sprite == 1)
    dy, dx = dy - half, dx - half

    video = np.zeros((len(positions), image_size, image_size), dtype=np.uint8)
    for t, (x, y) in enumerate(positions):
        rows, cols = y + dy, x + dx
        if rows.min() <= 0 or rows.max() >= image_size - 1 or cols.min() <= 0 or cols.max() >= image_size - 1:
            raise ValueError(f"shape would touch the border in frame {t}")
        video[t, rows, cols] = 1
    return video


#%% Independent, always-valid parameter supports

def parameter_supports(sim_config):
    """Return discrete supports whose Cartesian product is guaranteed to stay in-frame."""
    image_size = int(sim_config.get("image_size", 64))
    shape_size = float(sim_config.get("shape_size", 9.0))
    num_frames = int(sim_config.get("num_frames", 3))
    rotation_bins = int(sim_config.get("rotation_bins", 40))
    velocity_values = tuple(int(v) for v in sim_config.get("velocity_values", [-4, 0, 4]))
    supports = _parameter_supports_cached(image_size, shape_size, num_frames, rotation_bins, velocity_values)
    return {name: values.copy() for name, values in supports.items()}


@lru_cache(maxsize=16)
def _parameter_supports_cached(image_size, shape_size, num_frames, rotation_bins, velocity_values):
    """Cached implementation; sprite geometry is computed once per simulator configuration."""
    velocity_values = np.asarray(velocity_values, dtype=int)
    if num_frames < 2:
        raise ValueError("num_frames must be >= 2")
    if np.any(velocity_values % 4 != 0):
        raise ValueError("All sampled velocities must be multiples of four")

    rotations = 2.0 * np.pi * np.arange(rotation_bins, dtype=float) / rotation_bins
    min_dx, max_dx, min_dy, max_dy = 0, 0, 0, 0
    first = True
    for shape in (0, 1, 2):
        for rotation in rotations:
            sprite = make_sprite(shape, rotation, shape_size)
            half = sprite.shape[0] // 2
            yy, xx = np.where(sprite == 1)
            dx, dy = xx - half, yy - half
            if first:
                min_dx, max_dx = int(dx.min()), int(dx.max())
                min_dy, max_dy = int(dy.min()), int(dy.max())
                first = False
            else:
                min_dx, max_dx = min(min_dx, int(dx.min())), max(max_dx, int(dx.max()))
                min_dy, max_dy = min(min_dy, int(dy.min())), max(max_dy, int(dy.max()))

    max_disp = (num_frames - 1) * int(np.max(np.abs(velocity_values)))
    x_min = 1 - min_dx + max_disp
    x_max = image_size - 2 - max_dx - max_disp
    y_min = 1 - min_dy + max_disp
    y_max = image_size - 2 - max_dy - max_disp
    if x_min > x_max or y_min > y_max:
        raise ValueError(
            "No globally safe centre positions exist. Reduce shape_size, velocity range, "
            "or num_frames, or increase image_size."
        )

    names = PARAMETER_NAMES[:4 + 2 * (num_frames - 1)]
    supports = {
        "shape": np.asarray([0, 1, 2], dtype=int),
        "rotation": rotations,
        "x0": np.arange(x_min, x_max + 1, dtype=int),
        "y0": np.arange(y_min, y_max + 1, dtype=int),
    }
    for name in names[4:]:
        supports[name] = velocity_values.copy()
    return supports


def sample_parameters(rng, sim_config, fixed=None):
    """Sample one valid parameter vector; optional ``fixed`` maps factor names to values."""
    fixed = {} if fixed is None else dict(fixed)
    supports = parameter_supports(sim_config)
    num_frames = int(sim_config.get("num_frames", 3))
    names = PARAMETER_NAMES[:4 + 2 * (num_frames - 1)]

    unknown = set(fixed) - set(names)
    if unknown:
        raise KeyError(f"Unknown fixed parameter(s): {sorted(unknown)}")

    values = []
    for name in names:
        support = supports[name]
        if name in fixed and fixed[name] is not None:
            value = fixed[name]
            if name != "rotation" and not np.any(np.isclose(support, value)):
                raise ValueError(f"Fixed value {value!r} is outside the support for {name}")
            if name == "rotation" and not np.any(np.isclose(support, value, atol=1e-8)):
                raise ValueError(f"Fixed rotation {value!r} is outside the configured rotation grid")
        else:
            value = support[rng.integers(len(support))]
        values.append(value)

    return np.asarray(values, dtype=float)


def sample_video(rng, sim_config, fixed=None):
    """Sample parameters and call the simulator immediately."""
    parameters = sample_parameters(rng, sim_config, fixed=fixed)
    video = simulate_video(
        parameters,
        image_size=int(sim_config.get("image_size", 64)),
        shape_size=float(sim_config.get("shape_size", 9.0)),
    )
    return video[..., None].astype(np.float32), parameters.astype(np.float32)
