# src/utils.py
import equinox as eqx
from .config import CONFIG
from .models import WARP

def get_model(key):
    """
    Instantiates the complete Phase 2 WARP model.
    """
    model = WARP(
        root_width=CONFIG["root_width"], 
        root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], 
        frame_shape=CONFIG["frame_shape"], 
        lam_dim=CONFIG["lam_space"], 
        mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], 
        key=key, 
        phase=2
    )
    return model

def load_weights(model, path):
    """
    Loads saved Phase 2 weights into the Equinox model.
    """
    return eqx.tree_deserialise_leaves(path, model)
