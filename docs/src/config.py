# src/config.py
"""
Central configuration for the NOVA world model parameters.
Matches the defaults found in main.py for Moving MNIST.
"""

CONFIG = {
    "lam_space": 4,
    "mem_space": 256,
    "icl_decoding": True,
    "discrete_actions": False,
    "split_forward": True,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "use_time_in_root": False,
    "use_nll_loss": False,
    "pretrain_encoder": False,
    "frame_shape": (64, 64, 1) # Moving MNIST spatial dimensions
}
