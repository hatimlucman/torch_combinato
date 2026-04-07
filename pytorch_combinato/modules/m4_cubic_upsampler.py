"""
M4 - CubicUpsampler Module
============================
Upsamples each spike waveform by factor 3 using cubic spline interpolation.

Uses scipy's make_interp_spline to match original Combinato exactly.
"""

import torch
import numpy as np
from scipy.interpolate import make_interp_spline
from modules.block import Block


class CubicUpsampler(Block):
    def __init__(self, factor=3):
        super().__init__()
        self.factor = factor

    def forward(self, spikes):
        K, L = spikes.shape
        target_len = (L - 1) * self.factor + 1  # 74 -> 220
        
        # Handle empty input
        if K == 0:
            return torch.zeros((0, target_len), dtype=spikes.dtype, device=spikes.device)
        
        # Store original dtype and device
        dtype = spikes.dtype
        device = spikes.device
        
        # Convert to numpy
        spikes_np = spikes.cpu().numpy()
        
        # Original sample positions: 0, 3, 6, 9, ... 219
        axis = np.arange(0, target_len, self.factor)
        
        # Upsampled positions: 0, 1, 2, 3, 4, ... 219
        up_axis = np.arange(target_len)
        
        # Cubic spline interpolation (matches original Combinato)
        splines = make_interp_spline(axis, spikes_np.T)
        up_spikes = splines(up_axis).T
        
        # Convert back to tensor
        return torch.tensor(up_spikes, dtype=dtype, device=device)
