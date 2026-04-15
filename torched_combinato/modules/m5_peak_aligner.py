"""
M5 - PeakAligner Module
=========================
Aligns upsampled waveforms so peaks land at the same index,
removes outliers, then downsamples back to 64 samples.

VECTORIZED: Fully vectorized for speed while matching original Combinato exactly.
"""

import torch
from .block import Block


class PeakAligner(Block):
    def __init__(self,
                 factor=3,
                 indices_per_spike=64,
                 index_maximum=19,
                 border_pad=5):
        super().__init__()
        self.factor            = factor
        self.indices_per_spike = indices_per_spike
        self.index_maximum     = index_maximum
        self.border_pad        = border_pad
        self.width             = border_pad

        self.center     = (index_maximum + border_pad) * factor  # 72
        self.low        = factor                                  # 3
        self.high       = factor                                  # 3
        self.new_center = self.center - self.width * self.low    # 57

    def forward(self, spikes_up):
        """
        Align, clean, and downsample spikes.
        
        Matches original Combinato (interpolate.py):
            align() -> clean() -> downsample()
        """
        K, L = spikes_up.shape
        device = spikes_up.device
        dtype = spikes_up.dtype
        
        if K == 0:
            empty = torch.zeros((0, self.indices_per_spike), dtype=dtype, device=device)
            return empty, torch.zeros(0, dtype=torch.bool, device=device)
        
        # ============================================
        # ALIGN - Vectorized
        # ============================================
        search_start = self.center - self.width * self.low   # 57
        search_end = self.center + self.width * self.high    # 87
        
        window = spikes_up[:, search_start:search_end]
        local_peaks = window.argmax(dim=1)
        index_max = local_peaks + search_start
        
        aligned_len = L - self.width * self.low - self.width * self.high  # 190
        
        # Vectorized alignment using gather
        # starts[i] = index_max[i] - center + width * low
        starts = index_max - self.center + self.width * self.low  # Shape: (K,)
        
        # Create index matrix: each row is [start, start+1, start+2, ..., start+aligned_len-1]
        offsets = torch.arange(aligned_len, device=device)  # Shape: (aligned_len,)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)  # Shape: (K, aligned_len)
        
        # Gather aligned waveforms
        # Note: indices may be out of bounds for edge cases, same as original's behavior
        # We clamp to valid range but mark these as invalid later via the clean step
        indices_clamped = indices.clamp(0, L - 1)
        aligned = torch.gather(spikes_up, 1, indices_clamped)
        
        # ============================================
        # CLEAN - Vectorized (same as before)
        # ============================================
        index_max_aligned = aligned.argmax(dim=1)
        at_center = (index_max_aligned == self.new_center)
        removed_mask = ~at_center
        
        if at_center.sum() == 0:
            empty = torch.zeros((0, self.indices_per_spike), dtype=dtype, device=device)
            return empty, removed_mask
        
        cleaned = aligned[at_center]
        
        # ============================================
        # DOWNSAMPLE - Vectorized (same as before)
        # ============================================
        ds_idx = torch.arange(self.indices_per_spike, device=device) * self.factor
        spikes_final = cleaned[:, ds_idx]
        
        return spikes_final, removed_mask
