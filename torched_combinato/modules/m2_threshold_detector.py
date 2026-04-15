"""
M2 - ThresholdDetector Module (Batched Multi-Channel)
=======================================================
Takes data_detected from M1 and finds spike locations.

BATCHED: Processes all channels in parallel where possible.
         Falls back to per-channel for variable-length outputs.
"""

import torch
from .block import Block


class ThresholdDetector(Block):
    def __init__(self, sample_rate=24000, threshold_factor=5, max_spike_duration=0.0015):
        super().__init__()
        self.sample_rate = sample_rate
        self.threshold_factor = threshold_factor
        self.max_spike_samples = int(max_spike_duration * sample_rate)

    def compute_threshold(self, data_detected):
        """
        Compute threshold for single channel or batch.
        
        Args:
            data_detected: [N_samples] or [N_channels, N_samples]
            
        Returns:
            threshold: scalar or [N_channels] tensor
        """
        if data_detected.dim() == 1:
            noise_level = torch.median(torch.abs(data_detected)) / 0.6745
            return self.threshold_factor * noise_level
        else:
            # Batched: compute per-channel threshold
            noise_level = torch.median(torch.abs(data_detected), dim=1).values / 0.6745
            return self.threshold_factor * noise_level

    def find_crossings(self, data_detected, threshold, sign):
        """Find threshold crossings for single channel."""
        mask = (data_detected > threshold) if sign == 'pos' else (data_detected < -threshold)
        diff = torch.diff(mask.to(torch.int32))
        entries = (diff == 1).nonzero(as_tuple=True)[0]
        exits = (diff == -1).nonzero(as_tuple=True)[0]
        n = min(len(entries), len(exits))
        if n == 0:
            return None
        return torch.stack([entries[:n], exits[:n]], dim=1)

    def filter_duration(self, borders):
        """Filter crossings by duration."""
        durations = borders[:, 1] - borders[:, 0]
        return borders[durations <= self.max_spike_samples]

    def find_peaks_vectorized(self, data_detected, borders, sign):
        """
        Find peaks in ALL crossings with ONE batched operation.
        """
        if borders is None or len(borders) == 0:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        
        device = data_detected.device
        
        starts = borders[:, 0]
        ends = borders[:, 1]
        lengths = ends - starts
        
        valid_mask = lengths > 0
        if not valid_mask.any():
            return torch.zeros(0, dtype=torch.long, device=device)
        
        starts = starts[valid_mask]
        ends = ends[valid_mask]
        lengths = lengths[valid_mask]
        
        max_len = lengths.max().item()
        
        offsets = torch.arange(max_len, device=device).unsqueeze(0)
        indices = starts.unsqueeze(1) + offsets
        indices_clamped = indices.clamp(0, len(data_detected) - 1)
        
        windows = data_detected[indices_clamped]
        position_mask = offsets < lengths.unsqueeze(1)
        
        if sign == 'pos':
            windows_masked = torch.where(position_mask, windows, torch.tensor(float('-inf'), device=device, dtype=windows.dtype))
            local_peaks = windows_masked.argmax(dim=1)
        else:
            windows_masked = torch.where(position_mask, windows, torch.tensor(float('inf'), device=device, dtype=windows.dtype))
            local_peaks = windows_masked.argmin(dim=1)
        
        peaks = starts + local_peaks
        return peaks

    def _detect_one_channel(self, data_detected, threshold, sign):
        """Detect spikes in a single channel."""
        borders = self.find_crossings(data_detected, threshold, sign)
        if borders is None:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        
        borders = self.filter_duration(borders)
        if len(borders) == 0:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        
        peaks = self.find_peaks_vectorized(data_detected, borders, sign)
        
        # Match original Combinato: remove first 1 and last 2 spikes
        if len(peaks) > 3:
            peaks = peaks[1:-2]
        
        return peaks

    def forward(self, data_detected):
        """
        Forward pass for single channel.
        
        Args:
            data_detected: [N_samples] single channel
            
        Returns:
            pos_indices, neg_indices, threshold
        """
        threshold = self.compute_threshold(data_detected)
        pos_indices = self._detect_one_channel(data_detected, threshold, 'pos')
        neg_indices = self._detect_one_channel(data_detected, threshold, 'neg')
        return pos_indices, neg_indices, threshold

    def forward_batched(self, data_detected):
        """
        Forward pass for multiple channels.
        
        Args:
            data_detected: [N_channels, N_samples]
            
        Returns:
            List of (pos_indices, neg_indices) per channel, thresholds [N_channels]
        """
        n_channels = data_detected.shape[0]
        device = data_detected.device
        
        # Compute all thresholds at once (batched)
        thresholds = self.compute_threshold(data_detected)  # [N_channels]
        
        # Per-channel detection (can't easily batch due to variable spike counts)
        results = []
        for ch in range(n_channels):
            pos_idx = self._detect_one_channel(data_detected[ch], thresholds[ch], 'pos')
            neg_idx = self._detect_one_channel(data_detected[ch], thresholds[ch], 'neg')
            results.append((pos_idx, neg_idx))
        
        return results, thresholds