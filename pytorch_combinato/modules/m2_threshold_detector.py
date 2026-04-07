"""
M2 - ThresholdDetector Module
==============================
Takes data_detected from M1 and finds spike locations.

FIXED: Now uses actual crossing boundaries for peak finding,
matching original Combinato behavior exactly.

UPDATED: Removes first 1 and last 2 spikes to match original Combinato.
"""

import torch
from modules.block import Block

THRESHOLD_FACTOR   = 5
MAX_SPIKE_DURATION = 0.0015


class ThresholdDetector(Block):
    def __init__(self, sample_rate=24000, threshold_factor=5, max_spike_duration=0.0015):
        super().__init__()
        self.sample_rate       = sample_rate
        self.threshold_factor  = threshold_factor
        self.max_spike_samples = int(max_spike_duration * sample_rate)

    def compute_threshold(self, data_detected):
        noise_level = torch.median(torch.abs(data_detected)) / 0.6745
        return self.threshold_factor * noise_level

    def find_crossings(self, data_detected, threshold, sign):
        mask = (data_detected > threshold) if sign == 'pos' else (data_detected < -threshold)
        diff    = torch.diff(mask.to(torch.int32))
        entries = (diff ==  1).nonzero(as_tuple=True)[0]
        exits   = (diff == -1).nonzero(as_tuple=True)[0]
        n = min(len(entries), len(exits))
        if n == 0:
            return None
        return torch.stack([entries[:n], exits[:n]], dim=1)

    def filter_duration(self, borders):
        durations = borders[:, 1] - borders[:, 0]
        return borders[durations <= self.max_spike_samples]

    def find_peaks(self, data_detected, borders, sign):
        """
        Find peak within each crossing using ACTUAL crossing boundaries.
        
        This matches original Combinato:
            maxima = [detect_func(data_detect[range(borders[i,0], borders[i,1])])
                      + borders[i,0]
                      for i in range(borders.shape[0])]
        """
        if borders is None or len(borders) == 0:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        
        # Process each crossing individually using its actual boundaries
        peaks = []
        for i in range(borders.shape[0]):
            start = borders[i, 0].item()
            end = borders[i, 1].item()
            
            # Skip invalid crossings (start >= end)
            if start >= end:
                continue
            
            # Extract window using actual crossing boundaries
            window = data_detected[start:end]
            
            # Skip empty windows
            if window.numel() == 0:
                continue
            
            # Find peak (argmax for positive, argmin for negative)
            if sign == 'pos':
                local_peak = torch.argmax(window).item()
            else:
                local_peak = torch.argmin(window).item()
            
            peaks.append(start + local_peak)
        
        if len(peaks) == 0:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        
        return torch.tensor(peaks, dtype=torch.long, device=data_detected.device)

    def _detect_one(self, data_detected, threshold, sign):
        borders = self.find_crossings(data_detected, threshold, sign)
        if borders is None:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        borders = self.filter_duration(borders)
        if len(borders) == 0:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        peaks = self.find_peaks(data_detected, borders, sign)
        
        # Match original Combinato: remove first 1 and last 2 spikes
        if len(peaks) > 3:
            peaks = peaks[1:-2]
        
        return peaks

    def forward(self, data_detected):
        threshold   = self.compute_threshold(data_detected)
        pos_indices = self._detect_one(data_detected, threshold, 'pos')
        neg_indices = self._detect_one(data_detected, threshold, 'neg')
        return pos_indices, neg_indices, threshold
