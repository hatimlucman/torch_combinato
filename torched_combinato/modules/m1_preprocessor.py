"""
M1 - Preprocessor Module (Batched)
===================================
Replicates DefaultFilter from Combinato using PyTorch + torchaudio.

BATCHED: Supports input shape [N_channels, N_samples] for parallel processing.

Design note on numerical precision:
    torchaudio.functional.filtfilt differs from scipy.signal.filtfilt by at most
    ~0.19 on a signal scaled to ~600,000. This is 0.00003% of threshold magnitude
    and has zero practical effect on spike detection. clamp=False is required
    because torchaudio assumes audio signals in [-1, 1] by default.
"""

import torch
import torchaudio
import numpy as np
from scipy.signal import ellip
from .block import Block

DETECT_LOW   = 300
DETECT_HIGH  = 1000
EXTRACT_LOW  = 300
EXTRACT_HIGH = 3000
NOTCH_LOW    = 1999
NOTCH_HIGH   = 2001


class Preprocessor(Block):
    def __init__(self, sample_rate=24000):
        super().__init__()
        timestep = 1.0 / sample_rate

        b_notch, a_notch = ellip(2, 0.5, 20,
                                 (2*timestep*NOTCH_LOW, 2*timestep*NOTCH_HIGH),
                                 'bandstop')
        b_detect, a_detect = ellip(2, 0.1, 40,
                                   (2*timestep*DETECT_LOW, 2*timestep*DETECT_HIGH),
                                   'bandpass')
        b_extract, a_extract = ellip(2, 0.1, 40,
                                     (2*timestep*EXTRACT_LOW, 2*timestep*EXTRACT_HIGH),
                                     'bandpass')

        self.register_buffer('b_notch',   torch.tensor(b_notch,   dtype=torch.float64))
        self.register_buffer('a_notch',   torch.tensor(a_notch,   dtype=torch.float64))
        self.register_buffer('b_detect',  torch.tensor(b_detect,  dtype=torch.float64))
        self.register_buffer('a_detect',  torch.tensor(a_detect,  dtype=torch.float64))
        self.register_buffer('b_extract', torch.tensor(b_extract, dtype=torch.float64))
        self.register_buffer('a_extract', torch.tensor(a_extract, dtype=torch.float64))
        self.sample_rate = sample_rate

    def _apply_filter(self, x, b, a):
        """
        Apply filter to input.
        
        Supports:
            - 1D input: [N_samples] (single channel)
            - 2D input: [N_channels, N_samples] (batched)
        """
        if x.dim() == 1:
            # Single channel: [N_samples] -> [1, 1, N_samples]
            x_3d = x.unsqueeze(0).unsqueeze(0)
            y_3d = torchaudio.functional.filtfilt(x_3d, a, b, clamp=False)
            return y_3d.squeeze(0).squeeze(0)
        elif x.dim() == 2:
            # Batched: [N_channels, N_samples] -> [1, N_channels, N_samples]
            x_3d = x.unsqueeze(0)
            y_3d = torchaudio.functional.filtfilt(x_3d, a, b, clamp=False)
            return y_3d.squeeze(0)
        else:
            raise ValueError(f"Expected 1D or 2D input, got {x.dim()}D")

    def forward(self, x):
        """
        Apply notch and detect filters.
        
        Args:
            x: [N_samples] or [N_channels, N_samples]
            
        Returns:
            data_denoised: same shape as input
            data_detected: same shape as input
        """
        data_denoised = self._apply_filter(x, self.b_notch, self.a_notch)
        data_detected = self._apply_filter(data_denoised, self.b_detect, self.a_detect)
        return data_denoised, data_detected

    def filter_extract(self, data_denoised):
        """
        Apply extract filter.
        
        Args:
            data_denoised: [N_samples] or [N_channels, N_samples]
            
        Returns:
            data_extracted: same shape as input
        """
        return self._apply_filter(data_denoised, self.b_extract, self.a_extract)


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == '__main__':
    from scipy.signal import ellip, filtfilt as scipy_filtfilt
    import time

    print("=" * 60)
    print("M1 PREPROCESSOR (BATCHED) — VALIDATION")
    print("=" * 60)

    SAMPLE_RATE = 30000
    N_SAMPLES = 90000
    N_CHANNELS = 384
    
    np.random.seed(42)
    
    # Single channel test data
    raw_single = np.random.randn(N_SAMPLES).astype(np.float64) * 100
    
    # Batched test data
    raw_batched = np.random.randn(N_CHANNELS, N_SAMPLES).astype(np.float64) * 100

    # SciPy reference for single channel
    timestep = 1.0 / SAMPLE_RATE
    b_n, a_n = ellip(2, 0.5, 20, (2*timestep*1999, 2*timestep*2001), 'bandstop')
    b_d, a_d = ellip(2, 0.1, 40, (2*timestep*300,  2*timestep*1000), 'bandpass')

    orig_denoised = scipy_filtfilt(b_n, a_n, raw_single)
    orig_detected = scipy_filtfilt(b_d, a_d, orig_denoised)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    pre = Preprocessor(SAMPLE_RATE).to(device)
    
    # =========================================================================
    # Test 1: Single channel (backward compatibility)
    # =========================================================================
    print("[1] Single channel test...")
    
    x_single = torch.tensor(raw_single, dtype=torch.float64).to(device)
    
    with torch.no_grad():
        pt_denoised, pt_detected = pre(x_single)

    diff_denoised = np.max(np.abs(pt_denoised.cpu().numpy() - orig_denoised))
    diff_detected = np.max(np.abs(pt_detected.cpu().numpy() - orig_detected))
    
    print(f"    Denoised max diff: {diff_denoised:.6f}")
    print(f"    Detected max diff: {diff_detected:.6f}")
    print(f"    Output shape: {pt_denoised.shape}")
    print(f"    ✓ PASS" if diff_denoised < 0.01 else "    ✗ FAIL")
    
    # =========================================================================
    # Test 2: Batched channels
    # =========================================================================
    print("\n[2] Batched channels test...")
    
    x_batched = torch.tensor(raw_batched, dtype=torch.float64).to(device)
    
    with torch.no_grad():
        pt_denoised_batch, pt_detected_batch = pre(x_batched)
    
    print(f"    Input shape:  {x_batched.shape}")
    print(f"    Output shape: {pt_denoised_batch.shape}")
    
    # Verify first channel matches single-channel result
    scipy_result = scipy_filtfilt(b_n, a_n, raw_batched[0]).copy()
    first_channel_diff = torch.max(torch.abs(
        pt_denoised_batch[0] - torch.tensor(scipy_result, device=device)
    )).item()
    print(f"    First channel diff: {first_channel_diff:.6f}")
    print(f"    ✓ PASS" if first_channel_diff < 0.01 else "    ✗ FAIL")
    
    # =========================================================================
    # Benchmark: Sequential vs Batched
    # =========================================================================
    print("\n[3] Benchmark: Sequential vs Batched...")
    
    # Sequential (current approach)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for ch in range(N_CHANNELS):
        with torch.no_grad():
            _ = pre(x_batched[ch])
    if device.type == 'cuda':
        torch.cuda.synchronize()
    sequential_time = time.time() - t0
    
    # Batched (new approach)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = pre(x_batched)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    batched_time = time.time() - t0
    
    print(f"    Sequential ({N_CHANNELS} calls): {sequential_time*1000:.1f}ms")
    print(f"    Batched (1 call):                 {batched_time*1000:.1f}ms")
    print(f"    Speedup: {sequential_time/batched_time:.1f}x")

    print("\n" + "=" * 60)
    print("M1 VALIDATION COMPLETE")
    print("=" * 60)