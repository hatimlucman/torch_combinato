"""
C1 - WaveletFeatureExtractor (Pure PyTorch)
=============================================
Transforms spike waveforms into wavelet feature vectors using Haar wavelet.

PURE PYTORCH: No PyWavelets dependency. Fully GPU-compatible.

Uses precomputed Haar transform matrix for single matrix multiplication.
Produces identical results to pywt.wavedec(spike, 'haar', level=4).
"""

import torch
import numpy as np
from .block import Block

WAVELET = 'haar'
LEVEL = 4


def build_haar_matrix(n, level):
    """
    Build the full Haar wavelet transform matrix.
    
    For input size n=64 and level=4, produces a 64x64 orthogonal matrix
    that computes the same coefficients as pywt.wavedec(x, 'haar', level=4).
    
    Output order matches PyWavelets: [cA4, cD4, cD3, cD2, cD1]
    """
    sqrt2 = np.sqrt(2)
    
    detail_coeffs = []
    approx_basis = np.eye(n)
    current_len = n
    
    for lev in range(level):
        half = current_len // 2
        
        new_approx = np.zeros((half, n))
        new_detail = np.zeros((half, n))
        
        for i in range(half):
            new_approx[i] = (approx_basis[2*i] + approx_basis[2*i + 1]) / sqrt2
            new_detail[i] = (approx_basis[2*i] - approx_basis[2*i + 1]) / sqrt2
        
        detail_coeffs.append(new_detail)
        approx_basis = new_approx
        current_len = half
    
    final_approx = approx_basis
    
    output_rows = [final_approx]
    for i in range(level - 1, -1, -1):
        output_rows.append(detail_coeffs[i])
    
    output_matrix = np.vstack(output_rows)
    
    return output_matrix


class WaveletFeatureExtractor(Block):
    """
    GPU-accelerated Haar wavelet feature extraction.
    
    Computes 4-level Haar wavelet decomposition via single matrix multiplication.
    Output is identical to pywt.wavedec(spike, 'haar', level=4).
    """
    
    def __init__(self, wavelet=WAVELET, level=LEVEL, input_size=64):
        super().__init__()
        self.wavelet_name = wavelet
        self.level = level
        self.input_size = input_size
        self.feature_size = input_size
        
        haar_matrix = build_haar_matrix(input_size, level)
        
        self.register_buffer(
            'haar_matrix',
            torch.tensor(haar_matrix, dtype=torch.float64)
        )
    
    def forward(self, spikes):
        """
        Transform spike waveforms to wavelet features.
        
        Args:
            spikes: [N, 64] tensor of spike waveforms
            
        Returns:
            [N, 64] tensor of wavelet coefficients
        """
        haar = self.haar_matrix.to(dtype=spikes.dtype, device=spikes.device)
        return spikes @ haar.T


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == '__main__':
    import pywt
    import time
    
    print("=" * 60)
    print("C1 WAVELET FEATURE EXTRACTOR — VALIDATION")
    print("=" * 60)
    
    N_SPIKES = 1000
    INPUT_SIZE = 64
    LEVEL = 4
    
    np.random.seed(42)
    test_spikes = np.random.randn(N_SPIKES, INPUT_SIZE).astype(np.float32)
    
    # PyWavelets reference
    print("\n[1] Running PyWavelets reference...")
    
    pywt_output = np.zeros((N_SPIKES, INPUT_SIZE), dtype=np.float32)
    for i in range(N_SPIKES):
        coeffs = pywt.wavedec(test_spikes[i], 'haar', level=LEVEL)
        pywt_output[i] = np.hstack(coeffs)
    
    print(f"    Output shape: {pywt_output.shape}")
    
    # PyTorch implementation
    print("\n[2] Running Pure PyTorch implementation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    extractor = WaveletFeatureExtractor(wavelet='haar', level=LEVEL).to(device)
    spikes_tensor = torch.tensor(test_spikes, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        torch_output = extractor(spikes_tensor)
    
    torch_output_np = torch_output.cpu().numpy()
    print(f"    Output shape: {torch_output_np.shape}")
    
    # Compare
    print("\n[3] Comparing results...")
    
    diff = np.abs(pywt_output - torch_output_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"    Max difference:  {max_diff:.2e}")
    print(f"    Mean difference: {mean_diff:.2e}")
    
    print(f"\n    PyWavelets output range: [{pywt_output.min():.4f}, {pywt_output.max():.4f}]")
    print(f"    PyTorch output range:    [{torch_output_np.min():.4f}, {torch_output_np.max():.4f}]")
    
    TOLERANCE = 1e-5
    if max_diff < TOLERANCE:
        print(f"\n    ✓ PASS: Max difference {max_diff:.2e} < tolerance {TOLERANCE:.0e}")
    else:
        print(f"\n    ✗ FAIL: Max difference {max_diff:.2e} >= tolerance {TOLERANCE:.0e}")
    
    # Benchmark
    print("\n[4] Benchmarking...")
    
    start = time.time()
    for _ in range(10):
        for i in range(N_SPIKES):
            coeffs = pywt.wavedec(test_spikes[i], 'haar', level=LEVEL)
    pywt_time = (time.time() - start) / 10
    
    extractor_cpu = WaveletFeatureExtractor().to('cpu')
    spikes_cpu = torch.tensor(test_spikes, dtype=torch.float32)
    
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = extractor_cpu(spikes_cpu)
    torch_cpu_time = (time.time() - start) / 10
    
    if torch.cuda.is_available():
        extractor_gpu = WaveletFeatureExtractor().to('cuda')
        spikes_gpu = torch.tensor(test_spikes, dtype=torch.float32, device='cuda')
        
        with torch.no_grad():
            _ = extractor_gpu(spikes_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = extractor_gpu(spikes_gpu)
            torch.cuda.synchronize()
        torch_gpu_time = (time.time() - start) / 10
    else:
        torch_gpu_time = None
    
    print(f"\n    PyWavelets:   {pywt_time*1000:.2f} ms for {N_SPIKES} spikes")
    print(f"    PyTorch CPU:  {torch_cpu_time*1000:.2f} ms for {N_SPIKES} spikes")
    if torch_gpu_time:
        print(f"    PyTorch GPU:  {torch_gpu_time*1000:.2f} ms for {N_SPIKES} spikes")
        print(f"\n    Speedup (PyWavelets → PyTorch GPU): {pywt_time/torch_gpu_time:.1f}x")
    
    print("\n" + "=" * 60)
    print("C1 VALIDATION COMPLETE")
    print("=" * 60)