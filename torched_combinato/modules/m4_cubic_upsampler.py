"""
M4 - CubicUpsampler (Pure PyTorch)
===================================
Upsamples each spike waveform by factor 3 using cubic spline interpolation.

PURE PYTORCH: No SciPy dependency at runtime. Fully GPU-compatible.

Uses precomputed interpolation weight matrix for single matrix multiplication.
Produces identical results to scipy.interpolate.make_interp_spline.
"""

import torch
import numpy as np
from .block import Block


def build_cubic_spline_matrix(n_input, n_output, factor):
    """
    Build the cubic spline interpolation weight matrix.
    
    Uses SciPy once at initialization to compute exact weights,
    then stores them for reuse.
    
    Args:
        n_input: Number of input samples (74)
        n_output: Number of output samples (220)
        factor: Upsampling factor (3)
    
    Returns:
        [n_output, n_input] weight matrix
    """
    from scipy.interpolate import make_interp_spline
    
    # Input positions: [0, 3, 6, 9, ..., 219]
    input_positions = np.arange(0, n_output, factor)
    
    # Output positions: [0, 1, 2, 3, ..., 219]
    output_positions = np.arange(n_output)
    
    # Build weight matrix by probing with unit vectors
    # If input is [1, 0, 0, 0, ...], output tells us weights for input[0]
    # If input is [0, 1, 0, 0, ...], output tells us weights for input[1]
    # etc.
    
    weights = np.zeros((n_output, n_input), dtype=np.float64)
    
    for i in range(n_input):
        # Create unit vector: all zeros except 1 at position i
        unit = np.zeros(n_input, dtype=np.float64)
        unit[i] = 1.0
        
        # Build spline and evaluate at all output positions
        spline = make_interp_spline(input_positions, unit, k=3)
        weights[:, i] = spline(output_positions)
    
    return weights


class CubicUpsampler(Block):
    """
    GPU-accelerated cubic spline upsampling.
    
    Upsamples waveforms via single matrix multiplication.
    Output is identical to scipy.interpolate.make_interp_spline.
    """
    
    def __init__(self, factor=3, input_size=74, output_size=220):
        super().__init__()
        self.factor = factor
        self.input_size = input_size
        self.output_size = output_size
        
        # Precompute the interpolation weight matrix
        interp_matrix = build_cubic_spline_matrix(input_size, output_size, factor)
        
        # Register as buffer (moves with model to GPU, not a parameter)
        self.register_buffer(
            'interp_matrix',
            torch.tensor(interp_matrix, dtype=torch.float64)
        )
    
    def forward(self, spikes):
        """
        Upsample spike waveforms.
        
        Args:
            spikes: [N, 74] tensor of spike waveforms
            
        Returns:
            [N, 220] tensor of upsampled waveforms
        """
        # Handle empty input
        if spikes.shape[0] == 0:
            return torch.zeros((0, self.output_size), dtype=spikes.dtype, device=spikes.device)
        
        # Cast matrix to match input dtype
        interp = self.interp_matrix.to(dtype=spikes.dtype)
        
        # Single matrix multiplication - fully GPU compatible
        # [N, 74] @ [74, 220].T -> [N, 220]
        return spikes @ interp.T


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == '__main__':
    from scipy.interpolate import make_interp_spline
    import time
    
    print("=" * 60)
    print("M4 CUBIC UPSAMPLER — VALIDATION")
    print("=" * 60)
    
    # Test parameters
    N_SPIKES = 1000
    INPUT_SIZE = 74
    OUTPUT_SIZE = 220
    FACTOR = 3
    
    # Generate random test data
    np.random.seed(42)
    test_spikes = np.random.randn(N_SPIKES, INPUT_SIZE).astype(np.float64)
    
    # =========================================================================
    # SciPy reference implementation
    # =========================================================================
    print("\n[1] Running SciPy reference...")
    
    input_positions = np.arange(0, OUTPUT_SIZE, FACTOR)
    output_positions = np.arange(OUTPUT_SIZE)
    
    scipy_output = np.zeros((N_SPIKES, OUTPUT_SIZE), dtype=np.float64)
    for i in range(N_SPIKES):
        spline = make_interp_spline(input_positions, test_spikes[i], k=3)
        scipy_output[i] = spline(output_positions)
    
    print(f"    Output shape: {scipy_output.shape}")
    
    # =========================================================================
    # Pure PyTorch implementation
    # =========================================================================
    print("\n[2] Running Pure PyTorch implementation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    upsampler = CubicUpsampler(factor=FACTOR, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to(device)
    spikes_tensor = torch.tensor(test_spikes, dtype=torch.float64, device=device)
    
    with torch.no_grad():
        torch_output = upsampler(spikes_tensor)
    
    torch_output_np = torch_output.cpu().numpy()
    print(f"    Output shape: {torch_output_np.shape}")
    
    # =========================================================================
    # Compare results
    # =========================================================================
    print("\n[3] Comparing results...")
    
    diff = np.abs(scipy_output - torch_output_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"    Max difference:  {max_diff:.2e}")
    print(f"    Mean difference: {mean_diff:.2e}")
    
    # Check output ranges
    print(f"\n    SciPy output range:   [{scipy_output.min():.4f}, {scipy_output.max():.4f}]")
    print(f"    PyTorch output range: [{torch_output_np.min():.4f}, {torch_output_np.max():.4f}]")
    
    # Validation threshold
    TOLERANCE = 1e-10
    if max_diff < TOLERANCE:
        print(f"\n    ✓ PASS: Max difference {max_diff:.2e} < tolerance {TOLERANCE:.0e}")
    else:
        print(f"\n    ✗ FAIL: Max difference {max_diff:.2e} >= tolerance {TOLERANCE:.0e}")
    
    # =========================================================================
    # Benchmark
    # =========================================================================
    print("\n[4] Benchmarking...")
    
    # SciPy timing
    start = time.time()
    for _ in range(10):
        for i in range(N_SPIKES):
            spline = make_interp_spline(input_positions, test_spikes[i], k=3)
            _ = spline(output_positions)
    scipy_time = (time.time() - start) / 10
    
    # PyTorch CPU timing
    upsampler_cpu = CubicUpsampler(factor=FACTOR, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to('cpu')
    spikes_cpu = torch.tensor(test_spikes, dtype=torch.float64)
    
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = upsampler_cpu(spikes_cpu)
    torch_cpu_time = (time.time() - start) / 10
    
    # PyTorch GPU timing (if available)
    if torch.cuda.is_available():
        upsampler_gpu = CubicUpsampler(factor=FACTOR, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to('cuda')
        spikes_gpu = torch.tensor(test_spikes, dtype=torch.float64, device='cuda')
        
        # Warmup
        with torch.no_grad():
            _ = upsampler_gpu(spikes_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = upsampler_gpu(spikes_gpu)
            torch.cuda.synchronize()
        torch_gpu_time = (time.time() - start) / 10
    else:
        torch_gpu_time = None
    
    print(f"\n    SciPy:        {scipy_time*1000:.2f} ms for {N_SPIKES} spikes")
    print(f"    PyTorch CPU:  {torch_cpu_time*1000:.2f} ms for {N_SPIKES} spikes")
    if torch_gpu_time:
        print(f"    PyTorch GPU:  {torch_gpu_time*1000:.2f} ms for {N_SPIKES} spikes")
        print(f"\n    Speedup (SciPy → PyTorch GPU): {scipy_time/torch_gpu_time:.1f}x")
    
    print("\n" + "=" * 60)
    print("M4 VALIDATION COMPLETE")
    print("=" * 60)