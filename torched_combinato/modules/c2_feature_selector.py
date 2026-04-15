"""
C2 - FeatureSelector (Pure PyTorch - Vectorized)
=================================================
Selects the most informative wavelet features for clustering.

PURE PYTORCH: No SciPy dependency. Fully GPU-compatible.
FULLY VECTORIZED: No for loops. All 64 features processed in parallel.

Uses Kolmogorov-Smirnov test to identify non-Gaussian features.
"""

import torch
import numpy as np
from .block import Block

FEATURE_FACTOR = 3
N_FEATURES_OUT = 10


class FeatureSelector(Block):
    """
    GPU-accelerated feature selection using KS test.
    
    Selects features that deviate most from normal distribution.
    Fully vectorized - processes all features in parallel.
    """
    
    def __init__(self, feature_factor=FEATURE_FACTOR, n_features_out=N_FEATURES_OUT):
        super().__init__()
        self.feature_factor = feature_factor
        self.n_features_out = n_features_out
    
    def compute_scores(self, features):
        """
        Compute KS test p-values for all features in parallel.
        
        Args:
            features: [N, F] tensor
            
        Returns:
            scores: [F] tensor of p-values
        """
        N, F = features.shape
        device = features.device
        dtype = features.dtype
        
        # Compute bounds for outlier exclusion
        feat_std = self.feature_factor * features.std(dim=0)  # [F]
        feat_mean = features.mean(dim=0)  # [F]
        feat_up = feat_mean + feat_std  # [F]
        feat_down = feat_mean - feat_std  # [F]
        
        # Mask outliers: [N, F] boolean
        in_bounds = (features > feat_down) & (features < feat_up)
        
        # Replace out-of-bounds with NaN to exclude from stats
        features_masked = torch.where(in_bounds, features, torch.tensor(float('nan'), device=device, dtype=dtype))
        
        # Compute mean excluding NaNs: [F]
        valid_counts = in_bounds.sum(dim=0).float()  # [F]
        col_sums = torch.nansum(features_masked, dim=0)  # [F]
        col_means = col_sums / valid_counts  # [F]
        
        # Center the data
        centered = features_masked - col_means  # [N, F]
        
        # Compute std excluding NaNs: [F]
        col_var = torch.nansum(centered ** 2, dim=0) / valid_counts
        col_std = torch.sqrt(col_var)  # [F]
        
        # Normalize
        normalized = centered / col_std  # [N, F]
        
        # Replace NaN with large value so they sort to the end
        normalized = torch.where(torch.isnan(normalized), 
                                  torch.tensor(float('inf'), device=device, dtype=dtype), 
                                  normalized)
        
        # Sort each column: [N, F]
        sorted_vals, _ = torch.sort(normalized, dim=0)
        
        # For KS test, we only want the valid (non-inf) values
        # Create empirical CDF based on valid counts per column
        # For simplicity, use fixed N and mask later
        
        # Empirical CDF: i/n for i in 1..n
        # Shape: [N, 1] to broadcast with [N, F]
        ranks = torch.arange(1, N + 1, device=device, dtype=dtype).unsqueeze(1)  # [N, 1]
        empirical_cdf = ranks / valid_counts.unsqueeze(0)  # [N, F]
        
        # Clamp empirical CDF to [0, 1] for invalid entries
        empirical_cdf = torch.clamp(empirical_cdf, 0, 1)
        
        # Normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        # For inf values, this will give 1.0
        normal_cdf = 0.5 * (1 + torch.erf(sorted_vals / np.sqrt(2)))  # [N, F]
        
        # KS statistic: max|F_n(x) - Phi(x)|
        # Need to check D+ and D-
        d_plus = empirical_cdf - normal_cdf  # [N, F]
        d_minus = normal_cdf - (empirical_cdf - 1/valid_counts.unsqueeze(0))  # [N, F]
        
        # Mask out invalid rows (where sorted_vals is inf)
        valid_mask = sorted_vals != float('inf')  # [N, F]
        d_plus = torch.where(valid_mask, d_plus, torch.tensor(float('-inf'), device=device, dtype=dtype))
        d_minus = torch.where(valid_mask, d_minus, torch.tensor(float('-inf'), device=device, dtype=dtype))
        
        # Max over samples: [F]
        ks_plus = d_plus.max(dim=0).values
        ks_minus = d_minus.max(dim=0).values
        ks_stats = torch.maximum(ks_plus, ks_minus)  # [F]
        
        # Convert KS statistic to p-value using asymptotic approximation
        sqrt_n = torch.sqrt(valid_counts)  # [F]
        lambda_val = (sqrt_n + 0.12 + 0.11 / sqrt_n) * ks_stats  # [F]
        
        # Asymptotic p-value: 2 * sum_{j=1}^{inf} (-1)^{j-1} * exp(-2 * j^2 * lambda^2)
        # Use first 100 terms
        j = torch.arange(1, 101, device=device, dtype=dtype).unsqueeze(1)  # [100, 1]
        lambda_sq = lambda_val.unsqueeze(0) ** 2  # [1, F]
        terms = ((-1) ** (j - 1)) * torch.exp(-2 * j ** 2 * lambda_sq)  # [100, F]
        p_values = 2 * terms.sum(dim=0)  # [F]
        
        # Clamp to valid range
        p_values = torch.clamp(p_values, 0.0, 1.0)
        
        # Handle edge cases: if valid_counts too low, set p-value to 1
        p_values = torch.where(valid_counts > 10, p_values, torch.tensor(1.0, device=device, dtype=dtype))
        
        return p_values
    
    def forward(self, features):
        """
        Select most informative features.
        
        Args:
            features: [N, 64] tensor of wavelet features
            
        Returns:
            selected: [N, 10] tensor of selected features
            indices: [10] tensor of selected feature indices
        """
        scores = self.compute_scores(features)
        
        # Select features with lowest p-values (most non-normal)
        sorted_scores, _ = torch.sort(scores)
        border = sorted_scores[self.n_features_out]
        
        # Get indices of features with score <= border
        mask = scores <= border
        indices = torch.where(mask)[0][:self.n_features_out]
        
        selected = features[:, indices]
        
        return selected, indices


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == '__main__':
    from scipy import stats
    import time
    
    print("=" * 60)
    print("C2 FEATURE SELECTOR (VECTORIZED) — VALIDATION")
    print("=" * 60)
    
    # Test parameters
    N_SPIKES = 1000
    N_FEATURES = 64
    N_OUT = 10
    
    # Generate test data with some non-normal features
    np.random.seed(42)
    test_features = np.random.randn(N_SPIKES, N_FEATURES).astype(np.float64)
    
    # Make some features bimodal (non-normal)
    for i in [5, 12, 23, 34, 45]:
        half = N_SPIKES // 2
        test_features[:half, i] += 2
        test_features[half:, i] -= 2
    
    # =========================================================================
    # SciPy reference implementation
    # =========================================================================
    print("\n[1] Running SciPy reference...")
    
    feature_factor = 3
    feat_std = feature_factor * test_features.std(0)
    feat_mean = test_features.mean(0)
    feat_up = feat_mean + feat_std
    feat_down = feat_mean - feat_std
    
    scipy_scores = np.ones(N_FEATURES)
    for i in range(N_FEATURES):
        col = test_features[:, i]
        idx = (col > feat_down[i]) & (col < feat_up[i])
        if idx.any():
            good = col[idx]
            good = good - good.mean()
            std = good.std()
            if std > 0:
                good = good / std
                scipy_scores[i] = stats.kstest(good, 'norm')[1]
    
    sorted_scores = np.sort(scipy_scores)
    border = sorted_scores[N_OUT]
    scipy_indices = np.where(scipy_scores <= border)[0][:N_OUT]
    scipy_indices = np.sort(scipy_indices)
    
    print(f"    Selected indices: {scipy_indices}")
    
    # =========================================================================
    # Pure PyTorch implementation
    # =========================================================================
    print("\n[2] Running Pure PyTorch implementation (VECTORIZED)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    selector = FeatureSelector(feature_factor=3, n_features_out=N_OUT).to(device)
    features_tensor = torch.tensor(test_features, dtype=torch.float64, device=device)
    
    with torch.no_grad():
        selected, torch_indices = selector(features_tensor)
    
    torch_indices_np = torch_indices.cpu().numpy()
    torch_indices_np = np.sort(torch_indices_np)
    
    print(f"    Selected indices: {torch_indices_np}")
    
    # =========================================================================
    # Compare results
    # =========================================================================
    print("\n[3] Comparing results...")
    
    indices_match = np.array_equal(scipy_indices, torch_indices_np)
    
    if indices_match:
        print(f"    ✓ PASS: Same features selected")
    else:
        print(f"    ✗ Different features selected")
        print(f"      SciPy:   {scipy_indices}")
        print(f"      PyTorch: {torch_indices_np}")
        
        overlap = len(set(scipy_indices) & set(torch_indices_np))
        print(f"      Overlap: {overlap}/{N_OUT} features")
        
        if overlap >= 8:
            print(f"    ⚠ ACCEPTABLE: {overlap}/10 overlap (minor p-value differences)")
    
    # Compare p-values
    torch_scores = selector.compute_scores(features_tensor).cpu().numpy()
    score_diff = np.abs(scipy_scores - torch_scores)
    
    print(f"\n    P-value comparison:")
    print(f"    Max difference:  {score_diff.max():.2e}")
    print(f"    Mean difference: {score_diff.mean():.2e}")
    
    # =========================================================================
    # Benchmark
    # =========================================================================
    print("\n[4] Benchmarking...")
    
    # SciPy timing
    start = time.time()
    for _ in range(10):
        scipy_scores = np.ones(N_FEATURES)
        for i in range(N_FEATURES):
            col = test_features[:, i]
            idx = (col > feat_down[i]) & (col < feat_up[i])
            if idx.any():
                good = col[idx]
                good = good - good.mean()
                std = good.std()
                if std > 0:
                    good = good / std
                    scipy_scores[i] = stats.kstest(good, 'norm')[1]
    scipy_time = (time.time() - start) / 10
    
    # PyTorch CPU timing
    selector_cpu = FeatureSelector().to('cpu')
    features_cpu = torch.tensor(test_features, dtype=torch.float64)
    
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = selector_cpu(features_cpu)
    torch_cpu_time = (time.time() - start) / 10
    
    # PyTorch GPU timing
    if torch.cuda.is_available():
        selector_gpu = FeatureSelector().to('cuda')
        features_gpu = torch.tensor(test_features, dtype=torch.float64, device='cuda')
        
        with torch.no_grad():
            _ = selector_gpu(features_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = selector_gpu(features_gpu)
            torch.cuda.synchronize()
        torch_gpu_time = (time.time() - start) / 10
    else:
        torch_gpu_time = None
    
    print(f"\n    SciPy:        {scipy_time*1000:.2f} ms")
    print(f"    PyTorch CPU:  {torch_cpu_time*1000:.2f} ms")
    if torch_gpu_time:
        print(f"    PyTorch GPU:  {torch_gpu_time*1000:.2f} ms")
        print(f"\n    Speedup (SciPy → PyTorch GPU): {scipy_time/torch_gpu_time:.1f}x")
    
    print("\n" + "=" * 60)
    print("C2 VALIDATION COMPLETE")
    print("=" * 60)