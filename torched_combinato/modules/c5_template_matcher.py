"""
C5 - TemplateMatcher (Pure PyTorch - Optimized)
================================================
Assigns unmatched spikes to nearest cluster template via euclidean distance.

PURE PYTORCH: No NumPy dependency. 
VECTORIZED: Uses one-hot encoding + matmul for cluster means.
Fully GPU-compatible.
"""

import torch
from .block import Block

CLID_UNMATCHED = 0
FIRST_MATCH_FACTOR = 0.75
FIRST_MATCH_MAX_DIST = 4
EXCLUDE_VARIABLE_CLUSTERS = True


class TemplateMatcher(Block):
    def __init__(self,
                 first_match_factor=FIRST_MATCH_FACTOR,
                 first_match_max_dist=FIRST_MATCH_MAX_DIST,
                 exclude_variable_clusters=EXCLUDE_VARIABLE_CLUSTERS,
                 clid_unmatched=CLID_UNMATCHED):
        super().__init__()
        self.first_match_factor = first_match_factor
        self.first_match_max_dist = first_match_max_dist
        self.exclude_variable_clusters = exclude_variable_clusters
        self.clid_unmatched = clid_unmatched

    def get_means_vectorized(self, sort_idx, spikes):
        """
        Compute mean waveform and std for each cluster using vectorized ops.
        
        Uses one-hot encoding + matmul instead of loop.
        
        Args:
            sort_idx: [N] tensor of cluster IDs
            spikes: [N, 64] tensor of spike waveforms
            
        Returns:
            ids: [K] tensor of cluster IDs (excluding unmatched)
            means: [K, 64] tensor of mean waveforms
            stds: [K] tensor of std values
        """
        device = spikes.device
        dtype = spikes.dtype
        
        # Get unique cluster IDs (excluding unmatched)
        unique_ids = torch.unique(sort_idx)
        unique_ids = unique_ids[unique_ids != self.clid_unmatched]
        
        if len(unique_ids) == 0:
            return (torch.tensor([], device=device, dtype=torch.long),
                    torch.zeros((0, spikes.shape[1]), device=device, dtype=dtype),
                    torch.tensor([], device=device, dtype=dtype))
        
        K = len(unique_ids)
        N = spikes.shape[0]
        
        # One-hot encode: [N, K]
        # sort_idx[:, None] == unique_ids[None, :] gives [N, K] bool
        one_hot = (sort_idx.unsqueeze(1) == unique_ids.unsqueeze(0)).to(dtype)
        
        # Counts per cluster: [K]
        counts = one_hot.sum(dim=0)
        
        # Filter out empty clusters
        valid = counts > 0
        if not valid.all():
            unique_ids = unique_ids[valid]
            one_hot = one_hot[:, valid]
            counts = counts[valid]
            K = len(unique_ids)
        
        if K == 0:
            return (torch.tensor([], device=device, dtype=torch.long),
                    torch.zeros((0, spikes.shape[1]), device=device, dtype=dtype),
                    torch.tensor([], device=device, dtype=dtype))
        
        # Means: [K, 64] = [K, N] @ [N, 64]
        means = (one_hot.T @ spikes) / counts.unsqueeze(1)
        
        # Compute variance for stds
        # For each cluster k: var = sum((x - mean_k)^2) / count_k
        # Expand means to match spikes: [N, K, 64]
        # But more efficient: compute sum of squares directly
        
        # Sum of squares: [K, 64]
        sum_sq = (one_hot.T @ (spikes ** 2))
        
        # Variance: E[X^2] - E[X]^2
        variance = sum_sq / counts.unsqueeze(1) - means ** 2
        
        # Std: sqrt(sum of variances across features)
        stds = torch.sqrt(variance.sum(dim=1).clamp(min=1e-10))
        
        return unique_ids, means, stds

    def distances_euclidean(self, spikes, templates):
        """
        Compute euclidean distances between spikes and templates.
        
        Args:
            spikes: [M, 64] tensor
            templates: [K, 64] tensor
            
        Returns:
            distances: [M, K] tensor
        """
        # [M, 1, 64] - [1, K, 64] -> [M, K, 64]
        diff = spikes.unsqueeze(1) - templates.unsqueeze(0)
        return torch.sqrt((diff ** 2).sum(dim=2))

    def forward(self, spikes, sort_idx, match_idx, factor=None):
        """
        Assign unmatched spikes to nearest cluster template.
        
        Args:
            spikes: [N, 64] tensor or numpy array
            sort_idx: [N] tensor or numpy array (modified in-place)
            match_idx: [N] tensor or numpy array (modified in-place)
            factor: match factor (default: self.first_match_factor)
        """
        if factor is None:
            factor = self.first_match_factor

        # Convert inputs to tensors
        if not isinstance(spikes, torch.Tensor):
            spikes = torch.tensor(spikes, dtype=torch.float32)
        device = spikes.device
        dtype = spikes.dtype
        
        # Handle sort_idx
        sort_idx_is_numpy = not isinstance(sort_idx, torch.Tensor)
        if sort_idx_is_numpy:
            sort_idx_np = sort_idx
            sort_idx_t = torch.tensor(sort_idx, dtype=torch.long, device=device)
        else:
            sort_idx_t = sort_idx.to(device)
            sort_idx_np = None
        
        # Handle match_idx
        match_idx_is_numpy = not isinstance(match_idx, torch.Tensor)
        if match_idx_is_numpy:
            match_idx_np = match_idx
        
        num_samples = spikes.shape[1]
        
        # Find unmatched
        unmatched_mask = sort_idx_t == self.clid_unmatched
        if not unmatched_mask.any():
            return
        
        # Get cluster stats
        matched_mask = ~unmatched_mask
        if not matched_mask.any():
            return
            
        ids, means, stds = self.get_means_vectorized(sort_idx_t[matched_mask], spikes[matched_mask])
        
        if len(ids) == 0:
            return

        # Exclude variable clusters
        if self.exclude_variable_clusters and len(stds) > 0:
            median_std = torch.median(stds)
            valid = stds <= 3 * median_std
            if not valid.all():
                ids = ids[valid]
                means = means[valid]
                stds = stds[valid]
        
        if len(ids) == 0:
            return

        # Get unmatched spikes
        unmatched_spikes = spikes[unmatched_mask]
        
        # Compute distances: [M, K]
        all_distances = self.distances_euclidean(unmatched_spikes, means)
        
        # Mask distances > factor * std
        all_distances[all_distances > factor * stds.unsqueeze(0)] = float('inf')
        
        # Find minimums
        min_vals, minimizers_idx = all_distances.min(dim=1)
        minimizers = ids[minimizers_idx]
        
        # Apply max distance threshold
        minimizers[min_vals >= self.first_match_max_dist * num_samples] = self.clid_unmatched
        
        # Update sort_idx and match_idx
        if sort_idx_is_numpy:
            # Update numpy arrays directly
            unmatched_indices = unmatched_mask.cpu().numpy().nonzero()[0]
            sort_idx_np[unmatched_indices] = minimizers.cpu().numpy()
            match_idx_np[unmatched_indices] = minimizers.cpu().numpy()
        else:
            # Update tensors
            sort_idx[unmatched_mask] = minimizers
            match_idx[unmatched_mask] = minimizers.to(match_idx.dtype)


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    import time
    
    print("=" * 60)
    print("C5 TEMPLATE MATCHER (OPTIMIZED) — VALIDATION")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    
    NUM_SPIKES = 1162
    SPIKE_LEN = 64
    NUM_CLUSTERS = 5
    
    # Generate spikes with cluster structure
    spikes_np = np.random.randn(NUM_SPIKES, SPIKE_LEN).astype(np.float32)
    
    # Create templates for each cluster
    templates = []
    for i in range(NUM_CLUSTERS):
        template = np.random.randn(SPIKE_LEN).astype(np.float32) * 2
        templates.append(template)
    
    # Assign clusters (with some unmatched as cluster 0)
    sort_idx_np = np.zeros(NUM_SPIKES, dtype=np.uint16)
    for i in range(NUM_SPIKES):
        if i < 700:  # 700 unmatched
            sort_idx_np[i] = 0
        else:
            clid = (i % (NUM_CLUSTERS - 1)) + 1
            sort_idx_np[i] = clid
            spikes_np[i] += templates[clid]
    
    # =========================================================================
    # Test CPU
    # =========================================================================
    print("\n[1] Running on CPU...")
    
    matcher = TemplateMatcher()
    spikes_cpu = torch.tensor(spikes_np)
    sort_idx_test = sort_idx_np.copy()
    match_idx_test = np.zeros(NUM_SPIKES, dtype=np.int8)
    
    start = time.time()
    for _ in range(100):
        sort_idx_run = sort_idx_np.copy()
        match_idx_run = np.zeros(NUM_SPIKES, dtype=np.int8)
        matcher(spikes_cpu, sort_idx_run, match_idx_run)
    cpu_time = (time.time() - start) / 100
    
    # Run once for results
    sort_idx_cpu = sort_idx_np.copy()
    match_idx_cpu = np.zeros(NUM_SPIKES, dtype=np.int8)
    matcher(spikes_cpu, sort_idx_cpu, match_idx_cpu)
    
    unique, counts = np.unique(sort_idx_cpu, return_counts=True)
    print(f"    Clusters: {dict(zip(unique, counts))}")
    print(f"    Time: {cpu_time*1000:.3f} ms")
    
    # =========================================================================
    # Test GPU
    # =========================================================================
    if torch.cuda.is_available():
        print("\n[2] Running on GPU...")
        device = torch.device('cuda')
        
        matcher_gpu = TemplateMatcher().to(device)
        spikes_gpu = torch.tensor(spikes_np, device=device)
        
        # Warmup
        sort_idx_warm = sort_idx_np.copy()
        match_idx_warm = np.zeros(NUM_SPIKES, dtype=np.int8)
        matcher_gpu(spikes_gpu, sort_idx_warm, match_idx_warm)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            sort_idx_run = sort_idx_np.copy()
            match_idx_run = np.zeros(NUM_SPIKES, dtype=np.int8)
            matcher_gpu(spikes_gpu, sort_idx_run, match_idx_run)
            torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 100
        
        # Run once for results
        sort_idx_gpu = sort_idx_np.copy()
        match_idx_gpu = np.zeros(NUM_SPIKES, dtype=np.int8)
        matcher_gpu(spikes_gpu, sort_idx_gpu, match_idx_gpu)
        
        unique, counts = np.unique(sort_idx_gpu, return_counts=True)
        print(f"    Clusters: {dict(zip(unique, counts))}")
        print(f"    Time: {gpu_time*1000:.3f} ms")
        
        # Compare
        print("\n[3] Comparing...")
        match = np.array_equal(sort_idx_cpu, sort_idx_gpu)
        print(f"    Match: {'YES' if match else 'NO'}")
        if not match:
            diff = (sort_idx_cpu != sort_idx_gpu).sum()
            print(f"    Differences: {diff}/{NUM_SPIKES}")
    
    print("\n" + "=" * 60)