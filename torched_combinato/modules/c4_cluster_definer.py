"""
C4 - ClusterDefiner (Pure PyTorch - Optimized)
===============================================
Reads SPC temperature tree and extracts final cluster assignments.

PURE PYTORCH: No NumPy dependency. No .tolist() or Python sets.
Fully GPU-compatible with tensor-only operations.
"""

import torch
from .block import Block

MAX_CLUSTERS_PER_TEMP = 5
MIN_SPIKES_PER_CLUSTER = 15


class ClusterDefiner(Block):
    def __init__(self,
                 max_clusters_per_temp=MAX_CLUSTERS_PER_TEMP,
                 min_spikes=MIN_SPIKES_PER_CLUSTER):
        super().__init__()
        self.max_clusters_per_temp = max_clusters_per_temp
        self.min_spikes = min_spikes

    def find_relevant_tree_points(self, tree):
        """
        Find points in temperature tree where clusters should be selected.
        
        Pure tensor operations - no .tolist() or Python sets.
        
        Args:
            tree: [num_temps, num_cols] tensor
            
        Returns:
            Tensor of shape [M, 3] with columns (row, num_spikes, cluster_col)
        """
        device = tree.device
        dtype = tree.dtype
        results = []
        
        for shift in range(self.max_clusters_per_temp):
            col_idx = 5 + shift
            if col_idx >= tree.shape[1]:
                break
            
            col = tree[:, col_idx]
            
            # Find peaks using tensor ops
            # Rise: col[i] > col[i-1]
            rise = torch.zeros(col.shape[0], dtype=torch.bool, device=device)
            rise[1:] = col[1:] > col[:-1]
            
            # Fall: col[i] >= col[i+1]
            fall = torch.zeros(col.shape[0], dtype=torch.bool, device=device)
            fall[:-1] = col[:-1] >= col[1:]
            
            # Peaks: rise AND fall (intersection via logical AND)
            peaks_mask = rise & fall
            
            # Special case: falling at beginning (high amplitude clusters)
            if col.shape[0] > 1 and col[0] >= col[1]:
                peaks_mask[1] = True
            
            # Get peak indices
            peak_indices = torch.where(peaks_mask)[0]
            
            # Filter by min_spikes
            for peak_idx in peak_indices:
                nspk = col[peak_idx]
                if nspk >= self.min_spikes:
                    results.append((peak_idx.item(), nspk.item(), shift + 1))
        
        return results

    def forward(self, clu, tree):
        """
        Extract cluster assignments from SPC results.
        
        Args:
            clu: [num_temps, 2 + num_spikes] tensor - cluster labels at each temperature
            tree: [num_temps, num_cols] tensor - cluster sizes at each temperature
            
        Returns:
            idx: [num_spikes] tensor of cluster assignments
            tree: unchanged
            used_points: list of (row, col, type) for visualization
        """
        # Convert to tensor if numpy
        if not isinstance(clu, torch.Tensor):
            clu = torch.tensor(clu, dtype=torch.float32)
        if not isinstance(tree, torch.Tensor):
            tree = torch.tensor(tree, dtype=torch.float32)
        
        device = clu.device
        
        relevant_rows = self.find_relevant_tree_points(tree)
        num_features = clu.shape[1] - 2
        
        idx = torch.zeros(num_features, dtype=torch.uint8, device=device)
        used_points = []
        current_id = 2
        max_row = 0

        for row, _, col in relevant_rows:
            # Pure tensor operations
            row_labels = clu[row, 2:]
            row_idx = (row_labels == col) & (idx == 0)
            
            if row_idx.any():
                idx[row_idx] = current_id
                current_id += 1
                p_type = 'k'
                max_row = max(max_row, row)
            else:
                p_type = 'r'
            used_points.append((row, col + 4, p_type))

        if len(used_points):
            row_idx = clu[max_row, 2:] == 0
            used_points.append((max_row, 4, 'm'))
        else:
            row_idx = clu[1, 2:] == 0
            used_points.append((1, 4, 'c'))

        idx[row_idx] = 1
        
        return idx, tree, used_points


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    import time
    
    print("=" * 60)
    print("C4 CLUSTER DEFINER (OPTIMIZED) — VALIDATION")
    print("=" * 60)
    
    # Create mock SPC output data
    np.random.seed(42)
    
    NUM_TEMPS = 21
    NUM_SPIKES = 1162
    
    # Mock tree
    tree_np = np.zeros((NUM_TEMPS, 10), dtype=np.float32)
    tree_np[:, 0] = np.arange(NUM_TEMPS)
    tree_np[:, 1] = np.linspace(0, 0.2, NUM_TEMPS)
    tree_np[:, 4] = np.linspace(NUM_SPIKES, 100, NUM_TEMPS)
    tree_np[:, 5] = np.linspace(0, 200, NUM_TEMPS)
    tree_np[10, 5] = 250
    tree_np[:, 6] = np.linspace(0, 100, NUM_TEMPS)
    tree_np[8, 6] = 120
    
    # Mock clu
    clu_np = np.zeros((NUM_TEMPS, 2 + NUM_SPIKES), dtype=np.float32)
    clu_np[:, 0] = np.arange(NUM_TEMPS)
    clu_np[:, 1] = np.linspace(0, 0.2, NUM_TEMPS)
    for t in range(NUM_TEMPS):
        clu_np[t, 2:] = np.random.randint(0, 3, NUM_SPIKES)
    
    # CPU test
    print("\n[1] Running on CPU...")
    definer_cpu = ClusterDefiner()
    
    start = time.time()
    for _ in range(100):
        idx_cpu, _, _ = definer_cpu(clu_np.copy(), tree_np.copy())
    cpu_time = (time.time() - start) / 100
    
    idx_cpu_np = idx_cpu.numpy()
    print(f"    Output shape: {idx_cpu_np.shape}")
    print(f"    Unique clusters: {np.unique(idx_cpu_np)}")
    print(f"    Time: {cpu_time*1000:.3f} ms")
    
    # GPU test
    if torch.cuda.is_available():
        print("\n[2] Running on GPU...")
        device = torch.device('cuda')
        
        definer_gpu = ClusterDefiner().to(device)
        clu_tensor = torch.tensor(clu_np, device=device)
        tree_tensor = torch.tensor(tree_np, device=device)
        
        # Warmup
        _ = definer_gpu(clu_tensor.clone(), tree_tensor.clone())
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            idx_gpu, _, _ = definer_gpu(clu_tensor.clone(), tree_tensor.clone())
            torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 100
        
        idx_gpu_np = idx_gpu.cpu().numpy()
        print(f"    Output shape: {idx_gpu_np.shape}")
        print(f"    Unique clusters: {np.unique(idx_gpu_np)}")
        print(f"    Time: {gpu_time*1000:.3f} ms")
        
        # Compare
        print("\n[3] Comparing...")
        match = np.array_equal(idx_cpu_np, idx_gpu_np)
        print(f"    Match: {'YES' if match else 'NO'}")
    
    print("\n" + "=" * 60)