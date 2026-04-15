"""
C6 - ArtifactDetector (Pure PyTorch - Fixed)
==============================================
Scores each cluster for artifact-like waveform properties.

PURE PYTORCH: No NumPy dependency.
Uses torch.unique() instead of np.unique().
Peak finding matches original logic exactly.
"""

import torch
from .block import Block

ARTIFACT_CRITERIA = {
    'maxima': 5,
    'maxima_1_2_ratio': 2,
    'max_min_ratio': 1.5,
    'sem': 4,
    'ptp': 1,
}
TOLERANCE = 10


class ArtifactDetector(Block):
    def __init__(self, criteria=None, tolerance=TOLERANCE):
        super().__init__()
        self.criteria = criteria or ARTIFACT_CRITERIA
        self.tolerance = tolerance

    def find_maxima_ratio(self, mean):
        """
        Find number of peaks and ratio of highest peaks.
        Matches original combinato logic exactly.
        
        Args:
            mean: [64] tensor - mean spike waveform
            
        Returns:
            num_peaks: int
            ratio: float
        """
        device = mean.device
        dtype = mean.dtype
        tolerance = self.tolerance
        
        # Find where signal goes up and down
        up = (mean[1:] > mean[:-1]).nonzero(as_tuple=True)[0] + 1
        down = (mean[:-1] > mean[1:]).nonzero(as_tuple=True)[0]
        
        # Intersection = peaks
        up_set = set(up.tolist())
        down_set = set(down.tolist())
        peak_indices = sorted(up_set & down_set)
        peak_indices.append(len(mean))  # append end like original
        
        # Exclude nearby peaks (within tolerance)
        if len(peak_indices) > 1:
            diffs = []
            for i in range(1, len(peak_indices)):
                diffs.append(peak_indices[i] - peak_indices[i-1])
            
            valid_peaks = []
            for i, d in enumerate(diffs):
                if d >= tolerance:
                    valid_peaks.append(peak_indices[i])
            num = len(valid_peaks)
        else:
            num = 0
        
        if num > 1:
            vals = mean[valid_peaks]
            sorted_vals, _ = torch.sort(vals)
            ratio = torch.abs(sorted_vals[-1] / sorted_vals[-2]).item()
        else:
            ratio = float('inf')
        
        return num, ratio

    def max_min_ratio(self, mean):
        """Ratio of maximum and minimum."""
        return torch.abs(mean.max() / mean.min()).item()

    def std_err_mean(self, data):
        """Standard error of mean."""
        return (data.std(dim=0).mean() / torch.sqrt(torch.tensor(data.shape[0], dtype=data.dtype, device=data.device))).item()

    def peak_to_peak(self, mean):
        """Peak to peak ratio in second half."""
        cut = mean.shape[0] // 2
        second_half = mean[cut:] - mean[0]
        ptp = (second_half.max() - second_half.min()) / mean.max()
        return ptp.item()

    def artifact_score(self, data):
        """
        Run all artifact tests on one cluster.
        
        Args:
            data: [K, 64] tensor - all spikes in cluster
            
        Returns:
            score: int (0-5)
            reasons: list of failed criteria
            mean: [64] tensor
        """
        mean = data.mean(dim=0)
        score = 0
        reasons = []

        num_peaks, peak_ratio = self.find_maxima_ratio(mean)
        ratio = self.max_min_ratio(mean)
        std_err = self.std_err_mean(data)
        ptp = self.peak_to_peak(mean)

        if num_peaks > self.criteria['maxima']:
            score += 1
            reasons.append('maxima')
        if peak_ratio < self.criteria['maxima_1_2_ratio']:
            score += 1
            reasons.append('maxima_1_2_ratio')
        if ratio < self.criteria['max_min_ratio']:
            score += 1
            reasons.append('max_min_ratio')
        if std_err > self.criteria['sem']:
            score += 1
            reasons.append('sem')
        if ptp > self.criteria['ptp']:
            score += 1
            reasons.append('ptp')

        return score, reasons, mean

    def forward(self, spikes, sort_idx, sign='pos'):
        """
        Score each cluster for artifact properties.
        
        Args:
            spikes: [N, 64] tensor of spike waveforms
            sort_idx: [N] array/tensor of cluster assignments
            sign: 'pos' or 'neg'
            
        Returns:
            artifact_scores: dict {cluster_id: score}
            artifact_ids: list of artifact cluster IDs
        """
        # Convert to tensor if needed
        if not isinstance(spikes, torch.Tensor):
            spikes = torch.tensor(spikes, dtype=torch.float32)
        
        device = spikes.device
        dtype = spikes.dtype
        
        # Convert sort_idx to tensor
        if not isinstance(sort_idx, torch.Tensor):
            sort_idx_t = torch.tensor(sort_idx, dtype=torch.long, device=device)
        else:
            sort_idx_t = sort_idx.to(device)
        
        # Invert if negative spikes
        if sign == 'neg':
            spikes = -spikes
        
        # Get unique cluster IDs using torch (not numpy!)
        unique_ids = torch.unique(sort_idx_t)
        unique_ids = unique_ids[unique_ids != 0]  # exclude cluster 0
        
        artifact_scores = {}
        artifact_ids = []

        # Loop over clusters (typically only 4-5, so loop is fine)
        for class_id in unique_ids:
            class_id_val = class_id.item()
            
            # Get spikes for this cluster
            class_mask = sort_idx_t == class_id
            class_spikes = spikes[class_mask]
            
            if class_spikes.shape[0] == 0:
                continue
            
            score, reasons, _ = self.artifact_score(class_spikes)
            artifact_scores[class_id_val] = score
            
            if score > 0:
                artifact_ids.append(class_id_val)

        return artifact_scores, artifact_ids


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    import sys
    sys.path.insert(0, '/home/juwayni/combinato')
    
    print("=" * 60)
    print("C6 ARTIFACT DETECTOR (FIXED) — VALIDATION")
    print("=" * 60)
    
    # Load real test data
    spikes = np.load('neg_spikes_ch174.npy')
    sort_idx = np.array([0]*732 + [1]*70 + [2]*305 + [3]*32 + [4]*23, dtype=np.uint16)
    
    print(f"\nSpikes: {spikes.shape}")
    print(f"Clusters: {np.unique(sort_idx)}")
    
    # Original
    print("\n=== ORIGINAL ===")
    from combinato.cluster.artifacts import find_artifacts
    _, artifact_ids_orig = find_artifacts(spikes, sort_idx.copy(), np.unique(sort_idx), invert=True)
    print(f"Artifact IDs: {artifact_ids_orig}")
    
    # PyTorch
    print("\n=== PYTORCH (FIXED) ===")
    detector = ArtifactDetector()
    spikes_t = torch.tensor(spikes, dtype=torch.float64)
    scores, artifact_ids_pt = detector(spikes_t, sort_idx, sign='neg')
    print(f"Scores: {scores}")
    print(f"Artifact IDs: {artifact_ids_pt}")
    
    # Compare
    print("\n=== COMPARISON ===")
    match = set(artifact_ids_orig) == set(artifact_ids_pt)
    print(f"Match: {'YES' if match else 'NO'}")
    
    print("\n" + "=" * 60)