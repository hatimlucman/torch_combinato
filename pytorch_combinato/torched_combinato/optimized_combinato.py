"""
OptimizedCombinato - Hybrid GPU+CPU Pipeline (4.06x speedup)
============================================================
M1/M2: GPU batched (192 channels)
M3: CPU (avoids GPU overhead)
M4/M5: GPU
C1-C6: GPU (C3/SPC is CPU-bound anyway)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .modules.m1_preprocessor import Preprocessor
from .modules.m2_threshold_detector import ThresholdDetector
from .modules.m3_waveform_extractor import WaveformExtractor
from .modules.m4_cubic_upsampler import CubicUpsampler
from .modules.m5_peak_aligner import PeakAligner
from .modules.c1_wavelet_features import WaveletFeatureExtractor
from .modules.c2_feature_selector import FeatureSelector
from .modules.c3_spc_clusterer import SPCClusterer
from .modules.c4_cluster_definer import ClusterDefiner
from .modules.c5_template_matcher import TemplateMatcher
from .modules.c6_artifact_detector import ArtifactDetector


class OptimizedCombinato(nn.Module):
    def __init__(self, sample_rate=30000, spc_path='~/spc/cluster_linux64.exe', batch_size=192):
        super().__init__()
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        
        # GPU modules
        self.pre_gpu = Preprocessor(sample_rate=sample_rate).cuda()
        self.detector_gpu = ThresholdDetector(sample_rate=sample_rate).cuda()
        self.upsampler_gpu = CubicUpsampler().cuda()
        self.aligner_gpu = PeakAligner().cuda()
        self.c1_gpu = WaveletFeatureExtractor().cuda()
        self.c2_gpu = FeatureSelector().cuda()
        self.c6_gpu = ArtifactDetector().cuda()
        
        # CPU modules (M3 runs on CPU)
        self.pre_cpu = Preprocessor(sample_rate=sample_rate)
        self.extractor_cpu = WaveformExtractor(self.pre_cpu)
        
        # Clustering (C3 is CPU-bound external binary)
        self.c3 = SPCClusterer(cluster_path=os.path.expanduser(spc_path))
        self.c4 = ClusterDefiner()
        self.c5 = TemplateMatcher()

    def forward(self, raw_data, output_dir='/tmp/optimized_combinato'):
        n_samples, n_channels = raw_data.shape
        os.makedirs(output_dir, exist_ok=True)
        
        # M1: GPU BATCHED
        gpu_denoised = [None] * n_channels
        gpu_detected = [None] * n_channels
        
        n_batches = (n_channels + self.batch_size - 1) // self.batch_size
        
        for batch in range(n_batches):
            start_ch = batch * self.batch_size
            end_ch = min((batch + 1) * self.batch_size, n_channels)
            
            data_batch = torch.tensor(
                raw_data[:, start_ch:end_ch].T.astype(np.float64), 
                device='cuda'
            )
            
            with torch.no_grad():
                denoised_batch, detected_batch = self.pre_gpu(data_batch)
            
            for i, ch in enumerate(range(start_ch, end_ch)):
                gpu_denoised[ch] = denoised_batch[i].clone()
                gpu_detected[ch] = detected_batch[i].clone()
            
            del data_batch, denoised_batch, detected_batch
            torch.cuda.empty_cache()
        
        # M2: GPU
        neg_indices = {}
        for ch in range(n_channels):
            with torch.no_grad():
                _, neg_idx, _ = self.detector_gpu(gpu_detected[ch])
            if len(neg_idx) >= 15:
                neg_indices[ch] = neg_idx
        
        # M3: CPU
        spikes = {}
        for ch, neg_idx in neg_indices.items():
            denoised_cpu = gpu_denoised[ch].cpu()
            neg_idx_cpu = neg_idx.cpu() if isinstance(neg_idx, torch.Tensor) else neg_idx
            with torch.no_grad():
                spikes_raw, _, _ = self.extractor_cpu(denoised_cpu, neg_idx_cpu, 'neg')
            spikes[ch] = spikes_raw
        
        # M4/M5: GPU
        for ch in spikes:
            spikes[ch] = spikes[ch].cuda()
            with torch.no_grad():
                spikes[ch] = self.upsampler_gpu(spikes[ch])
                spikes[ch], _ = self.aligner_gpu(spikes[ch])
        
        # C1-C6: CLUSTERING
        results = {}
        for ch in spikes:
            spk = -spikes[ch].cpu().numpy()
            spk_gpu = torch.tensor(spk, dtype=torch.float64, device='cuda')
            
            folder = os.path.join(output_dir, f'ch{ch}')
            os.makedirs(folder, exist_ok=True)
            
            with torch.no_grad():
                features = self.c1_gpu(spk_gpu)
                features_sel, _ = self.c2_gpu(features)
            
            clu, tree = self.c3(features_sel.cpu(), folder, 'opt', 12345.0)
            sort_idx, _, _ = self.c4(clu, tree)
            
            sort_idx_np = sort_idx.numpy().astype(np.uint16).copy()
            match_idx = np.zeros(len(sort_idx_np), dtype=np.int8)
            self.c5(spk, sort_idx_np, match_idx)
            _, artifact_ids = self.c6_gpu(spk_gpu, sort_idx_np, sign='neg')
            
            results[ch] = {
                'spikes': spk,
                'clusters': sort_idx_np,
                'artifacts': artifact_ids
            }
        
        return results
