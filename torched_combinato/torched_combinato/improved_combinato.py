"""
ImprovedCombinato - Batched Extraction Pipeline
===============================================
Batches M1 (192 channels at once) for faster preprocessing.
On CPU: ~1.98x speedup
On GPU: ~0.71x (slower due to M3 per-channel overhead)
"""

import torch
import torch.nn as nn
import numpy as np
import os


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


class ImprovedCombinato(nn.Module):
    def __init__(self, sample_rate=30000, spc_path='~/spc/cluster_linux64.exe', device='cpu', batch_size=192):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        self.batch_size = batch_size
        
        self.pre = Preprocessor(sample_rate=sample_rate)
        self.detector = ThresholdDetector(sample_rate=sample_rate)
        self.extractor = WaveformExtractor(self.pre)
        self.upsampler = CubicUpsampler()
        self.aligner = PeakAligner()
        
        self.c1 = WaveletFeatureExtractor()
        self.c2 = FeatureSelector()
        self.c3 = SPCClusterer(cluster_path=os.path.expanduser(spc_path))
        self.c4 = ClusterDefiner()
        self.c5 = TemplateMatcher()
        self.c6 = ArtifactDetector()
        
        if device == 'cuda':
            self.pre = self.pre.cuda()
            self.detector = self.detector.cuda()
            self.extractor = self.extractor.cuda()
            self.upsampler = self.upsampler.cuda()
            self.aligner = self.aligner.cuda()
            self.c1 = self.c1.cuda()
            self.c2 = self.c2.cuda()
            self.c6 = self.c6.cuda()

    def forward(self, raw_data, output_dir='/tmp/improved_combinato'):
        n_samples, n_channels = raw_data.shape
        os.makedirs(output_dir, exist_ok=True)
        
        # M1: BATCHED
        denoised_all = [None] * n_channels
        detected_all = [None] * n_channels
        
        n_batches = (n_channels + self.batch_size - 1) // self.batch_size
        
        for batch in range(n_batches):
            start_ch = batch * self.batch_size
            end_ch = min((batch + 1) * self.batch_size, n_channels)
            
            data_batch = torch.tensor(
                raw_data[:, start_ch:end_ch].T.astype(np.float64),
                device=self.device
            )
            
            with torch.no_grad():
                denoised_batch, detected_batch = self.pre(data_batch)
            
            for i, ch in enumerate(range(start_ch, end_ch)):
                denoised_all[ch] = denoised_batch[i].clone()
                detected_all[ch] = detected_batch[i].clone()
            
            del data_batch, denoised_batch, detected_batch
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # M2-M5: PER-CHANNEL
        results = {}
        
        for ch in range(n_channels):
            with torch.no_grad():
                _, neg_idx, _ = self.detector(detected_all[ch])
            
            if len(neg_idx) < 15:
                continue
            
            folder = os.path.join(output_dir, f'ch{ch}')
            os.makedirs(folder, exist_ok=True)
            
            with torch.no_grad():
                spikes_raw, _, _ = self.extractor(denoised_all[ch], neg_idx, 'neg')
                spikes_up = self.upsampler(spikes_raw)
                spikes_aligned, _ = self.aligner(spikes_up)
            
            spk = -spikes_aligned.cpu().numpy()
            spk_t = torch.tensor(spk, dtype=torch.float64, device=self.device)
            
            with torch.no_grad():
                features = self.c1(spk_t)
                features_sel, _ = self.c2(features)
            
            clu, tree = self.c3(features_sel.cpu(), folder, 'imp', 12345.0)
            sort_idx, _, _ = self.c4(clu, tree)
            
            sort_idx_np = sort_idx.numpy().astype(np.uint16).copy()
            match_idx = np.zeros(len(sort_idx_np), dtype=np.int8)
            self.c5(spk, sort_idx_np, match_idx)
            _, artifact_ids = self.c6(spk_t, sort_idx_np, sign='neg')
            
            results[ch] = {
                'spikes': spk,
                'clusters': sort_idx_np,
                'artifacts': artifact_ids
            }
        
        return results
