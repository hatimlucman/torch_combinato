# m6_pipeline.py

import torch
import numpy as np

from modules.block import Block

from modules.m1_preprocessor import Preprocessor
from modules.m2_threshold_detector import ThresholdDetector
from modules.m3_waveform_extractor import WaveformExtractor
from modules.m4_cubic_upsampler import CubicUpsampler
from modules.m5_peak_aligner import PeakAligner


class CombinatoExtractor(Block):
    def __init__(self,
                 sample_rate=30000,
                 threshold_factor=5,
                 max_spike_duration=0.0015,
                 upsampling_factor=3,
                 indices_per_spike=64,
                 index_maximum=19,
                 border_pad=5):
        super().__init__()

        self.pre       = Preprocessor(sample_rate=sample_rate)
        self.detector  = ThresholdDetector(sample_rate=sample_rate,
                                           threshold_factor=threshold_factor,
                                           max_spike_duration=max_spike_duration)
        self.extractor = WaveformExtractor(self.pre,
                                           indices_per_spike=indices_per_spike,
                                           index_maximum=index_maximum,
                                           border_pad=border_pad)
        self.upsampler = CubicUpsampler(factor=upsampling_factor)
        self.aligner   = PeakAligner(factor=upsampling_factor,
                                     indices_per_spike=indices_per_spike,
                                     index_maximum=index_maximum,
                                     border_pad=border_pad)

    def _process_one(self, data_denoised, peak_indices, atimes, sign):
        if len(peak_indices) == 0:
            return torch.zeros((0, 64)), torch.zeros(0)

        spikes, valid_mask, _ = self.extractor(data_denoised, peak_indices, sign)

        valid_indices = peak_indices[valid_mask]
        times = atimes[valid_indices]

        if len(spikes) == 0:
            return torch.zeros((0, 64)), torch.zeros(0)

        spikes_up = self.upsampler(spikes)
        spikes_final, removed_mask = self.aligner(spikes_up)

        times_final = times[~removed_mask]

        if sign == 'neg':
            spikes_final = spikes_final * -1

        return spikes_final, times_final

    def forward(self, signal, atimes):
        data_denoised, data_detected = self.pre(signal)

        pos_idx, neg_idx, threshold = self.detector(data_detected)

        pos_spikes, pos_times = self._process_one(
            data_denoised, pos_idx, atimes, 'pos'
        )

        neg_spikes, neg_times = self._process_one(
            data_denoised, neg_idx, atimes, 'neg'
        )

        return {
            "pos_spikes": pos_spikes,
            "pos_times": pos_times,
            "neg_spikes": neg_spikes,
            "neg_times": neg_times,
            "threshold": threshold
        }


def save_to_h5(result, output_path):
    import tables

    f = tables.open_file(output_path, 'w')

    f.create_group('/', 'pos', 'positive spikes')
    f.create_group('/', 'neg', 'negative spikes')

    for sign in ['pos', 'neg']:
        f.create_earray(f'/{sign}', 'spikes', tables.Float32Atom(), (0, 64))
        f.create_earray(f'/{sign}', 'times', tables.FloatAtom(), (0,))

    f.create_earray('/', 'thr', tables.FloatAtom(), (0, 3))

    if len(result['pos_spikes']) > 0:
        f.root.pos.spikes.append(result['pos_spikes'].cpu().numpy())
        f.root.pos.times.append(result['pos_times'].cpu().numpy())

    if len(result['neg_spikes']) > 0:
        f.root.neg.spikes.append(result['neg_spikes'].cpu().numpy())
        f.root.neg.times.append(result['neg_times'].cpu().numpy())

    thr = result['threshold'].item()
    f.root.thr.append(np.array([[0, thr * 100, thr / 5 * 0.6745]]))

    f.close()
    print(f"Saved to {output_path}")