# c7_combinato_sorter.py

import os
import torch
import numpy as np

from modules.block import Block

from modules.c1_wavelet_features import WaveletFeatureExtractor
from modules.c2_feature_selector import FeatureSelector
from modules.c3_spc_clusterer import SPCClusterer
from modules.c4_cluster_definer import ClusterDefiner
from modules.c5_template_matcher import TemplateMatcher
from modules.c5b_total_matcher import TotalMatcher
from modules.c6_artifact_detector import ArtifactDetector

MIN_SPIKES_FOR_SPC = 15


class CombinatorSorter(Block):
    def __init__(self, cluster_path, seed=12345.0):
        super().__init__()
        self.seed = seed
        self.cluster_path = cluster_path

        self.c1  = WaveletFeatureExtractor()
        self.c2  = FeatureSelector()
        self.c3  = SPCClusterer(cluster_path=cluster_path)
        self.c4  = ClusterDefiner()
        self.c5  = TemplateMatcher()
        self.c5b = TotalMatcher()
        self.c6  = ArtifactDetector()

    def _cluster(self, spikes, folder):
        os.makedirs(folder, exist_ok=True)

        # C1
        features = self.c1(spikes)

        # C2
        features_sel, _ = self.c2(features)

        # C3
        clu, tree = self.c3(features_sel, folder, "run", self.seed)

        # C4
        sort_idx, _, _ = self.c4(clu, tree)

        return sort_idx

    def _post_process(self, spikes, sort_idx):
        N = len(sort_idx)

        match_idx = np.zeros(N, dtype=np.int8)
        distance  = np.zeros(N, dtype=np.float32)

        # C5
        self.c5(spikes, sort_idx, match_idx)

        # C5b
        sort_idx, match_idx, distance = self.c5b(
            spikes, sort_idx, match_idx, distance
        )

        # C6
        artifact_scores, artifact_ids = self.c6(spikes, sort_idx)

        return sort_idx, match_idx, distance, artifact_ids

    def sort_one(self, spikes, folder, sign="pos"):
        if isinstance(spikes, np.ndarray):
            spikes = torch.tensor(spikes, dtype=torch.float32)

        print(f"\nSorting {sign} spikes: {spikes.shape}")

        # Skip SPC if too few spikes
        if spikes.shape[0] < MIN_SPIKES_FOR_SPC:
            print(f"  Skipping SPC: only {spikes.shape[0]} spikes (need {MIN_SPIKES_FOR_SPC})")
            N = spikes.shape[0]
            return {
                "sort_idx": np.zeros(N, dtype=np.uint16),
                "match_idx": np.zeros(N, dtype=np.int8),
                "distance": np.zeros(N, dtype=np.float32),
                "artifact_ids": []
            }

        sort_idx = self._cluster(spikes, folder)

        sort_idx, match_idx, distance, artifact_ids = self._post_process(
            spikes, sort_idx
        )

        return {
            "sort_idx": sort_idx,
            "match_idx": match_idx,
            "distance": distance,
            "artifact_ids": artifact_ids
        }

    def forward(self, pos_spikes, neg_spikes, folder="pytorch_sort"):
        os.makedirs(folder, exist_ok=True)

        results = {}

        results["pos"] = self.sort_one(
            pos_spikes,
            os.path.join(folder, "pos"),
            "pos"
        )

        results["neg"] = self.sort_one(
            neg_spikes,
            os.path.join(folder, "neg"),
            "neg"
        )

        return results