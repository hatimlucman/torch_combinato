import os
import shutil
import time
import numpy as np
import torch

from modules.block import Block
from modules.m6_pipeline import CombinatoExtractor, save_to_h5
from modules.c7_combinato_sorter import CombinatorSorter


class PyTorchCombinato(Block):
    def __init__(self, cluster_path, sample_rate=30000):
        super().__init__()
        self.extractor = CombinatoExtractor(sample_rate=sample_rate)
        self.sorter = CombinatorSorter(cluster_path=cluster_path)
        self.sample_rate = sample_rate

    def forward(self, signal, atimes, folder="pytorch_sort"):
        extracted = self.extractor(signal, atimes)

        sorted_out = self.sorter(
            extracted["pos_spikes"],
            extracted["neg_spikes"],
            folder=folder
        )

        return {
            **extracted,
            **sorted_out
        }


def print_table(rows, headers):
    print("\n" + " | ".join(headers))
    print("-" * 90)
    for r in rows:
        print(" | ".join(str(x) for x in r))


def summarize(idx):
    unique = np.unique(idx)
    return {int(c): int((idx == c).sum()) for c in unique}


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_BIN = os.path.join(BASE_DIR, "io", "subset_data", "raw_10pct.bin")
    SPC_PATH = os.path.join(BASE_DIR, "io", "spc", "cluster.exe")

    CHANNEL = 188
    N_CHANNELS = 384
    SAMPLE_RATE = 30000.0
    OUTPUT_DIR = f"pytorch_core_ch{CHANNEL}"
    OUTPUT_H5 = os.path.join(OUTPUT_DIR, f"pytorch_output_ch{CHANNEL}.h5")

    print(SPC_PATH, os.path.exists(SPC_PATH))
    print(DATA_BIN, os.path.exists(DATA_BIN))

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw = np.memmap(DATA_BIN, dtype="int16", mode="r")
    signal_np = raw.reshape(-1, N_CHANNELS)[:, CHANNEL].astype(np.float64)
    atimes_np = np.arange(len(signal_np)) / (SAMPLE_RATE / 1000.0)

    signal = torch.tensor(signal_np)
    atimes = torch.tensor(atimes_np)

    model = PyTorchCombinato(cluster_path=SPC_PATH, sample_rate=SAMPLE_RATE)

    t0 = time.time()
    result = model(signal, atimes, folder=OUTPUT_DIR)
    total_time = time.time() - t0

    save_to_h5(
        {
            "pos_spikes": result["pos_spikes"],
            "pos_times": result["pos_times"],
            "neg_spikes": result["neg_spikes"],
            "neg_times": result["neg_times"],
            "threshold": result["threshold"],
        },
        OUTPUT_H5,
    )

    rows = [
        ["Channel", CHANNEL],
        ["Threshold", float(result["threshold"].item())],
        ["Positive spikes", int(result["pos_spikes"].shape[0])],
        ["Negative spikes", int(result["neg_spikes"].shape[0])],
        ["Pos distribution", summarize(result["pos"]["sort_idx"])],
        ["Neg distribution", summarize(result["neg"]["sort_idx"])],
        ["Pos artifact IDs", result["pos"]["artifact_ids"]],
        ["Neg artifact IDs", result["neg"]["artifact_ids"]],
        ["Total runtime (s)", total_time],
    ]

    print("\n=== PYTORCH COMBINATO RUN ===")
    print_table(rows, ["Metric", "Value"])