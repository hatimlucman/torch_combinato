import os
import time
import shutil
import numpy as np
import torch
import tables

from modules.m6_pipeline import CombinatoExtractor, save_to_h5
from modules.c7_combinato_sorter import CombinatorSorter


# =====================
# CONFIG
# =====================
DATA_BIN = "subset_data/raw_10pct.bin"
CHANNEL = 188
N_CHANNELS = 384
SAMPLE_RATE = 30000.0
OUTPUT_DIR = f"pytorch_core_ch{CHANNEL}"
OUTPUT_H5 = os.path.join(OUTPUT_DIR, f"pytorch_output_ch{CHANNEL}.h5")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPC_PATH = os.path.join(BASE_DIR, "io", "spc", "cluster.exe")
print(SPC_PATH, os.path.exists(SPC_PATH))


def print_table(rows, headers):
    print("\n" + " | ".join(headers))
    print("-" * 90)
    for r in rows:
        print(" | ".join(str(x) for x in r))


def summarize(idx):
    unique = np.unique(idx)
    return {int(c): int((idx == c).sum()) for c in unique}


# =====================
# PREP
# =====================
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

raw = np.memmap(DATA_BIN, dtype="int16", mode="r")
signal_np = raw.reshape(-1, N_CHANNELS)[:, CHANNEL].astype(np.float64)
atimes_np = np.arange(len(signal_np)) / (SAMPLE_RATE / 1000.0)

signal = torch.tensor(signal_np)
atimes = torch.tensor(atimes_np)

model = CombinatoExtractor(sample_rate=SAMPLE_RATE)
sorter = CombinatorSorter(cluster_path=SPC_PATH)

timings = {}

# =====================
# TIMED PYTORCH CORE
# =====================
t0_total = time.time()

# M1
t0 = time.time()
data_denoised, data_detected = model.pre(signal)
timings["M1_pytorch_preprocess"] = time.time() - t0

# M2
t0 = time.time()
pos_idx, neg_idx, threshold = model.detector(data_detected)
timings["M2_pytorch_detect"] = time.time() - t0

# M3-M5 positive
t0 = time.time()
pos_spikes, pos_times = model._process_one(data_denoised, pos_idx, atimes, "pos")
timings["M3-M5_pytorch_pos_extract"] = time.time() - t0

# M3-M5 negative
t0 = time.time()
neg_spikes, neg_times = model._process_one(data_denoised, neg_idx, atimes, "neg")
timings["M3-M5_pytorch_neg_extract"] = time.time() - t0

save_to_h5(
    {
        "pos_spikes": pos_spikes,
        "pos_times": pos_times,
        "neg_spikes": neg_spikes,
        "neg_times": neg_times,
        "threshold": threshold,
    },
    OUTPUT_H5,
)

# C positive
t0 = time.time()
pos_result = sorter.sort_one(pos_spikes, os.path.join(OUTPUT_DIR, "pos"), sign="pos")
timings["C_pytorch_pos_sort"] = time.time() - t0

# C negative
t0 = time.time()
neg_result = sorter.sort_one(neg_spikes, os.path.join(OUTPUT_DIR, "neg"), sign="neg")
timings["C_pytorch_neg_sort"] = time.time() - t0

timings["PyTorch_core_total"] = time.time() - t0_total

# =====================
# SUMMARY
# =====================
rows = [
    ["Channel", CHANNEL],
    ["Threshold", float(threshold.item())],
    ["Positive spikes", int(pos_spikes.shape[0])],
    ["Negative spikes", int(neg_spikes.shape[0])],
    ["Pos distribution", summarize(pos_result["sort_idx"])],
    ["Neg distribution", summarize(neg_result["sort_idx"])],
    ["Pos artifact IDs", pos_result["artifact_ids"]],
    ["Neg artifact IDs", neg_result["artifact_ids"]],
    ["--- Timing ---", ""],
    ["M1 PyTorch preprocess (s)", timings["M1_pytorch_preprocess"]],
    ["M2 PyTorch detect (s)", timings["M2_pytorch_detect"]],
    ["M3-M5 PyTorch pos extract (s)", timings["M3-M5_pytorch_pos_extract"]],
    ["M3-M5 PyTorch neg extract (s)", timings["M3-M5_pytorch_neg_extract"]],
    ["C PyTorch pos sort (s)", timings["C_pytorch_pos_sort"]],
    ["C PyTorch neg sort (s)", timings["C_pytorch_neg_sort"]],
    ["PyTorch core total (s)", timings["PyTorch_core_total"]],
]

print("\n=== PYTORCH CORE TIMED RUN ===")
print_table(rows, ["Metric", "Value"])