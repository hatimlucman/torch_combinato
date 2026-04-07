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
    print("-" * 120)
    for r in rows:
        print(" | ".join(str(x) for x in r))


def summarize(idx):
    unique = np.unique(idx)
    return {int(c): int((idx == c).sum()) for c in unique}


def warmup_cuda(device):
    """Force CUDA runtime initialization before any timed run."""
    dummy = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    del dummy


def timed_run(model, signal, atimes, folder, device):
    """
    Run model and return (result, elapsed_seconds).
    - CPU: plain wall-clock via time.time()
    - CUDA: CUDA events for accurate GPU timing, with synchronize() guards
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = model(signal, atimes, folder=folder)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed = start_event.elapsed_time(end_event) / 1000.0  # ms -> s
    else:
        t0 = time.time()
        result = model(signal, atimes, folder=folder)
        elapsed = time.time() - t0

    return result, elapsed


def run_pipeline_on_device(
    device_name,
    BASE_DIR,
    DATA_BIN,
    SPC_PATH,
    N_CHANNELS,
    SAMPLE_RATE,
    OUTPUT_DIR_BASE,
    save_h5=True
):
    device = torch.device(device_name)
    print(f"\n=== RUN ON {device} ===")

    output_dir = f"{OUTPUT_DIR_BASE}_{device_name.replace(':', '_')}"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model = PyTorchCombinato(cluster_path=SPC_PATH, sample_rate=SAMPLE_RATE).to(device)

    raw = np.memmap(DATA_BIN, dtype="int16", mode="r")
    data = raw.reshape(-1, N_CHANNELS).astype(np.float64)

    # Same time axis for every channel
    atimes_np = np.arange(data.shape[0]) / (SAMPLE_RATE / 1000.0)
    atimes = torch.tensor(atimes_np, dtype=torch.float64, device=device)

    torch.manual_seed(12345)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(12345)
        warmup_cuda(device)

    all_rows = []
    per_channel_times = []

    for ch in range(N_CHANNELS):
        print(f"\n--- Processing channel {ch} ---")

        ch_dir = os.path.join(output_dir, f"ch{ch}")
        os.makedirs(ch_dir, exist_ok=True)

        signal_np = data[:, ch]
        signal = torch.tensor(signal_np, dtype=torch.float64, device=device)

        result, ch_time = timed_run(model, signal, atimes, folder=ch_dir, device=device)
        per_channel_times.append(ch_time)

        if save_h5:
            out_h5 = os.path.join(
                ch_dir,
                f"pytorch_output_ch{ch}_{device_name.replace(':', '_')}.h5"
            )
            save_to_h5(
                {
                    "pos_spikes": result["pos_spikes"],
                    "pos_times":  result["pos_times"],
                    "neg_spikes": result["neg_spikes"],
                    "neg_times":  result["neg_times"],
                    "threshold":  result["threshold"],
                },
                out_h5,
            )

        row = [
            ch,
            float(result["threshold"].item()),
            int(result["pos_spikes"].shape[0]),
            int(result["neg_spikes"].shape[0]),
            summarize(result["pos"]["sort_idx"]),
            summarize(result["neg"]["sort_idx"]),
            result["pos"]["artifact_ids"],
            result["neg"]["artifact_ids"],
            round(ch_time, 4),
        ]
        all_rows.append(row)

    total_time = float(sum(per_channel_times))

    print("\n=== PYTORCH COMBINATO ALL-CHANNEL RUN ===")
    print_table(
        all_rows,
        [
            "Channel",
            "Threshold",
            "Pos spikes",
            "Neg spikes",
            "Pos distribution",
            "Neg distribution",
            "Pos artifact IDs",
            "Neg artifact IDs",
            "Runtime (s)"
        ]
    )

    print(f"\nTotal runtime on {device}: {total_time:.4f}s")

    return {
        "device": str(device),
        "total_time": total_time,
        "output_dir": output_dir
    }


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_BIN = os.path.join(BASE_DIR, "io", "subset_data", "raw_10pct.bin")
    SPC_PATH = os.path.join(BASE_DIR, "io", "spc", "cluster.exe")

    N_CHANNELS = 384
    SAMPLE_RATE = 30000.0
    OUTPUT_DIR_BASE = "pytorch_core_all_channels"

    print(SPC_PATH, os.path.exists(SPC_PATH))
    print(DATA_BIN, os.path.exists(DATA_BIN))

    cpu_res = run_pipeline_on_device(
        "cpu",
        BASE_DIR,
        DATA_BIN,
        SPC_PATH,
        N_CHANNELS,
        SAMPLE_RATE,
        OUTPUT_DIR_BASE,
        save_h5=True
    )

    cuda_res = None
    if torch.cuda.is_available():
        cuda_res = run_pipeline_on_device(
            "cuda",
            BASE_DIR,
            DATA_BIN,
            SPC_PATH,
            N_CHANNELS,
            SAMPLE_RATE,
            OUTPUT_DIR_BASE,
            save_h5=True
        )
    else:
        print("\nCUDA not available, skipped GPU run.")

    if cuda_res is not None:
        cpu_time = cpu_res["total_time"]
        cuda_time = cuda_res["total_time"]
        diff = cpu_time - cuda_time
        pct = (diff / cpu_time * 100.0) if cpu_time != 0 else 0.0
        faster_device = "GPU" if diff > 0 else "CPU"

        print("\n=== RUNTIME COMPARISON ===")
        print(f"CPU  total time : {cpu_time:.4f} s")
        print(f"CUDA total time : {cuda_time:.4f} s")
        print(f"Difference (CPU - CUDA): {diff:.4f} s -> {faster_device} is faster by {abs(pct):.2f}%")
    else:
        print("\nOnly CPU run completed; no CUDA comparison available.")