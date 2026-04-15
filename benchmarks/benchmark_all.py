"""
Comprehensive Torched Combinato Benchmark.

Compares three pipeline variants:
  - NaiveCombinato:    Per-channel sequential (CPU)
  - ImprovedCombinato: Batched M1 (CPU)
  - OptimizedCombinato: Hybrid GPU+CPU (M1/M2 GPU, M3 CPU, M4-C6 GPU)

For each variant:
  - Runs N repetitions, timing total wall-clock
  - Records resource usage (CPU%, RSS, GPU memory)
  - Assesses run-to-run determinism
  - Assesses output parity across variants
  - Produces JSON report and bar plot
"""

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from torched_combinato import NaiveCombinato, ImprovedCombinato, OptimizedCombinato


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_TIME = 103.75  # Original Combinato reference time

ALL_VARIANTS = ["naive", "improved", "optimized"]

VARIANT_LABELS = {
    "naive": "NaiveCombinato (CPU)",
    "improved": "ImprovedCombinato (CPU)",
    "optimized": "OptimizedCombinato (GPU+CPU)",
}

VARIANT_SHORT = {
    "naive": "Naive CPU",
    "improved": "Improved CPU",
    "optimized": "Optimized GPU+CPU",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    raw_path: Path = Path("raw.bin")
    sample_rate: int = 30000
    num_channels: int = 384
    results_dir: Path = Path("results")
    n_runs: int = 3
    variants: List[str] = field(default_factory=lambda: list(ALL_VARIANTS))
    check_channel: int = 174


# ---------------------------------------------------------------------------
# Resource Monitoring
# ---------------------------------------------------------------------------
@dataclass
class StageResources:
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    cpu_percent: float = 0.0
    process_threads: int = 0
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0

    @property
    def rss_delta_mb(self) -> float:
        return self.rss_after_mb - self.rss_before_mb


class StageMonitor:
    """Context manager for timing and resource monitoring."""
    
    def __init__(self, use_cuda_sync: bool = False):
        self.use_cuda_sync = use_cuda_sync
        self.elapsed: float = 0.0
        self.resources = StageResources()

    def __enter__(self):
        proc = psutil.Process()
        self.resources.rss_before_mb = proc.memory_info().rss / 1e6
        proc.cpu_percent()  # Prime measurement
        self._proc = proc
        
        if self.use_cuda_sync and HAS_CUDA:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.use_cuda_sync and HAS_CUDA:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
        
        proc = self._proc
        self.resources.rss_after_mb = proc.memory_info().rss / 1e6
        self.resources.cpu_percent = proc.cpu_percent()
        self.resources.process_threads = proc.num_threads()
        
        if self.use_cuda_sync and HAS_CUDA:
            self.resources.gpu_allocated_mb = torch.cuda.max_memory_allocated() / 1e6
            self.resources.gpu_reserved_mb = torch.cuda.max_memory_reserved() / 1e6


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    total_time: float
    resources: StageResources
    n_channels: int
    clusters: Optional[Dict] = None


@dataclass
class DeterminismReport:
    variant_name: str
    n_runs: int
    all_identical: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParityReport:
    variant_a: str
    variant_b: str
    labels_identical: bool = False
    n_spikes_a: int = 0
    n_spikes_b: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_variant(
    variant: str,
    raw: np.ndarray,
    cfg: BenchmarkConfig,
) -> Tuple[List[RunResult], List[np.ndarray]]:
    """Run a variant multiple times, collecting timing and outputs."""
    
    all_results = []
    all_clusters = []
    use_cuda = variant == "optimized"

    # CUDA warmup
    if use_cuda and HAS_CUDA:
        print("  CUDA warmup...")
        model = OptimizedCombinato(sample_rate=cfg.sample_rate)
        _ = model(raw[:, :10])
        torch.cuda.empty_cache()

    for run_idx in range(cfg.n_runs):
        print(f"  Run {run_idx + 1}/{cfg.n_runs}...", end=" ", flush=True)

        # Create model
        if variant == "naive":
            model = NaiveCombinato(sample_rate=cfg.sample_rate, device='cpu')
        elif variant == "improved":
            model = ImprovedCombinato(sample_rate=cfg.sample_rate, device='cpu')
        else:
            model = OptimizedCombinato(sample_rate=cfg.sample_rate)

        # Run with monitoring
        monitor = StageMonitor(use_cuda_sync=use_cuda)
        with monitor:
            outputs = model(raw)

        # Extract clusters for check channel
        clusters = outputs[cfg.check_channel]['clusters'] if cfg.check_channel in outputs else np.array([])

        result = RunResult(
            total_time=monitor.elapsed,
            resources=monitor.resources,
            n_channels=len(outputs),
            clusters={cfg.check_channel: clusters},
        )
        all_results.append(result)
        all_clusters.append(clusters)

        print(f"total={monitor.elapsed:.2f}s | channels={len(outputs)}", end="")
        if use_cuda and HAS_CUDA:
            print(f" | GPU={monitor.resources.gpu_allocated_mb:.0f}MB", end="")
        print()

    return all_results, all_clusters


# ---------------------------------------------------------------------------
# Fidelity Checks
# ---------------------------------------------------------------------------
def assess_determinism(variant: str, all_clusters: List[np.ndarray]) -> DeterminismReport:
    """Check if multiple runs produce identical outputs."""
    n = len(all_clusters)
    report = DeterminismReport(variant_name=variant, n_runs=n)
    
    if n < 2:
        report.all_identical = True
        return report
    
    ref = all_clusters[0]
    report.all_identical = all(np.array_equal(ref, c) for c in all_clusters[1:])
    return report


def assess_parity(
    name_a: str, clusters_a: np.ndarray,
    name_b: str, clusters_b: np.ndarray,
) -> ParityReport:
    """Compare outputs between two variants."""
    return ParityReport(
        variant_a=name_a,
        variant_b=name_b,
        labels_identical=np.array_equal(clusters_a, clusters_b),
        n_spikes_a=len(clusters_a),
        n_spikes_b=len(clusters_b),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary_table(timing_results: Dict[str, List[RunResult]]):
    """Print performance summary table."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<30} {'Time (s)':>15} {'Speedup':>12} {'Channels':>10}")
    print("-" * 70)
    print(f"{'Original Combinato':<30} {BASELINE_TIME:>15.2f} {1.0:>12.2f}x {'-':>10}")

    for variant, results in timing_results.items():
        times = [r.total_time for r in results]
        mean_time = np.mean(times)
        std_time = np.std(times)
        speedup = BASELINE_TIME / mean_time
        n_channels = results[0].n_channels if results else 0
        time_str = f"{mean_time:.2f} +/- {std_time:.2f}"
        print(f"{VARIANT_LABELS[variant]:<30} {time_str:>15} {speedup:>12.2f}x {n_channels:>10}")
    
    print("=" * 70)


def print_resource_table(timing_results: Dict[str, List[RunResult]]):
    """Print resource usage table."""
    print("\n" + "=" * 80)
    print("RESOURCE USAGE")
    print("=" * 80)
    print(f"{'Variant':<30} {'CPU%':>8} {'RSS delta':>12} {'GPU Alloc':>12}")
    print("-" * 80)
    
    for variant, results in timing_results.items():
        cpu = np.mean([r.resources.cpu_percent for r in results])
        rss = np.mean([r.resources.rss_delta_mb for r in results])
        gpu = np.mean([r.resources.gpu_allocated_mb for r in results])
        gpu_str = f"{gpu:.0f} MB" if gpu > 0 else "n/a"
        print(f"{VARIANT_LABELS[variant]:<30} {cpu:>7.1f}% {rss:>+11.0f} MB {gpu_str:>12}")
    
    print("=" * 80)


def print_determinism_report(reports: Dict[str, DeterminismReport]):
    """Print determinism analysis."""
    print("\n" + "=" * 60)
    print("DETERMINISM")
    print("=" * 60)
    for variant, report in reports.items():
        verdict = "DETERMINISTIC" if report.all_identical else "NON-DETERMINISTIC"
        print(f"  {VARIANT_LABELS[variant]}: {verdict}")
    print("=" * 60)


def print_parity_report(reports: List[ParityReport]):
    """Print output parity analysis."""
    print("\n" + "=" * 60)
    print("OUTPUT PARITY")
    print("=" * 60)
    for report in reports:
        status = "IDENTICAL" if report.labels_identical else "DIFFERENT"
        print(f"  {VARIANT_LABELS[report.variant_a]} vs {VARIANT_LABELS[report.variant_b]}: {status}")
    print("=" * 60)


def save_json_report(
    cfg: BenchmarkConfig,
    timing_results: Dict[str, List[RunResult]],
    determinism_reports: Dict[str, DeterminismReport],
    parity_reports: List[ParityReport],
):
    """Save comprehensive JSON report."""
    report = {
        "config": {
            "raw_path": str(cfg.raw_path),
            "n_runs": cfg.n_runs,
            "baseline": BASELINE_TIME,
        },
        "variants": {},
        "determinism": {k: v.to_dict() for k, v in determinism_reports.items()},
        "parity": [p.to_dict() for p in parity_reports],
    }
    
    for variant, results in timing_results.items():
        times = [r.total_time for r in results]
        report["variants"][variant] = {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "speedup": BASELINE_TIME / float(np.mean(times)),
        }
    
    out_path = cfg.results_dir / "benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_bar(timing_results: Dict[str, List[RunResult]], out_path: Path):
    """Create bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping plot (matplotlib not available)")
        return

    labels = ["Original"] + [VARIANT_SHORT[v] for v in timing_results]
    times = [BASELINE_TIME] + [np.mean([r.total_time for r in results]) for results in timing_results.values()]
    speedups = [1.0] + [BASELINE_TIME / t for t in times[1:]]
    colors = ["#9BBCE0", "#8FD3B8", "#E6A3BD", "#C7BCE6"]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, times, color=colors[:len(labels)], height=0.6)
    
    for i, (t, s) in enumerate(zip(times, speedups)):
        ax.text(t + max(times) * 0.02, y_pos[i], f"{s:.2f}x", va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Torched Combinato Benchmark")
    plt.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------
def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Torched Combinato Benchmark")
    parser.add_argument("raw_path", type=Path, help="Path to raw .bin file")
    parser.add_argument("--sample-rate", type=int, default=30000)
    parser.add_argument("--num-channels", type=int, default=384)
    parser.add_argument("--n-runs", type=int, default=3, help="Runs per variant")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument("--check-channel", type=int, default=174)
    args = parser.parse_args()
    
    return BenchmarkConfig(
        raw_path=args.raw_path,
        sample_rate=args.sample_rate,
        num_channels=args.num_channels,
        results_dir=args.results_dir,
        n_runs=args.n_runs,
        variants=args.variants,
        check_channel=args.check_channel,
    )


def main():
    cfg = parse_args()
    
    # Validate
    if "optimized" in cfg.variants and not HAS_CUDA:
        print("WARNING: CUDA not available, removing optimized variant")
        cfg.variants = [v for v in cfg.variants if v != "optimized"]
    
    if not cfg.variants:
        print("No valid variants. Exiting.")
        return
    
    # Load data
    print(f"Loading: {cfg.raw_path}")
    raw = np.memmap(cfg.raw_path, dtype='int16', mode='r').reshape(-1, cfg.num_channels)
    print(f"  Shape: {raw.shape}")
    
    # Run benchmarks
    timing_results = {}
    cluster_results = {}
    
    for variant in cfg.variants:
        print(f"\n{'=' * 60}")
        print(f"  {VARIANT_LABELS[variant]}  ({cfg.n_runs} runs)")
        print(f"{'=' * 60}")
        
        if variant == "optimized" and HAS_CUDA:
            torch.cuda.empty_cache()
        
        results, clusters = run_variant(variant, raw, cfg)
        timing_results[variant] = results
        cluster_results[variant] = clusters
    
    # Determinism analysis
    det_reports = {v: assess_determinism(v, cluster_results[v]) for v in cfg.variants}
    
    # Parity analysis
    parity_reports = []
    for i, v_a in enumerate(cfg.variants):
        for v_b in cfg.variants[i+1:]:
            report = assess_parity(v_a, cluster_results[v_a][0], v_b, cluster_results[v_b][0])
            parity_reports.append(report)
    
    # Reports
    print_summary_table(timing_results)
    print_resource_table(timing_results)
    print_determinism_report(det_reports)
    print_parity_report(parity_reports)
    save_json_report(cfg, timing_results, det_reports, parity_reports)
    plot_bar(timing_results, cfg.results_dir / "benchmark_runtime.png")


if __name__ == "__main__":
    main()
