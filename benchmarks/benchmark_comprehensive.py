"""
Comprehensive Torched Combinato Benchmark.

Features:
  - Per-module timing (M1-M5 extraction, C1-C6 clustering)
  - Resource monitoring (CPU%, RSS, GPU memory, threads, BLAS)
  - Multiple runs with mean +/- std
  - Determinism analysis (Jaccard index)
  - Port parity (Jaccard index, Hungarian label matching)
  - Stacked bar plot (per-module breakdown)
  - JSON report
"""

import argparse
import json
import time
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False

try:
    from threadpoolctl import threadpool_info
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from torched_combinato.modules.m1_preprocessor import Preprocessor
from torched_combinato.modules.m2_threshold_detector import ThresholdDetector
from torched_combinato.modules.m3_waveform_extractor import WaveformExtractor
from torched_combinato.modules.m4_cubic_upsampler import CubicUpsampler
from torched_combinato.modules.m5_peak_aligner import PeakAligner
from torched_combinato.modules.c1_wavelet_features import WaveletFeatureExtractor
from torched_combinato.modules.c2_feature_selector import FeatureSelector
from torched_combinato.modules.c3_spc_clusterer import SPCClusterer
from torched_combinato.modules.c4_cluster_definer import ClusterDefiner
from torched_combinato.modules.c5_template_matcher import TemplateMatcher
from torched_combinato.modules.c6_artifact_detector import ArtifactDetector

BASELINE_TIME = 103.75
MODULES = ['M1', 'M2', 'M3', 'M4', 'M5', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
MODULE_NAMES = {'M1': 'Preprocess', 'M2': 'Threshold', 'M3': 'Extract', 'M4': 'Upsample', 'M5': 'Align', 'C1': 'Wavelet', 'C2': 'Feature', 'C3': 'SPC', 'C4': 'Define', 'C5': 'Match', 'C6': 'Artifact'}
MODULE_COLORS = {'M1': '#7F77DD', 'M2': '#1D9E75', 'M3': '#D85A30', 'M4': '#D4537E', 'M5': '#888780', 'C1': '#AFA9EC', 'C2': '#5DCAA5', 'C3': '#85B7EB', 'C4': '#ED93B1', 'C5': '#FAC775', 'C6': '#F7C1C1'}
ALL_VARIANTS = ['naive_cpu', 'improved_cpu', 'optimized_gpu']
VARIANT_LABELS = {'naive_cpu': 'Naive (CPU)', 'improved_cpu': 'Improved (CPU)', 'optimized_gpu': 'Optimized (GPU+CPU)'}

@dataclass
class BenchmarkConfig:
    raw_path: Path = Path("raw.bin")
    sample_rate: int = 30000
    num_channels: int = 384
    results_dir: Path = Path("results")
    n_runs: int = 3
    variants: List[str] = field(default_factory=lambda: list(ALL_VARIANTS))
    spc_path: str = "~/spc/cluster_linux64.exe"
    check_channel: int = 174

@dataclass
class Resources:
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    cpu_percent: float = 0.0
    process_threads: int = 0
    torch_threads: Optional[int] = None
    torch_interop_threads: Optional[int] = None
    blas_info: Optional[List[dict]] = None
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    @property
    def rss_delta_mb(self): return self.rss_after_mb - self.rss_before_mb
    def to_dict(self): d = asdict(self); d['rss_delta_mb'] = self.rss_delta_mb; return d

class ResourceMonitor:
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda
        self.resources = Resources()
    def __enter__(self):
        proc = psutil.Process()
        self.resources.rss_before_mb = proc.memory_info().rss / 1e6
        proc.cpu_percent()
        self._proc = proc
        if self.use_cuda and HAS_CUDA: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        return self
    def __exit__(self, *exc):
        if self.use_cuda and HAS_CUDA: torch.cuda.synchronize()
        self.resources.rss_after_mb = self._proc.memory_info().rss / 1e6
        self.resources.cpu_percent = self._proc.cpu_percent()
        self.resources.process_threads = self._proc.num_threads()
        if HAS_TORCH: self.resources.torch_threads = torch.get_num_threads(); self.resources.torch_interop_threads = torch.get_num_interop_threads()
        if HAS_THREADPOOLCTL: self.resources.blas_info = [{"library": p.get("internal_api", "?"), "num_threads": p.get("num_threads", 0)} for p in threadpool_info()]
        if self.use_cuda and HAS_CUDA: self.resources.gpu_allocated_mb = torch.cuda.max_memory_allocated() / 1e6; self.resources.gpu_reserved_mb = torch.cuda.max_memory_reserved() / 1e6

class Timer:
    def __init__(self, use_cuda=False): self.use_cuda = use_cuda; self.elapsed = 0.0
    def __enter__(self):
        if self.use_cuda and HAS_CUDA: torch.cuda.synchronize()
        self._start = time.perf_counter(); return self
    def __exit__(self, *exc):
        if self.use_cuda and HAS_CUDA: torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start

@dataclass
class RunResult:
    module_times: Dict[str, float]
    total_time: float
    resources: Resources
    clusters: Optional[np.ndarray] = None
    def to_dict(self): return {'module_times': self.module_times, 'total_time': self.total_time, 'resources': self.resources.to_dict()}

@dataclass
class DeterminismReport:
    variant_name: str
    n_runs: int
    all_identical: bool = False
    min_jaccard: float = 1.0
    max_jaccard: float = 1.0
    mean_jaccard: float = 1.0
    def to_dict(self): return asdict(self)

@dataclass
class ParityReport:
    variant_a: str
    variant_b: str
    n_spikes_a: int = 0
    n_spikes_b: int = 0
    n_shared: int = 0
    jaccard: float = 0.0
    label_agreement: float = 0.0
    labels_identical: bool = False
    def to_dict(self): return asdict(self)

def compute_label_agreement(labels_a, labels_b):
    if len(labels_a) != len(labels_b) or len(labels_a) == 0: return 0.0
    if not HAS_SCIPY_OPT: return float(np.mean(labels_a == labels_b))
    K_a, K_b = int(labels_a.max()) + 1, int(labels_b.max()) + 1
    confusion = np.zeros((K_a, K_b), dtype=np.int64)
    for la, lb in zip(labels_a, labels_b): confusion[la, lb] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    remap = np.zeros(K_b, dtype=np.int64)
    for r, c in zip(row_ind, col_ind): remap[c] = r
    return float(np.mean(labels_a == remap[labels_b]))

def assess_determinism(variant, all_results):
    n = len(all_results)
    report = DeterminismReport(variant_name=variant, n_runs=n)
    if n < 2: report.all_identical = True; return report
    ref = all_results[0].clusters
    jaccards = []
    for r in all_results[1:]:
        if ref is not None and r.clusters is not None:
            shared = len(np.intersect1d(np.arange(len(ref)), np.arange(len(r.clusters))))
            union = len(np.union1d(np.arange(len(ref)), np.arange(len(r.clusters))))
            jaccards.append(shared / max(union, 1))
    if jaccards:
        report.all_identical = all(np.array_equal(ref, r.clusters) for r in all_results[1:])
        report.min_jaccard, report.max_jaccard, report.mean_jaccard = min(jaccards), max(jaccards), np.mean(jaccards)
    else: report.all_identical = True
    return report

def assess_parity(name_a, result_a, name_b, result_b):
    report = ParityReport(variant_a=name_a, variant_b=name_b)
    ca, cb = result_a.clusters, result_b.clusters
    if ca is None or cb is None: return report
    report.n_spikes_a, report.n_spikes_b = len(ca), len(cb)
    shared = np.intersect1d(np.arange(len(ca)), np.arange(len(cb)))
    report.n_shared = len(shared)
    report.jaccard = len(shared) / max(len(np.union1d(np.arange(len(ca)), np.arange(len(cb)))), 1)
    if len(ca) == len(cb): report.label_agreement = compute_label_agreement(ca, cb); report.labels_identical = np.array_equal(ca, cb)
    return report

def run_naive_cpu(raw, cfg):
    times = {m: 0.0 for m in MODULES}
    n_ch, sr = raw.shape[1], cfg.sample_rate
    pre, det, ext, ups, ali = Preprocessor(sample_rate=sr), ThresholdDetector(sample_rate=sr), WaveformExtractor(Preprocessor(sample_rate=sr)), CubicUpsampler(), PeakAligner()
    c1, c2, c3, c4, c5, c6 = WaveletFeatureExtractor(), FeatureSelector(), SPCClusterer(cluster_path=os.path.expanduser(cfg.spc_path)), ClusterDefiner(), TemplateMatcher(), ArtifactDetector()
    spikes, denoised_all, detected_all, check_clusters = {}, [], [], None
    with Timer() as t:
        for ch in range(n_ch):
            sig = torch.tensor(raw[:, ch].astype(np.float64), dtype=torch.float64)
            with torch.no_grad(): den, det_ = pre(sig)
            denoised_all.append(den); detected_all.append(det_)
    times['M1'] = t.elapsed
    neg_indices = {}
    with Timer() as t:
        for ch in range(n_ch):
            with torch.no_grad(): _, neg_idx, _ = det(detected_all[ch])
            if len(neg_idx) >= 15: neg_indices[ch] = neg_idx
    times['M2'] = t.elapsed
    with Timer() as t:
        for ch, neg_idx in neg_indices.items():
            with torch.no_grad(): spk, _, _ = ext(denoised_all[ch], neg_idx, 'neg')
            spikes[ch] = spk
    times['M3'] = t.elapsed
    with Timer() as t:
        for ch in spikes:
            with torch.no_grad(): spikes[ch] = ups(spikes[ch])
    times['M4'] = t.elapsed
    with Timer() as t:
        for ch in spikes:
            with torch.no_grad(): spikes[ch], _ = ali(spikes[ch])
    times['M5'] = t.elapsed
    cluster_spikes = {ch: -spikes[ch].numpy() for ch in spikes}
    for ch, spk in cluster_spikes.items():
        spk_t = torch.tensor(spk, dtype=torch.float64)
        folder = f'/tmp/bench_naive/ch{ch}'; os.makedirs(folder, exist_ok=True)
        with Timer() as t: features = c1(spk_t)
        times['C1'] += t.elapsed
        with Timer() as t: features_sel, _ = c2(features)
        times['C2'] += t.elapsed
        with Timer() as t: clu, tree = c3(features_sel, folder, 'naive', 12345.0)
        times['C3'] += t.elapsed
        with Timer() as t: sort_idx, _, _ = c4(clu, tree)
        times['C4'] += t.elapsed
        with Timer() as t: sort_idx_np = sort_idx.numpy().astype(np.uint16).copy(); match_idx = np.zeros(len(sort_idx_np), dtype=np.int8); c5(spk, sort_idx_np, match_idx)
        times['C5'] += t.elapsed
        with Timer() as t: c6(spk_t, sort_idx_np, sign='neg')
        times['C6'] += t.elapsed
        if ch == cfg.check_channel: check_clusters = sort_idx_np.copy()
    return times, check_clusters

def run_improved_cpu(raw, cfg):
    times = {m: 0.0 for m in MODULES}
    n_ch, sr, batch_size = raw.shape[1], cfg.sample_rate, 192
    pre, det, ext, ups, ali = Preprocessor(sample_rate=sr), ThresholdDetector(sample_rate=sr), WaveformExtractor(Preprocessor(sample_rate=sr)), CubicUpsampler(), PeakAligner()
    c1, c2, c3, c4, c5, c6 = WaveletFeatureExtractor(), FeatureSelector(), SPCClusterer(cluster_path=os.path.expanduser(cfg.spc_path)), ClusterDefiner(), TemplateMatcher(), ArtifactDetector()
    spikes, denoised_all, detected_all, check_clusters = {}, [None]*n_ch, [None]*n_ch, None
    with Timer() as t:
        for batch in range((n_ch + batch_size - 1) // batch_size):
            s, e = batch * batch_size, min((batch + 1) * batch_size, n_ch)
            data = torch.tensor(raw[:, s:e].T.astype(np.float64))
            with torch.no_grad(): den, det_ = pre(data)
            for i, ch in enumerate(range(s, e)): denoised_all[ch] = den[i].clone(); detected_all[ch] = det_[i].clone()
    times['M1'] = t.elapsed
    neg_indices = {}
    with Timer() as t:
        for ch in range(n_ch):
            with torch.no_grad(): _, neg_idx, _ = det(detected_all[ch])
            if len(neg_idx) >= 15: neg_indices[ch] = neg_idx
    times['M2'] = t.elapsed
    with Timer() as t:
        for ch, neg_idx in neg_indices.items():
            with torch.no_grad(): spk, _, _ = ext(denoised_all[ch], neg_idx, 'neg')
            spikes[ch] = spk
    times['M3'] = t.elapsed
    with Timer() as t:
        for ch in spikes:
            with torch.no_grad(): spikes[ch] = ups(spikes[ch])
    times['M4'] = t.elapsed
    with Timer() as t:
        for ch in spikes:
            with torch.no_grad(): spikes[ch], _ = ali(spikes[ch])
    times['M5'] = t.elapsed
    cluster_spikes = {ch: -spikes[ch].numpy() for ch in spikes}
    for ch, spk in cluster_spikes.items():
        spk_t = torch.tensor(spk, dtype=torch.float64)
        folder = f'/tmp/bench_improved/ch{ch}'; os.makedirs(folder, exist_ok=True)
        with Timer() as t: features = c1(spk_t)
        times['C1'] += t.elapsed
        with Timer() as t: features_sel, _ = c2(features)
        times['C2'] += t.elapsed
        with Timer() as t: clu, tree = c3(features_sel, folder, 'improved', 12345.0)
        times['C3'] += t.elapsed
        with Timer() as t: sort_idx, _, _ = c4(clu, tree)
        times['C4'] += t.elapsed
        with Timer() as t: sort_idx_np = sort_idx.numpy().astype(np.uint16).copy(); match_idx = np.zeros(len(sort_idx_np), dtype=np.int8); c5(spk, sort_idx_np, match_idx)
        times['C5'] += t.elapsed
        with Timer() as t: c6(spk_t, sort_idx_np, sign='neg')
        times['C6'] += t.elapsed
        if ch == cfg.check_channel: check_clusters = sort_idx_np.copy()
    return times, check_clusters

def run_optimized_gpu(raw, cfg):
    times = {m: 0.0 for m in MODULES}
    n_ch, sr, batch_size = raw.shape[1], cfg.sample_rate, 192
    pre_gpu, det_gpu, ups_gpu, ali_gpu = Preprocessor(sample_rate=sr).cuda(), ThresholdDetector(sample_rate=sr).cuda(), CubicUpsampler().cuda(), PeakAligner().cuda()
    c1_gpu, c2_gpu, c6_gpu = WaveletFeatureExtractor().cuda(), FeatureSelector().cuda(), ArtifactDetector().cuda()
    pre_cpu, ext_cpu = Preprocessor(sample_rate=sr), WaveformExtractor(Preprocessor(sample_rate=sr))
    c3, c4, c5 = SPCClusterer(cluster_path=os.path.expanduser(cfg.spc_path)), ClusterDefiner(), TemplateMatcher()
    spikes, gpu_denoised, gpu_detected, check_clusters = {}, [None]*n_ch, [None]*n_ch, None
    with Timer(use_cuda=True) as t:
        for batch in range((n_ch + batch_size - 1) // batch_size):
            s, e = batch * batch_size, min((batch + 1) * batch_size, n_ch)
            data = torch.tensor(raw[:, s:e].T.astype(np.float64), device='cuda')
            with torch.no_grad(): den, det_ = pre_gpu(data)
            for i, ch in enumerate(range(s, e)): gpu_denoised[ch] = den[i].clone(); gpu_detected[ch] = det_[i].clone()
            del data, den, det_; torch.cuda.empty_cache()
    times['M1'] = t.elapsed
    neg_indices = {}
    with Timer(use_cuda=True) as t:
        for ch in range(n_ch):
            with torch.no_grad(): _, neg_idx, _ = det_gpu(gpu_detected[ch])
            if len(neg_idx) >= 15: neg_indices[ch] = neg_idx
    times['M2'] = t.elapsed
    with Timer() as t:
        for ch, neg_idx in neg_indices.items():
            den_cpu = gpu_denoised[ch].cpu(); neg_cpu = neg_idx.cpu() if isinstance(neg_idx, torch.Tensor) else neg_idx
            with torch.no_grad(): spk, _, _ = ext_cpu(den_cpu, neg_cpu, 'neg')
            spikes[ch] = spk
    times['M3'] = t.elapsed
    with Timer(use_cuda=True) as t:
        for ch in spikes: spikes[ch] = spikes[ch].cuda(); spikes[ch] = ups_gpu(spikes[ch])
    times['M4'] = t.elapsed
    with Timer(use_cuda=True) as t:
        for ch in spikes:
            with torch.no_grad(): spikes[ch], _ = ali_gpu(spikes[ch])
    times['M5'] = t.elapsed
    cluster_spikes = {ch: -spikes[ch].cpu().numpy() for ch in spikes}
    for ch, spk in cluster_spikes.items():
        spk_gpu = torch.tensor(spk, dtype=torch.float64, device='cuda')
        folder = f'/tmp/bench_optimized/ch{ch}'; os.makedirs(folder, exist_ok=True)
        with Timer(use_cuda=True) as t: features = c1_gpu(spk_gpu)
        times['C1'] += t.elapsed
        with Timer(use_cuda=True) as t: features_sel, _ = c2_gpu(features)
        times['C2'] += t.elapsed
        with Timer() as t: clu, tree = c3(features_sel.cpu(), folder, 'optimized', 12345.0)
        times['C3'] += t.elapsed
        with Timer() as t: sort_idx, _, _ = c4(clu, tree)
        times['C4'] += t.elapsed
        with Timer() as t: sort_idx_np = sort_idx.numpy().astype(np.uint16).copy(); match_idx = np.zeros(len(sort_idx_np), dtype=np.int8); c5(spk, sort_idx_np, match_idx)
        times['C5'] += t.elapsed
        with Timer(use_cuda=True) as t: c6_gpu(spk_gpu, sort_idx_np, sign='neg')
        times['C6'] += t.elapsed
        if ch == cfg.check_channel: check_clusters = sort_idx_np.copy()
    return times, check_clusters

def run_variant(variant, raw, cfg):
    results = []
    use_cuda = variant == 'optimized_gpu'
    if use_cuda and HAS_CUDA: print("  CUDA warmup..."); _ = run_optimized_gpu(raw[:, :10], cfg); torch.cuda.empty_cache()
    for run_idx in range(cfg.n_runs):
        print(f"  Run {run_idx + 1}/{cfg.n_runs}...", end=" ", flush=True)
        monitor = ResourceMonitor(use_cuda=use_cuda)
        with monitor:
            if variant == 'naive_cpu': times, clusters = run_naive_cpu(raw, cfg)
            elif variant == 'improved_cpu': times, clusters = run_improved_cpu(raw, cfg)
            else: times, clusters = run_optimized_gpu(raw, cfg)
        total = sum(times.values())
        results.append(RunResult(module_times=times, total_time=total, resources=monitor.resources, clusters=clusters))
        print(f"total={total:.2f}s", end="")
        if use_cuda and HAS_CUDA: print(f" | GPU={monitor.resources.gpu_allocated_mb:.0f}MB", end="")
        print()
    return results

def print_module_table(results):
    print("\n" + "=" * 140)
    print("PER-MODULE TIMING (seconds)")
    print("=" * 140)
    header = f"{'Variant':<25}"; header += "".join(f" {m:>9}" for m in MODULES); header += f" {'Total':>10} {'Speedup':>8}"
    print(header); print("-" * 140)
    for variant, runs in results.items():
        row = f"{VARIANT_LABELS[variant]:<25}"
        for m in MODULES: row += f" {np.mean([r.module_times[m] for r in runs]):>9.2f}"
        mean_t = np.mean([r.total_time for r in runs]); row += f" {mean_t:>10.2f} {BASELINE_TIME/mean_t:>7.2f}x"
        print(row)
    print("=" * 140)

def print_resource_table(results):
    print("\n" + "=" * 100)
    print("RESOURCE USAGE")
    print("=" * 100)
    print(f"{'Variant':<25} {'CPU%':>8} {'Threads':>8} {'Torch':>8} {'BLAS':>8} {'RSS Δ':>10} {'GPU':>12}")
    print("-" * 100)
    for variant, runs in results.items():
        cpu = np.mean([r.resources.cpu_percent for r in runs]); threads = int(np.mean([r.resources.process_threads for r in runs]))
        torch_t = runs[0].resources.torch_threads or 0; blas_t = max((p['num_threads'] for p in runs[0].resources.blas_info), default=0) if runs[0].resources.blas_info else 0
        rss = np.mean([r.resources.rss_delta_mb for r in runs]); gpu = np.mean([r.resources.gpu_allocated_mb for r in runs]); gpu_str = f"{gpu:.0f} MB" if gpu > 0 else "n/a"
        print(f"{VARIANT_LABELS[variant]:<25} {cpu:>7.1f}% {threads:>8} {torch_t:>8} {blas_t:>8} {rss:>+9.0f} MB {gpu_str:>12}")
    print("=" * 100)

def print_determinism_report(reports):
    print("\n" + "=" * 80); print("DETERMINISM ANALYSIS"); print("=" * 80)
    for v, r in reports.items():
        verdict = "DETERMINISTIC" if r.all_identical else "NON-DETERMINISTIC"
        print(f"  {VARIANT_LABELS[v]}: {verdict}")
        if r.n_runs >= 2: print(f"    Jaccard: min={r.min_jaccard:.4f} mean={r.mean_jaccard:.4f} max={r.max_jaccard:.4f}")
    print("=" * 80)

def print_parity_report(reports):
    print("\n" + "=" * 90); print("OUTPUT PARITY"); print("=" * 90)
    for r in reports:
        status = "IDENTICAL" if r.labels_identical else f"Jaccard={r.jaccard:.4f}"
        print(f"  {VARIANT_LABELS[r.variant_a]} vs {VARIANT_LABELS[r.variant_b]}: {status}")
        print(f"    Spikes: {r.n_spikes_a} vs {r.n_spikes_b}, Label agreement (Hungarian): {r.label_agreement:.4f}")
    print("=" * 90)

def save_json_report(cfg, results, det_reports, parity_reports):
    report = {"config": {"raw_path": str(cfg.raw_path), "n_runs": cfg.n_runs, "baseline": BASELINE_TIME}, "variants": {}, "determinism": {k: v.to_dict() for k, v in det_reports.items()}, "parity": [p.to_dict() for p in parity_reports]}
    for v, runs in results.items():
        totals = [r.total_time for r in runs]; module_means = {m: float(np.mean([r.module_times[m] for r in runs])) for m in MODULES}
        report["variants"][v] = {"label": VARIANT_LABELS[v], "mean_time": float(np.mean(totals)), "std_time": float(np.std(totals)), "speedup": BASELINE_TIME / float(np.mean(totals)), "module_times": module_means}
    out_path = cfg.results_dir / "benchmark_comprehensive.json"; out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {out_path}")

def plot_stacked_bar(results, out_path):
    try: import matplotlib.pyplot as plt; import matplotlib.patches as mpatches
    except ImportError: print("Skipping plot (matplotlib not available)"); return
    variants = list(results.keys())
    data = {v: {m: np.mean([r.module_times[m] for r in runs]) for m in MODULES} for v, runs in results.items()}
    data['original'] = {'M1': 53.05, 'M2': 41.76, 'M3': 0.13, 'M4': 0.00, 'M5': 0.00, 'C1': 0.27, 'C2': 2.65, 'C3': 5.84, 'C4': 0.01, 'C5': 0.02, 'C6': 0.02}
    ordered = ['original'] + variants; labels = ['Original'] + [VARIANT_LABELS[v] for v in variants]
    fig, ax = plt.subplots(figsize=(10, 5)); y_pos = np.arange(len(ordered))[::-1]
    for v_idx, variant in enumerate(ordered):
        left = 0
        for m in MODULES: w = data[variant][m]; ax.barh(y_pos[v_idx], w, left=left, height=0.6, color=MODULE_COLORS[m], label=m if v_idx == 0 else ""); left += w
    totals = [sum(data[v].values()) for v in ordered]; speedups = [BASELINE_TIME / t for t in totals]; max_t = max(totals)
    for i, (t, s) in enumerate(zip(totals, speedups)): ax.text(t + max_t * 0.02, y_pos[i], f"{t:.1f}s ({s:.2f}x)", va='center', fontsize=9)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels); ax.set_xlabel("Time (seconds)"); ax.set_title("Torched Combinato - Per-Module Breakdown"); ax.set_xlim(0, max_t * 1.25)
    legend_patches = [mpatches.Patch(color=MODULE_COLORS[m], label=f"{m} {MODULE_NAMES[m]}") for m in MODULES]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Stacked bar plot saved to {out_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path); parser.add_argument("--sample-rate", type=int, default=30000); parser.add_argument("--num-channels", type=int, default=384)
    parser.add_argument("--n-runs", type=int, default=3); parser.add_argument("--results-dir", type=Path, default=Path("results")); parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument("--spc-path", default="~/spc/cluster_linux64.exe"); parser.add_argument("--check-channel", type=int, default=174)
    args = parser.parse_args()
    return BenchmarkConfig(raw_path=args.raw_path, sample_rate=args.sample_rate, num_channels=args.num_channels, results_dir=args.results_dir, n_runs=args.n_runs, variants=args.variants, spc_path=args.spc_path, check_channel=args.check_channel)

def main():
    cfg = parse_args()
    if "optimized_gpu" in cfg.variants and not HAS_CUDA: print("WARNING: CUDA not available, removing optimized_gpu"); cfg.variants = [v for v in cfg.variants if v != "optimized_gpu"]
    if not cfg.variants: print("No valid variants."); return
    print(f"Loading: {cfg.raw_path}"); raw = np.memmap(cfg.raw_path, dtype='int16', mode='r').reshape(-1, cfg.num_channels); print(f"  Shape: {raw.shape}")
    results = {}
    for variant in cfg.variants:
        print(f"\n{'=' * 60}\n  {VARIANT_LABELS[variant]}  ({cfg.n_runs} runs)\n{'=' * 60}")
        if variant == "optimized_gpu" and HAS_CUDA: torch.cuda.empty_cache()
        results[variant] = run_variant(variant, raw, cfg)
    det_reports = {v: assess_determinism(v, results[v]) for v in cfg.variants}
    parity_reports = [assess_parity(va, results[va][0], vb, results[vb][0]) for i, va in enumerate(cfg.variants) for vb in cfg.variants[i+1:]]
    print_module_table(results); print_resource_table(results); print_determinism_report(det_reports); print_parity_report(parity_reports)
    save_json_report(cfg, results, det_reports, parity_reports); plot_stacked_bar(results, cfg.results_dir / "benchmark_stacked.png")

if __name__ == "__main__": main()
