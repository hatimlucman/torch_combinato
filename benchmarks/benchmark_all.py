"""
Benchmark all Torched Combinato variants.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '..')

from torched_combinato import NaiveCombinato, ImprovedCombinato, OptimizedCombinato


def benchmark(raw_path, sample_rate=30000):
    raw = np.memmap(raw_path, dtype='int16', mode='r').reshape(-1, 384)
    print(f'Data: {raw.shape[0]} samples x {raw.shape[1]} channels')
    print('=' * 60)
    
    results = {}
    
    # Naive CPU
    print('\n[1/3] NaiveCombinato (CPU)...')
    model = NaiveCombinato(sample_rate=sample_rate, device='cpu')
    t0 = time.time()
    out = model(raw)
    results['Naive CPU'] = time.time() - t0
    print(f'       Time: {results["Naive CPU"]:.2f}s | Channels: {len(out)}')
    
    # Improved CPU
    print('\n[2/3] ImprovedCombinato (CPU)...')
    model = ImprovedCombinato(sample_rate=sample_rate, device='cpu')
    t0 = time.time()
    out = model(raw)
    results['Improved CPU'] = time.time() - t0
    print(f'       Time: {results["Improved CPU"]:.2f}s | Channels: {len(out)}')
    
    # Optimized GPU+CPU
    print('\n[3/3] OptimizedCombinato (GPU+CPU)...')
    model = OptimizedCombinato(sample_rate=sample_rate)
    # Warmup
    _ = model(raw[:, :10])
    t0 = time.time()
    out = model(raw)
    results['Optimized GPU'] = time.time() - t0
    print(f'       Time: {results["Optimized GPU"]:.2f}s | Channels: {len(out)}')
    
    # Summary
    baseline = 103.75  # Original Combinato
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'{"Pipeline":<20} {"Time":>10} {"Speedup":>10}')
    print('-' * 42)
    print(f'{"Original Combinato":<20} {baseline:>10.2f}s {1.0:>10.2f}x')
    for name, t in results.items():
        print(f'{name:<20} {t:>10.2f}s {baseline/t:>10.2f}x')
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', help='Path to raw .bin file')
    parser.add_argument('--sample_rate', type=int, default=30000)
    args = parser.parse_args()
    
    benchmark(args.raw_path, args.sample_rate)
