"""
Quick output verification for Torched Combinato variants.

Runs each variant once and compares outputs on a single channel.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from torched_combinato import NaiveCombinato, ImprovedCombinato, OptimizedCombinato


def verify(raw_path: str, channel: int = 174, sample_rate: int = 30000):
    raw = np.memmap(raw_path, dtype='int16', mode='r').reshape(-1, 384)
    print(f"Data: {raw.shape}")
    print(f"Checking channel {channel}")
    print("=" * 60)
    
    outputs = {}
    
    # Naive
    print("\n[1/3] NaiveCombinato...")
    model = NaiveCombinato(sample_rate=sample_rate, device='cpu')
    out = model(raw)
    outputs['naive'] = out[channel]['clusters']
    unique, counts = np.unique(outputs['naive'], return_counts=True)
    print(f"  Clusters: {dict(zip(unique, counts))}")
    
    # Improved  
    print("\n[2/3] ImprovedCombinato...")
    model = ImprovedCombinato(sample_rate=sample_rate, device='cpu')
    out = model(raw)
    outputs['improved'] = out[channel]['clusters']
    unique, counts = np.unique(outputs['improved'], return_counts=True)
    print(f"  Clusters: {dict(zip(unique, counts))}")
    
    # Optimized
    print("\n[3/3] OptimizedCombinato...")
    model = OptimizedCombinato(sample_rate=sample_rate)
    out = model(raw)
    outputs['optimized'] = out[channel]['clusters']
    unique, counts = np.unique(outputs['optimized'], return_counts=True)
    print(f"  Clusters: {dict(zip(unique, counts))}")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    naive_vs_improved = np.array_equal(outputs['naive'], outputs['improved'])
    improved_vs_optimized = np.array_equal(outputs['improved'], outputs['optimized'])
    
    print(f"Naive == Improved:     {'PASS' if naive_vs_improved else 'FAIL'}")
    print(f"Improved == Optimized: {'PASS' if improved_vs_optimized else 'FAIL'}")
    
    if naive_vs_improved and improved_vs_optimized:
        print("\n✓ All outputs identical!")
        return True
    else:
        print("\n✗ Outputs differ!")
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', help='Path to raw .bin file')
    parser.add_argument('--channel', type=int, default=174)
    parser.add_argument('--sample-rate', type=int, default=30000)
    args = parser.parse_args()
    
    verify(args.raw_path, args.channel, args.sample_rate)
