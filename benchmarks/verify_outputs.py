"""
Verify all variants produce identical outputs.
"""

import sys
import numpy as np

sys.path.insert(0, '..')

from torched_combinato import NaiveCombinato, ImprovedCombinato, OptimizedCombinato


def verify(raw_path, channel=174, sample_rate=30000):
    raw = np.memmap(raw_path, dtype='int16', mode='r').reshape(-1, 384)
    print(f'Data: {raw.shape[0]} samples x {raw.shape[1]} channels')
    print(f'Verifying channel {channel}')
    print('=' * 60)
    
    outputs = {}
    
    # Naive
    print('\n[1/3] NaiveCombinato...')
    model = NaiveCombinato(sample_rate=sample_rate, device='cpu')
    out = model(raw)
    outputs['Naive'] = out[channel]['clusters']
    unique, counts = np.unique(outputs['Naive'], return_counts=True)
    print(f'       Clusters: {dict(zip(unique, counts))}')
    
    # Improved
    print('\n[2/3] ImprovedCombinato...')
    model = ImprovedCombinato(sample_rate=sample_rate, device='cpu')
    out = model(raw)
    outputs['Improved'] = out[channel]['clusters']
    unique, counts = np.unique(outputs['Improved'], return_counts=True)
    print(f'       Clusters: {dict(zip(unique, counts))}')
    
    # Optimized
    print('\n[3/3] OptimizedCombinato...')
    model = OptimizedCombinato(sample_rate=sample_rate)
    out = model(raw)
    outputs['Optimized'] = out[channel]['clusters']
    unique, counts = np.unique(outputs['Optimized'], return_counts=True)
    print(f'       Clusters: {dict(zip(unique, counts))}')
    
    # Compare
    print('\n' + '=' * 60)
    print('COMPARISON')
    print('=' * 60)
    
    naive_vs_improved = np.array_equal(outputs['Naive'], outputs['Improved'])
    improved_vs_optimized = np.array_equal(outputs['Improved'], outputs['Optimized'])
    
    print(f'Naive == Improved:    {"PASS" if naive_vs_improved else "FAIL"}')
    print(f'Improved == Optimized: {"PASS" if improved_vs_optimized else "FAIL"}')
    
    if naive_vs_improved and improved_vs_optimized:
        print('\nAll outputs match!')
        return True
    else:
        print('\nOutputs differ!')
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', help='Path to raw .bin file')
    parser.add_argument('--channel', type=int, default=174)
    parser.add_argument('--sample_rate', type=int, default=30000)
    args = parser.parse_args()
    
    verify(args.raw_path, args.channel, args.sample_rate)
