# Torched Combinato

PyTorch reimplementation of the [Combinato](https://github.com/jniediek/combinato) spike sorting pipeline.

## Performance

Benchmarked on 384 channels, 10% of Neuropixel recording (~81s at 30kHz):

| Pipeline | Time | Speedup |
|----------|------|---------|
| Original Combinato | 103.75s | 1.00x |
| NaiveCombinato (CPU) | 55.99s | 1.85x |
| ImprovedCombinato (CPU) | 44.82s | 2.31x |
| **OptimizedCombinato (GPU+CPU)** | **24.12s** | **4.30x** |

## Variants

- **NaiveCombinato**: Per-channel sequential processing. Emulates original Combinato data flow.
- **ImprovedCombinato**: Batches M1 preprocessing (192 channels). Faster on CPU.
- **OptimizedCombinato**: Hybrid GPU+CPU. M1/M2 batched on GPU, M3 on CPU (avoids GPU overhead), M4-C6 on GPU.

## Usage

```python
from torched_combinato import OptimizedCombinato
import numpy as np

raw = np.memmap('recording.bin', dtype='int16', mode='r').reshape(-1, 384)
model = OptimizedCombinato(sample_rate=30000)
results = model(raw)
print(results[174]['clusters'])
```

## Benchmarks

```bash
cd benchmarks
python benchmark_all.py /path/to/raw.bin
python verify_outputs.py /path/to/raw.bin
```

## Requirements

- PyTorch
- torchaudio  
- numpy
- scipy
- SPC binary (for clustering)
