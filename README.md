# scMeformer — Luo Lab Fork (UCLA Hoffman2)

This branch contains modifications made by Aryana Satheesh (Luo Lab, UCLA) to adapt scMeformer for use on the UCLA Hoffman2 HPC cluster. See the [original repository](https://github.com/LieberInstitute/scMeformer) for the upstream codebase.

## Branch Status

The demo dataset pipeline (single_cell_regression task) has been partially reproduced on Hoffman2. Environment setup and data loading are working. Training is currently blocked by a PyTorch 2.1 inplace tensor operation error in `METH_CNN.forward()` (see Known Issues below).

---

## Hoffman2 Environment Setup

Due to system-level constraints on Hoffman2 (GLIBC 2.17, no Docker support), a conda environment is required. A setup script is provided at `setup_hoffman2.sh`.

**Key system-specific requirements:**
- GLIBC on Hoffman2 is too old for standard PyTorch/numpy builds — patched using `conda-forge` GCC/glib
- apex must be built from the **2022 version** of the source — the current version removed `amp`
- `PYTHONNOUSERSITE=1` must be set at runtime to prevent `~/.local` system Python packages from interfering with the conda environment
- The original repo assumes 4 GPUs for training; Hoffman2 nodes typically have 1

**To set up the environment:**
```bash
bash setup_hoffman2.sh
```

**To run the demo (single_cell_regression):**
```bash
bash run_hoffman2.sh
```

---

## Code Changes

The following changes were made to the original scMeformer source code to fix compatibility issues.

### `datasets.py`
**1. `np.long` removed in numpy 1.24+**
`dtype=np.long` is no longer valid in modern numpy. Replaced all instances with `np.int64`.
```bash
sed -i 's/np\.long/np.int64/g' datasets.py
```

**2. String iteration bug in `for chrom in split`**
The dataset class receives `split` as a plain string (e.g. `"chr1"`). Iterating directly over it produces individual characters (`"c"`, `"h"`, `"r"`, `"1"`), causing file-not-found errors like `position/c.npy`. Fixed by wrapping in a list.
```bash
sed -i 's/for chrom in split:/for chrom in [split]:/g' datasets.py
```

### `training.py`
**3. Missing `from apex import amp` import**
`training.py` calls `amp.initialize()` for fp16 training but never imports `amp`. Added the missing import.
```bash
sed -i 's/import torch$/import torch\nfrom apex import amp/' training.py
```

### `models/modeling_bert.py`
**4. dtype mismatch: float64 input vs float32 model weights**
`.npy` files are loaded as float64 by default, but the CNN layers expect float32. Cast `methyl_data` before passing to the CNN.
```bash
sed -i 's/sequence_outputs, feature_outputs = self.feature_cnn(methyl_data)/sequence_outputs, feature_outputs = self.feature_cnn(methyl_data.float())/' models/modeling_bert.py
```

---

## Data Format Note

The original code expects `genome.npy` and `position.npy` as single dictionary `.npy` files where keys are chromosome names. The demo data provided by the authors uses individual per-chromosome files in subdirectories (e.g. `datasets/genome/chr1.npy`). Use the following script to convert:

```python
import numpy as np
genome = {}
for chrom in ['chr1', 'chr21', 'chr22']:
    genome[chrom] = np.load(f'./datasets/genome/{chrom}.npy', allow_pickle=True)
np.save('./datasets/genome.npy', genome)
```

---

## Known Issues

**PyTorch 2.1 inplace tensor operation error (unresolved)**
`METH_CNN.forward()` repeatedly reassigns the same variable `sequence` across operations, which causes PyTorch 2.1 to lose track of intermediate values needed for backpropagation.
Attempted fix: added `.clone()` after transpose operations to force PyTorch to treat tensors as independent copies. Error persists. Under investigation.

**GPU memory constraints**
Available Hoffman2 GPU nodes (Tesla P4, 7680MB) are undersized relative to what the original repo was designed for (4x GPU training). Use `CUDA_VISIBLE_DEVICES` to target a free GPU and reduce `--batch_size` if needed.

---

## Next Steps

- Resolve PyTorch 2.1 inplace tensor operation error in `METH_CNN.forward()`
- Update hardcoded paths in `run_feature.py` for Luo Lab data (cluster assignment file and methylation TSV paths)
- Generate `methyl_data/<chrom>.npy` files (currently missing from pipeline — source unclear)
- Adapt pipeline for Luo Lab snm3C-seq dataset
