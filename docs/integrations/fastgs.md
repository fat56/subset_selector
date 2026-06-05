# FastGS Integration

Stage 1 uses FastGS as the first GS reconstruction backend.

## Source

- Repository: <https://github.com/fastgs/FastGS>
- Verified HEAD during scaffold setup: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`
- Selector-side config: [configs/integrations/fastgs.yaml](../../configs/integrations/fastgs.yaml)

FastGS is expected as an external checkout, not vendored into this repository:

```text
external/FastGS
```

Override paths with `FASTGS_ROOT`, `FASTGS_PYTHON`, or CLI flags.

## Preflight

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage fastgs-preflight --strict
```

The preflight checks the root, Python interpreter, `train.py`, PyTorch/CUDA,
visible GPUs, and imports for:

- `simple_knn._C`
- `fused_ssim`
- `diff_gaussian_rasterization_fastgs`

Current verified local environment:

```text
FastGS commit: 44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f
Python: 3.10.18 via uv
PyTorch: 2.11.0+cu128
torch CUDA runtime: 12.8
CUDA toolkit for extension build: /usr/local/cuda-12.8
TORCH_CUDA_ARCH_LIST: 12.0
NumPy: 1.26.4
GPUs: 2 x NVIDIA GeForce RTX 5090, capability 12.0
```

`git_dirty` may become true inside `external/FastGS` after local extension
builds because setup metadata and build folders are generated in the external
checkout. The selector repo ignores `external/FastGS/`.

## RTX 5090 Setup With uv

This setup follows the same migration principle as
<https://github.com/fat56/VFM_GS/blob/main/docs/migration_5090.md>: do not reuse
old 4090/cu116 virtualenvs; rebuild local CUDA extensions against a Blackwell
compatible PyTorch/CUDA stack.

```bash
git clone --recursive https://github.com/fastgs/FastGS.git external/FastGS
uv venv external/FastGS/.venv --python 3.10
```

Install PyTorch CUDA 12.8 wheels. The timeout avoids failures while extracting
large wheels:

```bash
UV_HTTP_TIMEOUT=600 uv pip install \
  --python external/FastGS/.venv/bin/python \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Install runtime and build dependencies:

```bash
UV_HTTP_TIMEOUT=600 uv pip install \
  --python external/FastGS/.venv/bin/python \
  "numpy<2" plyfile pyyaml tqdm websockets packaging wheel ninja
```

Compile FastGS CUDA extensions for RTX 5090:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export MAX_JOBS=8

uv pip install --python external/FastGS/.venv/bin/python --no-build-isolation \
  external/FastGS/submodules/simple-knn
uv pip install --python external/FastGS/.venv/bin/python --no-build-isolation \
  external/FastGS/submodules/fused-ssim
uv pip install --python external/FastGS/.venv/bin/python --no-build-isolation \
  external/FastGS/submodules/diff-gaussian-rasterization_fastgs
```

## Command Shape

`stage1-prepare` writes one `fastgs_train.sh` per prepared scene/method. The generated command follows FastGS' `train.py` interface:

```bash
python train.py \
  --source_path <prepared_fastgs_source> \
  --model_path <prepared_fastgs_output> \
  --images images \
  --eval
```

The prepared source is a sparse-view COLMAP scene: selected images are symlinked, and `sparse/0` text or binary sparse models are filtered to a selected-image text model.
