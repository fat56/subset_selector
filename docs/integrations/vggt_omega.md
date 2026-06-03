# VGGT-OMEGA Integration

This project integrates the local sibling checkout:

```text
/home/m/project/ltm/vggt-omega
```

The selector repository keeps a relative symlink:

```text
external/vggt-omega -> ../../vggt-omega
```

No VGGT-OMEGA source code or checkpoint is copied into this repository.

## Why Subprocess Integration

The selector environment does not need to install torch. VGGT-OMEGA inference is launched through:

```text
external/vggt-omega/.venv/bin/python
```

This keeps dependencies isolated while giving experiments a stable selector-side command.

## Preflight

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage vggt-preflight
```

The preflight checks:

- local VGGT-OMEGA root
- external Python interpreter
- importability of `vggt_omega.models.VGGTOmega`
- checkpoint paths and sizes
- VGGT-OMEGA git commit

## Cache VGGT Outputs

Example:

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage vggt-cache \
  --images data/raw/example_scene/images/0001.png data/raw/example_scene/images/0002.png \
  --output-dir caches/vggt_omega/example_scene/random_k_0001 \
  --checkpoint 512 \
  --image-resolution 512
```

Default cache outputs:

```text
<output-dir>/
├── camera_and_register_tokens.pt
├── camera_tokens.pt
├── register_tokens.pt
└── manifest.json
```

Optional outputs:

- `--include-pose` saves `pose_enc.pt`
- `--include-depth` saves `depth.pt` and `depth_conf.pt`
- `--enable-alignment --checkpoint 256_text` saves text-alignment tensors

## Output Contract

`camera_and_register_tokens.pt` has shape:

```text
[B, N, 1 + R, C]
```

where the first token is the camera token and the remaining `R` tokens are registers. `R` is read from the loaded model as `model.aggregator.patch_token_start - 1`; it is not hard-coded.

