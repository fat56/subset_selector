# Stage 2.0 Readout Train500 Manifest

- Created: 2026-06-07
- Source root: `/home/m/dataset/ltm_datasets`
- Selected scenes: 500
- Full frames per scene: 16
- Total full frames: 8000
- Random seed: 20260607
- Excluded validation scenes: 30

## Selected Scene Counts

| Key | Count |
|---|---:|
| dataset:DL3DV-ALL-480P | 250 |
| dataset:wildrgbd_harrison | 250 |
| depth:colmap_photometric_bin | 250 |
| depth:sensor_depth_png | 250 |

## Training Scope

- This manifest caches only the full-view token set for each scene.
- Training samples subset masks online from the cached full tokens.
- LTM30 hard subset native metrics remain validation-only for this MVP.
