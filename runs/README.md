# Runs

Each local execution gets a directory:

```text
runs/<experiment_id>/<run_id>/
├── manifest.yaml
├── config.yaml
├── metrics.json
├── notes.md
└── selected_indices.txt
```

Keep lightweight records in git. Put large outputs under ignored subdirectories such as `artifacts/`, `checkpoints/`, `cache/`, `logs/`, or `renders/`, and record their paths/checksums in `manifest.yaml`.

