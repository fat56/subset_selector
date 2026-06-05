# External Links

`vggt-omega` is a symlink to the sibling VGGT-OMEGA checkout:

```text
external/vggt-omega -> ../../vggt-omega
```

The selector project does not vendor VGGT-OMEGA source code or checkpoints. It records paths and launches the VGGT-OMEGA virtual environment when inference caches are needed.

FastGS is also treated as an external checkout:

```text
external/FastGS -> <your FastGS checkout>
```

Use `FASTGS_ROOT` / `FASTGS_PYTHON` or the CLI `--fastgs-root` /
`--fastgs-python` flags if the checkout lives elsewhere.
