#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m compileall src tests
"${PYTHON_BIN}" -m vggt_omega_selector.cli.manage --help >/dev/null
"${PYTHON_BIN}" -m vggt_omega_selector.cli.manage vggt-preflight --strict >/dev/null
"${PYTHON_BIN}" -m vggt_omega_selector.cli.manage record-run \
  --experiment 0001_stage1_register_quality_gate \
  --stage stage1 \
  --method smoke_test \
  --dataset placeholder_dataset \
  --config configs/experiments/0001_stage1_register_quality_gate.yaml \
  --run-id smoke_test \
  --notes "CLI smoke test" >/dev/null

rm -rf runs/0001_stage1_register_quality_gate/smoke_test
rmdir runs/0001_stage1_register_quality_gate 2>/dev/null || true
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

ledger = Path("docs/registry/run_ledger.csv")
lines = ledger.read_text(encoding="utf-8").splitlines()
filtered = [line for line in lines if ",smoke_test," not in line and not line.startswith("smoke_test,")]
ledger.write_text("\n".join(filtered) + "\n", encoding="utf-8")
PY
find src tests -type d -name __pycache__ -prune -exec rm -rf {} +

echo "smoke test passed"
