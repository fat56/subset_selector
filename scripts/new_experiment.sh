#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m vggt_omega_selector.cli.manage new-experiment "$@"
