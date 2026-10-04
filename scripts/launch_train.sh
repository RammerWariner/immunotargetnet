#!/usr/bin/env bash
# Train one reported configuration. Set PYTHON_BIN to pick an interpreter; the package is
# not installed, so src/ goes on the path explicitly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIGURATION="${1:-full}"
SEED="${2:-20260101}"
EPOCHS="${3:-60}"

if command -v torchrun >/dev/null 2>&1 && [[ "${WORLD_SIZE:-1}" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="${WORLD_SIZE}" \
    -m immunotargetnet.cli.train \
    --root "${ROOT}" --configuration "${CONFIGURATION}" --seed "${SEED}" --epochs "${EPOCHS}"
fi

exec "${PYTHON_BIN}" -m immunotargetnet.cli.train \
  --root "${ROOT}" --configuration "${CONFIGURATION}" --seed "${SEED}" --epochs "${EPOCHS}"
