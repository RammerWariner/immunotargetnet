#!/usr/bin/env bash
# Run the study and the shortlist dossier, then verify the release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${1:-20260101}"
EPOCHS="${2:-60}"
SEEDS="${3:-5}"
OUT="${ROOT}/runs"

"${PYTHON_BIN}" -m immunotargetnet.cli.evaluate --root "${ROOT}" --seed "${SEED}" \
  --epochs "${EPOCHS}" --seeds "${SEEDS}" --out "${OUT}"
"${PYTHON_BIN}" -m immunotargetnet.cli.prioritise --root "${ROOT}" --seed "${SEED}" \
  --epochs "${EPOCHS}" --top 12 --out "${OUT}"
"${PYTHON_BIN}" -m immunotargetnet.cli.verify --root "${ROOT}" --epochs 2 --stability-draws 3
