#!/usr/bin/env bash
# Rebuild the programme table from the gene-set corpus and, when asked, refresh the
# keystone snapshot from the public platform.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

REFRESH="${1:-no}"
CORPUS="${ROOT}/data/msigdb"

if [[ ! -d "${CORPUS}/GOBP" ]]; then
  echo "gene-set corpus missing under ${CORPUS}; place the three collections there first" >&2
  exit 1
fi

if [[ "${REFRESH}" == "refresh" ]]; then
  exec "${PYTHON_BIN}" -m immunotargetnet.cli.build_evidence \
    --assets "${ROOT}/assets" --corpus-root "${CORPUS}" \
    --refresh --budget-seconds 900 --max-queries 400
fi

exec "${PYTHON_BIN}" -m immunotargetnet.cli.build_evidence \
  --assets "${ROOT}/assets" --corpus-root "${CORPUS}"
