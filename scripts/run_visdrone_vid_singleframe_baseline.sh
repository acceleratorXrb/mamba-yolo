#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec env PYTHON_BIN="$ROOT/.conda/mambayolo/bin/python" DEVICE=0 \
  bash "$ROOT/scripts/run_visdrone_vid_singleframe_official_upstream.sh"
