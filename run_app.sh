#!/usr/bin/env bash
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJ/src:$PYTHONPATH"
streamlit run "$PROJ/app/app.py"
