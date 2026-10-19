#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-$PROJECT_DIR/untie_venv/bin/python}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/artifacts/qa_chunk_selection_v4_4}"
DEVICE="${DEVICE:-auto}"
SMOKE="${SMOKE:-0}"
SUPPORT="${SUPPORT:-1}"
SOFT="${SOFT:-0}"
LOG="${LOG:-$PROJECT_DIR/logs/qa_chunk_selection_v4_4_$(date +%Y%m%d_%H%M%S)_$$.log}"
ARGS=(--output-dir "$OUT_DIR" --device "$DEVICE")
if [[ "$SMOKE" == 1 ]]; then ARGS+=(--smoke); fi
if [[ "$SUPPORT" == 0 ]]; then ARGS+=(--no-support); fi
if [[ "$SOFT" == 1 ]]; then ARGS+=(--soft); fi
"$PYTHON" -B -m experiments.qa_chunk_selection.v4_4 --output-dir "$OUT_DIR" --check-output
# Validate before creating any log; log paths cannot target prior artifact trees.
"$PYTHON" -B - "$LOG" "$PROJECT_DIR" <<'PY'
import sys
from pathlib import Path
path, project = Path(sys.argv[1]).absolute(), Path(sys.argv[2]).resolve()
if any(p.is_symlink() for p in (path, *path.parents)):
    raise SystemExit("Refusing symlink log path")
if (project / "logs") not in path.resolve().parents or path.suffix != ".log":
    raise SystemExit("LOG must be a new .log file below project/logs")
if path.exists() or Path(str(path)+".pid").exists():
    raise SystemExit("Refusing to overwrite a log or PID file")
PY
mkdir -p -- "$(dirname -- "$LOG")"
set -o noclobber
# noclobber also protects against simultaneous launchers selecting the same LOG.
exec 3> "$LOG"
nohup "$PYTHON" -B -u -m experiments.qa_chunk_selection.v4_4 "${ARGS[@]}" >&3 2>&1 < /dev/null &
RUN_PID=$!
exec 3>&-
printf '%s\n' "$RUN_PID" > "$LOG.pid"
printf 'Started PID=%s\nOutput: %s\nLog: %s\n' "$RUN_PID" "$OUT_DIR" "$LOG"
printf 'Watch: tail -f %q\n' "$LOG"
