#!/usr/bin/env bash
# serve_auditions.sh — serve the audition web pages over http://localhost.
#
# WHY THIS EXISTS: the audition pages load their data with fetch("manifest.json").
# If you open index.html as a file:// URL (double-click), the browser blocks fetch()
# of local files for security and you get "TypeError: Failed to fetch" / "Failed to
# load manifest.json". Serving the folder over http://localhost fixes it — the
# manifest and FLAC clips are then same-origin HTTP requests.
#
# Usage:
#   bash scripts/serve_auditions.sh [PORT]        # default PORT=8000
#
# Then open in a browser:
#   http://localhost:<PORT>/renders/fusion_audition/index.html
#   http://localhost:<PORT>/renders/sweep_audition/index.html
#
# Stop with Ctrl-C. (Re-run scripts/build_manifest.py after adding new renders.)

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8000}"
PY="${PY:-$REPO/sat-venv/bin/python}"
[ -x "$PY" ] || PY=python3

cd "$REPO"

echo "Serving $REPO  ->  http://localhost:${PORT}"
echo "  Fusion audition: http://localhost:${PORT}/renders/fusion_audition/index.html"
echo "  Sweep  audition: http://localhost:${PORT}/renders/sweep_audition/index.html"
echo "Ctrl-C to stop."
exec "$PY" -m http.server "$PORT" --bind 127.0.0.1
