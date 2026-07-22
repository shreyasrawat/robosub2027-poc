#!/usr/bin/env bash
# Start the dashboard backend (REST :8000 + WS :8001). Binds 0.0.0.0 so the
# Mac reaches it over the direct Ethernet link. Env vars (DASH_*) override
# ports/host/quality — see dashboard/backend/settings.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$HERE/../backend"
cd "$BACKEND"
exec python3 run_backend.py
