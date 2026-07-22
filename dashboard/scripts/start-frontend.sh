#!/usr/bin/env bash
# Start the frontend. Serves the built dashboard (npm run preview) if a build
# exists, else runs the Vite dev server. Binds 0.0.0.0:3000 so the Mac reaches
# it at http://192.168.2.2:3000.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$HERE/../frontend"
cd "$FRONTEND"

if [ ! -d node_modules ]; then
  echo "[frontend] installing deps…"
  npm install
fi

if [ "${1:-}" = "dev" ]; then
  exec npm run dev
fi

if [ ! -d dist ]; then
  echo "[frontend] building…"
  npm run build
fi
exec npm run preview
