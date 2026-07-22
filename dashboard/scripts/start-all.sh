#!/usr/bin/env bash
# Launch backend + frontend together, each in a restart-on-crash supervisor
# loop. Ctrl-C stops both. This is the "plug in the cable and go" launcher:
# after it prints "ready", open http://192.168.2.2:3000 on the Mac.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pids=()
cleanup() {
  echo; echo "[start-all] stopping…"
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

supervise() {
  local name="$1"; shift
  while true; do
    echo "[start-all] ($name) starting"
    "$@"
    local code=$?
    echo "[start-all] ($name) exited ($code); restarting in 2s"
    sleep 2
  done
}

supervise backend  bash "$HERE/start-backend.sh"  & pids+=("$!")
supervise frontend bash "$HERE/start-frontend.sh" & pids+=("$!")

echo "[start-all] ready — backend :8000 / ws :8001 / frontend :3000"
echo "[start-all] open http://192.168.2.2:3000 on the Mac"
wait
