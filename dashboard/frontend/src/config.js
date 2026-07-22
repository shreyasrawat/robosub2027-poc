// Runtime endpoint resolution. Never hardcodes localhost: when the VITE_* env
// vars are unset, every URL is derived from the browser's current hostname, so
// the same build works in `vite dev` on a laptop and served from the Jetson at
// http://192.168.2.2:3000 without rebuilding.

const host = typeof window !== 'undefined' ? window.location.hostname : 'localhost'

const REST_PORT = import.meta.env.VITE_REST_PORT || 8000
const WS_PORT = import.meta.env.VITE_WS_PORT || 8001

export const API_URL =
  import.meta.env.VITE_API_URL || `http://${host}:${REST_PORT}`

const WS_BASE =
  import.meta.env.VITE_WS_URL || `ws://${host}:${WS_PORT}`

export const CAMERA_WS = `${WS_BASE}/camera`
export const TELEMETRY_WS = `${WS_BASE}/telemetry`
export const PING_WS = `${WS_BASE}/ping`
