import { useEffect, useRef, useState, useCallback } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'
import { CAMERA_WS, PING_WS, API_URL } from '../config'

// Live ZED feed with detection overlays. Frames arrive as binary messages:
//   [4-byte BE header length][JSON header][image bytes]
// The JSON header carries send_time, pipeline_ms, format, and the detections.
// End-to-end latency is approximated with a clock offset measured over /ping.
export function CameraView() {
  const imgRef = useRef(null)
  const urlRef = useRef(null)
  const offsetRef = useRef(0)          // serverTime - clientTime, ms
  const lastShownRef = useRef(performance.now())
  const [meta, setMeta] = useState({ detections: [] })
  const [latency, setLatency] = useState(null)
  const [fpsDisp, setFpsDisp] = useState(0)
  const [quality, setQuality] = useState(70)
  const [targetFps, setTargetFps] = useState(20)

  const onCamera = useCallback((ev) => {
    if (typeof ev.data === 'string') return
    const buf = new DataView(ev.data)
    const headerLen = buf.getUint32(0)
    const headerBytes = new Uint8Array(ev.data, 4, headerLen)
    const header = JSON.parse(new TextDecoder().decode(headerBytes))
    const payload = ev.data.slice(4 + headerLen)

    if (payload.byteLength === 0) {
      setMeta({ detections: [], status: header.status })
      return
    }
    setMeta(header)

    // Approximate end-to-end latency using the measured clock offset.
    if (header.send_time) {
      const est = performance.timeOrigin + performance.now()
      setLatency(Math.max(0, Math.round(est - (header.send_time - offsetRef.current))))
    }

    const now = performance.now()
    const dt = now - lastShownRef.current
    lastShownRef.current = now
    if (dt > 0) setFpsDisp((f) => Math.round((f * 0.8 + (1000 / dt) * 0.2)))

    const mime = header.format === 'webp' ? 'image/webp' : 'image/jpeg'
    const blob = new Blob([payload], { type: mime })
    const url = URL.createObjectURL(blob)
    if (imgRef.current) imgRef.current.src = url
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    urlRef.current = url
  }, [])

  const { status } = useWebSocket(CAMERA_WS, { binary: true, onMessage: onCamera })

  // Clock-offset probe: ask /ping, compare with local time (ignoring RTT/2).
  const pingWs = useWebSocket(PING_WS, {
    onMessage: (ev) => {
      try {
        const { server_time } = JSON.parse(ev.data)
        const clientNow = performance.timeOrigin + performance.now()
        offsetRef.current = server_time - clientNow
      } catch { /* ignore */ }
    },
  })
  useEffect(() => {
    const id = setInterval(() => pingWs.send('ping'), 2000)
    return () => clearInterval(id)
  }, [pingWs])

  useEffect(() => () => { if (urlRef.current) URL.revokeObjectURL(urlRef.current) }, [])

  const pushStream = useCallback(async (q, fps) => {
    try {
      await fetch(`${API_URL}/api/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jpeg_quality: q, target_fps: fps }),
      })
    } catch { /* backend may be briefly down; ignore */ }
  }, [])

  return (
    <div className="panel camera">
      <div className="panel-title">
        Camera
        <span className={`badge ${status === 'open' ? 'ok' : 'warn'}`}>{status}</span>
      </div>
      <div className="camera-frame">
        <img ref={imgRef} alt="ZED feed" />
        {!meta.width && (
          <div className="camera-overlay-msg">{meta.status || 'waiting for feed…'}</div>
        )}
      </div>
      <div className="camera-stats">
        <span>latency: {latency == null ? 'N/A' : `${latency} ms`}</span>
        <span>pipeline: {meta.pipeline_ms == null ? 'N/A' : `${meta.pipeline_ms} ms`}</span>
        <span>fps: {fpsDisp}</span>
        <span>objects: {meta.detections ? meta.detections.length : 0}</span>
      </div>
      <div className="camera-controls">
        <label>
          quality {quality}
          <input type="range" min="10" max="100" value={quality}
            onChange={(e) => { const v = +e.target.value; setQuality(v); pushStream(v, targetFps) }} />
        </label>
        <label>
          fps {targetFps}
          <input type="range" min="1" max="30" value={targetFps}
            onChange={(e) => { const v = +e.target.value; setTargetFps(v); pushStream(quality, v) }} />
        </label>
      </div>
      {meta.detections && meta.detections.length > 0 && (
        <ul className="det-list">
          {meta.detections.map((d, i) => (
            <li key={i}><b>{d.label}</b> {(d.conf * 100).toFixed(0)}%</li>
          ))}
        </ul>
      )}
    </div>
  )
}
