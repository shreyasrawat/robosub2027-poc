import { useRef, useState, useCallback } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { TELEMETRY_WS } from './config'
import { CameraView } from './components/CameraView'
import { TelemetryPanel } from './components/TelemetryPanel'
import { Vehicle3D } from './components/Vehicle3D'
import { MissionStatus } from './components/MissionStatus'

export default function App() {
  const [telemetry, setTelemetry] = useState(null)
  // Attitude feeds the 3D loop via a ref so high-rate samples never rerender it.
  const attitudeRef = useRef({ roll: 0, pitch: 0, yaw: 0 })

  const onTelemetry = useCallback((ev) => {
    try {
      const snap = JSON.parse(ev.data)
      setTelemetry(snap)
      const v = snap.vehicle || {}
      // quat (ZED orientation) is preferred by Vehicle3D; euler is the fallback.
      attitudeRef.current = {
        roll: v.roll || 0, pitch: v.pitch || 0, yaw: v.yaw || 0,
        quat: v.quat || null,
      }
    } catch { /* ignore malformed */ }
  }, [])

  const { status } = useWebSocket(TELEMETRY_WS, { onMessage: onTelemetry })

  return (
    <div className="app">
      <header className="app-header">
        <h1>RoboSub 2027 — Operator Dashboard</h1>
        <span className={`badge ${status === 'open' ? 'ok' : 'warn'}`}>telemetry: {status}</span>
      </header>
      <MissionStatus data={telemetry} />
      <main className="grid">
        <CameraView />
        <div className="side">
          <Vehicle3D attitudeRef={attitudeRef} />
          <TelemetryPanel data={telemetry} />
        </div>
      </main>
    </div>
  )
}
