// Renders the merged telemetry snapshot. Any field the backend reports as null
// (sensor absent / message not yet seen) shows "N/A" rather than a wrong zero.

function fmt(v, unit = '', digits = 1) {
  if (v === null || v === undefined) return 'N/A'
  if (typeof v === 'number') return `${v.toFixed(digits)}${unit}`
  return `${v}${unit}`
}

function Row({ label, value }) {
  return (
    <div className="tele-row"><span>{label}</span><b>{value}</b></div>
  )
}

export function TelemetryPanel({ data }) {
  const v = (data && data.vehicle) || {}
  const h = (data && data.host) || {}
  const thrusters = v.thrusters || []

  return (
    <div className="panel telemetry">
      <div className="panel-title">Telemetry</div>

      <div className="tele-grid">
        <Row label="Roll" value={fmt(v.roll, '°')} />
        <Row label="Pitch" value={fmt(v.pitch, '°')} />
        <Row label="Yaw" value={fmt(v.yaw, '°')} />
        <Row label="Heading" value={fmt(v.heading, '°')} />
        <Row label="Depth" value={fmt(v.depth, ' m', 2)} />
        <Row label="Vx / Vy / Vz" value={`${fmt(v.vx)} / ${fmt(v.vy)} / ${fmt(v.vz)}`} />
        <Row label="Battery" value={fmt(v.voltage, ' V', 2)} />
        <Row label="Current" value={fmt(v.current, ' A', 1)} />
        <Row label="Batt %" value={fmt(v.battery_remaining, ' %', 0)} />
        <Row label="Water temp" value={fmt(v.temperature, ' °C')} />
        <Row label="Leak" value={v.leak === null || v.leak === undefined ? 'N/A' : (v.leak ? '⚠ LEAK' : 'dry')} />
      </div>

      <div className="tele-sub">Thrusters (µs)</div>
      <div className="thrusters">
        {thrusters.length === 0 ? <span>N/A</span> : thrusters.map((t, i) => (
          <div key={i} className="thruster">
            <div className="bar" style={{ height: `${Math.min(100, Math.max(0, (t - 1100) / 8))}%` }} />
            <span>{t}</span>
          </div>
        ))}
      </div>

      <div className="tele-sub">System</div>
      <div className="tele-grid">
        <Row label="CPU" value={fmt(h.cpu, ' %', 0)} />
        <Row label="GPU" value={fmt(h.gpu, ' %', 0)} />
        <Row label="RAM" value={fmt(h.ram, ' %', 0)} />
        <Row label="Board temp" value={fmt(h.temperature, ' °C')} />
      </div>
    </div>
  )
}
