// Compact mission / vehicle-state strip: armed, flight mode, mission state,
// MAVLink link health. Colour reflects safety-relevant state at a glance.

export function MissionStatus({ data }) {
  const v = (data && data.vehicle) || {}
  const link = (data && data.link_status) || 'unknown'
  const mission = (data && data.mission_state) || 'IDLE'
  const armed = v.armed
  const linkOk = link === 'connected'

  return (
    <div className="panel mission">
      <div className="panel-title">Mission</div>
      <div className="status-strip">
        <div className={`chip ${armed ? 'danger' : 'ok'}`}>
          {armed === undefined ? 'ARMED: N/A' : armed ? 'ARMED' : 'DISARMED'}
        </div>
        <div className="chip">Mode: {v.mode || 'N/A'}</div>
        <div className="chip">State: {mission}</div>
        <div className={`chip ${linkOk ? 'ok' : 'warn'}`}>Link: {link}</div>
      </div>
    </div>
  )
}
