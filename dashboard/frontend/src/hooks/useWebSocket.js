import { useEffect, useRef, useState, useCallback } from 'react'

// Auto-reconnecting WebSocket hook. Survives the Ethernet cable being pulled:
// on close/error it retries with capped backoff, so replugging the cable brings
// the stream back with no user action. `binary` selects arraybuffer framing for
// the camera channel. `onMessage` gets each raw message event.
export function useWebSocket(url, { binary = false, onMessage } = {}) {
  const [status, setStatus] = useState('connecting')
  const wsRef = useRef(null)
  const retryRef = useRef(0)
  const closedRef = useRef(false)
  const cbRef = useRef(onMessage)
  cbRef.current = onMessage

  const connect = useCallback(() => {
    if (closedRef.current) return
    setStatus('connecting')
    let ws
    try {
      ws = new WebSocket(url)
    } catch {
      scheduleRetry()
      return
    }
    if (binary) ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      retryRef.current = 0
      setStatus('open')
    }
    ws.onmessage = (ev) => cbRef.current && cbRef.current(ev)
    ws.onclose = () => {
      setStatus('closed')
      scheduleRetry()
    }
    ws.onerror = () => ws.close()

    function scheduleRetry() {
      if (closedRef.current) return
      const delay = Math.min(500 * 2 ** retryRef.current, 5000)
      retryRef.current += 1
      setTimeout(connect, delay)
    }
  }, [url, binary])

  useEffect(() => {
    closedRef.current = false
    connect()
    return () => {
      closedRef.current = true
      if (wsRef.current) wsRef.current.close()
    }
  }, [connect])

  const send = useCallback((data) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(data)
  }, [])

  return { status, send }
}
