import { useEffect, useRef } from 'react'
import * as THREE from 'three'

// Real-time 3D vehicle attitude, driven by the ZED's fused pose.
//
// Orientation is applied as a quaternion rather than euler angles: the backend
// sends the ZED's orientation in its RIGHT_HANDED_Z_UP_X_FWD frame
// (X forward, Y left, Z up) and we rotate it into Three.js' frame
// (X right, Y up, Z toward the viewer) with a fixed change-of-basis. That
// removes all euler-order/sign ambiguity — the axes cannot be "swapped" by
// accident. The model is built nose-along -Z so an identity ZED pose renders
// as a level, forward-facing vehicle.
//
// Runs its own requestAnimationFrame loop and slerps toward the latest sample,
// so the model stays smooth and updates independently of the camera frame rate.
// Telemetry arrives via a ref (not state) so new samples never rerender the canvas.
export function Vehicle3D({ attitudeRef }) {
  const mountRef = useRef(null)

  useEffect(() => {
    const mount = mountRef.current
    const width = mount.clientWidth
    const height = mount.clientHeight || 260

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0d1117)

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100)
    // Sit on the -Z side so the nose faces the viewer.
    camera.position.set(2.4, 1.6, -2.6)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1))
    const dir = new THREE.DirectionalLight(0xffffff, 0.8)
    dir.position.set(3, 5, 2)
    scene.add(dir)

    // Simple AUV proxy: hull long along Z, nose cone marking -Z (forward).
    const rov = new THREE.Group()
    const hull = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 0.5, 1.4),
      new THREE.MeshStandardMaterial({ color: 0x1f6feb, metalness: 0.3, roughness: 0.6 }),
    )
    rov.add(hull)
    const nose = new THREE.Mesh(
      new THREE.ConeGeometry(0.28, 0.6, 24),
      new THREE.MeshStandardMaterial({ color: 0xf78166 }),
    )
    nose.rotation.x = -Math.PI / 2   // cone axis +Y -> -Z
    nose.position.set(0, 0, -0.9)
    rov.add(nose)
    scene.add(rov)

    scene.add(new THREE.GridHelper(6, 12, 0x30363d, 0x21262d))
    scene.add(new THREE.AxesHelper(1.2))

    // Change of basis: vehicle (X fwd, Y left, Z up) -> Three (X right, Y up,
    // Z back). Columns are the images of the vehicle axes in Three's frame:
    //   veh +X (fwd)  -> (0, 0, -1)
    //   veh +Y (left) -> (-1, 0, 0)
    //   veh +Z (up)   -> (0, 1, 0)
    const basisM = new THREE.Matrix4().makeBasis(
      new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(0, 1, 0),
    )
    const qB = new THREE.Quaternion().setFromRotationMatrix(basisM)
    const qBinv = qB.clone().invert()

    const qZed = new THREE.Quaternion()
    const qTarget = new THREE.Quaternion()
    const eTarget = new THREE.Euler()

    let raf
    const animate = () => {
      const a = attitudeRef.current || {}
      if (a.quat && a.quat.length === 4) {
        // q_three = qB * q_zed * qB^-1
        qZed.set(a.quat[0], a.quat[1], a.quat[2], a.quat[3])
        qTarget.copy(qB).multiply(qZed).multiply(qBinv)
      } else {
        // Fallback if the ZED has no pose yet: euler in the same convention.
        const d2r = Math.PI / 180
        eTarget.set((a.roll || 0) * d2r, (a.pitch || 0) * d2r, (a.yaw || 0) * d2r, 'XYZ')
        qZed.setFromEuler(eTarget)
        qTarget.copy(qB).multiply(qZed).multiply(qBinv)
      }
      // Slerp rather than per-axis easing: no gimbal artefacts, less visual lag.
      rov.quaternion.slerp(qTarget, 0.35)
      renderer.render(scene, camera)
      raf = requestAnimationFrame(animate)
    }
    animate()

    const onResize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight || 260
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [attitudeRef])

  return (
    <div className="panel viz">
      <div className="panel-title">3D Attitude</div>
      <div className="viz-canvas" ref={mountRef} />
    </div>
  )
}
