import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Frontend dev/preview server. Host + port come from env so the same build runs
// in local dev and served from the Jetson. Binds 0.0.0.0 so the Mac can reach
// it over the direct Ethernet link at http://192.168.2.2:3000.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: Number(env.DASH_FRONTEND_PORT || 3000),
      strictPort: true,
    },
    preview: {
      host: '0.0.0.0',
      port: Number(env.DASH_FRONTEND_PORT || 3000),
      strictPort: true,
    },
  }
})
