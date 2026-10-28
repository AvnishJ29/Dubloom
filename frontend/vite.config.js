import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: { chunkSizeWarningLimit: 900 },
  server: { proxy: { '/api': { target: 'http://127.0.0.1:8001', changeOrigin: true } } },
})
