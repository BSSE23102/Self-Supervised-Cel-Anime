import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  cacheDir: 'vite-cache',
  plugins: [react(), tailwindcss()],
  server: {
    host: '127.0.0.1',
    allowedHosts: ['localhost', '127.0.0.1'],
  },
  preview: {
    host: '127.0.0.1',
    allowedHosts: ['localhost', '127.0.0.1'],
  },
  optimizeDeps: {
    include: ['axios', 'lucide-react', 'react', 'react-dom/client'],
  },
  build: {
    emptyOutDir: false,
  },
})
