import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxies /video-stream and the control endpoints to the FastAPI backend
// running on the GPU spot instance so the dev server can hit it without
// CORS headaches. Point VITE_BACKEND_URL at the instance's public IP.
export default defineConfig(({ mode }) => {
  const backendUrl = process.env.VITE_BACKEND_URL ?? "http://localhost:8000";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/video-stream": { target: backendUrl, changeOrigin: true, ws: false },
        "/start-run": { target: backendUrl, changeOrigin: true },
        "/stop-run": { target: backendUrl, changeOrigin: true },
        "/dev-reload": { target: backendUrl, changeOrigin: true },
        "/health": { target: backendUrl, changeOrigin: true },
      },
    },
  };
});
