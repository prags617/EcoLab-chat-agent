import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
  },
  server: {
    port: 3000,
    proxy: {
      // Local dev: proxy /api to local backend
      "/api": {
        target: "http://localhost:8086",
        changeOrigin: true,
      },
    },
  },
});
