import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/dashboard": "http://localhost:8002",
      "/admin": "http://localhost:8002",
    },
  },
  base: "/dashboard/",
  build: {
    outDir: "../cserve/dashboard/static/dist",
    emptyOutDir: true,
  },
});
