import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Build the React bundle into kim_app/web_dist/ so the Python package can
// serve it directly (and PyInstaller can pick it up via package_data).
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: path.resolve(__dirname, "../kim_app/web_dist"),
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
});
