import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Base path: when deployed to GitHub Pages at georgebon5.github.io/greekkv/
// the app lives under /greekkv/. In dev the base stays "/".
export default defineConfig({
  base: process.env.GREEKKV_BASE ?? "/",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: true },
  test: {
    environment: "node",
    globals: true,
  },
});
