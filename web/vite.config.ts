import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// base "./" so the build works from any folder on shared hosting (e.g. public_html/).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1200 },
});
