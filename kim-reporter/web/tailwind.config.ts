import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // KIM axis palette mirrored from AXIS_SPECS in the Python backend.
        "axis-lr": "#1f77b4",
        "axis-si": "#2ca02c",
        "axis-ap": "#d62728",
        "kim-bg": "#0F172A",
        "kim-panel": "#1E293B",
        "kim-edge": "#334155",
        "kim-ink": "#E2E8F0",
        "kim-muted": "#94A3B8",
        "kim-accent": "#F59E0B",
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
