// Shared layout/colour presets so DeviationPlot and any future
// per-marker sparkline component pick up identical styling.

export const AXIS_COLORS = {
  meas_x: "#1f77b4", // LR
  meas_y: "#2ca02c", // SI
  meas_z: "#d62728", // AP
} as const;

export const DARK_LAYOUT_BASE: Partial<Plotly.Layout> = {
  paper_bgcolor: "#1E293B",
  plot_bgcolor: "#0F172A",
  font: { color: "#E2E8F0", family: "Inter, system-ui, sans-serif" },
  margin: { t: 24, r: 16, b: 56, l: 64 },
  showlegend: false,
  hovermode: "x unified",
  hoverlabel: {
    bgcolor: "#0F172A",
    bordercolor: "#334155",
    font: { color: "#E2E8F0" },
  },
};

export const DARK_GRID_AXIS: Partial<Plotly.LayoutAxis> = {
  gridcolor: "#334155",
  zerolinecolor: "#475569",
  linecolor: "#475569",
  tickcolor: "#94A3B8",
  tickfont: { color: "#CBD5E1" },
};
