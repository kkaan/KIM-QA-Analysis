// Three-panel deviation plot: LR / SI / AP. Built with react-plotly.js so we
// inherit pan, zoom, drag-select, and modebar download for free. Highlight
// bands are rendered as Plotly `shapes`; users can drag the band edges in the
// UI by editing the input fields in the highlight editor (Plotly's native
// shape-editing is more complex than this UI needs).
//
// The chart plots in *display-space* seconds (compressed beam-off pauses) but
// the x-axis ticks are labelled with the equivalent *real-time* minutes — so
// physicists read the real timestamps even though the layout collapses pauses.

import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { useMemo } from "react";
import { FractionPayload, HighlightSpec, Intervention } from "../lib/api";
import {
  AXIS_COLORS,
  DARK_GRID_AXIS,
  DARK_LAYOUT_BASE,
} from "../lib/plotlyPresets";

const Plot = createPlotlyComponent(Plotly);

type Props = {
  fraction: FractionPayload;
  highlights: Record<string, HighlightSpec>;
  showOverlay: boolean;
};

const AXES: Array<{
  key: "meas_x" | "meas_y" | "meas_z";
  label: string;
  yref: "y" | "y2" | "y3";
  color: string;
  rowSpan: [number, number];
}> = [
  { key: "meas_x", label: "LR (mm)", yref: "y", color: AXIS_COLORS.meas_x, rowSpan: [0.69, 1.0] },
  { key: "meas_y", label: "SI (mm)", yref: "y2", color: AXIS_COLORS.meas_y, rowSpan: [0.355, 0.66] },
  { key: "meas_z", label: "AP (mm)", yref: "y3", color: AXIS_COLORS.meas_z, rowSpan: [0.02, 0.33] },
];

function realToDisplay(t_real: number, real: number[], display: number[]): number {
  if (real.length === 0) return 0;
  if (t_real <= real[0]) return display[0];
  if (t_real >= real[real.length - 1]) return display[real.length - 1];
  // Linear interp via binary search.
  let lo = 0,
    hi = real.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (real[mid] <= t_real) lo = mid;
    else hi = mid;
  }
  const span = real[hi] - real[lo];
  if (span === 0) return display[lo];
  const f = (t_real - real[lo]) / span;
  return display[lo] + f * (display[hi] - display[lo]);
}

function groupBursts(burstId: number[]): Map<number, number[]> {
  const out = new Map<number, number[]>();
  for (let i = 0; i < burstId.length; i++) {
    const arr = out.get(burstId[i]) ?? [];
    arr.push(i);
    out.set(burstId[i], arr);
  }
  return out;
}

function buildDeltaForAxis(
  iv: Intervention,
  axisKey: "meas_x" | "meas_y" | "meas_z",
): number {
  switch (axisKey) {
    case "meas_x":
      return iv.dlr_mm;
    case "meas_y":
      return iv.dsi_mm;
    case "meas_z":
      return iv.dap_mm;
  }
}

export function DeviationPlot({ fraction, highlights, showOverlay }: Props) {
  const { traces, layout } = useMemo(() => {
    const real = fraction.time_real_s;
    const display = fraction.time_display_s;
    const minutes = display.map((s) => s / 60.0);
    const bursts = groupBursts(fraction.burst_id);
    const interventions = fraction.interventions.filter((iv) => !iv.is_localisation);

    // ----- traces: per-axis main + counterfactual overlay segments -----
    const traces: Plotly.Data[] = [];

    AXES.forEach((axis, axisIdx) => {
      const yRef = `y${axisIdx + 1}` as "y" | "y2" | "y3";
      const series: number[] =
        axis.key === "meas_x"
          ? fraction.meas_x
          : axis.key === "meas_y"
            ? fraction.meas_y
            : fraction.meas_z;
      // Main trace, broken into bursts (insert NaN between bursts so Plotly
      // doesn't draw a connecting line across the compressed gap).
      const xs: (number | null)[] = [];
      const ys: (number | null)[] = [];
      const sortedBurstIds = [...bursts.keys()].sort((a, b) => a - b);
      sortedBurstIds.forEach((bid, j) => {
        const indices = bursts.get(bid) ?? [];
        for (const i of indices) {
          xs.push(minutes[i]);
          ys.push(series[i]);
        }
        if (j < sortedBurstIds.length - 1) {
          xs.push(null);
          ys.push(null);
        }
      });
      traces.push({
        x: xs,
        y: ys,
        type: "scattergl",
        mode: "lines+markers",
        line: { color: axis.color, width: 1.4 },
        marker: { color: axis.color, size: 4 },
        name: axis.label,
        yaxis: yRef,
        hovertemplate: `${axis.label}: %{y:+0.2f} mm<extra></extra>`,
      } as Plotly.Data);

      // Per-axis no-correction overlays.
      if (showOverlay) {
        for (let ivIdx = 0; ivIdx < interventions.length; ivIdx++) {
          const iv = interventions[ivIdx];
          const delta = buildDeltaForAxis(iv, axis.key);
          if (delta === 0) continue;
          // Limit the overlay to t in (iv.t_real_s, nextIvOrEnd).
          const nextT = interventions
            .slice(ivIdx + 1)
            .reduce<number>((acc, x) => Math.min(acc, x.t_real_s), Infinity);
          const ox: (number | null)[] = [];
          const oy: (number | null)[] = [];
          for (const bid of sortedBurstIds) {
            const indices = bursts.get(bid) ?? [];
            let segOpen = false;
            for (const i of indices) {
              if (real[i] <= iv.t_real_s || real[i] >= nextT) {
                if (segOpen) {
                  ox.push(null);
                  oy.push(null);
                  segOpen = false;
                }
                continue;
              }
              ox.push(minutes[i]);
              oy.push(series[i] - delta);
              segOpen = true;
            }
            if (segOpen) {
              ox.push(null);
              oy.push(null);
              segOpen = false;
            }
          }
          if (ox.length === 0) continue;
          traces.push({
            x: ox,
            y: oy,
            type: "scattergl",
            mode: "lines",
            line: { color: axis.color, width: 1.0, dash: "dot" },
            opacity: 0.35,
            yaxis: yRef,
            name: `${axis.label} (no-correction)`,
            hovertemplate: `no-correction ${axis.label}: %{y:+0.2f} mm<extra></extra>`,
          } as Plotly.Data);
        }
      }
    });

    // ----- shapes: gap bands, highlights, intervention rules ----------
    const shapes: Partial<Plotly.Shape>[] = [];

    // Compressed beam-off pauses: hatched grey band spanning all three axes.
    for (const [d0, d1] of fraction.gap_display_intervals) {
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: d0 / 60.0,
        x1: d1 / 60.0,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(148, 163, 184, 0.18)",
        line: { width: 0 },
        layer: "below",
      });
    }

    // Highlight bands (gold).
    for (const hl of Object.values(highlights)) {
      const x0 = realToDisplay(hl.t_start_real_s, real, display) / 60.0;
      const x1 = realToDisplay(hl.t_end_real_s, real, display) / 60.0;
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0,
        x1,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(245, 158, 11, 0.20)",
        line: { color: "rgba(245, 158, 11, 0.6)", width: 1 },
        layer: "below",
      });
    }

    // Intervention rules (dotted vertical line per non-localisation event).
    for (const iv of interventions) {
      const x = iv.t_display_s / 60.0;
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: x,
        x1: x,
        y0: 0,
        y1: 1,
        line: { color: "rgba(148, 163, 184, 0.6)", width: 1, dash: "dot" },
      });
    }

    // ±2 mm tolerance lines per axis.
    AXES.forEach((axis, idx) => {
      const yRef = `y${idx + 1}` as "y" | "y2" | "y3";
      for (const y of [-2, 2]) {
        shapes.push({
          type: "line",
          xref: "x",
          yref: yRef,
          x0: minutes[0] ?? 0,
          x1: minutes[minutes.length - 1] ?? 1,
          y0: y,
          y1: y,
          line: { color: "rgba(148, 163, 184, 0.6)", width: 1, dash: "dash" },
        });
      }
    });

    // ----- layout: 3 stacked y-axes sharing x ------------------------
    const yLimits = fraction.y_limits;
    const layout: Partial<Plotly.Layout> = {
      ...DARK_LAYOUT_BASE,
      height: 560,
      shapes,
      xaxis: {
        ...DARK_GRID_AXIS,
        title: { text: "Time from kV beam-on (min)", standoff: 10 },
        domain: [0, 1],
      },
      yaxis: {
        ...DARK_GRID_AXIS,
        title: { text: "LR (mm)" },
        domain: AXES[0].rowSpan,
        range: yLimits.meas_x ?? [-5, 5],
        zerolinecolor: AXIS_COLORS.meas_x + "55",
      },
      yaxis2: {
        ...DARK_GRID_AXIS,
        title: { text: "SI (mm)" },
        domain: AXES[1].rowSpan,
        range: yLimits.meas_y ?? [-5, 5],
        zerolinecolor: AXIS_COLORS.meas_y + "55",
      },
      yaxis3: {
        ...DARK_GRID_AXIS,
        title: { text: "AP (mm)" },
        domain: AXES[2].rowSpan,
        range: yLimits.meas_z ?? [-5, 5],
        zerolinecolor: AXIS_COLORS.meas_z + "55",
      },
      annotations: interventions.map((iv) => ({
        xref: "x",
        yref: "paper",
        x: iv.t_display_s / 60.0,
        y: 1.02,
        xanchor: "center",
        yanchor: "bottom",
        text: `<b>${iv.label}</b>`,
        showarrow: false,
        font: { color: "#FCD34D", size: 10 },
      })),
    };

    return { traces, layout };
  }, [fraction, highlights, showOverlay]);

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{
        displaylogo: false,
        responsive: true,
        scrollZoom: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        toImageButtonOptions: { format: "png", scale: 2 },
      }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
