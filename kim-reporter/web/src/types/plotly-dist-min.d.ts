// `plotly.js-dist-min` is a pre-bundled subset of plotly.js. It ships no types
// of its own, but its public API is a strict subset of plotly.js, so we point
// TypeScript at the same types that @types/plotly.js provides.
declare module "plotly.js-dist-min" {
  import * as Plotly from "plotly.js";
  export = Plotly;
}
