import { el, mount } from "../dom.js";
import { pct, fmtRelative, fmtDateTime } from "../format.js";
import { modelRepo } from "../model.js";

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const HEIGHT = 180;
const PAD = { top: 12, right: 14, bottom: 24, left: 38 };

function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

function niceStep(range, maxTicks) {
  for (const s of [1, 2, 5, 10, 20, 25, 50]) {
    if (range / s <= maxTicks) return s;
  }
  return 100;
}

function legendKey(cls, label) {
  return el("span", { class: "hero-chart-key" }, el("span", { class: `swatch ${cls}` }), label);
}

function tipRow(value, cls, label) {
  return el("div", { class: "hero-chart-tip-row" },
    el("span", { class: `swatch ${cls}` }),
    el("b", {}, pct(value)),
    el("span", { class: "lbl" }, label));
}

export function renderHeroChart(container, runs) {
  const points = (runs || [])
    .filter(r => r.finished_at && r.score_challenger != null)
    .reverse(); // newest-first -> chronological
  if (points.length < 2) {
    mount(container, el("div", { class: "empty" }, "no eval history yet."));
    return;
  }

  const width = Math.max(container.clientWidth || 0, 320);
  const w = width - PAD.left - PAD.right;
  const h = HEIGHT - PAD.top - PAD.bottom;

  const ts = points.map(r => new Date(r.finished_at).getTime());
  const t0 = ts[0], t1 = ts[ts.length - 1];
  const vals = points.flatMap(r => [r.score_challenger * 100, r.score_king != null ? r.score_king * 100 : null])
    .filter(v => v != null);
  let vMin = Math.min(...vals), vMax = Math.max(...vals);
  const vPad = Math.max((vMax - vMin) * 0.06, 0.5);
  vMin -= vPad; vMax += vPad;

  const x = t => PAD.left + ((t - t0) / Math.max(t1 - t0, 1)) * w;
  const y = v => PAD.top + (1 - (v - vMin) / (vMax - vMin)) * h;

  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });

  // y gridlines + labels
  const yStep = niceStep(vMax - vMin, 5);
  for (let v = Math.ceil(vMin / yStep) * yStep; v <= vMax; v += yStep) {
    svg.append(svgEl("line", { x1: PAD.left, y1: y(v), x2: width - PAD.right, y2: y(v), class: "grid" }));
    svg.append(svgEl("text", { x: PAD.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, v));
  }

  // x ticks at UTC midnights
  const dayMs = 86400000;
  const days = Math.max((t1 - t0) / dayMs, 1);
  const dayStep = Math.max(1, Math.ceil(days / Math.max(Math.floor(w / 110), 2)));
  for (let t = Math.ceil(t0 / dayMs) * dayMs; t <= t1; t += dayStep * dayMs) {
    const d = new Date(t);
    svg.append(svgEl("text", { x: x(t), y: HEIGHT - 6, class: "tick", "text-anchor": "middle" },
      `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`));
  }

  // challenger series (gray line)
  svg.append(svgEl("polyline", {
    points: points.map((r, i) => `${x(ts[i]).toFixed(1)},${y(r.score_challenger * 100).toFixed(1)}`).join(" "),
    class: "line-chal",
  }));

  // king series (gold line)
  const kingPts = points
    .map((r, i) => (r.score_king != null ? `${x(ts[i]).toFixed(1)},${y(r.score_king * 100).toFixed(1)}` : null))
    .filter(Boolean);
  if (kingPts.length > 1) svg.append(svgEl("polyline", { points: kingPts.join(" "), class: "line-king" }));

  // coronations: gold dots with surface ring, on top
  points.forEach((r, i) => {
    if (!r.coronated) return;
    svg.append(svgEl("circle", { cx: x(ts[i]).toFixed(1), cy: y(r.score_challenger * 100).toFixed(1), r: 4, class: "dot-crown" }));
  });

  // hover: crosshair snapping to the nearest run + one tooltip for both series
  const crosshair = svgEl("line", { y1: PAD.top, y2: HEIGHT - PAD.bottom, class: "crosshair", visibility: "hidden" });
  svg.append(crosshair);
  const tip = el("div", { class: "hero-chart-tip", hidden: true });
  let hover = -1;

  function nearestRun(px) {
    let best = 0;
    for (let i = 1; i < ts.length; i++) {
      if (Math.abs(x(ts[i]) - px) < Math.abs(x(ts[best]) - px)) best = i;
    }
    return best;
  }

  svg.addEventListener("pointermove", e => {
    const rect = svg.getBoundingClientRect();
    const i = nearestRun(e.clientX - rect.left);
    hover = i;
    const r = points[i], px = x(ts[i]);
    crosshair.setAttribute("x1", px.toFixed(1));
    crosshair.setAttribute("x2", px.toFixed(1));
    crosshair.removeAttribute("visibility");
    mount(tip,
      tipRow(r.score_challenger, "k-chal", `challenger · ${modelRepo(r.model_uri)}`),
      r.score_king != null ? tipRow(r.score_king, "k-king", "king") : null,
      el("div", { class: "hero-chart-tip-meta", title: fmtDateTime(r.finished_at) },
        `uid ${r.uid ?? "—"} · ${fmtRelative(r.finished_at)} · ${r.coronated ? "crowned" : r.challenger_won ? "won" : "lost"}`));
    tip.hidden = false;
    const left = px + 12 + tip.offsetWidth > width ? px - tip.offsetWidth - 12 : px + 12;
    tip.style.left = `${Math.max(left, 0)}px`;
    tip.style.top = `${PAD.top}px`;
  });
  svg.addEventListener("pointerleave", () => {
    hover = -1;
    crosshair.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  svg.addEventListener("click", () => {
    const r = points[hover];
    if (r?.eval_run_id) location.href = `detail.html?eval_run_id=${encodeURIComponent(r.eval_run_id)}`;
  });

  mount(container,
    el("div", { class: "hero-chart-top" },
      el("span", { class: "hero-chart-legend" },
        legendKey("k-king", "king"),
        legendKey("k-chal", "challenger"),
        legendKey("k-crown", "crowned"))),
    el("div", { class: "hero-chart-plot" }, svg, tip));
}
