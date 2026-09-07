import { el, mount } from "../dom.js";
import { pct } from "../format.js";
import { modelRepo } from "../model.js";
import { collapsePasses } from "./history.js";

const HEIGHT = 180;
const PAD = { top: 12, right: 34, bottom: 12, left: 40 };
const DEFAULT_THRESHOLD = 0.025;
const MAX_BAR_W = 18;

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

const signed = m => `${m > 0 ? "+" : ""}${pct(m)}`;

// One bar per duel: a submission that won pass 1 is re-evaluated and only the
// confirming pass decides the outcome, so the later pass represents the duel.
function decidingRuns(runs) {
  return collapsePasses(runs || [])
    .filter(r => r.win_margin != null)
    .reverse();
}

export function renderHeroChart(container, runs) {
  const points = decidingRuns(runs);
  if (!points.length) {
    mount(container, el("div", { class: "empty" }, "no eval history yet."));
    return;
  }

  const threshold = (points.find(r => r.required_win_margin != null)?.required_win_margin
    ?? DEFAULT_THRESHOLD) * 100;
  const width = Math.max(container.clientWidth || 0, 320);
  const w = width - PAD.left - PAD.right;
  const h = HEIGHT - PAD.top - PAD.bottom;

  const vals = points.map(r => Number(r.win_margin) * 100);
  let vMin = Math.min(0, ...vals), vMax = Math.max(threshold, ...vals);
  const vPad = Math.max((vMax - vMin) * 0.08, 0.5);
  vMin -= vPad; vMax += vPad;

  const slot = w / points.length;
  const barW = Math.max(1, Math.min(slot - 2, MAX_BAR_W));
  const cx = i => PAD.left + slot * (i + 0.5);
  const y = v => PAD.top + (1 - (v - vMin) / (vMax - vMin)) * h;
  const y0 = y(0);

  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });
  svg.append(svgEl("rect", { x: PAD.left, y: PAD.top, width: w, height: h, fill: "transparent", "pointer-events": "all" }));

  const yStep = niceStep(vMax - vMin, 5);
  for (let v = Math.ceil(vMin / yStep) * yStep; v <= vMax; v += yStep) {
    svg.append(svgEl("line", { x1: PAD.left, y1: y(v), x2: width - PAD.right, y2: y(v), class: "grid" }));
    svg.append(svgEl("text", { x: PAD.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, v));
  }

  const bars = points.map((r, i) => {
    const v = Number(r.win_margin) * 100;
    // a near-zero margin still needs a visible stub, and it must stay on its own side of the baseline
    const top = v >= 0 ? Math.min(y(v), y0 - 1) : y0;
    const height = v >= 0 ? y0 - top : Math.max(y(v) - y0, 1);
    const bar = svgEl("rect", {
      x: (cx(i) - barW / 2).toFixed(1),
      y: top.toFixed(1),
      width: barW.toFixed(1),
      height: height.toFixed(1),
      class: v >= threshold ? "bar bar-win" : "bar bar-loss",
    });
    svg.append(bar);
    return bar;
  });

  svg.append(svgEl("line", { x1: PAD.left, y1: y0, x2: width - PAD.right, y2: y0, class: "zero-line" }));
  svg.append(svgEl("line", { x1: PAD.left, y1: y(threshold), x2: width - PAD.right, y2: y(threshold), class: "threshold-line" }));
  svg.append(svgEl("text", {
    x: width - PAD.right + 5, y: y(threshold), class: "threshold-tick", "dominant-baseline": "middle",
  }, `+${threshold.toFixed(1)}`));

  const crosshair = svgEl("line", { y1: PAD.top, y2: HEIGHT - PAD.bottom, class: "crosshair", visibility: "hidden" });
  svg.append(crosshair);
  const tip = el("div", { class: "hero-chart-tip", hidden: true });
  let hover = -1;

  function highlight(i) {
    if (hover === i) return;
    if (bars[hover]) bars[hover].classList.remove("active");
    if (bars[i]) bars[i].classList.add("active");
    hover = i;
  }

  svg.addEventListener("pointermove", e => {
    const rect = svg.getBoundingClientRect();
    const pointerX = ((e.clientX - rect.left) / Math.max(rect.width, 1)) * width;
    const i = Math.min(points.length - 1, Math.max(0, Math.floor((pointerX - PAD.left) / slot)));
    highlight(i);
    const r = points[i], px = cx(i);
    const margins = Array.isArray(r.pass_margins) && r.pass_margins.length >= 2 ? r.pass_margins : null;
    crosshair.setAttribute("x1", px.toFixed(1));
    crosshair.setAttribute("x2", px.toFixed(1));
    crosshair.removeAttribute("visibility");
    mount(tip,
      el("div", { class: "hero-chart-tip-row" },
        el("span", { class: `swatch ${Number(r.win_margin) * 100 >= threshold ? "k-win" : "k-loss"}` }),
        el("b", {}, `${signed(r.win_margin)} pts`),
        el("span", { class: "lbl" }, modelRepo(r.model_uri))),
      el("div", { class: "hero-chart-tip-meta" },
        `uid ${r.uid ?? "—"} · ${r.coronated ? "crowned" : r.challenger_won ? "won" : "lost"}`),
      margins
        ? el("div", { class: "hero-chart-tip-meta" },
            `pass 1 ${signed(margins[0])} · pass 2 ${signed(margins[1])}`)
        : null);
    tip.hidden = false;
    const cssX = (px / width) * rect.width;
    const left = cssX + 12 + tip.offsetWidth > rect.width ? cssX - tip.offsetWidth - 12 : cssX + 12;
    tip.style.left = `${Math.max(left, 0)}px`;
    tip.style.top = `${PAD.top}px`;
  });
  svg.addEventListener("pointerleave", () => {
    highlight(-1);
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
        legendKey("k-win", `≥ +${threshold.toFixed(1)} pts`),
        legendKey("k-loss", "below threshold")),
      el("span", {}, `win margin · ${points.length} duels`)),
    el("div", { class: "hero-chart-plot" }, svg, tip));
}
