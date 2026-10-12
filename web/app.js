/* Tide Pool — explore EmbeddingGemma 2 embeddings of NOAA benthic patches.
   Plain ES2020, no build step. Data comes from web/data/ (written by pipeline/build.py). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false)
    el.append(kid instanceof Node ? kid : document.createTextNode(kid));
  return el;
};
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (+v).toFixed(d);
const pct = v => (v === null || v === undefined) ? "—" : `${Math.round(v * 100)}%`;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------------------------------------------------------------- palette */
// Validated reference categorical palette (light / dark steps), fixed order.
const CAT = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};
const SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const STATUS = { good: "#0ca30c", critical: "#d03b3b" };
const OTHER = { light: "#b9b8b0", dark: "#5b5a55" };
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};
const theme = () => (isDark() ? "dark" : "light");
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
// sequential ramp: low values recede toward the surface in both modes
const seqRamp = () => (isDark() ? SEQ.slice(0, 11).reverse() : SEQ.slice(1));

function hexToHsl(hex) {
  const n = parseInt(hex.slice(1), 16);
  let r = (n >> 16) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  let hh = 0, s = 0;
  if (mx !== mn) {
    const d = mx - mn;
    s = l > .5 ? d / (2 - mx - mn) : d / (mx + mn);
    hh = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
    hh *= 60;
  }
  return [hh, s * 100, l * 100];
}

/* ---------------------------------------------------------------- state */
const S = {
  M: null, P: null, N: 0, E: null, T: null, dim: 256, sprites: [],
  view: "umap", colorBy: "group", render: "dots", size: 4, labels: true, spin: true,
  tx: { k: 1, x: 0, y: 0 }, rot: { yaw: 0.6, pitch: 0.35 },
  hiddenSplits: new Set(), hidden: new Map(), // colorBy key -> Set of hidden category ids
  visible: null, sx: null, sy: null, sd: null, order: null,
  hover: -1, focus: -1, isolate: null, selection: null, selLabel: "",
  scores: null, query: "", server: false, dirty: true,
};

/* ---------------------------------------------------------------- loading */
async function load() {
  const bust = `?v=${Date.now()}`;
  let M;
  try {
    M = await (await fetch("data/manifest.json" + bust)).json();
  } catch {
    $("#loading").innerHTML = "No data bundle found.<br>Run <code>python pipeline/build.py</code> first, then reload.";
    return;
  }
  const [P, ebuf, tbuf] = await Promise.all([
    fetch("data/points.json" + bust).then(r => r.json()),
    fetch("data/embeddings.f32" + bust).then(r => r.arrayBuffer()),
    fetch("data/text.f32" + bust).then(r => r.arrayBuffer()),
  ]);
  Object.assign(S, { M, P, N: P.file.length, dim: M.web_dim, E: new Float32Array(ebuf), T: new Float32Array(tbuf) });
  S.sprites = await Promise.all([...Array(M.sheets).keys()].map(i => new Promise(res => {
    const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = `data/sprite_${i}.jpg${bust}`;
  })));
  S.view = M.projections.includes("umap") ? "umap" : M.projections[0];
  S.sx = new Float32Array(S.N); S.sy = new Float32Array(S.N); S.sd = new Float32Array(S.N);
  S.order = new Uint32Array(S.N).map((_, i) => i);
  derive();
  setupUI();
  $("#loading").hidden = true;
  readHash();
  checkServer();
  resize();
  requestAnimationFrame(frame);
}

function derive() {
  const { P, M } = S;
  P.cn_agree = P.cn_pred.map((c, i) => c == null ? null : (c === M.classes[P.label[i]].code ? 1 : 0));
  P.depthNum = P.depth;
  P.group_name = P.group.map(g => M.groups[g].code);
  // per-class means of quality signals (for the Classes tab)
  const C = M.classes.map(() => ({ n: 0, agree: 0, lp: 0, zs: 0, knn: 0, cn: 0, cnN: 0, atyp: 0 }));
  for (let i = 0; i < S.N; i++) {
    const c = C[P.label[i]];
    c.n++; c.agree += P.knn_agree[i]; c.lp += P.lp_correct[i]; c.zs += P.zs_correct[i]; c.knn += P.knn_correct[i]; c.atyp += P.atypicality[i];
    if (P.cn_agree[i] != null) { c.cn += P.cn_agree[i]; c.cnN++; }
  }
  S.classStats = C.map(c => ({ agree: c.agree / c.n, lp: c.lp / c.n, zs: c.zs / c.n, knn: c.knn / c.n, cn: c.cnN ? c.cn / c.cnN : null, atyp: c.atyp / c.n }));
  // dataset-wide percentiles for the feature sliders in the detail panel
  S.ranges = {};
  for (const k of ["brightness", "contrast", "saturation", "colorfulness", "sharpness", "entropy", "atypicality", "knn_agree", "centroid_sim", "depth"]) {
    const v = P[k].filter(x => x != null).sort((a, b) => a - b);
    S.ranges[k] = v.length ? [v[Math.floor(v.length * .01)], v[Math.floor(v.length * .99)]] : [0, 1];
  }
}

/* ---------------------------------------------------------------- colour encodings */
function classColors() {
  // T3 classes: hue from their T1 group's palette slot, lightness spread within the group
  const pal = CAT[theme()], out = [];
  const byGroup = new Map();
  S.M.classes.forEach((c, i) => { if (!byGroup.has(c.group)) byGroup.set(c.group, []); byGroup.get(c.group).push(i); });
  for (const [g, idx] of byGroup) {
    const [hh, ss, ll] = hexToHsl(pal[g % pal.length]);
    idx.forEach((ci, j) => {
      const t = idx.length === 1 ? 0 : j / (idx.length - 1) - .5;
      out[ci] = `hsl(${(hh + t * 16).toFixed(0)} ${ss.toFixed(0)}% ${Math.max(22, Math.min(78, ll + t * 34)).toFixed(0)}%)`;
    });
  }
  return out;
}

function categorical(values, opts = {}) {
  // values: array of keys (string/number/null). Returns {ids, levels:[{key,label,color,n}]}
  const counts = new Map();
  values.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
  let keys = [...counts.keys()].filter(k => k !== null && k !== undefined);
  keys.sort(opts.sort || ((a, b) => counts.get(b) - counts.get(a)));
  const pal = opts.palette || CAT[theme()];
  const max = opts.max ?? pal.length;
  let shown = keys, folded = [];
  if (keys.length > max) { shown = keys.slice(0, max - 1); folded = keys.slice(max - 1); }
  const levels = shown.map((k, i) => ({ key: k, label: opts.label ? opts.label(k) : String(k), color: opts.colorOf ? opts.colorOf(k, i) : pal[i % pal.length], n: counts.get(k) }));
  const otherIdx = levels.length;
  if (folded.length) levels.push({ key: "__other", label: `Other (${folded.length})`, color: OTHER[theme()], n: folded.reduce((s, k) => s + counts.get(k), 0), other: true });
  const hasNull = counts.has(null) || counts.has(undefined);
  const nullIdx = levels.length;
  if (hasNull) levels.push({ key: "__null", label: "Unknown", color: OTHER[theme()], n: (counts.get(null) || 0) + (counts.get(undefined) || 0), other: true });
  const lookup = new Map(levels.map((l, i) => [l.key, i]));
  const ids = new Int16Array(values.length);
  values.forEach((v, i) => { ids[i] = v == null ? nullIdx : (lookup.get(v) ?? otherIdx); });
  return { type: "cat", ids, levels };
}

function numeric(values, opts = {}) {
  const v = values.map(x => x == null ? NaN : +x);
  const finite = v.filter(Number.isFinite).sort((a, b) => a - b);
  const lo = opts.lo ?? (finite[Math.floor(finite.length * .02)] ?? 0);
  const hi = opts.hi ?? (finite[Math.floor(finite.length * .98) - 1] ?? 1);
  const ramp = opts.ramp || seqRamp();
  const bins = new Int16Array(v.length);
  v.forEach((x, i) => { bins[i] = Number.isFinite(x) ? Math.max(0, Math.min(ramp.length - 1, Math.floor((x - lo) / ((hi - lo) || 1) * ramp.length))) : -1; });
  const hist = new Array(24).fill(0);
  finite.forEach(x => { const b = Math.max(0, Math.min(23, Math.floor((x - lo) / ((hi - lo) || 1) * 24))); hist[b]++; });
  return { type: "num", bins, ramp, lo, hi, hist, missing: v.length - finite.length, fmtv: opts.fmt || (x => fmt(x, 2)), desc: opts.desc };
}

const COLOR_BY = [
  { group: "Labels" },
  { key: "group", label: "T1 group (broad)", build: () => categorical(S.P.group, { sort: (a, b) => a - b, label: g => S.M.groups[g].name, max: 8 }) },
  { key: "label", label: "T3 class (fine)", build: () => { const cc = classColors(); return categorical(S.P.label, { sort: (a, b) => a - b, label: c => `${S.M.classes[c].code}`, colorOf: c => cc[c], max: 999 }); } },
  { group: "Model behaviour" },
  { key: "score", label: "Search similarity", build: () => S.scores ? numeric(S.scores, { desc: "cosine similarity to the current query" }) : null },
  { key: "knn_agree", label: "Neighbour label agreement", build: () => numeric(S.P.knn_agree, { lo: 0, hi: 1, fmt: pct, desc: `share of the ${S.M.k} nearest neighbours with the same T3 label. Low = ambiguous or possibly mislabelled.` }) },
  { key: "lp_correct", label: "Linear probe correct?", build: () => yesNo(S.P.lp_correct, "correct", "wrong") },
  { key: "knn_correct", label: "k-NN vote correct?", build: () => yesNo(S.P.knn_correct, "correct", "wrong") },
  { key: "zs_correct", label: "Zero-shot (text) correct?", build: () => yesNo(S.P.zs_correct, "correct", "wrong") },
  { key: "lp_group_pred", label: "Linear probe: predicted group", build: () => categorical(S.P.lp_group_pred, { sort: (a, b) => a - b, label: g => S.M.groups[g].name, max: 8 }) },
  { key: "zs_group_pred", label: "Zero-shot: predicted group", build: () => categorical(S.P.zs_group_pred, { sort: (a, b) => a - b, label: g => S.M.groups[g].name, max: 8 }) },
  { key: "lp_group_conf", label: "Linear probe confidence", build: () => numeric(S.P.lp_group_conf, { lo: 0, hi: 1, fmt: pct }) },
  { key: "atypicality", label: "Atypicality (1 − mean NN sim)", build: () => numeric(S.P.atypicality, { desc: "how far a patch sits from its nearest neighbours. High = unusual." }) },
  { key: "centroid_sim", label: "Similarity to class centroid", build: () => numeric(S.P.centroid_sim) },
  { key: "cluster", label: "k-means cluster", build: () => categorical(S.P.cluster, { label: c => `Cluster ${c}` }) },
  { group: "Survey metadata" },
  { key: "split", label: "Split", build: () => categorical(S.P.split, { sort: (a, b) => ["train", "val", "test"].indexOf(a) - ["train", "val", "test"].indexOf(b) }) },
  { key: "region", label: "Region", build: () => categorical(S.P.region) },
  { key: "island", label: "Island", build: () => categorical(S.P.island) },
  { key: "year", label: "Survey year", build: () => numeric(S.P.year, { fmt: x => Math.round(x) }) },
  { key: "depth", label: "Depth (m)", build: () => numeric(S.P.depth, { fmt: x => fmt(x, 1) }) },
  { key: "cn_agree", label: "CoralNet suggestion agrees?", build: () => yesNo(S.P.cn_agree, "agrees", "disagrees") },
  { key: "cn_conf", label: "CoralNet confidence", build: () => numeric(S.P.cn_conf, { lo: 0, hi: 100, fmt: x => `${Math.round(x)}%` }) },
  { group: "Pixels" },
  { key: "rgb", label: "Mean colour (true)", build: () => ({ type: "direct", colors: S.P.rgb }) },
  { key: "hue", label: "Dominant hue (true)", build: () => ({ type: "direct", colors: S.P.hue.map((x, i) => `hsl(${x} ${Math.min(90, 25 + S.P.saturation[i] * 120)}% 50%)`) }) },
  { key: "brightness", label: "Brightness", build: () => numeric(S.P.brightness) },
  { key: "contrast", label: "Contrast", build: () => numeric(S.P.contrast) },
  { key: "saturation", label: "Saturation", build: () => numeric(S.P.saturation) },
  { key: "colorfulness", label: "Colourfulness", build: () => numeric(S.P.colorfulness) },
  { key: "sharpness", label: "Sharpness (log Laplacian var.)", build: () => numeric(S.P.sharpness) },
  { key: "entropy", label: "Texture entropy", build: () => numeric(S.P.entropy) },
];
function yesNo(arr, yes, no) {
  return categorical(arr, { sort: (a, b) => b - a, label: v => v ? yes : no, colorOf: v => v ? STATUS.good : STATUS.critical });
}

let ENC = null;
function encode() {
  const spec = COLOR_BY.find(c => c.key === S.colorBy);
  ENC = (spec && spec.build()) || COLOR_BY[1].build();
  if (ENC.type === "cat") ENC.levelColors = ENC.levels.map(l => l.color);
  computeVisible();
  renderLegend();
  S.dirty = true;
}

function hiddenSet() {
  if (!S.hidden.has(S.colorBy)) S.hidden.set(S.colorBy, new Set());
  return S.hidden.get(S.colorBy);
}

function computeVisible() {
  const vis = new Uint8Array(S.N), hid = hiddenSet();
  const coords = S.P.proj[S.view];
  for (let i = 0; i < S.N; i++) {
    let ok = !S.hiddenSplits.has(S.P.split[i]);
    if (ok && ENC && ENC.type === "cat" && hid.has(ENC.ids[i])) ok = false;
    if (ok && coords && coords[0][i] == null) ok = false;
    vis[i] = ok;
  }
  S.visible = vis;
}

function colorOf(i) {
  if (ENC.type === "cat") return ENC.levelColors[ENC.ids[i]];
  if (ENC.type === "direct") return ENC.colors[i];
  const b = ENC.bins[i];
  return b < 0 ? OTHER[theme()] : ENC.ramp[b];
}

/* ---------------------------------------------------------------- projection to screen */
let W = 0, H = 0, DPR = 1;
const canvas = () => $("#plot");

function resize() {
  const c = canvas(), r = c.getBoundingClientRect();
  DPR = window.devicePixelRatio || 1;
  W = r.width; H = r.height;
  c.width = Math.round(W * DPR); c.height = Math.round(H * DPR);
  S.dirty = true;
}

function extent(view) {
  // robust bounding box of a 2-D projection (1st-99th percentile), cached per view
  S.extents = S.extents || {};
  if (S.extents[view]) return S.extents[view];
  const [X, Y] = S.P.proj[view];
  const q = a => { const v = a.filter(x => x != null).sort((p, r) => p - r); return [v[Math.floor(v.length * .005)], v[Math.ceil(v.length * .995) - 1]]; };
  const [x0, x1] = q(X), [y0, y1] = q(Y);
  return (S.extents[view] = { mx: (x0 + x1) / 2, my: (y0 + y1) / 2, w: (x1 - x0) || 1, h: (y1 - y0) || 1 });
}

function screenMap() {
  // data -> screen transform for the current 2-D view, filling the stage
  const e = extent(S.view);
  const s = Math.min((W - 40) / e.w, (H - 40) / e.h) * S.tx.k;
  return { s, ox: W / 2 + S.tx.x - e.mx * s, oy: H / 2 + S.tx.y + e.my * s };
}

function project() {
  const coords = S.P.proj[S.view];
  const base = Math.min(W, H) * 0.44 * S.tx.k;
  const cx = W / 2 + S.tx.x, cy = H / 2 + S.tx.y;
  if (S.view === "umap3d") {
    const { yaw, pitch } = S.rot, cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const [X, Y, Z] = coords;
    for (let i = 0; i < S.N; i++) {
      const x = X[i], y = Y[i], z = Z[i];
      const x1 = x * cyw - z * syw, z1 = x * syw + z * cyw;
      const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const f = CAM / (CAM + z2);
      S.sx[i] = cx + x1 * f * base; S.sy[i] = cy - y1 * f * base; S.sd[i] = z2;
    }
    S.order.sort((a, b) => S.sd[b] - S.sd[a]);
  } else {
    const [X, Y] = coords, m = screenMap();
    for (let i = 0; i < S.N; i++) {
      S.sx[i] = X[i] == null ? NaN : m.ox + X[i] * m.s;
      S.sy[i] = Y[i] == null ? NaN : m.oy - Y[i] * m.s;
      S.sd[i] = 0;
    }
  }
}

/* ---------------------------------------------------------------- drawing */
function emphasis() {
  // returns a Uint8Array mask of emphasised points, or null when nothing is emphasised
  if (S.isolate !== null && ENC.type === "cat") {
    const m = new Uint8Array(S.N);
    for (let i = 0; i < S.N; i++) m[i] = ENC.ids[i] === S.isolate;
    return m;
  }
  if (S.selection) {
    const m = new Uint8Array(S.N);
    S.selection.forEach(i => { m[i] = 1; });
    return m;
  }
  return null;
}

function frame() {
  if (S.view === "umap3d" && S.spin && !drag) { S.rot.yaw += 0.0025; S.dirty = true; }
  if (S.dirty) { draw(); S.dirty = false; }
  requestAnimationFrame(frame);
}

function draw() {
  const c = canvas(), ctx = c.getContext("2d");
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!S.P) return;
  project();
  const emph = emphasis();
  const vis = S.visible;
  const is3d = S.view === "umap3d";
  if (S.view === "geo") drawGeoGuides(ctx);
  if (is3d) drawAxes3d(ctx);

  const r = S.size * 0.55 + 0.6;
  const patchMode = S.render === "patches";
  const ps = Math.max(10, Math.min(96, (S.size * 5 + 6) * Math.sqrt(S.tx.k)));
  const dimAlpha = isDark() ? 0.13 : 0.18;

  // pass 1: de-emphasised; pass 2: emphasised (drawn on top)
  for (const pass of emph ? [0, 1] : [1]) {
    ctx.globalAlpha = emph && pass === 0 ? dimAlpha : (patchMode ? 1 : 0.88);
    if (patchMode) {
      for (let o = 0; o < S.N; o++) {
        const i = is3d ? S.order[o] : o;
        if (!vis[i] || (emph && emph[i] !== pass)) continue;
        const x = S.sx[i], y = S.sy[i];
        if (!(x > -ps && x < W + ps && y > -ps && y < H + ps)) continue;
        const s = is3d ? ps * (CAM / (CAM + S.sd[i])) : ps;
        drawPatch(ctx, i, x - s / 2, y - s / 2, s);
        ctx.lineWidth = Math.max(1.5, s / 22);
        ctx.strokeStyle = colorOf(i);
        ctx.strokeRect(x - s / 2, y - s / 2, s, s);
      }
    } else if (is3d) {
      for (let o = 0; o < S.N; o++) {
        const i = S.order[o];
        if (!vis[i] || (emph && emph[i] !== pass)) continue;
        const rr = r * (CAM / (CAM + S.sd[i]));
        ctx.fillStyle = colorOf(i);
        ctx.beginPath(); ctx.arc(S.sx[i], S.sy[i], rr, 0, 6.2832); ctx.fill();
      }
    } else {
      // batch by colour for speed
      const buckets = new Map();
      for (let i = 0; i < S.N; i++) {
        if (!vis[i] || (emph && emph[i] !== pass)) continue;
        const x = S.sx[i], y = S.sy[i];
        if (!(x > -5 && x < W + 5 && y > -5 && y < H + 5)) continue;
        const col = colorOf(i);
        let b = buckets.get(col); if (!b) buckets.set(col, b = []);
        b.push(i);
      }
      // numeric encodings: draw high values last so they stay visible
      const entries = [...buckets.entries()];
      if (ENC.type === "num") entries.sort((a, b) => ENC.bins[a[1][0]] - ENC.bins[b[1][0]]);
      for (const [col, idx] of entries) {
        ctx.fillStyle = col; ctx.beginPath();
        for (const i of idx) { ctx.moveTo(S.sx[i] + r, S.sy[i]); ctx.arc(S.sx[i], S.sy[i], r, 0, 6.2832); }
        ctx.fill();
      }
    }
  }
  ctx.globalAlpha = 1;
  if (S.labels) drawLabels(ctx, emph);

  // focus + hover rings
  for (const [i, w] of [[S.focus, 2.5], [S.hover, 2]]) {
    if (i < 0 || !Number.isFinite(S.sx[i])) continue;
    const rr = patchMode ? ps / 2 + 3 : r + 4;
    ctx.lineWidth = w; ctx.strokeStyle = css("--ink");
    ctx.beginPath();
    if (patchMode) ctx.rect(S.sx[i] - rr, S.sy[i] - rr, rr * 2, rr * 2); else ctx.arc(S.sx[i], S.sy[i], rr, 0, 6.2832);
    ctx.stroke();
  }
  if (S.focus >= 0 && S.view !== "geo") { // lines to neighbours
    ctx.strokeStyle = css("--ink-2"); ctx.globalAlpha = .45; ctx.lineWidth = 1;
    ctx.beginPath();
    for (const j of S.P.nn[S.focus].slice(0, 8)) { if (!S.visible[j]) continue; ctx.moveTo(S.sx[S.focus], S.sy[S.focus]); ctx.lineTo(S.sx[j], S.sy[j]); }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  if (lasso && lasso.length > 1) {
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.fillStyle = "rgba(57,135,229,.08)";
    ctx.beginPath(); lasso.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawPatch(ctx, i, x, y, s) {
  const per = S.M.sheet_cols * S.M.sheet_cols, sheet = S.sprites[Math.floor(i / per)];
  if (!sheet) { ctx.fillStyle = colorOf(i); ctx.fillRect(x, y, s, s); return; }
  const j = i % per, t = S.M.thumb;
  ctx.drawImage(sheet, (j % S.M.sheet_cols) * t, Math.floor(j / S.M.sheet_cols) * t, t, t, x, y, s, s);
}

function drawLabels(ctx, emph) {
  if (ENC.type !== "cat" || ENC.levels.length > 80) return;
  // direct labels at the median position of each category
  const xs = ENC.levels.map(() => []), ys = ENC.levels.map(() => []);
  for (let i = 0; i < S.N; i++) {
    if (!S.visible[i] || !Number.isFinite(S.sx[i])) continue;
    const id = ENC.ids[i]; if (ENC.levels[id].other) continue;
    xs[id].push(S.sx[i]); ys[id].push(S.sy[i]);
  }
  const med = a => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
  ctx.font = `600 ${S.colorBy === "label" ? 11 : 12}px ${css("--font")}`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  const minN = S.colorBy === "label" ? 3 : 1;
  const placed = [];
  const fits = (x, y, w, hh) => !placed.some(b => x < b[0] + b[2] && x + w > b[0] && y < b[1] + b[3] && y + hh > b[1]);
  // biggest categories claim their spot first; colliding labels are skipped
  ENC.levels.map((l, id) => [l, id]).sort((a, b) => xs[b[1]].length - xs[a[1]].length).forEach(([l, id]) => {
    if (xs[id].length < minN) return;
    if (S.isolate !== null && S.isolate !== id) return;
    const x = med(xs[id]), y = med(ys[id]), txt = l.label;
    const w = ctx.measureText(txt).width + 6, hh = 15;
    if (S.isolate === null && !fits(x - w / 2, y - hh / 2, w, hh)) return;
    placed.push([x - w / 2, y - hh / 2, w, hh]);
    ctx.lineWidth = 4; ctx.strokeStyle = css("--surface"); ctx.strokeText(txt, x, y);
    ctx.fillStyle = css("--ink"); ctx.fillText(txt, x, y);
  });
  if (S.view === "geo" && S.colorBy !== "island") { // island names on the map
    const isl = new Map();
    for (let i = 0; i < S.N; i++) {
      if (!S.visible[i] || !S.P.island[i] || !Number.isFinite(S.sx[i])) continue;
      if (!isl.has(S.P.island[i])) isl.set(S.P.island[i], [[], []]);
      const e = isl.get(S.P.island[i]); e[0].push(S.sx[i]); e[1].push(S.sy[i]);
    }
    ctx.font = `11px ${css("--font")}`; ctx.textAlign = "left";
    for (const [name, [a, b]] of isl) {
      const x = med(a) + 8, y = med(b) - 8, w = ctx.measureText(name).width + 4;
      if (!fits(x, y - 7, w, 14)) continue;
      placed.push([x, y - 7, w, 14]);
      ctx.lineWidth = 3; ctx.strokeStyle = css("--surface"); ctx.strokeText(name, x, y);
      ctx.fillStyle = css("--ink-2"); ctx.fillText(name, x, y);
    }
  }
}

function drawGeoGuides(ctx) {
  // graticule in lon (0..360) / lat, using the same normalisation as the build
  const coords = S.P.proj.geo, P = S.P;
  let i0 = -1, i1 = -1;
  for (let i = 0; i < S.N; i++) if (P.lat[i] != null) { if (i0 < 0) i0 = i; else if (Math.abs(P.lat[i] - P.lat[i0]) > 1) { i1 = i; break; } }
  if (i0 < 0 || i1 < 0) return;
  // recover affine map from two samples (lat->y)
  const ly = (coords[1][i1] - coords[1][i0]) / (P.lat[i1] - P.lat[i0]), y0 = coords[1][i0] - ly * P.lat[i0];
  const lon360 = v => v < 0 ? v + 360 : v;
  let j1 = -1;
  for (let i = 0; i < S.N; i++) if (P.lon[i] != null && Math.abs(lon360(P.lon[i]) - lon360(P.lon[i0])) > 1) { j1 = i; break; }
  if (j1 < 0) return;
  const lx = (coords[0][j1] - coords[0][i0]) / (lon360(P.lon[j1]) - lon360(P.lon[i0])), x0 = coords[0][i0] - lx * lon360(P.lon[i0]);
  const m = screenMap();
  ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--muted"); ctx.lineWidth = 1;
  ctx.font = `10px ${css("--font")}`; ctx.textAlign = "left"; ctx.textBaseline = "top";
  for (let lat = -30; lat <= 40; lat += 5) {
    const y = m.oy - (y0 + ly * lat) * m.s; if (y < 0 || y > H) continue;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillText(`${Math.abs(lat)}°${lat >= 0 ? "N" : "S"}`, 4, y + 2);
  }
  for (let lon = 120; lon <= 220; lon += 5) {
    const x = m.ox + (x0 + lx * lon) * m.s; if (x < 0 || x > W) continue;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    const L = lon > 180 ? `${360 - lon}°W` : `${lon}°E`;
    ctx.fillText(L, x + 3, 4);
  }
}

function drawAxes3d(ctx) {
  const base = Math.min(W, H) * 0.44 * S.tx.k, cx = W / 2 + S.tx.x, cy = H / 2 + S.tx.y;
  const { yaw, pitch } = S.rot;
  const P3 = (x, y, z) => {
    const x1 = x * Math.cos(yaw) - z * Math.sin(yaw), z1 = x * Math.sin(yaw) + z * Math.cos(yaw);
    const y1 = y * Math.cos(pitch) - z1 * Math.sin(pitch), z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch);
    const f = CAM / (CAM + z2); return [cx + x1 * f * base, cy - y1 * f * base];
  };
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  const e = 1, edges = [[[-e, -e, -e], [e, -e, -e]], [[-e, -e, -e], [-e, e, -e]], [[-e, -e, -e], [-e, -e, e]],
    [[e, e, e], [-e, e, e]], [[e, e, e], [e, -e, e]], [[e, e, e], [e, e, -e]],
    [[e, -e, -e], [e, e, -e]], [[e, -e, -e], [e, -e, e]], [[-e, e, -e], [e, e, -e]], [[-e, e, -e], [-e, e, e]], [[-e, -e, e], [e, -e, e]], [[-e, -e, e], [-e, e, e]]];
  ctx.beginPath();
  for (const [a, b] of edges) { const p = P3(...a), q = P3(...b); ctx.moveTo(...p); ctx.lineTo(...q); }
  ctx.stroke();
}

/* ---------------------------------------------------------------- interaction */
let drag = null, lasso = null;

function nearest(mx, my, maxd = 12) {
  let best = -1, bd = maxd * maxd;
  const patch = S.render === "patches";
  const half = patch ? Math.max(10, Math.min(96, (S.size * 5 + 6) * Math.sqrt(S.tx.k))) / 2 : 0;
  const order = S.view === "umap3d" ? S.order : null;
  for (let o = S.N - 1; o >= 0; o--) { // topmost first in 3D
    const i = order ? order[o] : o;
    if (!S.visible[i]) continue;
    const dx = S.sx[i] - mx, dy = S.sy[i] - my;
    if (patch && Math.abs(dx) <= half && Math.abs(dy) <= half) return i;
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function setupCanvas() {
  const c = canvas();
  c.addEventListener("pointerdown", e => {
    c.setPointerCapture(e.pointerId);
    const p = [e.offsetX, e.offsetY];
    if (e.shiftKey || e.altKey) { lasso = [p]; return; }
    drag = { x: e.clientX, y: e.clientY, moved: false, tx: { ...S.tx }, rot: { ...S.rot }, rotate: S.view === "umap3d" && !e.ctrlKey && !e.metaKey };
  });
  c.addEventListener("pointermove", e => {
    if (lasso) { lasso.push([e.offsetX, e.offsetY]); S.dirty = true; return; }
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.rotate) { S.rot.yaw = drag.rot.yaw - dx * 0.008; S.rot.pitch = Math.max(-1.5, Math.min(1.5, drag.rot.pitch + dy * 0.008)); }
      else { S.tx.x = drag.tx.x + dx; S.tx.y = drag.tx.y + dy; }
      S.dirty = true; hideTip(); return;
    }
    const i = nearest(e.offsetX, e.offsetY);
    if (i !== S.hover) { S.hover = i; S.dirty = true; }
    if (i >= 0) showTip(i, e.offsetX, e.offsetY); else hideTip();
  });
  c.addEventListener("pointerup", e => {
    if (lasso) {
      const poly = lasso; lasso = null;
      if (poly.length > 4) {
        const sel = [];
        for (let i = 0; i < S.N; i++) if (S.visible[i] && inPoly(S.sx[i], S.sy[i], poly)) sel.push(i);
        setSelection(sel, `Lasso: ${sel.length} patches`);
      }
      S.dirty = true; return;
    }
    if (drag && !drag.moved) {
      const i = nearest(e.offsetX, e.offsetY);
      if (i >= 0) focus(i); else { focus(-1); }
    }
    drag = null;
  });
  c.addEventListener("pointerleave", () => { S.hover = -1; hideTip(); S.dirty = true; });
  c.addEventListener("wheel", e => {
    e.preventDefault();
    const f = Math.exp(-e.deltaY * 0.0015), k = Math.max(0.3, Math.min(60, S.tx.k * f)), r = k / S.tx.k;
    const mx = e.offsetX - W / 2, my = e.offsetY - H / 2;
    S.tx.x = mx - (mx - S.tx.x) * r; S.tx.y = my - (my - S.tx.y) * r; S.tx.k = k;
    S.dirty = true;
  }, { passive: false });
  new ResizeObserver(resize).observe($("#stage"));
}

function inPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function thumbEl(i, size = 64, cls = "thumb") {
  const per = S.M.sheet_cols * S.M.sheet_cols, j = i % per, cols = S.M.sheet_cols;
  return h("div", {
    class: cls, style: {
      width: size ? `${size}px` : null, height: size ? `${size}px` : null,
      backgroundImage: `url(data/sprite_${Math.floor(i / per)}.jpg)`,
      backgroundSize: `${cols * 100}% ${cols * 100}%`,
      backgroundPosition: `${(j % cols) / (cols - 1) * 100}% ${Math.floor(j / cols) / (cols - 1) * 100}%`,
    }
  });
}

function showTip(i, x, y) {
  const P = S.P, c = S.M.classes[P.label[i]], t = $("#tooltip");
  t.replaceChildren(thumbEl(i, 72), h("div", { class: "t-body" },
    h("b", {}, `${c.code} · ${c.name}`),
    h("div", { class: "muted" }, `${S.M.groups[c.group].name} · ${P.split[i]}`),
    h("div", { class: "muted" }, [P.island[i], P.year[i]].filter(Boolean).join(" · ")),
    h("div", { class: "muted" }, `${legendValue(i)}`),
  ));
  t.hidden = false;
  const tw = t.offsetWidth, th = t.offsetHeight;
  t.style.left = `${Math.min(x + 14, W - tw - 6)}px`;
  t.style.top = `${y + 14 + th > H ? y - th - 10 : y + 14}px`;
}
function hideTip() { $("#tooltip").hidden = true; }

function legendValue(i) {
  const spec = COLOR_BY.find(c => c.key === S.colorBy);
  if (!spec || ENC.type === "direct") return "";
  if (ENC.type === "cat") return `${spec.label}: ${ENC.levels[ENC.ids[i]].label}`;
  const raw = S.colorBy === "score" ? S.scores[i] : S.P[S.colorBy][i];
  return `${spec.label}: ${raw == null ? "—" : ENC.fmtv(raw)}`;
}

/* ---------------------------------------------------------------- legend */
function renderLegend() {
  const el = $("#legend"), spec = COLOR_BY.find(c => c.key === S.colorBy);
  el.replaceChildren(h("h3", {}, spec.label));
  if (ENC.type === "direct") {
    el.append(h("p", { class: "note" }, "Each dot is painted with the patch’s own pixel colour."));
    return;
  }
  if (ENC.type === "num") {
    el.append(h("div", { class: "ramp", style: { background: `linear-gradient(90deg, ${ENC.ramp.join(",")})` } }),
      h("div", { class: "ramp-labels" }, h("span", {}, ENC.fmtv(ENC.lo)), h("span", {}, ENC.fmtv(ENC.hi))));
    const mx = Math.max(...ENC.hist, 1);
    el.append(h("div", { class: "hist", title: "distribution" }, ENC.hist.map(v => h("i", { style: { height: `${v / mx * 100}%` } }))));
    if (ENC.desc) el.append(h("p", { class: "note" }, ENC.desc));
    if (ENC.missing) el.append(h("p", { class: "note" }, `${ENC.missing} patches have no value (grey).`));
    return;
  }
  const hid = hiddenSet();
  el.append(h("div", { class: "actions" },
    h("button", { onclick: () => { hid.clear(); computeVisible(); renderLegend(); S.dirty = true; } }, "Show all"),
    h("button", { onclick: () => { ENC.levels.forEach((_, i) => hid.add(i)); computeVisible(); renderLegend(); S.dirty = true; } }, "Hide all")));
  const item = (l, id) => h("div", {
    class: "item" + (hid.has(id) ? " off" : ""), title: "Click to toggle · double-click to show only this · hover to highlight",
    onclick: () => { hid.has(id) ? hid.delete(id) : hid.add(id); computeVisible(); renderLegend(); S.dirty = true; },
    ondblclick: () => { hid.clear(); ENC.levels.forEach((_, k) => k !== id && hid.add(k)); computeVisible(); renderLegend(); S.dirty = true; },
    onmouseenter: () => { S.isolate = id; S.dirty = true; },
    onmouseleave: () => { S.isolate = null; S.dirty = true; },
  }, h("span", { class: "sw", style: { background: l.color } }),
    h("span", { class: "lab" }, l.label, S.colorBy === "label" && !l.other ? h("small", {}, ` ${S.M.classes[l.key].name}`) : null),
    h("span", { class: "n" }, l.n));
  if (S.colorBy === "label") { // group the 60+ fine classes under their T1 group
    S.M.groups.forEach((g, gi) => {
      const ids = ENC.levels.map((l, id) => [l, id]).filter(([l]) => !l.other && S.M.classes[l.key].group === gi);
      if (!ids.length) return;
      el.append(h("div", { class: "grp" }, g.name), ...ids.map(([l, id]) => item(l, id)));
    });
  } else {
    ENC.levels.forEach((l, id) => el.append(item(l, id)));
  }
  if (ENC.levels.some(l => l.other && l.key === "__other"))
    el.append(h("p", { class: "note" }, "Smaller categories are folded into “Other” so colours stay distinguishable. Hover an item to highlight it."));
}

/* ---------------------------------------------------------------- selection & search */
function setSelection(idx, label) {
  S.selection = idx && idx.length ? idx : null;
  S.selLabel = label || "";
  renderGallery();
  showTab("gallery");
  S.dirty = true;
}

function dot(a, ai, b, bi, d) {
  let s = 0; const oa = ai * d, ob = bi * d;
  for (let k = 0; k < d; k++) s += a[oa + k] * b[ob + k];
  return s;
}

function rankBy(vec, label) {
  const d = S.dim, sc = new Float32Array(S.N);
  for (let i = 0; i < S.N; i++) { let s = 0; const o = i * d; for (let k = 0; k < d; k++) s += S.E[o + k] * vec[k]; sc[i] = s; }
  S.scores = sc; S.query = label;
  $("#queryChip").hidden = false; $("#queryText").textContent = label;
  const prev = S.colorBy; S.colorBy = "score"; $("#colorBy").value = "score";
  if (prev !== "score") S.prevColor = prev;
  encode();
  const idx = [...Array(S.N).keys()].filter(i => S.visible[i]).sort((a, b) => sc[b] - sc[a]).slice(0, 120);
  setSelection(idx, `Top ${idx.length} for “${label}”`);
}

function clearQuery() {
  S.scores = null; S.query = ""; $("#queryChip").hidden = true;
  if (S.colorBy === "score") { S.colorBy = S.prevColor || "group"; $("#colorBy").value = S.colorBy; }
  setSelection(null); encode();
}

async function textSearch(q) {
  q = q.trim(); if (!q) return;
  if (S.server) {
    try {
      const r = await fetch("api/embed_text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: q, kind: "query" }) });
      if (!r.ok) throw new Error(await r.text());
      return rankBy((await r.json()).vector, q);
    } catch (e) { console.warn(e); }
  }
  // offline fallback: use a class text prototype computed at build time
  const ql = q.toLowerCase();
  const ci = S.M.classes.findIndex(c => c.code.toLowerCase() === ql || c.name.toLowerCase().includes(ql));
  const gi = S.M.groups.findIndex(g => g.code.toLowerCase() === ql || g.name.toLowerCase().includes(ql));
  const row = ci >= 0 ? ci : gi >= 0 ? S.M.classes.length + gi : -1;
  if (row < 0) { alert("Free-text search needs the server (python server/app.py). Offline, type a class code or name, e.g. “POCS” or “sand”."); return; }
  const v = S.T.subarray(row * S.dim, (row + 1) * S.dim);
  rankBy(v, `${ci >= 0 ? S.M.classes[ci].code : S.M.groups[gi].name} (text prototype)`);
}

async function imageSearch(file) {
  if (!S.server) { alert("Image search needs the server: python server/app.py"); return; }
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch("api/embed_image", { method: "POST", body: fd });
  if (!r.ok) { alert(`Image search failed: ${await r.text()}`); return; }
  rankBy((await r.json()).vector, `image: ${file.name}`);
}

function similarTo(i) {
  rankBy(S.E.subarray(i * S.dim, (i + 1) * S.dim), `like #${i} (${S.M.classes[S.P.label[i]].code})`);
}

async function checkServer() {
  const chip = $("#serverChip");
  const set = (txt, ok) => { chip.replaceChildren(h("span", { class: "dot", style: { background: ok ? STATUS.good : OTHER[theme()] } }), txt); };
  try {
    const r = await fetch("api/health");
    if (!r.ok) throw 0;
    const j = await r.json();
    S.server = j.status === "ready";
    set(S.server ? "search server ready" : `server: ${j.status}`, S.server);
    chip.title = j.model || "";
    $("#uploadBtn").classList.toggle("disabled", !S.server);
    if (j.status === "loading" || j.status === "not loaded") setTimeout(checkServer, 3000);
  } catch {
    S.server = false; set("static mode", false);
    chip.title = "No search server: free-text and image search are off. Class-name search uses build-time text prototypes.";
    $("#uploadBtn").classList.add("disabled");
  }
}

/* ---------------------------------------------------------------- detail panel */
function cropUrl(i) {
  if (S.M.local_crops) return `data/crops/${i}.jpg`;
  return `https://storage.googleapis.com/${S.M.bucket}/${S.P.object[i].split("/").map(encodeURIComponent).join("/")}`;
}
function photoUrl(i) {
  const P = S.P;
  if (!P.source[i]) return null;
  return `https://storage.googleapis.com/${S.M.bucket}/${S.M.mirror}/${P.source[i]}/images/${encodeURIComponent(P.image_name[i])}`;
}

function focus(i) {
  S.focus = i; S.dirty = true;
  const el = $("#detail");
  if (i < 0) { el.hidden = true; writeHash(); return; }
  el.hidden = false;
  const P = S.P, M = S.M, c = M.classes[P.label[i]], g = M.groups[c.group];
  const bars = (items, matchIdx) => h("div", { class: "bars" }, items.map(([lab, v, ci, title]) =>
    h("div", { class: "row", title: title || "", onclick: () => ci != null && selectClass(ci) },
      h("span", { class: "name" + (ci === matchIdx ? " match" : "") }, lab),
      h("span", { class: "track" }, h("span", { class: "fill", style: { width: `${Math.max(0, Math.min(1, v)) * 100}%` } })),
      h("span", { class: "val" }, v <= 1 ? pct(v) : fmt(v, 2)))));
  // zero-shot scores are cosines; rescale for display within the top-5
  const zs = P.zs_score[i], zmin = Math.min(...zs) - 0.02, zmax = Math.max(...zs);
  const feat = (k, label, d = 2) => {
    const v = P[k][i], [lo, hi] = S.ranges[k];
    return h("div", { class: "feat" }, h("span", {}, label),
      h("span", { class: "track" }, v == null ? null : h("i", { style: { left: `${Math.max(0, Math.min(1, (v - lo) / ((hi - lo) || 1))) * 100}%` } })),
      h("span", { class: "val" }, fmt(v, d)));
  };
  const machine = P.machine[i] || [];
  const codeIdx = code => M.classes.findIndex(x => x.code === code);
  const photo = photoUrl(i);
  el.replaceChildren(
    h("div", { class: "head" },
      h("div", {}, h("h2", {}, `${c.code} · ${c.name}`), h("div", { class: "muted" }, `${g.name} · patch #${i}`)),
      h("button", { class: "btn ghost x", onclick: () => focus(-1), "aria-label": "Close details" }, "×")),
    h("div", { class: "patch" },
      h("img", { src: cropUrl(i), alt: `${c.name} patch`, loading: "lazy", onerror: e => { e.target.replaceWith(thumbEl(i, 0, "thumb lg")); } }),
      h("span", { class: "cross", title: "annotated point" })),
    h("div", { class: "actions-row" },
      h("button", { class: "btn", onclick: () => similarTo(i) }, "Find similar"),
      photo ? h("button", { class: "btn ghost", onclick: () => openContext(i) }, "Show in photo") : null,
      h("button", { class: "btn ghost", onclick: () => selectClass(P.label[i]) }, `All ${c.code}`),
      h("button", { class: "btn ghost", title: "Copy gs:// URI", onclick: () => navigator.clipboard?.writeText(`gs://${M.bucket}/${P.object[i]}`) }, "Copy URI")),
    h("div", {},
      h("span", { class: "pill " + (P.lp_correct[i] ? "good" : "bad") }, "linear probe"),
      h("span", { class: "pill " + (P.knn_correct[i] ? "good" : "bad") }, "k-NN"),
      h("span", { class: "pill " + (P.zs_correct[i] ? "good" : "bad") }, "zero-shot"),
      P.cn_agree[i] == null ? null : h("span", { class: "pill " + (P.cn_agree[i] ? "good" : "bad") }, "CoralNet")),
    h("h4", {}, "Survey"),
    h("dl", { class: "kv" },
      ...kv("Split", P.split[i]), ...kv("Region", P.region[i]), ...kv("Island", P.island[i]),
      ...kv("Site", P.site[i]), ...kv("Date", P.date[i] || P.year[i]),
      ...kv("Lat / lon", P.lat[i] == null ? null : `${fmt(P.lat[i], 4)}, ${fmt(P.lon[i], 4)}`),
      ...kv("Depth", P.depth[i] == null ? null : `${fmt(P.depth[i], 1)} m`),
      ...kv("Photo", P.image_name[i]), ...kv("Point", P.row[i] == null ? `#${P.point[i]}` : `#${P.point[i]} (row ${P.row[i]}, col ${P.col[i]})`),
      ...kv("Cluster", `k-means ${P.cluster[i]}`)),
    h("h4", { title: "Out-of-fold logistic regression on the embeddings" }, "Linear probe (top 5)"),
    bars(P.lp_top[i].map((ci, k) => [`${M.classes[ci].code} · ${M.classes[ci].name}`, P.lp_prob[i][k], ci]), P.label[i]),
    h("h4", { title: `Cosine similarity to text: “${M.prompt_template}”` }, "Zero-shot vs. text descriptions"),
    bars(P.zs_top[i].map((ci, k) => [`${M.classes[ci].code} · ${M.classes[ci].name}`, (zs[k] - zmin) / ((zmax - zmin) || 1), ci, `cosine ${fmt(zs[k], 3)}`]), P.label[i]),
    machine.length ? h("h4", {}, "CoralNet machine suggestions") : null,
    machine.length ? bars(machine.map(([code, conf]) => [code + (codeIdx(code) >= 0 ? ` · ${M.classes[codeIdx(code)].name}` : ""), (conf ?? 0) / 100, codeIdx(code) >= 0 ? codeIdx(code) : null]), P.label[i]) : null,
    h("h4", {}, `Nearest neighbours · ${pct(P.knn_agree[i])} share the label`),
    h("div", { class: "nn" }, P.nn[i].map((j, k) => h("div", { class: "cell" + (P.label[j] !== P.label[i] ? " diff" : ""), onclick: () => focus(j), title: `${M.classes[P.label[j]].name} · cos ${P.nn_sim[i][k]}` },
      thumbEl(j, 0), h("div", {}, `${M.classes[P.label[j]].code} ${fmt(P.nn_sim[i][k], 2)}`)))),
    h("h4", {}, "Pixels"),
    h("div", { class: "swatch-lg", style: { background: P.rgb[i] }, title: `mean colour ${P.rgb[i]}` }),
    feat("brightness", "Brightness"), feat("contrast", "Contrast"), feat("saturation", "Saturation"),
    feat("colorfulness", "Colourfulness"), feat("sharpness", "Sharpness"), feat("entropy", "Entropy"),
    h("h4", {}, "Embedding geometry"),
    feat("knn_agree", "NN agreement"), feat("atypicality", "Atypicality", 3), feat("centroid_sim", "Centroid sim.", 3),
  );
  writeHash();
}
const kv = (k, v) => v == null || v === "" ? [] : [h("dt", {}, k), h("dd", {}, String(v))];

function selectClass(ci) {
  const idx = [];
  for (let i = 0; i < S.N; i++) if (S.P.label[i] === ci && S.visible[i]) idx.push(i);
  setSelection(idx, `${S.M.classes[ci].code} · ${S.M.classes[ci].name}`);
}

/* ---------------------------------------------------------------- photo context */
async function openContext(i) {
  const dlg = $("#contextDlg"), P = S.P, url = photoUrl(i);
  $("#ctxTitle").textContent = P.image_name[i];
  $("#ctxInfo").textContent = [P.island[i], P.date[i] || P.year[i], "sampled patches from this photo are outlined"].filter(Boolean).join(" · ");
  $("#ctxOpen").href = url;
  $("#ctxLoading").hidden = false;
  dlg.showModal();
  const cv = $("#ctxCanvas"), ctx = cv.getContext("2d");
  const img = new Image(); // no crossOrigin: the bucket sends no CORS headers, and we only draw (never read) pixels
  const others = [];
  for (let j = 0; j < S.N; j++) if (P.image_name[j] === P.image_name[i] && P.row[j] != null) others.push(j);
  const view = { k: 1, x: 0, y: 0 };
  const render = () => {
    const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    cv.width = r.width * d; cv.height = r.height * d; ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    if (!img.naturalWidth) return;
    const s0 = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight), s = s0 * view.k;
    const ox = (r.width - img.naturalWidth * s) / 2 + view.x, oy = (r.height - img.naturalHeight * s) / 2 + view.y;
    ctx.drawImage(img, ox, oy, img.naturalWidth * s, img.naturalHeight * s);
    ctx.font = `600 12px ${css("--font")}`; ctx.textBaseline = "bottom";
    for (const j of others) {
      const x = ox + P.col[j] * s, y = oy + P.row[j] * s, half = 112 * s;
      ctx.lineWidth = j === i ? 3 : 1.5;
      ctx.strokeStyle = j === i ? "#fff" : "rgba(255,255,255,.75)";
      ctx.strokeRect(x - half, y - half, half * 2, half * 2);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 6.3); ctx.fillStyle = j === i ? "#eda100" : "#fff"; ctx.fill();
      ctx.fillStyle = "#fff"; ctx.strokeStyle = "rgba(0,0,0,.7)"; ctx.lineWidth = 3;
      const t = S.M.classes[P.label[j]].code; ctx.strokeText(t, x - half, y - half - 2); ctx.fillText(t, x - half, y - half - 2);
    }
  };
  img.onload = () => {
    $("#ctxLoading").hidden = true;
    // start zoomed on the focused patch
    const r = cv.getBoundingClientRect(), s0 = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
    view.k = 2.5; const s = s0 * view.k;
    view.x = (img.naturalWidth / 2 - P.col[i]) * s; view.y = (img.naturalHeight / 2 - P.row[i]) * s;
    render();
  };
  img.onerror = () => { $("#ctxLoading").textContent = "Could not load the full photo."; };
  img.src = url;
  let dr = null;
  cv.onpointerdown = e => { dr = { x: e.clientX - view.x, y: e.clientY - view.y }; cv.setPointerCapture(e.pointerId); };
  cv.onpointermove = e => { if (dr) { view.x = e.clientX - dr.x; view.y = e.clientY - dr.y; render(); } };
  cv.onpointerup = () => { dr = null; };
  cv.onwheel = e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(), f = Math.exp(-e.deltaY * 0.0015), k = Math.max(.5, Math.min(20, view.k * f)), q = k / view.k;
    const mx = e.clientX - r.left - r.width / 2, my = e.clientY - r.top - r.height / 2;
    view.x = mx - (mx - view.x) * q; view.y = my - (my - view.y) * q; view.k = k; render();
  };
  window.addEventListener("resize", render, { once: true });
}

/* ---------------------------------------------------------------- drawer tabs */
function showTab(name) {
  $$(".tabs button[data-tab]").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  $$(".tabbody").forEach(b => { b.hidden = b.id !== `tab-${name}`; });
  $("#drawer").classList.remove("collapsed");
  ({ gallery: renderGallery, classes: renderClasses, confusion: renderConfusion, clusters: renderClusters, about: renderAbout })[name]();
}

function composition(idx) {
  const counts = new Map();
  idx.forEach(i => counts.set(S.P.group[i], (counts.get(S.P.group[i]) || 0) + 1));
  const pal = CAT[theme()];
  return h("div", { class: "composition", title: "T1 group composition" },
    [...counts.entries()].sort((a, b) => a[0] - b[0]).map(([g, n]) =>
      h("i", { style: { flex: n, background: pal[g % pal.length] }, title: `${S.M.groups[g].name}: ${n}` })));
}

function renderGallery() {
  const el = $("#tab-gallery");
  $("#galleryCount").textContent = S.selection ? S.selection.length : "";
  if (!S.selection) {
    el.replaceChildren(h("p", { class: "empty" }, "Shift-drag on the plot to lasso patches, search, or click a class in the Classes tab. Results show here."));
    return;
  }
  const idx = S.selection.slice(0, 400);
  const lab = new Map();
  S.selection.forEach(i => lab.set(S.P.label[i], (lab.get(S.P.label[i]) || 0) + 1));
  const top = [...lab.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([c, n]) => `${S.M.classes[c].code} ${Math.round(n / S.selection.length * 100)}%`).join(" · ");
  el.replaceChildren(
    h("div", { class: "gallery-head" }, h("b", {}, S.selLabel), h("span", {}, top), composition(S.selection),
      h("button", { class: "btn ghost", onclick: () => setSelection(null) }, "Clear"),
      h("button", { class: "btn ghost", onclick: () => downloadCsv(S.selection) }, "Export CSV")),
    h("div", { class: "gallery" }, idx.map(i => h("div", { class: "cell", onclick: () => focus(i), onmouseenter: () => { S.hover = i; S.dirty = true; } },
      thumbEl(i, 0),
      h("div", { class: "cap" }, h("span", {}, S.M.classes[S.P.label[i]].code), h("span", {}, S.scores ? fmt(S.scores[i], 3) : (S.P.split[i])))))),
    S.selection.length > idx.length ? h("p", { class: "muted" }, `Showing first ${idx.length} of ${S.selection.length}.`) : "",
  );
}

function downloadCsv(idx) {
  const P = S.P, M = S.M;
  const cols = ["index", "gcs_uri", "split", "label", "label_name", "group", "island", "site", "date", "lat", "lon", "depth", "knn_agree", "lp_pred", "lp_prob", "zs_pred", "cn_pred", "cn_conf", "score"];
  const rows = idx.map(i => [i, `gs://${M.bucket}/${P.object[i]}`, P.split[i], M.classes[P.label[i]].code, M.classes[P.label[i]].name, M.groups[P.group[i]].code,
    P.island[i], P.site[i], P.date[i], P.lat[i], P.lon[i], P.depth[i], P.knn_agree[i], M.classes[P.lp_top[i][0]].code, P.lp_prob[i][0],
    M.classes[P.zs_top[i][0]].code, P.cn_pred[i], P.cn_conf[i], S.scores ? S.scores[i].toFixed(4) : ""]);
  const csv = [cols, ...rows].map(r => r.map(v => v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v).join(",")).join("\n");
  const a = h("a", { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: "tide-pool-selection.csv" });
  a.click(); URL.revokeObjectURL(a.href);
}

let classSort = { key: "n", dir: -1 };
function renderClasses() {
  const M = S.M, pal = CAT[theme()], cc = classColors();
  const cols = [
    ["code", "Code"], ["name", "Name"], ["group", "Group"], ["n", "Sampled"], ["n_total", "In dataset"],
    ["agree", "NN agree"], ["lp", "Probe acc"], ["knn", "k-NN acc"], ["zs", "Zero-shot acc"], ["cn", "CoralNet agree"], ["atyp", "Atypicality"],
  ];
  const rows = M.classes.map((c, i) => ({ i, code: c.code, name: c.name, sure: c.sure, group: M.groups[c.group].name, gi: c.group, n: c.n, n_total: c.n_total, ...S.classStats[i] }));
  rows.sort((a, b) => { const x = a[classSort.key], y = b[classSort.key]; return (x > y ? 1 : x < y ? -1 : 0) * classSort.dir; });
  const bar = v => v == null ? "—" : [h("span", { class: "minibar", style: { width: `${v * 50}px` } }), pct(v)];
  $("#tab-classes").replaceChildren(h("table", { class: "t" },
    h("thead", {}, h("tr", {}, cols.map(([k, l]) => h("th", { onclick: () => { classSort = { key: k, dir: classSort.key === k ? -classSort.dir : -1 }; renderClasses(); } }, l + (classSort.key === k ? (classSort.dir > 0 ? " ▲" : " ▼") : ""))))),
    h("tbody", {}, rows.map(r => h("tr", { onclick: () => selectClass(r.i), title: "Select this class" },
      h("td", {}, h("span", { class: "sw", style: { background: cc[r.i] } }), r.code),
      h("td", { class: r.sure ? "" : "unsure", title: r.sure ? "" : "name is a best guess - edit pipeline/labels.py" }, r.name + (r.sure ? "" : " ?")),
      h("td", {}, h("span", { class: "sw", style: { background: pal[r.gi % pal.length] } }), r.group),
      h("td", {}, r.n), h("td", {}, r.n_total.toLocaleString()),
      h("td", {}, bar(r.agree)), h("td", {}, bar(r.lp)), h("td", {}, bar(r.knn)), h("td", {}, bar(r.zs)), h("td", {}, bar(r.cn)),
      h("td", {}, fmt(r.atyp, 3))))),
  ));
}

let confSource = "lp";
function renderConfusion() {
  const M = S.M, P = S.P, G = M.groups.length;
  const sources = { lp: ["Linear probe", i => P.lp_top[i][0]], knn: ["k-NN vote", i => P.knn_pred[i]], zs: ["Zero-shot", i => P.zs_top[i][0]],
    cn: ["CoralNet", i => { const c = P.cn_pred[i]; const k = M.classes.findIndex(x => x.code === c); return k < 0 ? null : k; }] };
  const pred = sources[confSource][1];
  const mat = Array.from({ length: G }, () => new Array(G).fill(0)), pairs = new Map(), cells = new Map();
  for (let i = 0; i < S.N; i++) {
    if (!S.visible[i]) continue;
    const p = pred(i); if (p == null) continue;
    const tg = P.group[i], pg = M.classes[p].group;
    mat[tg][pg]++;
    const ck = `${tg},${pg}`; if (!cells.has(ck)) cells.set(ck, []); cells.get(ck).push(i);
    if (p !== P.label[i]) { const k = `${P.label[i]},${p}`; if (!pairs.has(k)) pairs.set(k, []); pairs.get(k).push(i); }
  }
  const ramp = seqRamp();
  const cell = (v, tot) => {
    const f = tot ? v / tot : 0, col = ramp[Math.min(ramp.length - 1, Math.floor(f * ramp.length))];
    const n = parseInt(col.slice(1), 16), lum = (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
    return { background: v ? col : "transparent", color: v ? (lum > .55 ? "#0b0b0b" : "#fff") : css("--ink") };
  };
  const top = [...pairs.entries()].sort((a, b) => b[1].length - a[1].length).slice(0, 18);
  $("#tab-confusion").replaceChildren(
    h("div", { class: "gallery-head" }, h("b", {}, "Predictions vs. labels"),
      h("div", { class: "seg" }, Object.entries(sources).map(([k, [l]]) => h("button", { class: k === confSource ? "on" : "", onclick: () => { confSource = k; renderConfusion(); } }, l)))),
    h("div", { class: "conf-wrap" },
      h("div", {}, h("p", { class: "muted" }, "T1 groups · rows = true label, columns = prediction, shaded by row share. Click a cell to select."),
        h("table", { class: "matrix" },
          h("tr", {}, h("th", {}), M.groups.map(g => h("th", { title: g.name }, g.code))),
          mat.map((row, r) => { const tot = row.reduce((a, b) => a + b, 0); return h("tr", {}, h("th", { class: "rowh", title: M.groups[r].name }, M.groups[r].code),
            row.map((v, c) => h("td", { style: cell(v, tot), title: `${M.groups[r].name} → ${M.groups[c].name}: ${v} (${pct(tot ? v / tot : 0)})`, onclick: () => setSelection(cells.get(`${r},${c}`) || [], `${M.groups[r].code} predicted as ${M.groups[c].code}`) }, v ? pct(v / tot) : ""))); }))),
      h("div", { class: "toplist", style: { minWidth: "380px", flex: 1 } }, h("p", { class: "muted" }, "Most frequent T3 confusions (true → predicted). Click to inspect."),
        top.map(([k, idx]) => { const [t, p] = k.split(",").map(Number); return h("div", { class: "row", onclick: () => setSelection(idx, `${M.classes[t].code} predicted as ${M.classes[p].code}`) },
          h("span", {}, `${M.classes[t].code} · ${M.classes[t].name}`), h("span", {}, `→ ${M.classes[p].code} · ${M.classes[p].name}`), h("b", {}, idx.length)); })),
    ));
}

function renderClusters() {
  const M = S.M, P = S.P, K = M.clusters;
  const members = Array.from({ length: K }, () => []);
  for (let i = 0; i < S.N; i++) if (S.visible[i]) members[P.cluster[i]].push(i);
  $("#tab-clusters").replaceChildren(h("div", { class: "clusters" }, members.map((idx, k) => {
    if (!idx.length) return null;
    const lab = new Map(); idx.forEach(i => lab.set(P.label[i], (lab.get(P.label[i]) || 0) + 1));
    const [dom, dn] = [...lab.entries()].sort((a, b) => b[1] - a[1])[0];
    const purity = dn / idx.length;
    // show members closest to the cluster's typical patch: highest neighbour agreement first
    const sample = [...idx].sort((a, b) => P.atypicality[a] - P.atypicality[b]).slice(0, 12);
    return h("div", { class: "ccard", onclick: () => setSelection(idx, `Cluster ${k}`) },
      h("div", { style: { display: "flex", justifyContent: "space-between" } }, h("b", {}, `Cluster ${k}`), h("span", { class: "muted" }, `${idx.length} patches`)),
      h("div", { class: "muted" }, `${pct(purity)} ${M.classes[dom].code} · ${M.classes[dom].name} · ${lab.size} classes`),
      composition(idx),
      h("div", { class: "thumbs" }, sample.map(i => thumbEl(i, 0))));
  })));
}

function renderAbout() {
  const M = S.M, m = M.metrics;
  const metric = (v, l, t) => h("div", { class: "metric", title: t || "" }, h("b", {}, pct(v)), h("span", {}, l));
  $("#tab-about").replaceChildren(h("div", { class: "about" },
    h("div", { class: "metrics" },
      metric(m.linear_probe_acc, "Linear probe · T3", "Out-of-fold logistic regression accuracy (5 folds, grouped by source photo)"),
      metric(m.linear_probe_group_acc, "Linear probe · T1"),
      metric(m.knn_acc, `k-NN (k=${M.k}) · T3`),
      metric(m.zero_shot_acc, "Zero-shot text · T3", "Nearest class description in the shared text-image space"),
      metric(m.zero_shot_group_acc, "Zero-shot text · T1"),
      metric(m.coralnet_agree, "CoralNet top-1 agrees", "From the survey annotation CSVs; CoralNet’s classifier at annotation time")),
    h("h3", {}, "What you’re looking at"),
    h("p", {}, `Each point is a 224×224 crop centred on one human-annotated point in a NOAA PIFSC benthic photo-quadrat. `,
      `${M.n.toLocaleString()} of ${M.n_total.toLocaleString()} crops were sampled (${M.strategy}, seed ${M.seed}) from `,
      h("a", { href: M.dataset_url, target: "_blank", rel: "noopener" }, M.dataset), ` and embedded with `, h("b", {}, M.model),
      ` (${M.dim}-d Matryoshka for analysis, ${M.web_dim}-d in the browser).`),
    h("p", {}, "Projections: UMAP (2-D and 3-D, cosine), t-SNE, PCA, and a map from the survey coordinates. ",
      "Zero-shot scores compare each image to the text “", h("i", {}, M.prompt_template), "” for every class, in the same embedding space. ",
      "The linear probe and k-NN show how much class structure the embedding carries without any fine-tuning; patches whose neighbours disagree with their label are worth a second look."),
    h("p", {}, "Metadata (island, site, date, coordinates, CoralNet suggestions) is joined from the ESD REA annotation CSVs in the same public bucket; “Show in photo” loads the full-resolution quadrat photo and outlines every sampled patch from it."),
    h("p", { class: "muted" }, `Built ${M.built}. Class names marked “?” are best guesses from PIFSC codes; fix them in pipeline/labels.py.`),
  ));
}

/* ---------------------------------------------------------------- UI wiring */
const CAM = 4.5; // 3-D camera distance (in normalised data units)
const VIEWS = { umap: "UMAP", umap3d: "UMAP 3D", tsne: "t-SNE", pca: "PCA", geo: "Map" };

function setView(v) {
  S.view = v; S.tx = { k: 1, x: 0, y: 0 };
  $$("#viewSeg button").forEach(b => b.classList.toggle("on", b.dataset.v === v));
  $("#spinWrap").hidden = v !== "umap3d";
  $("#hint").innerHTML = v === "umap3d"
    ? "Drag to orbit · Ctrl-drag to pan · scroll to zoom · <b>Shift</b>-drag to lasso"
    : "Scroll to zoom · drag to pan · <b>Shift</b>-drag to lasso · click a point for details";
  computeVisible(); S.dirty = true; writeHash();
}

function setupUI() {
  const M = S.M;
  $("#subtitle").textContent = `${M.title} · ${M.n.toLocaleString()} of ${M.n_total.toLocaleString()} patches · ${M.classes.length} classes`;
  $("#modelChip").textContent = M.model_is_stub ? "stub embedder" : `${M.model.split("/").pop()} · ${M.dim}d`;
  $("#stubBanner").hidden = !M.model_is_stub;

  $("#viewSeg").replaceChildren(...M.projections.map(p => h("button", { "data-v": p, onclick: () => setView(p) }, VIEWS[p] || p)));
  const sel = $("#colorBy");
  let og = null;
  for (const c of COLOR_BY) {
    if (c.group) { og = h("optgroup", { label: c.group }); sel.append(og); continue; }
    (og || sel).append(h("option", { value: c.key }, c.label));
  }
  sel.value = S.colorBy;
  sel.onchange = () => { S.colorBy = sel.value; if (S.colorBy === "score" && !S.scores) { alert("Run a search first."); S.colorBy = "group"; sel.value = "group"; } encode(); writeHash(); };
  $$("#renderSeg button").forEach(b => b.onclick = () => { S.render = b.dataset.v; $$("#renderSeg button").forEach(x => x.classList.toggle("on", x === b)); S.dirty = true; writeHash(); });
  $("#size").oninput = e => { S.size = +e.target.value; S.dirty = true; };
  $("#labelsToggle").onchange = e => { S.labels = e.target.checked; S.dirty = true; };
  $("#spin").onchange = e => { S.spin = e.target.checked; };
  $("#resetView").onclick = () => { S.tx = { k: 1, x: 0, y: 0 }; S.dirty = true; };
  $("#splitFilters").replaceChildren(...["train", "val", "test"].map(s => h("button", {
    class: "on", onclick: e => { S.hiddenSplits.has(s) ? S.hiddenSplits.delete(s) : S.hiddenSplits.add(s); e.target.classList.toggle("on"); computeVisible(); S.dirty = true; }
  }, s)));
  $("#searchForm").onsubmit = e => { e.preventDefault(); textSearch($("#q").value); };
  $("#upload").onchange = e => { if (e.target.files[0]) imageSearch(e.target.files[0]); e.target.value = ""; };
  $("#uploadBtn").addEventListener("click", e => { if (!S.server) { e.preventDefault(); alert("Image search needs the server: python server/app.py"); } });
  $("#clearQuery").onclick = clearQuery;
  $$(".tabs button[data-tab]").forEach(b => b.onclick = () => showTab(b.dataset.tab));
  $("#drawerToggle").onclick = () => { const d = $("#drawer"); d.classList.toggle("collapsed"); $("#drawerToggle").textContent = d.classList.contains("collapsed") ? "▴" : "▾"; };
  $("#ctxClose").onclick = () => $("#contextDlg").close();
  document.addEventListener("keydown", e => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "Escape") { focus(-1); setSelection(null); }
    if (e.key === "r") { S.tx = { k: 1, x: 0, y: 0 }; S.dirty = true; }
    if (e.key === "p") $$("#renderSeg button").find(b => b.dataset.v !== S.render)?.click();
    if (e.key === "s" && S.focus >= 0) similarTo(S.focus);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => encode());
  setupCanvas();
  setView(S.view);
  encode();
  renderGallery();
}

function writeHash() {
  const p = new URLSearchParams({ view: S.view, color: S.colorBy, render: S.render });
  if (S.focus >= 0) p.set("p", S.focus);
  history.replaceState(null, "", `#${p}`);
}
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("view") && S.M.projections.includes(p.get("view"))) setView(p.get("view"));
  if (p.get("color") && p.get("color") !== "score" && COLOR_BY.some(c => c.key === p.get("color"))) { S.colorBy = p.get("color"); $("#colorBy").value = S.colorBy; encode(); }
  if (p.get("render")) $$("#renderSeg button").find(b => b.dataset.v === p.get("render"))?.click();
  if (p.has("p") && +p.get("p") < S.N) focus(+p.get("p"));
}

load();
