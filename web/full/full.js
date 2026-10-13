/* Tide Pool, all patches: a WebGL map of every embedded patch (written by pipeline/full.py).
   Plain ES2020, no build step. Per-point columns are binary files in data/; per-patch details
   are fetched shard by shard when a patch is opened; similarity search runs in the server. */
"use strict";

const $ = s => document.querySelector(s);
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) el.append(c.nodeType ? c : String(c));
  return el;
}
const fmt = (v, d = 2) => (v == null || !Number.isFinite(+v)) ? "—" : (+v).toFixed(d);
const pct = v => (v == null || !Number.isFinite(+v)) ? "—" : `${Math.round(v * 100)}%`;
const num = v => v.toLocaleString();

/* ---------------------------------------------------------------- palette (same as the sample explorer) */
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
const seqRamp = () => (isDark() ? SEQ.slice(0, 11).reverse() : SEQ.slice(1));
const hexRgb = hex => { const n = parseInt(hex.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; };
function hslRgb(hh, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + hh / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0) * 255, f(8) * 255, f(4) * 255].map(Math.round);
}
function rgbHsl([r, g, b]) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  let hh = 0, s = 0;
  if (mx !== mn) {
    const d = mx - mn;
    s = l > .5 ? d / (2 - mx - mn) : d / (mx + mn);
    hh = (mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4) * 60;
  }
  return [hh, s * 100, l * 100];
}
const rgbCss = c => `rgb(${c[0]},${c[1]},${c[2]})`;

/* ---------------------------------------------------------------- state */
const S = {
  M: null, P: null, C: {}, n: 0,
  view: "umap", colorBy: "group", render: "dots", size: 2.5,
  cam: { x: 0, y: 0, k: 1 }, hidden: new Map(), scores: null, query: "",
  focus: -1, hover: -1, details: new Map(), server: false, dirty: true,
};
let ENC = null;          // current colour encoding
const imgCache = new Map();

/* ---------------------------------------------------------------- loading */
async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); }
async function load() {
  const L = $("#loading");
  try {
    S.M = await getJSON("data/meta.json");
  } catch {
    L.replaceChildren(h("div", { style: { maxWidth: "520px", textAlign: "center" } },
      h("p", {}, "No full-dataset build yet."),
      h("p", {}, "Run ", h("code", {}, "python pipeline/full.py"), " (or ", h("code", {}, "docker compose run --rm full"),
        ") to embed every patch, then reload this page.")));
    return;
  }
  const M = S.M, T = { f32: Float32Array, u8: Uint8Array, u16: Uint16Array, u32: Uint32Array };
  S.n = M.n;
  const names = Object.keys(M.columns);
  let done = 0;
  await Promise.all([
    ...names.map(async k => {
      const c = M.columns[k], r = await fetch(`data/${c.file}`);
      if (!r.ok) throw new Error(`data/${c.file}: ${r.status}`);
      S.C[k] = new T[c.type](await r.arrayBuffer());
      L.textContent = `Loading ${num(M.n)} patches… ${++done}/${names.length}`;
    }),
    getJSON("data/photos.json").then(p => { S.P = p; }),
  ]);
  derive();
  initGL();
  initUI();
  L.hidden = true;
  checkServer();
  setView(M.views.includes("umap") ? "umap" : M.views[0]);
}

function derive() {
  const M = S.M, C = S.C, n = S.n, cls = M.classes;
  const pick = key => { const a = new Uint8Array(n); for (let i = 0; i < n; i++) a[i] = cls[C.label[i]][key]; return a; };
  C.group = pick("group"); C.t2 = pick("t2"); C.t2b = pick("t2b");
  C.zs_correct = new Uint8Array(n); C.cn_agree = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    C.zs_correct[i] = C.zs_pred[i] === C.label[i] ? 1 : 0;
    C.cn_agree[i] = C.cn_pred[i] === 255 ? 255 : (C.cn_pred[i] === C.label[i] ? 1 : 0);
  }
  const met = M.metrics;
  $("#subtitle").textContent = `${num(M.n)} patches · 67 classes · ${M.model}` +
    ` · k-NN acc ${pct(met.knn_acc)} · zero-shot ${pct(met.zero_shot_acc)}`;
  $("#stubBanner").hidden = !M.model_is_stub;
}

/* ---------------------------------------------------------------- image urls */
function objectPath(i) {
  const C = S.C, code = S.M.classes[C.label[i]].code;
  return `${S.P.t3_prefix}/${S.M.splits[C.split[i]]}/${code}/${code}_${S.P.photos[C.photo[i]]}_${C.point[i]}.png`;
}
const cropUrl = i => `https://storage.googleapis.com/${S.M.bucket}/${objectPath(i).split("/").map(encodeURIComponent).join("/")}`;
function photoUrl(i) {
  const src = S.P.sources[S.C.source[i]];
  return src ? `https://storage.googleapis.com/${S.M.bucket}/${S.M.mirror}/${src}/images/${encodeURIComponent(S.P.photos[S.C.photo[i]] + ".JPG")}` : null;
}
const crop = (i, cls = "crop") => h("img", { class: cls, src: cropUrl(i), alt: "", loading: "lazy", decoding: "async" });

/* ---------------------------------------------------------------- colour encodings */
function categorical(values, levels, opts = {}) {
  // values: per-point level id (255 / missing = Unknown). levels: [{label, color, sub?, group?}]
  const counts = new Array(levels.length).fill(0);
  let unknown = 0;
  for (let i = 0; i < S.n; i++) { const v = values[i]; if (v < levels.length) counts[v]++; else unknown++; }
  const lv = levels.map((l, id) => ({ ...l, id, n: counts[id], rgb: hexOrRgb(l.color) }));
  if (unknown) lv.push({ id: levels.length, label: "Unknown", n: unknown, rgb: hexRgb(OTHER[theme()]), other: true });
  return { type: "cat", values, levels: lv.filter(l => l.n > 0 || !opts.dropEmpty), unknownId: levels.length, grouped: opts.grouped };
}
const hexOrRgb = c => Array.isArray(c) ? c : hexRgb(c);
function nestedColors(items) { // items: [{group}] -> rgb per item: hue of the T1 group, lightness spread within it
  const pal = CAT[theme()], byG = new Map(), out = [];
  items.forEach((it, i) => { if (!byG.has(it.group)) byG.set(it.group, []); byG.get(it.group).push(i); });
  for (const [g, idx] of byG) {
    const [hh, ss, ll] = rgbHsl(hexRgb(pal[g % pal.length]));
    idx.forEach((i, j) => {
      const t = idx.length === 1 ? 0 : j / (idx.length - 1) - .5;
      out[i] = hslRgb(hh + t * 16, ss, Math.max(22, Math.min(78, ll + t * 34)));
    });
  }
  return out;
}
const cycle = k => { const pal = CAT[theme()]; return k < pal.length ? hexRgb(pal[k]) : hslRgb((k * 137.508) % 360, 62, isDark() ? 58 : 48); };
function numeric(values, opts = {}) {
  const v = Float32Array.from(values, x => (opts.missing != null && x === opts.missing) ? NaN : x);
  const fin = v.filter(Number.isFinite).sort();
  const q = p => fin.length ? fin[Math.min(fin.length - 1, Math.floor(fin.length * p))] : 0;
  const lo = opts.lo ?? q(.02), hi = opts.hi ?? q(.98);
  return { type: "num", v, lo, hi, ramp: seqRamp().map(hexRgb), fmtv: opts.fmt || (x => fmt(x, 2)), desc: opts.desc };
}

const COLOR_BY = [
  { group: "Labels" },
  { key: "group", label: "T1 group (broad)", build: () => categorical(S.C.group, S.M.groups.map((g, i) => ({ label: g.name, color: CAT[theme()][i % 8] }))) },
  { key: "t2", label: "T2 growth form / type", build: () => taxCat("t2") },
  { key: "t2b", label: "T2b genus / subgroup", build: () => taxCat("t2b") },
  { key: "label", label: "T3 class (fine)", build: () => taxCat("classes", "label") },
  { group: "Model behaviour" },
  { key: "score", label: "Search similarity", build: () => numeric(S.scores, { lo: S.scoreLo, hi: S.scoreHi, fmt: x => fmt(x, 3), desc: "Cosine similarity to the current search." }) },
  { key: "knn_agree", label: "Neighbour label agreement", build: () => numeric(S.C.knn_agree, { lo: 0, hi: 1, fmt: pct, desc: "Share of the k nearest neighbours (in all patches) with the same T3 class." }) },
  { key: "knn_correct", label: "k-NN vote correct", build: () => yesNo(S.C.knn_correct, "k-NN majority = label", "k-NN majority ≠ label") },
  { key: "zs_correct", label: "Zero-shot correct", build: () => yesNo(S.C.zs_correct, "zero-shot top-1 = label", "zero-shot top-1 ≠ label") },
  { key: "atypicality", label: "Atypicality", build: () => numeric(S.C.atypicality, { desc: "1 − mean similarity to the nearest neighbours. High = unusual patch." }) },
  { key: "centroid_sim", label: "Similarity to class centroid", build: () => numeric(S.C.centroid_sim) },
  { key: "cluster", label: "k-means cluster", build: () => categorical(S.C.cluster, Array.from({ length: S.M.clusters }, (_, k) => ({ label: `Cluster ${k}`, color: cycle(k) }))) },
  { group: "Survey" },
  { key: "split", label: "Split", build: () => categorical(S.C.split, S.M.splits.map((s, i) => ({ label: s, color: CAT[theme()][i] }))) },
  { key: "region", label: "Region", build: () => categorical(S.C.region, S.M.regions.map((s, i) => ({ label: s, color: cycle(i) }))) },
  { key: "island", label: "Island", build: () => categorical(S.C.island, S.M.islands.map((s, i) => ({ label: s, color: cycle(i) }))) },
  { key: "year", label: "Year", build: () => numeric(S.C.year, { missing: 0, fmt: x => String(Math.round(x)) }) },
  { key: "cn_agree", label: "CoralNet suggestion agrees", build: () => yesNo(S.C.cn_agree, "CoralNet agrees", "CoralNet differs") },
  { key: "cn_conf", label: "CoralNet confidence", build: () => numeric(S.C.cn_conf, { fmt: x => `${Math.round(x)}%` }) },
  { group: "Pixels" },
  { key: "rgb", label: "True mean colour", build: () => ({ type: "direct" }) },
];
function taxCat(listKey, colKey = listKey) {
  const list = S.M[listKey], cols = nestedColors(list);
  return categorical(S.C[colKey], list.map((it, i) => ({ label: it.code, sub: it.name, group: it.group, color: cols[i] })), { grouped: true });
}
function yesNo(values, yes, no) {
  return categorical(Uint8Array.from(values, v => v === 255 ? 255 : 1 - v),
    [{ label: yes, color: STATUS.good }, { label: no, color: STATUS.critical }]);
}
const hiddenSet = () => { if (!S.hidden.has(S.colorBy)) S.hidden.set(S.colorBy, new Set()); return S.hidden.get(S.colorBy); };

function encode() {
  const spec = COLOR_BY.find(c => c.key === S.colorBy);
  ENC = spec.build();
  const n = S.n, col = new Uint8Array(n * 4), ok = S.C.ok, hid = hiddenSet();
  let order = null;
  if (ENC.type === "cat") {
    const lut = new Map(ENC.levels.map(l => [l.id, l]));
    for (let i = 0; i < n; i++) {
      const v = ENC.values[i], l = lut.get(v < ENC.unknownId ? v : ENC.unknownId);
      const o = i * 4;
      if (!l) continue;
      col[o] = l.rgb[0]; col[o + 1] = l.rgb[1]; col[o + 2] = l.rgb[2];
      col[o + 3] = ok[i] && !hid.has(l.id) ? 255 : 0;
    }
  } else if (ENC.type === "num") {
    const R = ENC.ramp, span = (ENC.hi - ENC.lo) || 1, miss = hexRgb(OTHER[theme()]);
    for (let i = 0; i < n; i++) {
      const x = ENC.v[i], o = i * 4;
      const c = Number.isFinite(x) ? R[Math.max(0, Math.min(R.length - 1, Math.floor((x - ENC.lo) / span * R.length)))] : miss;
      col[o] = c[0]; col[o + 1] = c[1]; col[o + 2] = c[2]; col[o + 3] = ok[i] ? 255 : 0;
    }
    if (S.colorBy === "score") { // draw the best matches last so they sit on top
      order = new Uint32Array(n).map((_, i) => i);
      const key = i => Number.isFinite(ENC.v[i]) ? ENC.v[i] : -Infinity;
      order.sort((a, b) => key(a) - key(b));
    }
  } else {
    const rgb = S.C.rgb;
    for (let i = 0; i < n; i++) { const o = i * 4; col[o] = rgb[i * 3]; col[o + 1] = rgb[i * 3 + 1]; col[o + 2] = rgb[i * 3 + 2]; col[o + 3] = ok[i] ? 255 : 0; }
  }
  S.alpha = Uint8Array.from({ length: n }, (_, i) => col[i * 4 + 3]);
  GL.setColors(col, order);
  renderLegend();
  S.dirty = true;
}

/* ---------------------------------------------------------------- WebGL */
const GL = {};
function initGL() {
  const cv = $("#gl"), gl = cv.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
  if (!gl) { $("#loading").hidden = false; $("#loading").textContent = "This viewer needs WebGL 2."; throw new Error("no webgl2"); }
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, `#version 300 es
    in vec2 a_pos; in vec4 a_col;
    uniform vec2 u_scale; uniform vec2 u_off; uniform float u_size;
    out vec4 v_col;
    void main() {
      v_col = a_col;
      gl_Position = a_col.a == 0.0 ? vec4(2.0, 2.0, 2.0, 1.0) : vec4(a_pos * u_scale + u_off, 0.0, 1.0);
      gl_PointSize = u_size;
    }`));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, `#version 300 es
    precision mediump float;
    in vec4 v_col; uniform float u_alpha; out vec4 o;
    void main() {
      vec2 c = gl_PointCoord * 2.0 - 1.0;
      float r = dot(c, c);
      if (r > 1.0) discard;
      o = vec4(v_col.rgb, v_col.a * u_alpha * (1.0 - smoothstep(0.7, 1.0, r) * 0.6));
    }`));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const pos = gl.createBuffer(), col = gl.createBuffer(), idx = gl.createBuffer();
  const aPos = gl.getAttribLocation(prog, "a_pos"), aCol = gl.getAttribLocation(prog, "a_col");
  gl.bindBuffer(gl.ARRAY_BUFFER, pos); gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, col); gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol, 4, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const U = n => gl.getUniformLocation(prog, n);
  Object.assign(GL, {
    gl, cv, U: { scale: U("u_scale"), off: U("u_off"), size: U("u_size"), alpha: U("u_alpha") },
    setPositions(a) { gl.bindBuffer(gl.ARRAY_BUFFER, pos); gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW); },
    setColors(c, order) {
      gl.bindBuffer(gl.ARRAY_BUFFER, col); gl.bufferData(gl.ARRAY_BUFFER, c, gl.DYNAMIC_DRAW);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, order || identity(), gl.DYNAMIC_DRAW);
    },
  });
}
let _id = null;
const identity = () => _id || (_id = new Uint32Array(S.n).map((_, i) => i));

function resize() {
  const st = $("#stage"), d = devicePixelRatio || 1, w = st.clientWidth, hh = st.clientHeight;
  for (const c of [$("#gl"), $("#overlay")]) { c.width = Math.max(1, w * d); c.height = Math.max(1, hh * d); }
  S.W = w; S.H = hh; S.dirty = true;
}
// data coords live in [-1, 1]; m = pixels per data unit at zoom 1
const unit = () => Math.min(S.W, S.H) / 2 * 0.92;
const toScreen = (x, y) => [S.W / 2 + (x - S.cam.x) * S.cam.k * unit(), S.H / 2 - (y - S.cam.y) * S.cam.k * unit()];
const toData = (px, py) => [S.cam.x + (px - S.W / 2) / (S.cam.k * unit()), S.cam.y - (py - S.H / 2) / (S.cam.k * unit())];
const pointPx = () => Math.max(1, S.size * Math.min(3, Math.pow(S.cam.k, 0.35)));

function draw() {
  if (!S.dirty || !ENC) return requestAnimationFrame(draw);
  S.dirty = false;
  const { gl } = GL, d = devicePixelRatio || 1, k = S.cam.k * unit();
  gl.viewport(0, 0, GL.cv.width, GL.cv.height);
  gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  const sx = k / (S.W / 2), sy = k / (S.H / 2);
  gl.uniform2f(GL.U.scale, sx, sy);
  gl.uniform2f(GL.U.off, -S.cam.x * sx, -S.cam.y * sy);
  gl.uniform1f(GL.U.size, pointPx() * d);
  gl.uniform1f(GL.U.alpha, S.n > 100000 && S.cam.k < 2 ? 0.75 : 0.9);
  gl.drawElements(gl.POINTS, S.n, gl.UNSIGNED_INT, 0);
  drawOverlay();
  requestAnimationFrame(draw);
}

/* ---------------------------------------------------------------- overlay: focus, neighbours, patches */
function drawOverlay() {
  const cv = $("#overlay"), ctx = cv.getContext("2d"), d = devicePixelRatio || 1;
  ctx.setTransform(d, 0, 0, d, 0, 0);
  ctx.clearRect(0, 0, S.W, S.H);
  if (S.render === "patches") drawPatches(ctx);
  const ring = (i, r, w, color) => {
    const [x, y] = toScreen(S.xy[i * 2], S.xy[i * 2 + 1]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.lineWidth = w; ctx.strokeStyle = color; ctx.stroke();
  };
  const ink = isDark() ? "#fff" : "#0b0b0b";
  if (S.focus >= 0) {
    const det = detailOf(S.focus);
    if (det) for (const j of det.nn) ring(j, pointPx() / 2 + 3, 1.5, S.C.label[j] === S.C.label[S.focus] ? ink : STATUS.critical);
    ring(S.focus, pointPx() / 2 + 6, 2.5, ink);
  }
  if (S.hover >= 0 && S.hover !== S.focus) ring(S.hover, pointPx() / 2 + 4, 1.5, ink);
}
function drawPatches(ctx) {
  const vis = visible(450);
  if (vis === null) { $("#hint").textContent = "Patches: zoom in further (they are drawn once fewer than 450 are on screen)"; return; }
  $("#hint").textContent = `Patches: ${vis.length} on screen`;
  const s = Math.max(18, Math.min(96, 3.2 * pointPx() + 6));
  for (const i of vis) {
    const [x, y] = toScreen(S.xy[i * 2], S.xy[i * 2 + 1]);
    let im = imgCache.get(i);
    if (!im) {
      if (imgCache.size > 1500) imgCache.delete(imgCache.keys().next().value);
      im = new Image(); im.decoding = "async"; im.onload = () => { S.dirty = true; }; im.src = cropUrl(i); imgCache.set(i, im);
    }
    if (im.complete && im.naturalWidth) ctx.drawImage(im, x - s / 2, y - s / 2, s, s);
    else { ctx.fillStyle = isDark() ? "#2c2c2a" : "#e1e0d9"; ctx.fillRect(x - s / 2, y - s / 2, s, s); }
  }
}

/* ---------------------------------------------------------------- spatial index for picking */
const G = 256, LO = -1.4, SPAN = 2.8;
function buildGrid() {
  const n = S.n, xy = S.xy, cell = new Int32Array(n), counts = new Uint32Array(G * G + 1);
  for (let i = 0; i < n; i++) {
    const x = xy[i * 2], y = xy[i * 2 + 1];
    if (x > 8 || !Number.isFinite(x)) { cell[i] = -1; continue; }
    const cx = Math.max(0, Math.min(G - 1, Math.floor((x - LO) / SPAN * G)));
    const cy = Math.max(0, Math.min(G - 1, Math.floor((y - LO) / SPAN * G)));
    cell[i] = cy * G + cx; counts[cell[i] + 1]++;
  }
  for (let c = 0; c < G * G; c++) counts[c + 1] += counts[c];
  const fill = counts.slice(0, G * G), items = new Uint32Array(n);
  for (let i = 0; i < n; i++) if (cell[i] >= 0) items[fill[cell[i]]++] = i;
  S.grid = { start: counts, items };
}
function cellRange(x0, y0, x1, y1) {
  const c = v => Math.max(0, Math.min(G - 1, Math.floor((v - LO) / SPAN * G)));
  return [c(Math.min(x0, x1)), c(Math.min(y0, y1)), c(Math.max(x0, x1)), c(Math.max(y0, y1))];
}
function nearest(px, py, rpx = 8) {
  const [x, y] = toData(px, py), r = (rpx + pointPx() / 2) / (S.cam.k * unit());
  const [a, b, c, d] = cellRange(x - r, y - r, x + r, y + r);
  let best = -1, bd = r * r;
  for (let cy = b; cy <= d; cy++) for (let cx = a; cx <= c; cx++) {
    const id = cy * G + cx;
    for (let p = S.grid.start[id]; p < S.grid.start[id + 1]; p++) {
      const i = S.grid.items[p];
      if (!S.alpha[i]) continue;
      const dx = S.xy[i * 2] - x, dy = S.xy[i * 2 + 1] - y, dd = dx * dx + dy * dy;
      if (dd < bd) { bd = dd; best = i; }
    }
  }
  return best;
}
function visible(limit) { // indices on screen, or null if more than `limit`
  const [x0, y0] = toData(0, S.H), [x1, y1] = toData(S.W, 0);
  const [a, b, c, d] = cellRange(x0, y0, x1, y1), out = [];
  for (let cy = b; cy <= d; cy++) for (let cx = a; cx <= c; cx++) {
    const id = cy * G + cx;
    for (let p = S.grid.start[id]; p < S.grid.start[id + 1]; p++) {
      const i = S.grid.items[p], x = S.xy[i * 2], y = S.xy[i * 2 + 1];
      if (S.alpha[i] && x >= x0 && x <= x1 && y >= y0 && y <= y1) { out.push(i); if (out.length > limit) return null; }
    }
  }
  return out;
}

/* ---------------------------------------------------------------- views & camera */
function setView(v) {
  S.view = v;
  S.xy = S.C[`xy_${v}`];
  GL.setPositions(S.xy);
  buildGrid();
  document.querySelectorAll("#viewSeg button").forEach(b => b.classList.toggle("on", b.dataset.v === v));
  resetCam();
  if (!ENC) encode();
}
function resetCam() { S.cam = { x: 0, y: 0, k: 1 }; S.dirty = true; }
function panTo(i, k) {
  S.cam.x = S.xy[i * 2]; S.cam.y = S.xy[i * 2 + 1];
  if (k) S.cam.k = Math.max(S.cam.k, k);
  S.dirty = true;
}

/* ---------------------------------------------------------------- legend */
function renderLegend() {
  const el = $("#legend"), spec = COLOR_BY.find(c => c.key === S.colorBy);
  el.replaceChildren(h("h3", {}, spec.label));
  if (ENC.type === "direct") { el.append(h("p", { class: "note" }, "Each dot is painted with the patch’s mean pixel colour.")); return aboutMini(el); }
  if (ENC.type === "num") {
    el.append(h("div", { class: "ramp", style: { background: `linear-gradient(90deg, ${ENC.ramp.map(rgbCss).join(",")})` } }),
      h("div", { class: "ramp-labels" }, h("span", {}, ENC.fmtv(ENC.lo)), h("span", {}, ENC.fmtv(ENC.hi))));
    if (ENC.desc) el.append(h("p", { class: "note" }, ENC.desc));
    return aboutMini(el);
  }
  const hid = hiddenSet(), refresh = () => encode();
  el.append(h("div", { class: "actions" },
    h("button", { onclick: () => { hid.clear(); refresh(); } }, "Show all"),
    h("button", { onclick: () => { ENC.levels.forEach(l => hid.add(l.id)); refresh(); } }, "Hide all")));
  const item = l => h("div", {
    class: "item" + (hid.has(l.id) ? " off" : ""), title: "Click to hide or show · double-click to show only this",
    onclick: () => { hid.has(l.id) ? hid.delete(l.id) : hid.add(l.id); refresh(); },
    ondblclick: () => { hid.clear(); ENC.levels.forEach(o => o.id !== l.id && hid.add(o.id)); refresh(); },
  }, h("span", { class: "sw", style: { background: rgbCss(l.rgb) } }),
    h("span", { class: "lab" }, l.label, l.sub && l.sub !== l.label ? h("small", {}, ` ${l.sub}`) : null),
    h("span", { class: "n" }, num(l.n)));
  if (ENC.grouped) {
    S.M.groups.forEach((g, gi) => {
      const ls = ENC.levels.filter(l => l.group === gi);
      if (ls.length) el.append(h("div", { class: "grp" }, g.name), ...ls.map(item));
    });
    ENC.levels.filter(l => l.other).forEach(l => el.append(item(l)));
  } else ENC.levels.forEach(l => el.append(item(l)));
  aboutMini(el);
}
function aboutMini(el) {
  const M = S.M, m = M.metrics, row = (k, v) => h("div", { class: "row" }, h("span", {}, k), h("b", {}, v));
  el.append(h("div", { class: "about-mini" }, h("h3", {}, "This build"),
    row("Patches", num(M.n)), M.n_ok < M.n ? row("Crops that failed to load", num(M.n - M.n_ok)) : null,
    row("k-NN accuracy", pct(m.knn_acc)), row("Neighbour agreement", pct(m.knn_agree)),
    row("Zero-shot accuracy", pct(m.zero_shot_acc)), row("CoralNet agrees with label", pct(m.coralnet_agree)),
    row("Built", M.built)));
}

/* ---------------------------------------------------------------- tooltip & detail */
async function detailShard(i) {
  const k = Math.floor(i / S.M.shard);
  if (!S.details.has(k)) S.details.set(k, getJSON(`data/detail/${String(k).padStart(5, "0")}.json`).then(d => (S.details.set(k, d), d)));
  return S.details.get(k);
}
function detailOf(i) { // synchronous view of an already-loaded record
  const d = S.details.get(Math.floor(i / S.M.shard));
  if (!d || d instanceof Promise) return null;
  const j = i % S.M.shard;
  return Object.fromEntries(Object.entries(d).map(([k, v]) => [k, v[j]]));
}
function taxPath(c) {
  const parts = [S.M.groups[c.group].name];
  for (const n of [S.M.t2[c.t2].name, S.M.t2b[c.t2b].name]) if (n !== parts[parts.length - 1] && n !== c.name) parts.push(n);
  return parts.join(" › ");
}

let tipTimer = 0;
function showTip(i, px, py) {
  const t = $("#tooltip"), C = S.C, c = S.M.classes[C.label[i]];
  clearTimeout(tipTimer);
  tipTimer = setTimeout(() => {
    t.replaceChildren(crop(i), h("div", { class: "t-body" }, h("b", {}, `${c.code} · ${c.name}`),
      h("div", { class: "muted" }, taxPath(c)),
      h("div", { class: "muted" }, [S.M.splits[C.split[i]], S.M.islands[C.island[i]], C.year[i] || null].filter(Boolean).join(" · ")),
      S.scores ? h("div", { class: "muted" }, `similarity ${fmt(S.scores[i], 3)}`) : null));
    t.hidden = false;
    const tw = t.offsetWidth, th = t.offsetHeight;
    t.style.left = `${Math.min(px + 14, S.W - tw - 6)}px`;
    t.style.top = `${py + 14 + th > S.H ? py - th - 10 : py + 14}px`;
  }, 60);
}
function hideTip() { clearTimeout(tipTimer); $("#tooltip").hidden = true; }

async function focus(i, { pan = false } = {}) {
  S.focus = i; S.dirty = true;
  hideTip();
  const el = $("#detail");
  if (i < 0) { el.hidden = true; return resize(); }
  if (pan) panTo(i, 4);
  const wasHidden = el.hidden;
  el.hidden = false;
  if (wasHidden) resize();
  const C = S.C, M = S.M, c = M.classes[C.label[i]];
  el.replaceChildren(h("div", { class: "head" }, h("div", {}, h("h2", {}, `${c.code} · ${c.name}`), h("div", { class: "muted" }, `${taxPath(c)} · #${i}`)),
    h("button", { class: "btn ghost x", onclick: () => focus(-1), "aria-label": "Close details" }, "×")), h("p", { class: "muted" }, "Loading…"));
  const d = (await detailShard(i), detailOf(i));
  if (S.focus !== i) return;
  S.dirty = true;
  const kv = (k, v) => v == null || v === "" ? [] : [h("dt", {}, k), h("dd", {}, v)];
  const bars = rows => h("div", { class: "bars" }, rows.map(([lab, v, ci, title]) =>
    h("div", { class: "row", title: title || "", onclick: () => ci != null && isolateClass(ci) },
      h("span", { class: "name" + (ci === C.label[i] ? " match" : "") }, lab),
      h("span", { class: "track" }, h("span", { class: "fill", style: { width: `${Math.max(0, Math.min(1, v)) * 100}%` } })),
      h("span", { class: "val" }, pct(v)))));
  const codeIdx = code => M.classes.findIndex(x => x.code === code);
  const zs = d.zs_score, zmin = Math.min(...zs) - .02, zmax = Math.max(...zs);
  const photo = photoUrl(i);
  el.replaceChildren(
    h("div", { class: "head" },
      h("div", {}, h("h2", {}, `${c.code} · ${c.name}`), h("div", { class: "muted" }, `${taxPath(c)} · #${i}`)),
      h("button", { class: "btn ghost x", onclick: () => focus(-1), "aria-label": "Close details" }, "×")),
    h("div", { class: "patch" }, h("img", { src: cropUrl(i), alt: `${c.name} patch` }), h("span", { class: "cross", title: "annotated point" })),
    h("div", { class: "actions-row" },
      h("button", { class: "btn", onclick: () => similarTo(i), disabled: !S.server, title: S.server ? "Rank all patches by similarity (S)" : "Needs the server" }, "Find similar"),
      h("button", { class: "btn ghost", onclick: () => isolateClass(C.label[i]) }, `Only ${c.code}`),
      photo ? h("a", { class: "btn ghost", href: photo, target: "_blank", rel: "noopener" }, "Open photo") : null,
      h("button", { class: "btn ghost", onclick: () => navigator.clipboard?.writeText(`gs://${M.bucket}/${objectPath(i)}`).catch(() => {}) }, "Copy URI")),
    h("div", {},
      h("span", { class: "pill " + (S.C.knn_correct[i] ? "good" : "bad") }, "k-NN vote"),
      h("span", { class: "pill " + (C.zs_correct[i] ? "good" : "bad") }, "zero-shot"),
      C.cn_agree[i] === 255 ? null : h("span", { class: "pill " + (C.cn_agree[i] ? "good" : "bad") }, "CoralNet")),
    h("h4", {}, "Survey"),
    h("dl", { class: "kv" }, ...kv("Split", M.splits[C.split[i]]), ...kv("Region", M.regions[C.region[i]]),
      ...kv("Island", M.islands[C.island[i]]), ...kv("Site", d.site), ...kv("Date", d.date || C.year[i] || null),
      ...kv("Lat / lon", d.lat == null ? null : `${fmt(d.lat, 4)}, ${fmt(d.lon, 4)}`),
      ...kv("Photo", `${S.P.photos[C.photo[i]]}.JPG`), ...kv("Point", d.row == null ? `#${C.point[i]}` : `#${C.point[i]} (row ${d.row}, col ${d.col})`),
      ...kv("Cluster", `k-means ${C.cluster[i]}`), ...kv("Neighbour agreement", pct(C.knn_agree[i]))),
    h("h4", { title: "Nearest neighbours among all patches (cosine). Red = different T3 class." }, `Nearest ${d.nn.length} of ${num(M.n)}`),
    h("div", { class: "nn" }, d.nn.map((j, k) => h("div", {
      class: "cell" + (C.label[j] !== C.label[i] ? " diff" : ""), onclick: () => focus(j, { pan: true }),
      title: `${M.classes[C.label[j]].code} · ${M.classes[C.label[j]].name} · cos ${d.nn_sim[k]}`,
    }, crop(j), h("div", {}, `${M.classes[C.label[j]].code} ${fmt(d.nn_sim[k], 2)}`)))),
    h("h4", { title: "Cosine similarity to each class’s text description, rescaled within the top 3" }, "Zero-shot (top 3)"),
    bars(d.zs_top.map((ci, k) => [`${M.classes[ci].code} · ${M.classes[ci].name}`, (zs[k] - zmin) / ((zmax - zmin) || 1), ci, `cosine ${fmt(zs[k], 3)}`])),
    d.machine.length ? h("h4", {}, "CoralNet suggestions") : null,
    d.machine.length ? bars(d.machine.map(([code, conf]) => [code + (codeIdx(code) >= 0 ? ` · ${M.classes[codeIdx(code)].name}` : ""), (conf ?? 0) / 100, codeIdx(code) >= 0 ? codeIdx(code) : null])) : null,
  );
}

function isolateClass(ci) {
  S.colorBy = "label"; $("#colorBy").value = "label";
  const hid = hiddenSet(); hid.clear();
  S.M.classes.forEach((_, k) => k !== ci && hid.add(k));
  hid.add(S.M.classes.length); // Unknown
  encode();
}

/* ---------------------------------------------------------------- search (server-side, over every vector) */
const F16 = (() => { // float16 -> float32 lookup table
  const t = new Float32Array(65536);
  for (let u = 0; u < 65536; u++) {
    const s = u & 0x8000 ? -1 : 1, e = (u >> 10) & 31, f = u & 1023;
    t[u] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return t;
})();
async function runSearch(req, label) {
  const chip = $("#queryChip");
  chip.hidden = false; $("#queryText").textContent = `${label} …`;
  try {
    const r = await fetch(req.url, req.init);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `server error ${r.status}`);
    const u = new Uint16Array(await r.arrayBuffer());
    if (u.length !== S.n) throw new Error(`server returned ${u.length} scores for ${S.n} patches`);
    const sc = new Float32Array(S.n);
    for (let i = 0; i < S.n; i++) sc[i] = S.C.ok[i] ? F16[u[i]] : NaN;
    const top = topK(sc, 60);
    S.scores = sc; S.query = label;
    S.scoreHi = sc[top[0]]; S.scoreLo = sc[top[top.length - 1]] - (sc[top[0]] - sc[top[top.length - 1]]) * 2;
    if (S.colorBy !== "score") S.prevColor = S.colorBy;
    S.colorBy = "score"; $("#colorBy").value = "score";
    $("#queryText").textContent = label;
    encode();
    showResults(top, label);
  } catch (e) {
    $("#queryText").textContent = `${label}: ${e.message}`;
  }
}
function topK(sc, k) {
  const best = [];
  for (let i = 0; i < sc.length; i++) {
    const v = sc[i];
    if (!(v > (best.length === k ? sc[best[k - 1]] : -Infinity))) continue;
    let p = Math.min(best.length, k - 1);
    best[p] = i;
    while (p > 0 && sc[best[p - 1]] < v) { [best[p - 1], best[p]] = [best[p], best[p - 1]]; p--; }
  }
  return best;
}
function showResults(idx, label) {
  $("#results").hidden = false;
  $("#resultsTitle").textContent = `Top ${idx.length} of ${num(S.n)} for “${label}”`;
  $("#resultsNote").textContent = " · click a patch to open it · all patches are coloured by similarity";
  $("#strip").replaceChildren(...idx.map(i => h("div", { class: "cell", onclick: () => focus(i, { pan: true }), title: S.M.classes[S.C.label[i]].name },
    crop(i), h("div", { class: "cap" }, h("span", {}, S.M.classes[S.C.label[i]].code), h("span", {}, fmt(S.scores[i], 3))))));
  resize();
}
function clearSearch() {
  S.scores = null; S.query = "";
  $("#queryChip").hidden = true; $("#results").hidden = true;
  if (S.colorBy === "score") { S.colorBy = S.prevColor || "group"; $("#colorBy").value = S.colorBy; }
  encode(); resize();
}
const similarTo = i => S.server && runSearch({ url: "../api/full/search", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ index: i }) } },
  `like #${i} (${S.M.classes[S.C.label[i]].code})`);

async function checkServer() {
  const chip = $("#serverChip"), set = (txt, ok) => chip.replaceChildren(h("span", { class: "dot", style: { background: ok ? STATUS.good : OTHER[theme()] } }), txt);
  try {
    const r = await fetch("../api/health");
    if (!r.ok) throw 0;
    const j = await r.json();
    S.server = j.status === "ready" && j.full === S.n;
    set(S.server ? "search ready" : j.full == null ? "server: loading vectors…" : j.full !== S.n ? "server: vectors out of date" : `server: ${j.status}`, S.server);
    chip.title = S.server ? `${j.model} · ${num(j.full)} vectors` : "Text, image and similar-patch search need the server with this build's vectors.";
    $("#uploadBtn").classList.toggle("disabled", !S.server);
    if (!S.server && (j.status === "loading" || j.full == null)) setTimeout(checkServer, 3000);
  } catch {
    S.server = false; set("no server: search off", false);
    chip.title = "Run python server/app.py (or docker compose up app) to search.";
    $("#uploadBtn").classList.add("disabled");
  }
  if (S.focus >= 0) focus(S.focus);
}

/* ---------------------------------------------------------------- UI wiring */
function initUI() {
  const vs = $("#viewSeg"), names = { umap: "UMAP", pca: "PCA", geo: "Map" };
  for (const v of S.M.views) vs.append(h("button", { "data-v": v, onclick: () => setView(v) }, names[v] || v));
  const sel = $("#colorBy");
  let og = null;
  for (const c of COLOR_BY) {
    if (c.group) { og = h("optgroup", { label: c.group }); sel.append(og); continue; }
    og.append(h("option", { value: c.key }, c.label));
  }
  sel.value = S.colorBy;
  sel.onchange = () => {
    if (sel.value === "score" && !S.scores) { sel.value = S.colorBy; $("#q").focus(); return; }
    S.colorBy = sel.value; encode();
  };
  $("#renderSeg").onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    S.render = b.dataset.v;
    document.querySelectorAll("#renderSeg button").forEach(x => x.classList.toggle("on", x === b));
    if (S.render === "dots") $("#hint").textContent = "Scroll to zoom · drag to pan · click a point for details · S = find similar";
    S.dirty = true;
  };
  $("#size").oninput = e => { S.size = +e.target.value; S.dirty = true; };
  $("#resetView").onclick = resetCam;
  $("#clearQuery").onclick = clearSearch;
  $("#searchForm").onsubmit = e => {
    e.preventDefault();
    const q = $("#q").value.trim(); if (!q) return;
    const ci = S.M.classes.findIndex(c => c.code.toLowerCase() === q.toLowerCase());
    if (ci >= 0) return isolateClass(ci);
    if (!S.server) { $("#queryChip").hidden = false; $("#queryText").textContent = "Text search needs the server; type a class code (e.g. POMA) to filter instead"; return; }
    runSearch({ url: "../api/full/search", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: q }) } }, `“${q}”`);
  };
  $("#upload").onchange = e => {
    const f = e.target.files[0]; if (!f || !S.server) return;
    const fd = new FormData(); fd.append("file", f);
    runSearch({ url: "../api/full/search_image", init: { method: "POST", body: fd } }, `image ${f.name}`);
    e.target.value = "";
  };

  const cv = $("#gl"), st = $("#stage");
  let drag = null;
  cv.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, cx: S.cam.x, cy: S.cam.y, moved: false }; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => {
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; st.classList.add("dragging"); hideTip(); }
      if (drag.moved) { S.cam.x = drag.cx - dx / (S.cam.k * unit()); S.cam.y = drag.cy + dy / (S.cam.k * unit()); S.dirty = true; }
      return;
    }
    const i = nearest(px, py);
    if (i !== S.hover) { S.hover = i; S.dirty = true; i >= 0 ? showTip(i, px, py) : hideTip(); }
  });
  cv.addEventListener("pointerup", e => {
    st.classList.remove("dragging");
    if (drag && !drag.moved) {
      const r = cv.getBoundingClientRect(), i = nearest(e.clientX - r.left, e.clientY - r.top);
      focus(i);
    }
    drag = null;
  });
  cv.addEventListener("pointerleave", () => { S.hover = -1; hideTip(); S.dirty = true; });
  cv.addEventListener("wheel", e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    const [x, y] = toData(px, py), k = Math.max(0.5, Math.min(4000, S.cam.k * Math.exp(-e.deltaY * 0.0015)));
    S.cam.k = k;
    const [x2, y2] = toData(px, py);
    S.cam.x += x - x2; S.cam.y += y - y2; S.dirty = true;
  }, { passive: false });
  addEventListener("keydown", e => {
    if (e.target.closest("input, select, textarea")) return;
    if (e.key === "Escape") focus(-1);
    else if (e.key === "r" || e.key === "R") resetCam();
    else if ((e.key === "s" || e.key === "S") && S.focus >= 0) similarTo(S.focus);
  });
  new ResizeObserver(resize).observe(st);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => encode());
  resize();
  requestAnimationFrame(draw);
}

load().catch(e => { $("#loading").hidden = false; $("#loading").textContent = `Could not load: ${e.message}`; console.error(e); });
