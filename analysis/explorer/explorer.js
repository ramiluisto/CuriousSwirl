/* Interactive explorer for the UMAP layouts of difference vectors.
 *
 * Plain JavaScript on <canvas>: no libraries and no network access, so the
 * rendered Quarto page stays self-contained.  The data come from
 * explorer_data.js (written by scripts/09_build_explorer.py), which defines
 * window.SWIRL_DATA with base64-encoded typed arrays.
 */
(function () {
  "use strict";
  const DATA = window.SWIRL_DATA;
  const root = document.getElementById("swirl-explorer");
  if (!root) return;
  if (!DATA) {
    root.textContent = "Explorer data missing: run scripts/09_build_explorer.py before rendering.";
    return;
  }

  // ------------------------------------------------------------------ constants
  const PAIR_STYLE = {
    antonyms: { name: "Antonyms", colour: "#DD8452" },
    shuffled_antonym_words: { name: "Shuffled antonyms", colour: "#C44E52" },
    shuffled_synonym_words: { name: "Shuffled synonyms", colour: "#55A868" },
    synonyms: { name: "Synonyms", colour: "#4C72B0" },
  };
  const LAYOUTS = {
    stored: {
      title: "Rows as stored (the paper's figures)",
      blurb: "The order used for the paper: antonyms, shuffled antonyms, shuffled synonyms, synonyms, each block in file order.",
    },
    permuted_a: {
      title: "Rows shuffled (permutation A)",
      blurb: "The same matrix with all rows in a random order. Nothing else changes.",
    },
    permuted_b: {
      title: "Rows shuffled (permutation B)",
      blurb: "A second, independent random order of the same rows.",
    },
    blocks_reordered: {
      title: "Class blocks reordered",
      blurb: "The same four blocks stacked as synonyms, antonyms, shuffled synonyms, shuffled antonyms.",
    },
    alphabetical: {
      title: "Rows sorted alphabetically",
      blurb: "Rows sorted by the first word of the pair, ignoring pair type.",
    },
    by_length: {
      title: "Rows sorted by vector length",
      blurb: "Shortest difference vectors first, ignoring pair type.",
    },
    edges_random: {
      title: "Edges visited in random order",
      blurb: "The paper's neighbour graph, initialisation and seeds; only the order in which the optimiser visits the edges is shuffled, so no row order can leak.",
    },
    adam: {
      title: "umap-learn 0.6, Adam optimiser",
      blurb: "Rows as stored, laid out with an optimiser that shuffles the node order every epoch.",
    },
    noise: {
      title: "Random directions, same lengths",
      blurb: "Every difference vector replaced by a random direction of the same length; rows as stored. No semantics, no shared words.",
    },
  };
  const LAYOUT_ORDER = ["stored", "permuted_a", "permuted_b", "blocks_reordered", "alphabetical",
    "by_length", "edges_random", "adam", "noise"];
  const COLOURINGS = {
    pair: { title: "Pair type", kind: "pair" },
    fed: { title: "Position in the order given to UMAP", kind: "ramp", ramp: "turbo", ends: ["first row fed", "last row fed"] },
    stored: { title: "Row index in the paper's matrix", kind: "ramp", ramp: "turbo", ends: ["row 0", "last row"] },
    norm: { title: "Length of the difference vector", kind: "ramp", ramp: "viridis", ends: ["shortest", "longest (quantiles)"] },
    inDegree: { title: "Popularity as a neighbour (hubs)", kind: "ramp", ramp: "viridis", ends: ["nobody's neighbour", "most popular (log scale)"] },
    cosine: { title: "Cosine similarity of the two words", kind: "ramp", ramp: "viridis", ends: ["least similar", "most similar (quantiles)"] },
    split: { title: "Stuttgart split", kind: "codes", names: ["shuffled pair", "train", "validation", "test"], colours: ["#c9c9c9", "#4C72B0", "#E3B23C", "#C44E52"] },
    pos: { title: "Part of speech", kind: "codes", names: ["shuffled pair", "adjectives", "nouns", "verbs"], colours: ["#c9c9c9", "#8172B3", "#64B5CD", "#937860"] },
  };
  const PRESETS = [
    { label: "1 · The paper's picture", left: "stored", right: "permuted_a", colour: "pair",
      note: "Left: the swirl as published. Right: the same rows in random order. The curve survives, the colours mix." },
    { label: "2 · Colour by row index", left: "stored", right: "permuted_a", colour: "stored",
      note: "Left: a smooth gradient, so the curve is the row index. Right: the same colouring is noise once the rows were fed in another order." },
    { label: "3 · Colour by feed order", left: "permuted_a", right: "permuted_b", colour: "fed",
      note: "Each panel coloured by the order in which its rows were given to UMAP: both shuffles give a smooth gradient along their own curve." },
    { label: "4 · UMAP draws the alphabet", left: "alphabetical", right: "by_length", colour: "fed",
      note: "Sort the rows alphabetically, or by vector length, and the curve follows that order instead." },
    { label: "5 · Move the blocks", left: "stored", right: "blocks_reordered", colour: "pair",
      note: "Restacking the four blocks changes which colours sit next to each other along the curve." },
    { label: "6 · Who the hubs are", left: "stored", right: "edges_random", colour: "inDegree",
      note: "A few very popular neighbours (yellow) drive the layout. Hover them, or search for 'north'." },
    { label: "7 · Noise does it too", left: "noise", right: "stored", colour: "pair",
      note: "Left: random directions with the original lengths, in the stored order. Four clean colours without any content." },
    { label: "8 · Order-free layouts", left: "edges_random", right: "adam", colour: "pair",
      note: "Two layouts that cannot see the row order. What separates here is in the neighbour graph itself." },
  ];

  // ------------------------------------------------------------------ helpers
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const key in attrs || {}) {
      if (key === "class") node.className = attrs[key];
      else if (key === "text") node.textContent = attrs[key];
      else node.setAttribute(key, attrs[key]);
    }
    (children || []).forEach((child) => node.appendChild(child));
    return node;
  }

  function decode(text, Type) {
    const binary = atob(text);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Type(bytes.buffer);
  }

  function turbo(t) {  // polynomial approximation of the Turbo colour map
    const r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
    const g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
    const b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66))));
    const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
    return "rgb(" + clamp(r) + "," + clamp(g) + "," + clamp(b) + ")";
  }
  const VIRIDIS = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];
  function viridis(t) {
    const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
    const i = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - i;
    const c = [0, 1, 2].map((k) => Math.round(VIRIDIS[i][k] + f * (VIRIDIS[i + 1][k] - VIRIDIS[i][k])));
    return "rgb(" + c.join(",") + ")";
  }
  const RAMPS = {
    turbo: Array.from({ length: 256 }, (_, i) => turbo(i / 255)),
    viridis: Array.from({ length: 256 }, (_, i) => viridis(i / 255)),
  };

  function quantileIndex(values) {  // 0..255 by rank
    const n = values.length, order = Array.from({ length: n }, (_, i) => i);
    order.sort((a, b) => values[a] - values[b]);
    const out = new Uint8Array(n);
    for (let r = 0; r < n; r++) out[order[r]] = Math.floor((r / n) * 256);
    return out;
  }

  // ------------------------------------------------------------------ data access
  const modelCache = {};
  function getModel(key) {
    if (modelCache[key]) return modelCache[key];
    const raw = DATA.models[key], n = raw.n;
    const model = {
      key: key, name: raw.name, n: n, dim: raw.dim, vocab: raw.vocab, raw: raw,
      w1: decode(raw.w1, Uint16Array), w2: decode(raw.w2, Uint16Array),
      label: decode(raw.label, Uint8Array), split: decode(raw.split, Uint8Array),
      pos: decode(raw.pos, Uint8Array), norm: decode(raw.norm, Float32Array),
      cosine: decode(raw.cosine, Float32Array), inDegree: decode(raw.inDegree, Uint16Array),
      layouts: {}, colourIndex: {},
    };
    // A fixed pseudo-random drawing order, so that no pair type is systematically on top.
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    let seed = 12345;
    for (let i = n - 1; i > 0; i--) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const j = seed % (i + 1), tmp = order[i];
      order[i] = order[j]; order[j] = tmp;
    }
    model.drawOrder = order;
    model.hubOrder = Array.from(order).sort((a, b) => model.inDegree[a] - model.inDegree[b]);
    let maxDegree = 1;
    for (let i = 0; i < n; i++) maxDegree = Math.max(maxDegree, model.inDegree[i]);
    model.maxDegree = maxDegree;
    modelCache[key] = model;
    return model;
  }

  function getLayout(model, key) {
    if (model.layouts[key]) return model.layouts[key];
    const raw = model.raw.layouts[key], n = model.n;
    const fed = raw.fed ? decode(raw.fed, Uint16Array) : null;  // position of each row in the feed order
    const rowAt = new Uint32Array(n);                           // stored row sitting at each feed position
    for (let i = 0; i < n; i++) rowAt[fed ? fed[i] : i] = i;
    model.layouts[key] = { key: key, xy: decode(raw.xy, Uint16Array), fed: fed, rowAt: rowAt, stats: raw.stats };
    return model.layouts[key];
  }

  function colourIndex(model, layout, colouring) {
    const n = model.n, spec = COLOURINGS[colouring];
    const cacheKey = colouring === "fed" ? "fed:" + layout.key : colouring;
    if (model.colourIndex[cacheKey]) return model.colourIndex[cacheKey];
    let index;
    if (colouring === "pair") index = model.label;
    else if (spec.kind === "codes") index = model[colouring];
    else if (colouring === "stored" || colouring === "fed") {
      index = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const position = colouring === "fed" && layout.fed ? layout.fed[i] : i;
        index[i] = Math.floor((position / n) * 256);
      }
    } else if (colouring === "inDegree") {
      index = new Uint8Array(n);
      const top = Math.log1p(model.maxDegree);
      for (let i = 0; i < n; i++) index[i] = Math.min(255, Math.floor((Math.log1p(model.inDegree[i]) / top) * 256));
    } else index = quantileIndex(model[colouring]);
    model.colourIndex[cacheKey] = index;
    return index;
  }

  function palette(colouring) {
    const spec = COLOURINGS[colouring];
    if (spec.kind === "pair") return DATA.pairTypes.map((t) => PAIR_STYLE[t].colour);
    if (spec.kind === "codes") return spec.colours;
    return RAMPS[spec.ramp];
  }

  // ------------------------------------------------------------------ state and controls
  const modelKeys = Object.keys(DATA.models);
  const state = {
    model: modelKeys.indexOf("text-embedding-3-large") >= 0 ? "text-embedding-3-large" : modelKeys[0],
    left: "stored", right: "permuted_a", colour: "pair",
    shown: DATA.pairTypes.map(() => true), size: 2, alpha: 0.5, query: "", hover: -1,
  };

  function select(options, value, onChange) {
    const node = el("select", { class: "form-select form-select-sm" });
    options.forEach((o) => node.appendChild(el("option", { value: o.value, text: o.text })));
    node.value = value;
    node.addEventListener("change", () => onChange(node.value));
    return node;
  }
  function field(label, control, extraClass) {
    return el("label", { class: "sx-field" + (extraClass ? " " + extraClass : "") },
      [el("span", { class: "sx-label", text: label }), control]);
  }
  function availableLayouts() {
    const present = DATA.models[state.model].layouts;
    return LAYOUT_ORDER.filter((k) => present[k]).map((k) => ({ value: k, text: LAYOUTS[k].title }));
  }

  const presetBar = el("div", { class: "sx-presets" });
  const presetNote = el("p", { class: "sx-note" });
  const controls = el("div", { class: "sx-controls" });
  const legend = el("div", { class: "sx-legend" });
  const panelsBox = el("div", { class: "sx-panels" });
  const tooltip = el("div", { class: "sx-tooltip" });
  const searchInfo = el("span", { class: "sx-searchinfo" });
  root.appendChild(el("div", { class: "sx-presetrow" }, [el("span", { class: "sx-label", text: "Guided tour" }), presetBar]));
  root.appendChild(presetNote);
  root.appendChild(controls);
  root.appendChild(legend);
  root.appendChild(panelsBox);
  root.appendChild(tooltip);

  let modelSelect, leftSelect, rightSelect, colourSelect;
  function buildControls() {
    controls.innerHTML = "";
    modelSelect = select(modelKeys.map((k) => ({ value: k, text: DATA.models[k].name })), state.model, (v) => {
      state.model = v; state.hover = -1;
      const present = DATA.models[v].layouts;
      if (!present[state.left]) state.left = "stored";
      if (!present[state.right]) state.right = "permuted_a";
      buildControls(); render();
    });
    leftSelect = select(availableLayouts(), state.left, (v) => { state.left = v; render(); });
    rightSelect = select(availableLayouts(), state.right, (v) => { state.right = v; render(); });
    colourSelect = select(Object.keys(COLOURINGS).map((k) => ({ value: k, text: COLOURINGS[k].title })), state.colour,
      (v) => { state.colour = v; render(); });
    controls.appendChild(field("Model", modelSelect));
    controls.appendChild(field("Left panel", leftSelect));
    controls.appendChild(field("Right panel", rightSelect));
    controls.appendChild(field("Colour by", colourSelect));

    const size = el("input", { type: "range", min: "0.6", max: "4", step: "0.2", value: String(state.size), class: "form-range" });
    size.addEventListener("input", () => { state.size = parseFloat(size.value); render(); });
    const alpha = el("input", { type: "range", min: "0.05", max: "1", step: "0.05", value: String(state.alpha), class: "form-range" });
    alpha.addEventListener("input", () => { state.alpha = parseFloat(alpha.value); render(); });
    controls.appendChild(field("Point size", size));
    controls.appendChild(field("Opacity", alpha));

    const search = el("input", { type: "search", class: "form-control form-control-sm", placeholder: "a word, e.g. north", value: state.query });
    search.addEventListener("input", () => { state.query = search.value.trim().toLowerCase(); drawOverlays(); });
    controls.appendChild(field("Find pairs containing", el("span", { class: "sx-search" }, [search, searchInfo])));

    const boxes = el("span", { class: "sx-checks" });
    DATA.pairTypes.forEach((type, k) => {
      const box = el("input", { type: "checkbox", class: "form-check-input" });
      box.checked = state.shown[k];
      box.addEventListener("change", () => { state.shown[k] = box.checked; render(); });
      const swatch = el("span", { class: "sx-swatch" });
      swatch.style.background = PAIR_STYLE[type].colour;
      boxes.appendChild(el("label", { class: "sx-check" }, [box, swatch, el("span", { text: PAIR_STYLE[type].name })]));
    });
    controls.appendChild(field("Show", boxes, "sx-wide"));
  }

  PRESETS.forEach((preset) => {
    const button = el("button", { type: "button", class: "btn btn-sm btn-outline-secondary", text: preset.label });
    button.addEventListener("click", () => {
      const present = DATA.models[state.model].layouts;
      state.left = present[preset.left] ? preset.left : "stored";
      state.right = present[preset.right] ? preset.right : "permuted_a";
      state.colour = preset.colour;
      presetNote.textContent = preset.note;
      Array.from(presetBar.children).forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      buildControls(); render();
    });
    presetBar.appendChild(button);
  });

  // ------------------------------------------------------------------ panels
  function makePanel(side) {
    const panel = { side: side };
    panel.title = el("div", { class: "sx-title" });
    panel.blurb = el("div", { class: "sx-blurb" });
    panel.canvas = el("canvas", { class: "sx-canvas" });
    panel.overlay = el("canvas", { class: "sx-overlay" });
    panel.stage = el("div", { class: "sx-stage" }, [panel.canvas, panel.overlay]);
    panel.stats = el("div", { class: "sx-stats" });
    panel.stripLabel = el("div", { class: "sx-caption", text: "Order given to UMAP, first row on the left, coloured by pair type" });
    panel.strip = el("canvas", { class: "sx-strip" });
    panel.profileLabel = el("div", { class: "sx-caption", text: "Horizontal plot coordinate (up) against position in that order (right)" });
    panel.profile = el("canvas", { class: "sx-profile" });
    panel.node = el("div", { class: "sx-panel" }, [panel.title, panel.blurb, panel.stage, panel.stats,
      panel.stripLabel, panel.strip, panel.profileLabel, panel.profile]);
    panel.overlay.addEventListener("mousemove", (event) => onHover(panel, event));
    panel.overlay.addEventListener("mouseleave", () => { state.hover = -1; tooltip.style.display = "none"; drawOverlays(); });
    return panel;
  }
  const panels = [makePanel("left"), makePanel("right")];
  panels.forEach((p) => panelsBox.appendChild(p.node));

  function fit(canvas, width, height) {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return context;
  }

  const PAD = 10;
  function project(panel, i) {
    const scale = (panel.side_px - 2 * PAD) / 65535, xy = panel.layout.xy;
    return [PAD + xy[2 * i] * scale, PAD + (65535 - xy[2 * i + 1]) * scale];
  }

  function drawPanel(panel, layoutKey) {
    const model = getModel(state.model), layout = getLayout(model, layoutKey);
    panel.layout = layout;
    panel.title.textContent = LAYOUTS[layoutKey].title;
    panel.blurb.textContent = LAYOUTS[layoutKey].blurb;
    const index = colourIndex(model, layout, state.colour), colours = palette(state.colour);
    const side = panel.side_px, n = model.n, label = model.label, shown = state.shown;
    const order = state.colour === "inDegree" ? model.hubOrder : model.drawOrder;
    const size = state.size, half = size / 2, scale = (side - 2 * PAD) / 65535, xy = layout.xy;

    let context = fit(panel.canvas, side, side);
    context.fillStyle = "#fafafa";
    context.fillRect(0, 0, side, side);
    const byHub = state.colour === "inDegree";  // popular neighbours are drawn larger and more opaque
    context.globalAlpha = state.alpha;
    for (let k = 0; k < n; k++) {
      const i = order[k];
      if (!shown[label[i]]) continue;
      let s = size, h = half;
      if (byHub) {
        const t = index[i] / 255;
        s = size * (1 + 3 * t * t * t);
        h = s / 2;
        context.globalAlpha = state.alpha + (1 - state.alpha) * t * t;
      }
      context.fillStyle = colours[index[i]];
      context.fillRect(PAD + xy[2 * i] * scale - h, PAD + (65535 - xy[2 * i + 1]) * scale - h, s, s);
    }
    context.globalAlpha = 1;
    fit(panel.overlay, side, side);

    // Strip: which pair type sits at each position of the feed order.
    const stripHeight = 14, pairColours = palette("pair");
    context = fit(panel.strip, side, stripHeight);
    for (let px = 0; px < side; px++) {
      context.fillStyle = pairColours[label[layout.rowAt[Math.min(n - 1, Math.floor((px / side) * n))]]];
      context.fillRect(px, 0, 1, stripHeight);
    }

    // Profile: horizontal plot coordinate against position in the feed order.
    const profileHeight = 120;
    context = fit(panel.profile, side, profileHeight);
    context.fillStyle = "#fafafa";
    context.fillRect(0, 0, side, profileHeight);
    context.globalAlpha = Math.min(1, state.alpha + 0.1);
    for (let k = 0; k < n; k++) {
      const i = order[k];
      if (!shown[label[i]]) continue;
      const position = layout.fed ? layout.fed[i] : i;
      context.fillStyle = colours[index[i]];
      context.fillRect((position / n) * side, 4 + (1 - xy[2 * i] / 65535) * (profileHeight - 8), 1.2, 1.2);
    }
    context.globalAlpha = 1;

    const s = layout.stats;
    panel.stats.innerHTML =
      "<span title='Median distance in the feed order between neighbours in the plot, relative to an unrelated ordering. 1 = the plot ignores the order; near 0 = plot neighbours were fed one after the other.'>" +
      "follows feed order: <b>" + s.localityFed.toFixed(3) + "</b> <i>(1 = not at all)</i></span>" +
      "<span title='Share of the 10 nearest plot neighbours that have the same pair type.'>" +
      "same-type neighbours: <b>" + s.agreement.toFixed(2) + "</b> <i>(chance 0.25)</i></span>";
  }

  function matches(model) {
    const query = state.query;
    if (!query) return null;
    const hit = model.vocab.indexOf(query);
    const wanted = hit >= 0 ? new Set([hit]) : new Set(model.vocab.map((w, i) => (w.toLowerCase() === query ? i : -1)).filter((i) => i >= 0));
    const found = [];
    if (wanted.size) for (let i = 0; i < model.n; i++) if (wanted.has(model.w1[i]) || wanted.has(model.w2[i])) found.push(i);
    return found;
  }

  function drawOverlays() {
    const model = getModel(state.model), found = matches(model);
    searchInfo.textContent = found === null ? "" : found.length + (found.length === 1 ? " pair" : " pairs");
    panels.forEach((panel) => {
      if (!panel.layout) return;
      const context = fit(panel.overlay, panel.side_px, panel.side_px);
      const colours = palette("pair");
      if (found) {
        found.forEach((i) => {
          if (!state.shown[model.label[i]]) return;
          const p = project(panel, i);
          context.beginPath();
          context.arc(p[0], p[1], 4.5, 0, 2 * Math.PI);
          context.fillStyle = colours[model.label[i]];
          context.fill();
          context.lineWidth = 1.5;
          context.strokeStyle = "#111";
          context.stroke();
        });
      }
      if (state.hover >= 0) {
        const p = project(panel, state.hover);
        context.beginPath();
        context.arc(p[0], p[1], 8, 0, 2 * Math.PI);
        context.lineWidth = 2.5;
        context.strokeStyle = "#000";
        context.stroke();
        context.beginPath();
        context.arc(p[0], p[1], 3, 0, 2 * Math.PI);
        context.fillStyle = "#fff";
        context.fill();
        context.lineWidth = 1;
        context.stroke();
      }
    });
  }

  function onHover(panel, event) {
    const model = getModel(state.model), box = panel.overlay.getBoundingClientRect();
    const x = event.clientX - box.left, y = event.clientY - box.top;
    const scale = (panel.side_px - 2 * PAD) / 65535, xy = panel.layout.xy;
    let best = -1, bestDistance = 12 * 12;
    for (let i = 0; i < model.n; i++) {
      if (!state.shown[model.label[i]]) continue;
      const dx = PAD + xy[2 * i] * scale - x, dy = PAD + (65535 - xy[2 * i + 1]) * scale - y;
      const d = dx * dx + dy * dy;
      if (d < bestDistance) { bestDistance = d; best = i; }
    }
    if (best !== state.hover) { state.hover = best; drawOverlays(); }
    if (best < 0) { tooltip.style.display = "none"; return; }
    const type = DATA.pairTypes[model.label[best]];
    const fed = panels.map((p) => (p.layout.fed ? p.layout.fed[best] : best));
    const extras = [];
    if (model.split[best]) extras.push(DATA.splits[model.split[best]] + ", " + DATA.pos[model.pos[best]]);
    tooltip.innerHTML =
      "<b>" + model.vocab[model.w1[best]] + " → " + model.vocab[model.w2[best]] + "</b><br>" +
      "<span style='color:" + PAIR_STYLE[type].colour + "'>■</span> " + PAIR_STYLE[type].name +
      (extras.length ? " (" + extras.join("; ") + ")" : "") + "<br>" +
      "row " + best + " of " + model.n + " in the paper's matrix<br>" +
      "fed to UMAP at position " + fed[0] + " (left) / " + fed[1] + " (right)<br>" +
      "vector length " + model.norm[best].toFixed(1) + " · cosine " + model.cosine[best].toFixed(2) + "<br>" +
      "a nearest neighbour of " + model.inDegree[best] + " other pairs";
    tooltip.style.display = "block";
    const rootBox = root.getBoundingClientRect();
    let left = event.clientX - rootBox.left + 16, top = event.clientY - rootBox.top + 16;
    if (left + 300 > rootBox.width) left = Math.max(0, left - 330);
    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }

  function drawLegend() {
    const spec = COLOURINGS[state.colour];
    legend.innerHTML = "";
    if (spec.kind === "ramp") {
      const bar = el("canvas", { class: "sx-ramp" });
      const context = fit(bar, 256, 12);
      RAMPS[spec.ramp].forEach((colour, i) => { context.fillStyle = colour; context.fillRect(i, 0, 1, 12); });
      legend.appendChild(el("span", { text: spec.ends[0] }));
      legend.appendChild(bar);
      legend.appendChild(el("span", { text: spec.ends[1] }));
      if (state.colour === "fed") legend.appendChild(el("span", { class: "sx-hint", text: "each panel uses its own feed order" }));
      if (state.colour === "inDegree") legend.appendChild(el("span", { class: "sx-hint", text: "popular pairs are drawn larger" }));
    } else if (spec.kind === "codes") {  // pair types are already labelled by the "Show" boxes
      const names = spec.names;
      palette(state.colour).forEach((colour, k) => {
        const swatch = el("span", { class: "sx-swatch" });
        swatch.style.background = colour;
        legend.appendChild(el("span", { class: "sx-legenditem" }, [swatch, el("span", { text: names[k] })]));
      });
    }
  }

  function render() {
    const width = root.clientWidth;
    if (!width) return;  // hidden tab: wait until it is shown
    const twoColumns = width >= 760;
    panelsBox.style.gridTemplateColumns = twoColumns ? "1fr 1fr" : "1fr";
    const side = Math.max(260, Math.min(660, Math.floor(twoColumns ? (width - 32) / 2 : width - 2)));
    panels.forEach((p) => { p.side_px = side; p.stage.style.width = side + "px"; p.stage.style.height = side + "px"; });
    drawLegend();
    drawPanel(panels[0], state.left);
    drawPanel(panels[1], state.right);
    drawOverlays();
  }

  buildControls();
  presetNote.textContent = "Pick a step of the tour, or choose model, row orderings and colouring yourself. Hover a point to see the pair and where it sits in the other panel.";
  render();
  let pending = null;
  window.addEventListener("resize", () => { clearTimeout(pending); pending = setTimeout(render, 120); });
  document.addEventListener("shown.bs.tab", render);
})();
