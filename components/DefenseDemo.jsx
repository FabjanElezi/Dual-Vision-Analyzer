"use client";
import { useState, useRef, useCallback, useEffect } from "react";

// ── HELPERS ──────────────────────────────────────────────────────────────────

function toGray(imageData, W, H) {
  const gray = new Float32Array(W * H);
  for (let i = 0; i < imageData.data.length; i += 4) {
    gray[i >> 2] =
      (0.299 * imageData.data[i] +
        0.587 * imageData.data[i + 1] +
        0.114 * imageData.data[i + 2]) / 255;
  }
  return gray;
}

function drawGray(ctx, gray, W, H) {
  const id = ctx.createImageData(W, H);
  for (let i = 0; i < gray.length; i++) {
    const v = Math.round(gray[i] * 255);
    id.data[i * 4] = v;
    id.data[i * 4 + 1] = v;
    id.data[i * 4 + 2] = v;
    id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

function drawEdges(ctx, gray, W, H) {
  const id = ctx.createImageData(W, H);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const gx = gray[i + 1] - gray[i - 1];
      const gy = gray[i + W] - gray[i - W];
      const mag = Math.min(Math.sqrt(gx * gx + gy * gy) * 4 * 255, 255);
      const p = i * 4;
      id.data[p] = id.data[p + 1] = id.data[p + 2] = mag;
      id.data[p + 3] = 255;
    }
  }
  ctx.putImageData(id, 0, 0);
}

function drawHOG(ctx, gray, W, H) {
  const cell = 8;
  ctx.fillStyle = "#080810";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#c9a96e";
  ctx.lineWidth = 0.9;

  for (let row = 0; row < H / cell; row++) {
    for (let col = 0; col < W / cell; col++) {
      let gx = 0, gy = 0, mag = 0, n = 0;
      for (let dy = 0; dy < cell; dy++) {
        for (let dx = 0; dx < cell; dx++) {
          const x = col * cell + dx;
          const y = row * cell + dy;
          if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) continue;
          const i = y * W + x;
          const lx = gray[i + 1] - gray[i - 1];
          const ly = gray[i + W] - gray[i - W];
          gx += lx; gy += ly;
          mag += Math.sqrt(lx * lx + ly * ly);
          n++;
        }
      }
      if (!n) continue;
      gx /= n; gy /= n; mag /= n;
      const angle = Math.atan2(gy, gx);
      const cx = col * cell + cell / 2;
      const cy = row * cell + cell / 2;
      const len = Math.min(mag * 22, cell * 0.46);
      if (len < 0.25) continue;
      ctx.globalAlpha = Math.min(mag * 5, 0.95);
      ctx.beginPath();
      ctx.moveTo(cx - Math.cos(angle) * len, cy - Math.sin(angle) * len);
      ctx.lineTo(cx + Math.cos(angle) * len, cy + Math.sin(angle) * len);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

// ── COMPONENT ─────────────────────────────────────────────────────────────────
const SZ = 160;

// Two palettes. Every UI color is referenced through the active theme so the
// dark/light toggle is a single source of truth — no per-element overrides.
const THEMES = {
  dark: {
    bg: "#080810", panel: "#0e0e1a", hover: "#0d0d18",
    line: "#1a1a2e", lineDash: "#1e1e32", lineStrong: "#3a3a5a",
    text: "#ddd8cc", dim: "#b6b6d2", mute: "#9a9ab8",
    accent: "#c9a96e", accentHi: "#e8c88a",
    classical: "#c4c4e4", gapAlt: "#8a8ac0",
    ok: "#4a9e70", dotModel: "#555570", dotIdle: "#333348",
    overlay: "rgba(8,8,16,0.75)", iconStroke: "#2a2a44", error: "#a05050",
  },
  light: {
    bg: "#f4f1ea", panel: "#ebe6db", hover: "#e7e1d3",
    line: "#dcd5c6", lineDash: "#d0c8b6", lineStrong: "#c2b8a0",
    text: "#2c2820", dim: "#55503f", mute: "#7a7363",
    accent: "#946f2c", accentHi: "#7a5a1a",
    classical: "#54526e", gapAlt: "#8a87ad",
    ok: "#2f7a4f", dotModel: "#b0a890", dotIdle: "#cfc7b5",
    overlay: "rgba(244,241,234,0.82)", iconStroke: "#c6bfae", error: "#a83a3a",
  },
};

export default function DefenseDemo() {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [imageLoaded, setImageLoaded] = useState(false);
  const [topLabel, setTopLabel] = useState("");
  const [modelReady, setModelReady] = useState(false);
  const [theme, setTheme] = useState("dark");

  const modelRef = useRef(null);
  const dropRef = useRef(null);
  const fileRef = useRef(null);
  const previewRef = useRef(null);
  const cOrigRef = useRef(null);
  const cGrayRef = useRef(null);
  const cHogRef = useRef(null);
  const cEdgeRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        await import("@tensorflow/tfjs");
        const { load } = await import("@tensorflow-models/mobilenet");
        // MobileNet v2 + alpha 1.0: wider network, inverted-residual blocks,
        // better top-5 accuracy on ImageNet than v1 at the same alpha.
        const m = await load({ version: 2, alpha: 1.0 });
        if (!cancelled) {
          modelRef.current = m;
          setModelReady(true);
        }
      } catch (e) {
        console.error("Model load failed:", e);
      }
    }
    loadModel();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    function onKey(e) {
      if (e.key !== "Escape") return;
      setPredictions(null);
      setLoading(false);
      setError("");
      setImageLoaded(false);
      setTopLabel("");
      if (previewRef.current) previewRef.current.src = "";
      if (fileRef.current) fileRef.current.value = "";
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Restore the saved theme after mount (initial state stays "dark" to match
  // the server-rendered HTML, so there's no hydration mismatch).
  useEffect(() => {
    try {
      const saved = localStorage.getItem("dva-theme");
      if (saved === "light" || saved === "dark") setTheme(saved);
    } catch {}
  }, []);
  useEffect(() => {
    try { localStorage.setItem("dva-theme", theme); } catch {}
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  const processImage = useCallback(async (file) => {
    if (!file?.type.startsWith("image/") || !modelRef.current) return;
    setLoading(true);
    setError("");
    setPredictions(null);
    setImageLoaded(false);
    setTopLabel("");

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      const img = new Image();
      img.onload = async () => {
        previewRef.current.src = dataUrl;
        setImageLoaded(true);

        const c1 = cOrigRef.current.getContext("2d");
        c1.drawImage(img, 0, 0, SZ, SZ);

        const c2 = cGrayRef.current.getContext("2d");
        c2.drawImage(img, 0, 0, SZ, SZ);
        const id = c2.getImageData(0, 0, SZ, SZ);
        const gray = toGray(id, SZ, SZ);
        drawGray(c2, gray, SZ, SZ);
        drawHOG(cHogRef.current.getContext("2d"), gray, SZ, SZ);
        drawEdges(cEdgeRef.current.getContext("2d"), gray, SZ, SZ);

        try {
          const raw = await modelRef.current.classify(img, 5);
          const preds = raw.map((p) => ({
            label: p.className.split(",")[0].trim(),
            confidence: p.probability,
          }));
          setPredictions(preds);
          if (preds[0]) setTopLabel(preds[0].label);
        } catch (err) {
          setError("Classification failed.");
          console.error(err);
        }
        setLoading(false);
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    dropRef.current?.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f) processImage(f);
  }, [processImage]);

  const onDragOver = (e) => {
    e.preventDefault();
    dropRef.current?.classList.add("over");
  };
  const onDragLeave = () => dropRef.current?.classList.remove("over");

  const c = THEMES[theme];
  const s = {
    root: {
      display: "flex", flexDirection: "column", height: "100vh",
      background: c.bg, color: c.text,
      fontFamily: "Georgia, serif", overflow: "hidden",
      transition: "background 0.3s ease, color 0.3s ease",
      "--dva-accent": c.accent, "--dva-hover": c.hover, "--dva-line": c.line
    },
    header: {
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "12px 28px", borderBottom: `1px solid ${c.line}`,
      background: c.panel, flexShrink: 0,
      transition: "background 0.3s ease, border-color 0.3s ease"
    },
    h1: { fontSize: 14, fontWeight: "normal", color: c.accent },
    sub: { fontSize: 11, color: c.dim, marginTop: 2, letterSpacing: 0.4 },
    badge: {
      fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase",
      color: c.dim, border: `1px solid ${c.lineStrong}`, padding: "3px 8px", borderRadius: 3
    },
    themeBtn: {
      display: "flex", alignItems: "center", justifyContent: "center",
      width: 30, height: 30, padding: 0, borderRadius: 6, cursor: "pointer",
      background: "transparent", border: `1px solid ${c.lineStrong}`, color: c.accent,
      transition: "color 0.3s ease, border-color 0.2s ease"
    },
    statusPill: { display: "flex", alignItems: "center", gap: 7, fontSize: 11, color: c.dim },
    dot: (state) => ({
      width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
      background: state === "ready" ? c.ok : state === "loading" ? c.accent : state === "model-loading" ? c.dotModel : c.dotIdle,
      animation: state !== "idle" ? "pulse 1.4s ease-in-out infinite" : "none"
    }),
    main: {
      display: "grid", gridTemplateColumns: "1fr 1fr",
      flex: 1, overflow: "hidden"
    },
    panel: {
      padding: "24px 28px", borderRight: `1px solid ${c.line}`,
      display: "flex", flexDirection: "column", overflow: "hidden"
    },
    panelR: { padding: "24px 28px", display: "flex", flexDirection: "column", overflow: "hidden" },
    label: {
      fontSize: 10, letterSpacing: 2, textTransform: "uppercase",
      color: c.dim, marginBottom: 16, flexShrink: 0
    },
    dropZone: {
      border: `1px dashed ${c.lineDash}`, borderRadius: 8,
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", flex: 1, cursor: modelReady ? "pointer" : "default",
      transition: "border-color 0.2s, background 0.2s", position: "relative",
      overflow: "hidden", minHeight: 200,
      opacity: modelReady ? 1 : 0.5
    },
    hint: { textAlign: "center" },
    hintP: { fontSize: 13, color: c.dim },
    hintS: { fontSize: 11, color: c.mute, marginTop: 4 },
    preview: {
      maxWidth: "100%", maxHeight: "100%", objectFit: "contain",
      borderRadius: 4, display: imageLoaded ? "block" : "none"
    },
    predList: { display: "flex", flexDirection: "column", gap: 16, flex: 1, overflow: "auto" },
    emptyState: {
      flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
      color: c.mute, fontSize: 13, fontStyle: "italic", textAlign: "center"
    },
    predRow: { display: "flex", justifyContent: "space-between", marginBottom: 6 },
    predName: (isTop) => ({ fontSize: isTop ? 15 : 13, color: isTop ? c.accentHi : c.text }),
    predPct: { fontSize: 12, fontFamily: "monospace", color: c.accent },
    barTrack: { height: 3, background: c.line, borderRadius: 2, overflow: "hidden" },
    barFill: (w, isTop) => ({
      height: "100%", borderRadius: 2,
      background: isTop ? c.accentHi : c.accent,
      width: w + "%", transition: "width 0.9s cubic-bezier(0.4,0,0.2,1)"
    }),
    bottom: {
      borderTop: `1px solid ${c.line}`,
      display: "grid", gridTemplateColumns: "3fr 2fr",
      flexShrink: 0
    },
    vizPanel: { padding: "16px 28px", borderRight: `1px solid ${c.line}` },
    canvasRow: { display: "flex", gap: 14, marginTop: 4 },
    canvasBlock: { flex: 1, textAlign: "center" },
    canvasLabel: {
      display: "block", fontSize: 10, letterSpacing: 1.5,
      textTransform: "uppercase", color: c.dim, marginBottom: 6
    },
    canvas: {
      border: `1px solid ${c.line}`, borderRadius: 3,
      width: "100%", maxWidth: SZ, imageRendering: "pixelated"
    },
    explainPanel: {
      padding: "16px 24px", display: "flex",
      flexDirection: "column", justifyContent: "center"
    },
    explainText: { fontSize: 13, color: c.dim, lineHeight: 1.75 },
    gold: { color: c.accent },
    foot: {
      borderTop: `1px solid ${c.line}`, background: c.panel,
      padding: "8px 28px", display: "flex", gap: 24,
      alignItems: "center", flexShrink: 0,
      transition: "background 0.3s ease, border-color 0.3s ease"
    },
    cmpItem: { display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: c.dim },
    cmpName: { fontSize: 10, letterSpacing: 1, textTransform: "uppercase", color: c.dim },
    cmpOn: { fontSize: 11, color: c.mute },
    sep: { color: c.lineStrong },
    footNote: { fontSize: 11, color: c.mute, fontStyle: "italic" },
    cmpAcc: (isNN) => ({
      fontFamily: "monospace", fontSize: 18,
      color: isNN ? c.accentHi : c.classical
    }),
    gapWrap: { marginTop: 18 },
    gapCaption: { fontSize: 11, color: c.dim, marginBottom: 12, fontStyle: "italic" },
    gapRow: { display: "flex", alignItems: "center", gap: 10, marginBottom: 9 },
    gapName: { fontSize: 11, color: c.dim, width: 78, flexShrink: 0, textTransform: "uppercase", letterSpacing: 0.5 },
    gapTrack: { flex: 1, height: 8, background: c.line, borderRadius: 3, overflow: "hidden" },
    gapFill: (w, top) => ({ height: "100%", width: w + "%", borderRadius: 3, background: top ? c.accentHi : c.gapAlt, transition: "width 1s cubic-bezier(0.4,0,0.2,1)" }),
    gapPct: (top) => ({ fontSize: 13, fontFamily: "monospace", color: top ? c.accentHi : c.classical, width: 52, textAlign: "right", flexShrink: 0 }),
    gapBadge: { marginTop: 4, fontSize: 11, color: c.accent, letterSpacing: 0.5 },
    loadingOverlay: {
      position: "absolute", inset: 0, background: c.overlay,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 11, color: c.accent, letterSpacing: 2, textTransform: "uppercase"
    }
  };

  const statusState = !modelReady ? "model-loading" : loading ? "loading" : predictions ? "ready" : "idle";
  const statusText = !modelReady ? "Loading model…" : loading ? "Classifying…" : predictions ? "Ready" : "Waiting for image";

  return (
    <div style={s.root} className="dva-root">
      <style>{`
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
        .drop-zone-inner:hover,.drop-zone-inner.over{border-color:var(--dva-accent)!important;background:var(--dva-hover)!important;}
        .dva-theme-btn:hover{border-color:var(--dva-accent)!important;}
        @media(max-width:768px){
          .dva-root{height:auto!important;min-height:100dvh;overflow-y:auto!important;}
          .dva-header{flex-direction:column!important;align-items:flex-start!important;gap:10px;padding:12px 16px!important;}
          .dva-main{grid-template-columns:1fr!important;overflow:visible!important;flex:none!important;}
          .dva-panel{border-right:none!important;border-bottom:1px solid var(--dva-line);padding:20px 16px!important;}
          .dva-panelR{padding:20px 16px!important;}
          .dva-bottom{grid-template-columns:1fr!important;}
          .dva-vizpanel{border-right:none!important;border-bottom:1px solid var(--dva-line);}
          .dva-foot{flex-wrap:wrap!important;gap:10px!important;padding:12px 16px!important;}
        }
      `}</style>

      {/* HEADER */}
      <header style={s.header} className="dva-header">
        <div>
          <div style={s.h1}>The Role of Deep Learning in Digital Image Analysis</div>
          <div style={s.sub}>Bachelor Thesis Defense · Fabjan Elezi · Universiteti Metropolitan Tirana · June 2026</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button
            type="button"
            onClick={toggleTheme}
            style={s.themeBtn}
            className="dva-theme-btn"
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            title={theme === "dark" ? "Light mode" : "Dark mode"}
          >
            {theme === "dark" ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
              </svg>
            )}
          </button>
          <span style={s.badge}>Dual Vision Analyzer</span>
          <div style={s.statusPill}>
            <div style={s.dot(statusState)} />
            <span>{statusText}</span>
          </div>
        </div>
      </header>

      {/* MAIN */}
      <div style={s.main} className="dva-main">
        {/* LEFT: Upload */}
        <div style={s.panel} className="dva-panel">
          <div style={s.label}>Input Image</div>
          <div
            ref={dropRef}
            className="drop-zone-inner"
            style={s.dropZone}
            onClick={() => modelReady && fileRef.current?.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
          >
            {!imageLoaded && (
              <div style={s.hint}>
                <svg width="42" height="42" viewBox="0 0 24 24" fill="none"
                  stroke={c.iconStroke} strokeWidth="1" style={{ display: "block", margin: "0 auto 12px" }}>
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
                <p style={s.hintP}>{modelReady ? "Tap or drop an image" : "Loading model…"}</p>
                <span style={s.hintS}>{modelReady ? "photos, camera, or any image" : "please wait"}</span>
              </div>
            )}
            <img ref={previewRef} alt="preview" style={s.preview} />
            {loading && <div style={s.loadingOverlay}>Classifying…</div>}
          </div>
          <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
            onChange={e => e.target.files[0] && processImage(e.target.files[0])} />
        </div>

        {/* RIGHT: Predictions */}
        <div style={s.panelR} className="dva-panelR">
          <div style={s.label}>Classification Results — Top 5 Predictions</div>
          {error && <div style={{ color: c.error, fontSize: 12, marginBottom: 12 }}>{error}</div>}
          {!predictions && (
            <div style={s.emptyState}>
              {loading ? "Analyzing image…" : modelReady ? "Upload an image to see predictions" : "Loading MobileNet model…"}
            </div>
          )}
          {predictions && (
            <div style={s.predList}>
              {predictions.map((p, i) => {
                const pct = Math.round(p.confidence * 100);
                const isTop = i === 0;
                return (
                  <div key={i}>
                    <div style={s.predRow}>
                      <span style={s.predName(isTop)}>{i + 1}. {p.label}</span>
                      <span style={s.predPct}>{pct}%</span>
                    </div>
                    <div style={s.barTrack}>
                      <div style={s.barFill(pct, isTop)} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* BOTTOM */}
      <div style={s.bottom} className="dva-bottom">
        <div style={s.vizPanel} className="dva-vizpanel">
          <div style={{ ...s.label, marginBottom: 10 }}>Image Analysis Visualizations</div>
          <div style={s.canvasRow}>
            {[
              { ref: cOrigRef, label: "Original" },
              { ref: cGrayRef, label: "Grayscale" },
              { ref: cHogRef, label: "HOG Features" },
              { ref: cEdgeRef, label: "Edge Map" },
            ].map(({ ref, label }) => (
              <div key={label} style={s.canvasBlock}>
                <label style={s.canvasLabel}>{label}</label>
                <canvas ref={ref} width={SZ} height={SZ} style={s.canvas} />
              </div>
            ))}
          </div>
        </div>

        <div style={s.explainPanel}>
          {topLabel && predictions ? (
            <p style={s.explainText}>
              <span style={{ ...s.gold, fontStyle: "normal" }}>
                Identified: {topLabel} — {Math.round(predictions[0].confidence * 100)}% confidence
              </span>
              <br /><br />
              The <span style={s.gold}>edge map</span> traces the outlines that define this {topLabel}'s shape.
              The <span style={s.gold}>HOG canvas</span> bins those gradients into 8×8 cell histograms —
              exactly the hand-engineered features a classical SVM would use.
              <span style={s.gold}> MobileNet</span> — the CNN running live here — instead learned its own
              filters from millions of labeled images: the same advantage that let a CNN beat HOG+SVM by
              <span style={s.gold}> ~19 points</span> in my CIFAR-10 tests.
            </p>
          ) : (
            <div>
              <p style={s.explainText}>
                <span style={{ ...s.gold, fontStyle: "normal" }}>How this works</span><br /><br />
                Upload any image — the live classifier (<span style={s.gold}>MobileNet</span>, a pretrained CNN)
                returns its top-5 ImageNet predictions in real time, while the Grayscale, HOG and Edge panels
                show the hand-engineered features a classical pipeline relies on.
              </p>
              <div style={s.gapWrap}>
                <div style={s.gapCaption}>My measured CIFAR-10 accuracy (Chapter 7) — same data &amp; evaluation, both pipelines</div>
                {[
                  { name: "HOG + SVM", val: 50.15, top: false },
                  { name: "CNN", val: 69.25, top: true },
                ].map(({ name, val, top }) => (
                  <div key={name} style={s.gapRow}>
                    <span style={s.gapName}>{name}</span>
                    <div style={s.gapTrack}><div style={s.gapFill(val, top)} /></div>
                    <span style={s.gapPct(top)}>{val}%</span>
                  </div>
                ))}
                <div style={s.gapBadge}>▲ ~19-point accuracy gap</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* FOOTER */}
      <div style={s.foot} className="dva-foot">
        <div style={s.cmpItem}>
          <span style={s.cmpName}>Classical HOG + SVM</span>
          <span style={s.cmpAcc(false)}>~50.15%</span>
          <span style={s.cmpOn}>on CIFAR-10</span>
        </div>
        <span style={s.sep}>|</span>
        <div style={s.cmpItem}>
          <span style={s.cmpName}>Deep Learning CNN</span>
          <span style={s.cmpAcc(true)}>~69.25%</span>
          <span style={s.cmpOn}>on CIFAR-10</span>
        </div>
        <span style={s.sep}>|</span>
        <span style={s.footNote}>
          My own measured results — Chapter 7: classical vs. deep learning image classification
        </span>
      </div>
    </div>
  );
}
