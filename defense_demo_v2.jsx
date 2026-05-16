import { useState, useRef, useCallback } from "react";

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

async function classifyWithClaude(base64, mimeType) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 300,
      messages: [{
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: mimeType, data: base64 }
          },
          {
            type: "text",
            text: `You are a computer vision model. Classify this image and return ONLY a JSON array of exactly 5 predictions, ordered by confidence. Format strictly:
[{"label":"specific object or scene name","confidence":0.92},{"label":"...","confidence":0.04},...]
Confidence values must be between 0 and 1. Use specific ImageNet-style labels. No other text outside the JSON.`
          }
        ]
      }]
    })
  });
  const data = await res.json();
  const text = data.content?.[0]?.text || "[]";
  const clean = text.replace(/```json|```/g, "").trim();
  return JSON.parse(clean);
}

// ── COMPONENT ─────────────────────────────────────────────────────────────────
const SZ = 110; // canvas size

export default function DefenseDemo() {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [imageLoaded, setImageLoaded] = useState(false);
  const [topLabel, setTopLabel] = useState("");

  const dropRef = useRef(null);
  const fileRef = useRef(null);
  const previewRef = useRef(null);
  const cOrigRef = useRef(null);
  const cGrayRef = useRef(null);
  const cHogRef = useRef(null);
  const cEdgeRef = useRef(null);

  const processImage = useCallback(async (file) => {
    if (!file?.type.startsWith("image/")) return;
    setLoading(true);
    setError("");
    setPredictions(null);
    setImageLoaded(false);

    const reader = new FileReader();
    reader.onload = async (e) => {
      const dataUrl = e.target.result;
      const base64 = dataUrl.split(",")[1];
      const mimeType = file.type;

      // Show preview
      const img = new Image();
      img.onload = () => {
        previewRef.current.src = dataUrl;
        setImageLoaded(true);

        // Draw visualizations
        const S = SZ;

        // Original
        const c1 = cOrigRef.current.getContext("2d");
        c1.drawImage(img, 0, 0, S, S);

        // Grayscale
        const c2 = cGrayRef.current.getContext("2d");
        c2.drawImage(img, 0, 0, S, S);
        const id = c2.getImageData(0, 0, S, S);
        const gray = toGray(id, S, S);
        drawGray(c2, gray, S, S);

        // HOG
        drawHOG(cHogRef.current.getContext("2d"), gray, S, S);

        // Edge
        drawEdges(cEdgeRef.current.getContext("2d"), gray, S, S);
      };
      img.src = dataUrl;

      // Classify
      try {
        const preds = await classifyWithClaude(base64, mimeType);
        setPredictions(preds);
        if (preds[0]) setTopLabel(preds[0].label);
      } catch (err) {
        setError("Classification failed — check connection.");
        console.error(err);
      }
      setLoading(false);
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

  const s = {
    root: {
      display: "flex", flexDirection: "column", height: "100vh",
      background: "#080810", color: "#ddd8cc",
      fontFamily: "Georgia, serif", overflow: "hidden"
    },
    header: {
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "12px 28px", borderBottom: "1px solid #1a1a2e",
      background: "#0e0e1a", flexShrink: 0
    },
    h1: { fontSize: 14, fontWeight: "normal", color: "#c9a96e" },
    sub: { fontSize: 11, color: "#333348", marginTop: 2, letterSpacing: 0.4 },
    badge: {
      fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase",
      color: "#333348", border: "1px solid #1a1a2e", padding: "3px 8px", borderRadius: 3
    },
    statusPill: { display: "flex", alignItems: "center", gap: 7, fontSize: 11, color: "#333348" },
    dot: (state) => ({
      width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
      background: state === "ready" ? "#4a9e70" : state === "loading" ? "#c9a96e" : "#333348",
      animation: state !== "idle" ? "pulse 1.4s ease-in-out infinite" : "none"
    }),
    main: {
      display: "grid", gridTemplateColumns: "1fr 1fr",
      flex: 1, overflow: "hidden"
    },
    panel: {
      padding: "24px 28px", borderRight: "1px solid #1a1a2e",
      display: "flex", flexDirection: "column", overflow: "hidden"
    },
    panelR: { padding: "24px 28px", display: "flex", flexDirection: "column", overflow: "hidden" },
    label: {
      fontSize: 10, letterSpacing: 2, textTransform: "uppercase",
      color: "#333348", marginBottom: 16, flexShrink: 0
    },
    dropZone: {
      border: "1px dashed #1e1e32", borderRadius: 8,
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", flex: 1, cursor: "pointer",
      transition: "border-color 0.2s, background 0.2s", position: "relative",
      overflow: "hidden", minHeight: 200
    },
    hint: { textAlign: "center", color: "#1e1e32" },
    hintP: { fontSize: 13, color: "#2a2a44" },
    hintS: { fontSize: 11, color: "#1e1e32", marginTop: 4 },
    preview: {
      maxWidth: "100%", maxHeight: "100%", objectFit: "contain",
      borderRadius: 4, display: imageLoaded ? "block" : "none"
    },
    predList: { display: "flex", flexDirection: "column", gap: 16, flex: 1, overflow: "auto" },
    emptyState: {
      flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
      color: "#1e1e32", fontSize: 13, fontStyle: "italic", textAlign: "center"
    },
    predItem: {},
    predRow: { display: "flex", justifyContent: "space-between", marginBottom: 6 },
    predName: (isTop) => ({ fontSize: isTop ? 15 : 13, color: isTop ? "#e8c88a" : "#ddd8cc" }),
    predPct: { fontSize: 12, fontFamily: "monospace", color: "#c9a96e" },
    barTrack: { height: 3, background: "#1a1a2e", borderRadius: 2, overflow: "hidden" },
    barFill: (w, isTop) => ({
      height: "100%", borderRadius: 2,
      background: isTop ? "#e8c88a" : "#c9a96e",
      width: w + "%", transition: "width 0.9s cubic-bezier(0.4,0,0.2,1)"
    }),
    bottom: {
      borderTop: "1px solid #1a1a2e",
      display: "grid", gridTemplateColumns: "3fr 2fr",
      flexShrink: 0
    },
    vizPanel: { padding: "16px 28px", borderRight: "1px solid #1a1a2e" },
    canvasRow: { display: "flex", gap: 14, marginTop: 4 },
    canvasBlock: { flex: 1, textAlign: "center" },
    canvasLabel: {
      display: "block", fontSize: 9, letterSpacing: 1.5,
      textTransform: "uppercase", color: "#2a2a44", marginBottom: 6
    },
    canvas: {
      border: "1px solid #1a1a2e", borderRadius: 3,
      width: "100%", maxWidth: SZ, imageRendering: "pixelated"
    },
    explainPanel: {
      padding: "16px 24px", display: "flex",
      flexDirection: "column", justifyContent: "center"
    },
    explainText: { fontSize: 11, color: "#2a2a44", lineHeight: 1.75 },
    gold: { color: "#c9a96e" },
    foot: {
      borderTop: "1px solid #1a1a2e", background: "#0e0e1a",
      padding: "8px 28px", display: "flex", gap: 24,
      alignItems: "center", flexShrink: 0
    },
    cmpItem: { display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "#2a2a44" },
    cmpAcc: (isNN) => ({
      fontFamily: "monospace", fontSize: 14,
      color: isNN ? "#e8c88a" : "#2a2a44"
    }),
    loadingOverlay: {
      position: "absolute", inset: 0, background: "rgba(8,8,16,0.75)",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 11, color: "#c9a96e", letterSpacing: 2, textTransform: "uppercase"
    }
  };

  const statusState = loading ? "loading" : predictions ? "ready" : "idle";

  return (
    <div style={s.root}>
      <style>{`
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
        .drop-zone-inner:hover,.drop-zone-inner.over{border-color:#c9a96e!important;background:#0d0d18!important;}
      `}</style>

      {/* HEADER */}
      <header style={s.header}>
        <div>
          <div style={s.h1}>The Role of Deep Learning in Digital Image Analysis</div>
          <div style={s.sub}>Bachelor Thesis Defense Demo · Fabjan Elezi · Universiteti Metropolitan Tirana · June 2026</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={s.badge}>Claude Vision · Real-Time Classification</span>
          <div style={s.statusPill}>
            <div style={s.dot(statusState)} />
            <span>{loading ? "Classifying…" : predictions ? "Ready" : "Waiting for image"}</span>
          </div>
        </div>
      </header>

      {/* MAIN */}
      <div style={s.main}>
        {/* LEFT: Upload */}
        <div style={s.panel}>
          <div style={s.label}>Input Image</div>
          <div
            ref={dropRef}
            className="drop-zone-inner"
            style={s.dropZone}
            onClick={() => fileRef.current?.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
          >
            {!imageLoaded && (
              <div style={s.hint}>
                <svg width="42" height="42" viewBox="0 0 24 24" fill="none"
                  stroke="#2a2a44" strokeWidth="1" style={{ display: "block", margin: "0 auto 12px" }}>
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
                <p style={s.hintP}>Drop an image here</p>
                <span style={s.hintS}>or click to browse</span>
              </div>
            )}
            <img ref={previewRef} alt="preview" style={s.preview} />
            {loading && <div style={s.loadingOverlay}>Classifying…</div>}
          </div>
          <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
            onChange={e => e.target.files[0] && processImage(e.target.files[0])} />
          <div style={{ marginTop: 10, fontSize: 11, color: "#1e1e32", textAlign: "center", fontStyle: "italic" }}>
            Try: animals, vehicles, food, everyday objects
          </div>
        </div>

        {/* RIGHT: Predictions */}
        <div style={s.panelR}>
          <div style={s.label}>Classification Results — Top 5 Predictions</div>
          {error && <div style={{ color: "#a05050", fontSize: 12, marginBottom: 12 }}>{error}</div>}
          {!predictions && !loading && (
            <div style={s.emptyState}>Upload an image to see predictions</div>
          )}
          {loading && !predictions && (
            <div style={s.emptyState}>Analyzing image…</div>
          )}
          {predictions && (
            <div style={s.predList}>
              {predictions.map((p, i) => {
                const pct = Math.round(p.confidence * 100);
                const isTop = i === 0;
                return (
                  <div key={i} style={s.predItem}>
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
      <div style={s.bottom}>
        <div style={s.vizPanel}>
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
          {topLabel ? (
            <p style={s.explainText}>
              <span style={{ ...s.gold, fontStyle: "normal" }}>Identified: {topLabel}</span>
              <br /><br />
              The <span style={s.gold}>HOG canvas</span> shows what a classical SVM classifier uses — 
              edge direction histograms in 8×8 pixel cells. The CNN learned its own features from 
              labeled examples, which is why it outperforms HOG+SVM by ~20 percentage points on CIFAR-10.
            </p>
          ) : (
            <p style={s.explainText}>
              <span style={{ ...s.gold, fontStyle: "normal" }}>How this works:</span><br /><br />
              Upload an image to see real-time classification. The HOG visualization shows 
              what classical feature extractors see. The predictions use a vision AI system — 
              the same approach underlying production image classifiers.
            </p>
          )}
        </div>
      </div>

      {/* FOOTER */}
      <div style={s.foot}>
        <div style={s.cmpItem}>
          <span style={{ fontSize: 9, letterSpacing: 1, textTransform: "uppercase" }}>Classical HOG + SVM</span>
          <span style={s.cmpAcc(false)}>~54%</span>
          <span style={{ fontSize: 10, color: "#1e1e32" }}>on CIFAR-10</span>
        </div>
        <span style={{ color: "#1a1a2e" }}>|</span>
        <div style={s.cmpItem}>
          <span style={{ fontSize: 9, letterSpacing: 1, textTransform: "uppercase" }}>Deep Learning CNN</span>
          <span style={s.cmpAcc(true)}>~76%</span>
          <span style={{ fontSize: 10, color: "#1e1e32" }}>on CIFAR-10</span>
        </div>
        <span style={{ color: "#1a1a2e" }}>|</span>
        <span style={{ fontSize: 10, color: "#1e1e32", fontStyle: "italic" }}>
          This demo illustrates Chapter 7 of the thesis — classical vs. deep learning image classification
        </span>
      </div>
    </div>
  );
}
