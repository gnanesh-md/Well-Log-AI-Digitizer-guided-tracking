import React, { useState, useRef, useEffect, useCallback } from 'react';

/**
 * HumanGuidedCurveTracker
 * -----------------------
 * Human-in-the-loop AI curve tracing:
 *   1. The user clicks a few anchor points on ONE curve.
 *   2. "AI Track" sends the image + anchors to the backend, which learns the
 *      curve's color/darkness from the clicked points and traces the ACTUAL
 *      curve pixels between every pair of anchors (anchored optimal path).
 *   3. Low-confidence segments are drawn in amber: the user clicks one more
 *      anchor there and re-tracks. A few clicks converge to pixel accuracy.
 *
 * Supports multiple curves per image (each with its own color + anchors).
 *
 * Props:
 *   imageUrl        - URL / blob URL of the graph image being digitized
 *   onCurveTracked  - callback(curves) fired after every successful track:
 *                     [{ id, name, color, anchors, points, confidence }]
 *   apiUrl          - optional override of the tracking endpoint
 */

const TRACK_API_URL =
  (typeof import.meta !== 'undefined' &&
    import.meta.env &&
    import.meta.env.VITE_GRAPH_GUIDED_TRACK) ||
  'http://127.0.0.1:8000/guided-curve-track';

const CURVE_COLORS = ['#00C853', '#2962FF', '#FF6D00', '#D500F9', '#00B8D4', '#FFD600'];

// Fallback only (used when the backend is unreachable): smooth spline
// through the anchors. NOT real curve tracking.
function catmullRomSpline(points, numSegments = 30) {
  if (points.length < 2) return points;
  const p = [points[0], ...points, points[points.length - 1]];
  const result = [];
  for (let i = 1; i < p.length - 2; i++) {
    const [p0, p1, p2, p3] = [p[i - 1], p[i], p[i + 1], p[i + 2]];
    for (let t = 0; t < 1; t += 1 / numSegments) {
      const t2 = t * t, t3 = t2 * t;
      const x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3);
      const y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3);
      result.push([Math.round(x), Math.round(y)]);
    }
  }
  result.push(points[points.length - 1]);
  return result;
}

let curveCounter = 0;
const newCurve = () => ({
  id: `curve-${Date.now()}-${curveCounter++}`,
  name: `Curve ${curveCounter}`,
  color: CURVE_COLORS[(curveCounter - 1) % CURVE_COLORS.length],
  anchors: [],          // [[x, y], ...] user clicks (snapped client-side)
  points: [],           // AI-traced path [[x, y], ...]
  segments: [],         // per-segment confidence from backend
  confidence: null,
  source: null,         // 'ai' | 'fallback'
});

const HumanGuidedCurveTracker = ({ imageUrl, onCurveTracked, onSave, apiUrl }) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const containerRef = useRef(null);

  const [curves, setCurves] = useState([newCurve()]);
  const [activeId, setActiveId] = useState(curves[0].id);
  const [mousePos, setMousePos] = useState(null);
  const [dragging, setDragging] = useState(null); // { curveId, anchorIdx }
  const [tracking, setTracking] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);     // for undo

  const endpoint = apiUrl || TRACK_API_URL;
  const activeCurve = curves.find(c => c.id === activeId) || curves[0];

  /* ---------------- image loading ---------------- */
  useEffect(() => {
    if (!imageUrl) return;
    const canvas = canvasRef.current;
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.src = imageUrl;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      imageRef.current = img;
      redrawRef.current && redrawRef.current();
    };
  }, [imageUrl]);

  /* ---------------- client-side snap (visual nicety only;
        the backend re-snaps with the learned appearance model) ------- */
  const snapToDarkestPixel = (x, y, radius = 15) => {
    const canvas = canvasRef.current;
    if (!canvas || !imageRef.current) return [x, y];
    const ctx = canvas.getContext('2d');
    const x0 = Math.max(0, x - radius), y0 = Math.max(0, y - radius);
    const w = Math.min(canvas.width - x0, radius * 2 + 1);
    const h = Math.min(canvas.height - y0, radius * 2 + 1);
    if (w <= 0 || h <= 0) return [x, y];
    // sample from a pristine copy of the image (not the overlay drawing)
    const off = document.createElement('canvas');
    off.width = w; off.height = h;
    off.getContext('2d').drawImage(imageRef.current, x0, y0, w, h, 0, 0, w, h);
    const data = off.getContext('2d').getImageData(0, 0, w, h).data;
    let best = Infinity, bx = x, by = y;
    for (let i = 0; i < data.length; i += 4) {
      const idx = i / 4;
      const px = x0 + (idx % w), py = y0 + Math.floor(idx / w);
      const darkness = data[i] + data[i + 1] + data[i + 2];
      const d2 = (px - x) ** 2 + (py - y) ** 2;
      const score = darkness + d2 * 0.8; // prefer dark AND close
      if (score < best) { best = score; bx = px; by = py; }
    }
    return [bx, by];
  };

  /* ---------------- coordinate helpers ---------------- */
  const canvasCoords = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = canvasRef.current.width / rect.width;
    const sy = canvasRef.current.height / rect.height;
    return [
      Math.round((e.clientX - rect.left) * sx),
      Math.round((e.clientY - rect.top) * sy),
    ];
  };

  const pushHistory = () => setHistory(h => [...h.slice(-30), JSON.stringify(curves)]);

  const updateActive = (fn) =>
    setCurves(prev => prev.map(c => (c.id === activeId ? fn(c) : c)));

  /* ---------------- mouse interaction ---------------- */
  const ANCHOR_HIT_RADIUS = 10;

  const findAnchorAt = (x, y) => {
    for (const c of curves) {
      for (let i = 0; i < c.anchors.length; i++) {
        const [ax, ay] = c.anchors[i];
        if ((ax - x) ** 2 + (ay - y) ** 2 <= ANCHOR_HIT_RADIUS ** 2) {
          return { curveId: c.id, anchorIdx: i };
        }
      }
    }
    return null;
  };

  const handleMouseDown = (e) => {
    const [x, y] = canvasCoords(e);
    const hit = findAnchorAt(x, y);

    // Alt+click or right-click on an anchor -> delete it
    if (hit && (e.altKey || e.button === 2)) {
      e.preventDefault();
      pushHistory();
      const targetCurve = curves.find(c => c.id === hit.curveId);
      const updatedCurve = {
        ...targetCurve,
        anchors: targetCurve.anchors.filter((_, i) => i !== hit.anchorIdx),
      };
      
      if (updatedCurve.anchors.length >= 2) {
        setCurves(prev => prev.map(c => c.id !== hit.curveId ? c : updatedCurve));
        handleTrackCurve(updatedCurve);
      } else {
        const clearedCurve = { ...updatedCurve, points: [], segments: [], confidence: null };
        setCurves(prev => prev.map(c => c.id !== hit.curveId ? c : clearedCurve));
      }
      return;
    }
    if (hit) { setDragging(hit); setActiveId(hit.curveId); return; }
    if (e.button === 2) return;

    // plain click -> add an anchor to the active curve
    pushHistory();
    const [sx, sy] = snapToDarkestPixel(x, y, 15);
    const updatedCurve = { ...activeCurve, anchors: [...activeCurve.anchors, [sx, sy]] };
    updateActive(() => updatedCurve);
    if (updatedCurve.anchors.length >= 2) handleTrackCurve(updatedCurve);
  };

  const handleMouseMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = canvasRef.current.width / rect.width;
    const sy = canvasRef.current.height / rect.height;
    setMousePos({
      x: (e.clientX - rect.left) * sx,
      y: (e.clientY - rect.top) * sy,
      rawX: e.clientX, rawY: e.clientY,
    });
    if (dragging) {
      const [x, y] = canvasCoords(e);
      setCurves(prev => prev.map(c => c.id !== dragging.curveId ? c : {
        ...c,
        anchors: c.anchors.map((p, i) => (i === dragging.anchorIdx ? [x, y] : p)),
      }));
    }
  };

  const handleMouseUp = () => {
    if (dragging) {
      // snap the dropped anchor
      const targetCurve = curves.find(c => c.id === dragging.curveId);
      if (targetCurve) {
        const updatedCurve = {
          ...targetCurve,
          anchors: targetCurve.anchors.map((p, i) =>
            i === dragging.anchorIdx ? snapToDarkestPixel(p[0], p[1], 15) : p),
        };
        setCurves(prev => prev.map(c => c.id !== dragging.curveId ? c : updatedCurve));
        if (updatedCurve.anchors.length >= 2) handleTrackCurve(updatedCurve);
      }
      setDragging(null);
    }
  };

  /* ---------------- undo ---------------- */
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        setHistory(h => {
          if (!h.length) return h;
          setCurves(JSON.parse(h[h.length - 1]));
          return h.slice(0, -1);
        });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  /* ---------------- drawing ---------------- */
  const redrawRef = useRef(null);
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageRef.current) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageRef.current, 0, 0);

    for (const c of curves) {
      // traced path - low-confidence segments in amber dashes
      if (c.points.length > 1) {
        const lowConfRanges = (c.segments || [])
          .filter(s => s.confidence < 0.55)
          .map(s => [Math.min(s.from[1], s.to[1]), Math.max(s.from[1], s.to[1])]);
        const isLowConf = (y) => lowConfRanges.some(([a, b]) => y >= a && y <= b);

        let runLow = isLowConf(c.points[0][1]);
        let start = 0;
        const drawRun = (from, to, low) => {
          ctx.beginPath();
          ctx.moveTo(c.points[from][0], c.points[from][1]);
          for (let i = from + 1; i <= to; i++) ctx.lineTo(c.points[i][0], c.points[i][1]);
          ctx.strokeStyle = low ? '#F59E0B' : c.color;
          ctx.setLineDash(low ? [6, 4] : []);
          ctx.lineWidth = c.id === activeId ? 5.0 : 3.0;
          ctx.stroke();
          ctx.setLineDash([]);
        };
        for (let i = 1; i < c.points.length; i++) {
          const low = isLowConf(c.points[i][1]);
          if (low !== runLow) { drawRun(start, i, runLow); start = i; runLow = low; }
        }
        drawRun(start, c.points.length - 1, runLow);
      }
      // anchors
      c.anchors.forEach((p, idx) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], c.id === activeId ? 5 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = c.color;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        if (c.id === activeId) {
          ctx.fillStyle = '#111';
          ctx.font = '10px sans-serif';
          ctx.fillText(String(idx + 1), p[0] + 7, p[1] - 7);
        }
      });
    }
  }, [curves, activeId]);
  redrawRef.current = redraw;
  useEffect(() => { redraw(); }, [redraw]);

  /* ---------------- AI tracking ---------------- */
  const handleTrackCurve = async (curveOverride) => {
    const c = curveOverride && curveOverride.id ? curveOverride : activeCurve;
    if (!c || c.anchors.length < 2) return;
    setTracking(true);
    setError(null);
    try {
      const blob = await fetch(imageUrl).then(r => r.blob());
      const form = new FormData();
      const name = blob.type.includes('tiff') ? 'image.tif'
        : blob.type.includes('jpeg') ? 'image.jpg' : 'image.png';
      form.append('file', blob, name);
      form.append('points', JSON.stringify(c.anchors));

      const res = await fetch(endpoint, { method: 'POST', body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server returned ${res.status}`);
      }
      const data = await res.json();
      const tracked = {
        ...c,
        points: data.points,
        segments: data.segments,
        confidence: data.confidence,
        anchors: data.snapped_anchors || c.anchors,
        source: 'ai',
      };
      setCurves(prev => prev.map(cu => (cu.id === c.id ? tracked : cu)));
      if (onCurveTracked) {
        onCurveTracked(curves.map(cu => (cu.id === c.id ? tracked : cu)));
      }
    } catch (err) {
      // Backend unreachable -> spline fallback so the user is never blocked
      const sorted = [...c.anchors].sort((a, b) => a[1] - b[1]);
      const spline = catmullRomSpline(sorted, 50);
      updateActive(cur => ({
        ...cur, points: spline, segments: [], confidence: null, source: 'fallback',
      }));
      setError(`AI tracking unavailable (${err.message}). Drew a spline through your points instead - start the Python backend for real pixel tracking.`);
    } finally {
      setTracking(false);
    }
  };

  /* ---------------- curve management ---------------- */
  const addCurve = () => {
    const c = newCurve();
    setCurves(prev => [...prev, c]);
    setActiveId(c.id);
  };
  const removeCurve = (id) => {
    pushHistory();
    setCurves(prev => {
      const next = prev.filter(c => c.id !== id);
      if (!next.length) next.push(newCurve());
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  };
  const clearActive = () => {
    pushHistory();
    updateActive(c => ({ ...c, anchors: [], points: [], segments: [], confidence: null }));
  };

  const lowConfCount = (activeCurve?.segments || []).filter(s => s.confidence < 0.55).length;

  return (
    <div className="flex flex-col gap-3 p-4 bg-white rounded-lg shadow-sm border border-gray-200">
      {/* header */}
      <div className="flex justify-between items-start flex-wrap gap-2">
        <div>
          <h3 className="font-semibold text-gray-800 text-lg">Human-Guided AI Tracking</h3>
          <p className="text-sm text-gray-500">
            Click a few points along one curve, then press <b>AI Track</b>. The AI learns the
            curve's color from your clicks and traces the real curve pixels between them.
            Drag anchors to move - Alt+click to delete - Ctrl+Z to undo.
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={clearActive}
            className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded text-gray-700 font-medium">
            Clear Curve
          </button>
          <button
            onClick={handleTrackCurve}
            disabled={tracking || (activeCurve?.anchors.length || 0) < 2}
            className={`px-4 py-2 text-sm rounded font-medium flex items-center gap-2 ${
              !tracking && (activeCurve?.anchors.length || 0) >= 2
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}>
            {tracking ? 'Tracing…' : '🪄 AI Track'}
          </button>
          <button
            onClick={() => {
              if (onSave) {
                // only pass curves that have actual traced points
                const validCurves = curves.filter(c => c.points && c.points.length > 0).map(c => c.points);
                onSave(validCurves);
              }
            }}
            disabled={!curves.some(c => c.points && c.points.length > 0)}
            className={`px-4 py-2 text-sm rounded font-medium flex items-center gap-2 ${
              curves.some(c => c.points && c.points.length > 0)
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}>
            💾 Save & Return to Main View
          </button>
        </div>
      </div>

      {/* curve tabs */}
      <div className="flex items-center gap-2 flex-wrap">
        {curves.map(c => (
          <div key={c.id}
            onClick={() => setActiveId(c.id)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm cursor-pointer border ${
              c.id === activeId ? 'border-gray-800 bg-gray-50 font-medium' : 'border-gray-200 text-gray-600'
            }`}>
            <span className="w-3 h-3 rounded-full" style={{ background: c.color }} />
            {c.name}
            <span className="text-xs text-gray-400">{c.anchors.length} pts</span>
            {c.confidence != null && (
              <span className={`text-xs ${c.confidence > 0.7 ? 'text-green-600' : 'text-amber-600'}`}>
                {(c.confidence * 100).toFixed(0)}%
              </span>
            )}
            {curves.length > 1 && (
              <button onClick={(e) => { e.stopPropagation(); removeCurve(c.id); }}
                className="text-gray-400 hover:text-red-500">×</button>
            )}
          </div>
        ))}
        <button onClick={addCurve}
          className="px-3 py-1.5 text-sm rounded-full border border-dashed border-gray-300 text-gray-500 hover:border-gray-500">
          + New Curve
        </button>
      </div>

      {/* canvas */}
      <div ref={containerRef}
        className="relative overflow-auto border border-gray-300 rounded cursor-crosshair max-h-[700px] w-full">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { setMousePos(null); handleMouseUp(); }}
          onContextMenu={(e) => e.preventDefault()}
          className="max-w-none shadow-inner"
        />
        {/* magnifying loupe */}
        {mousePos && imageUrl && !dragging && (
          <div
            className="fixed pointer-events-none border-2 border-blue-500 rounded-full overflow-hidden shadow-2xl z-50 bg-white"
            style={{
              left: mousePos.rawX + 20, top: mousePos.rawY + 20,
              width: 120, height: 120,
              backgroundImage: `url(${imageUrl})`,
              backgroundPosition: `-${mousePos.x * 3 - 60}px -${mousePos.y * 3 - 60}px`,
              backgroundSize: `${canvasRef.current?.width * 3 || 0}px ${canvasRef.current?.height * 3 || 0}px`,
              backgroundRepeat: 'no-repeat',
            }}>
            <div className="absolute top-1/2 left-0 w-full h-[1px] bg-red-500 opacity-60" />
            <div className="absolute left-1/2 top-0 w-[1px] h-full bg-red-500 opacity-60" />
          </div>
        )}
      </div>

      {/* status */}
      {error && (
        <div className="text-xs text-amber-800 bg-amber-50 p-2 rounded border border-amber-200">
          ⚠ {error}
        </div>
      )}
      {activeCurve?.source === 'ai' && activeCurve.points.length > 0 && (
        <div className="text-xs text-green-800 bg-green-50 p-2 rounded border border-green-200">
          ✓ AI traced <b>{activeCurve.points.length}</b> points (one per depth row),
          confidence <b>{(activeCurve.confidence * 100).toFixed(1)}%</b>.
          {lowConfCount > 0 && (
            <span className="text-amber-700">
              {' '}{lowConfCount} segment{lowConfCount > 1 ? 's' : ''} drawn in
              <b> amber dashes</b> look uncertain - click an extra anchor there and re-track.
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default HumanGuidedCurveTracker;
