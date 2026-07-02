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
  'http://127.0.0.1:8123/guided-curve-track';

const CURVE_COLORS = [
  '#2962FF', '#00C853', '#FF6D00', '#D500F9', '#00B8D4',
  '#FFD600', '#F50057', '#00BFA5', '#FF6E40', '#6200EA',
  '#E53935', '#8E24AA', '#3949AB', '#039BE5', '#43A047',
  '#FDD835', '#FB8C00', '#6D4C41', '#546E7A', '#000000'
];

const COLOR_NAMES = [
  'Blue', 'Green', 'Orange', 'Purple', 'Cyan',
  'Yellow', 'Pink', 'Teal', 'Deep Orange', 'Violet',
  'Red', 'Plum', 'Indigo', 'Light Blue', 'Leaf Green',
  'Lemon', 'Dark Orange', 'Brown', 'Blue Grey', 'Black'
];

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
  style: 'solid',
  wrapMode: false,
  activeFragmentId: null,
  fragments: [],        // { id, anchors, points, segments, confidence, wrapLevel, edgeIn, edgeOut, wrapLevelManual }
  anchors: [],          // legacy (non-wrap) [[x, y], ...] user clicks (snapped client-side)
  points: [],           // legacy (non-wrap) AI-traced path [[x, y], ...]
  segments: [],         // legacy (non-wrap) per-segment confidence from backend
  confidence: null,
  source: null,         // 'ai' | 'fallback'
  hidden: false,
});

const HumanGuidedCurveTracker = ({
  imageUrl,
  onCurveTracked,
  onSave,
  apiUrl,
  trackBounds = [],
  zoom: parentZoom,
  setZoom: parentSetZoom,
  panOffset: parentPanOffset,
  setPanOffset: parentSetPanOffset
}) => {
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
  const [localZoom, setLocalZoom] = useState(1);
  const zoom = parentZoom ?? localZoom;
  const setZoom = parentSetZoom ?? setLocalZoom;

  const [colorPickerOpen, setColorPickerOpen] = useState(null);
  const [editingCurveId, setEditingCurveId] = useState(null);
  const [panMode, setPanMode] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  
  const [localPanOffset, setLocalPanOffset] = useState({ x: 0, y: 0 });
  const panOffset = parentPanOffset ?? localPanOffset;
  const setPanOffset = parentSetPanOffset ?? setLocalPanOffset;
  
  const panStart = useRef(null);

  const endpoint = apiUrl || TRACK_API_URL;

  const getActiveTrackBounds = (points) => {
    if (!points || !points.length || !trackBounds.length) return [-1, -1];
    const [x] = points[0];
    // Exact containment first.
    for (const b of trackBounds) {
      if (x >= b.left && x <= b.right) return [b.left, b.right];
    }
    // FIX: a wrapped fragment's edge point is frequently a pixel or two
    // outside the track rectangle (anti-aliasing / snap-to-darkest-pixel
    // can nudge it past the boundary line). Previously this fell through
    // to [-1, -1], which corrupts computeFragmentEdges's tolerance window
    // (tol = 0.06 * max(right-left,1) with left=right=-1) and makes the
    // "is this the right edge?" test (x >= trackRight - tol) true for
    // almost any real image coordinate — silently freezing wrapLevel
    // detection for the whole curve. Instead, snap to the closest bound.
    let best = null;
    let bestDist = Infinity;
    for (const b of trackBounds) {
      const d = x < b.left ? b.left - x : x > b.right ? x - b.right : 0;
      if (d < bestDist) { bestDist = d; best = b; }
    }
    if (best && bestDist <= 25) return [best.left, best.right]; // within 25px, treat as that track
    return [-1, -1];
  };
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


  // STEP 1: Compute fragment entry/exit edges from its tracked points.
  // Used as a fallback when the backend edge_in/edge_out is null.
  const computeFragmentEdges = (frag, trackLeft, trackRight) => {
    const pts = (frag.points || []).filter(Boolean);
    if (pts.length < 2) return { edgeIn: null, edgeOut: null };
    // FIX: bail out instead of guessing when bounds are unresolved. Silently
    // treating x >= -1.06 as "at the right edge" (the old behavior when
    // trackLeft/trackRight came back as -1) made every point look like a
    // right-edge exit, which corrupted wrapLevel for the whole curve.
    if (trackLeft == null || trackRight == null || trackLeft < 0 || trackRight < 0 || trackRight <= trackLeft) {
      return { edgeIn: null, edgeOut: null };
    }
    const width = trackRight - trackLeft;
    // FIX: 6% of a wide track (e.g. 1000px) is 60px of slack — too loose.
    // 6% of a narrow track (e.g. 80px) is under 5px — too tight for a
    // hand-clicked or snapped endpoint. Use a hybrid absolute+relative
    // tolerance instead so detection is consistent across track widths.
    const tol = Math.max(6, Math.min(0.08 * width, 20));
    const edgeOf = (x) => {
      if (x <= trackLeft + tol) return 'left';
      if (x >= trackRight - tol) return 'right';
      return null;
    };
    const sorted = [...pts].sort((a, b) => a[1] - b[1]); // sort by y (depth)
    // FIX: average a small window of points near each end instead of a
    // single endpoint, so one noisy/mis-clicked point can't flip the
    // edge classification (and therefore the wrap level) for the piece.
    const windowAvgX = (arr) => arr.reduce((s, p) => s + p[0], 0) / arr.length;
    const topX = windowAvgX(sorted.slice(0, Math.min(3, sorted.length)));
    const botX = windowAvgX(sorted.slice(-Math.min(3, sorted.length)));
    return { edgeIn: edgeOf(topX), edgeOut: edgeOf(botX) };
  };

  // STEP 2: Recompute wrap levels from edge transitions between fragments.
  // Piece 1 = L0; each subsequent piece increments by +1 if prev exited right→left,
  // or -1 if left→right. Manual overrides are preserved.
  const recomputeWrapLevels = (curve, trackLeft, trackRight) => {
    if (!curve.fragments || !curve.fragments.length) return curve;
    const frags = curve.fragments.map(f => {
      // Auto-fill edges from points if backend didn't supply them
      if ((f.edgeIn === null || f.edgeIn === undefined) ||
          (f.edgeOut === null || f.edgeOut === undefined)) {
        if (trackLeft != null && trackRight != null && f.points?.length >= 2) {
          const computed = computeFragmentEdges(f, trackLeft, trackRight);
          return {
            ...f,
            edgeIn:  f.edgeIn  ?? computed.edgeIn,
            edgeOut: f.edgeOut ?? computed.edgeOut,
          };
        }
      }
      return f;
    });

    // Piece 1: always L0 unless manually overridden
    frags[0] = { ...frags[0], wrapLevel: frags[0].wrapLevelManual ? frags[0].wrapLevel : 0 };

    for (let i = 1; i < frags.length; i++) {
      if (frags[i].wrapLevelManual) continue; // respect manual +/-
      const prevOut = frags[i - 1].edgeOut;
      const thisIn  = frags[i].edgeIn;
      let step = 0;
      let ambiguous = false;
      if (prevOut === 'right' && thisIn === 'left')  step = +1; // value crossed Vmax upward
      else if (prevOut === 'left'  && thisIn === 'right') step = -1; // value crossed Vmin downward
      else if (prevOut !== thisIn) {
        // FIX: previously any transition that wasn't a clean right→left or
        // left→right match silently fell through to step = 0 (treated as
        // "same decade as before"). That is only correct if the piece
        // genuinely didn't wrap — but it's indistinguishable from a failed
        // edge-detection. Flag it so the UI can warn instead of exporting
        // a silently-wrong value for this piece.
        ambiguous = true;
      }
      frags[i] = { ...frags[i], wrapLevel: frags[i - 1].wrapLevel + step, wrapAmbiguous: ambiguous };
    }
    return { ...curve, fragments: frags };
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


  const toggleWrapMode = async () => {
    const c = activeCurve;
    if (!c) return;
    
    // STEP 3: Exit wrap mode
    if (c.wrapMode) {
      updateActive(() => ({ ...c, wrapMode: false }));
      return;
    }
    
    // STEP 1 & 2: Enter wrap mode
    updateActive((cc) => {
      let fragments = (cc.fragments && cc.fragments.length) ? [...cc.fragments] : [];
      
      // Preserve the existing tracked curve EXACTLY, as a locked fragment.
      if (!fragments.length && cc.points && cc.points.length > 1) {
        fragments = [{
          id: `frag-${Date.now()}-0`,
          points: cc.points.map(p => p ? [...p] : null), // verbatim copy, preserving null breaks
          anchors: (cc.anchors || []).map(p => p ? [...p] : null), // keep few user anchors
          segments: cc.segments || [],
          confidence: cc.confidence ?? null,
          wrapLevel: 0,
          edgeIn: null, edgeOut: null,
          wrapLevelManual: false,
          locked: true,
        }];
      }

      // Start a FRESH, EMPTY piece and make it the active one.
      const fresh = {
        id: `frag-${Date.now()}-1`,
        points: [],
        anchors: [],
        segments: [],
        confidence: null,
        wrapLevel: 0,
        edgeIn: null, edgeOut: null,
        wrapLevelManual: false,
        locked: false,
      };
      fragments = [...fragments, fresh];

      return {
        ...cc,
        wrapMode: true,
        fragments,
        activeFragmentId: fresh.id,
        // clear legacy path
        points: [],
        anchors: [],
      };
    });
  };


  /* ---------------- mouse interaction ---------------- */
  const ANCHOR_HIT_RADIUS = 10;

  const findAnchorAt = (x, y) => {
    for (const c of curves) {
      if (c.wrapMode) {
        for (const f of c.fragments) {
          for (let i = 0; i < f.anchors.length; i++) {
            const [ax, ay] = f.anchors[i];
            if ((ax - x) ** 2 + (ay - y) ** 2 <= ANCHOR_HIT_RADIUS ** 2) {
              return { curveId: c.id, fragmentId: f.id, anchorIdx: i };
            }
          }
        }
      } else {
        for (let i = 0; i < c.anchors.length; i++) {
          const [ax, ay] = c.anchors[i];
          if ((ax - x) ** 2 + (ay - y) ** 2 <= ANCHOR_HIT_RADIUS ** 2) {
            return { curveId: c.id, anchorIdx: i };
          }
        }
      }
    }
    return null;
  };

  const handleMouseDown = (e) => {
    if (e.button === 2) { e.preventDefault(); return; }

    if (panMode) {
      setIsPanning(true);
      panStart.current = { x: e.clientX, y: e.clientY };
      return;
    }

    const [x, y] = canvasCoords(e);
    const hit = findAnchorAt(x, y);

    // Alt+click or right-click on an anchor -> delete it
    if (hit && (e.altKey || e.button === 2)) {
      e.preventDefault();
      pushHistory();
      
      const targetCurve = curves.find(c => c.id === hit.curveId);
      const isWrap = targetCurve.wrapMode;
      const tAnchors = isWrap ? targetCurve.fragments.find(f => f.id === hit.fragmentId).anchors : targetCurve.anchors;

      const newAnchors = tAnchors.filter((_, i) => i !== hit.anchorIdx);
      const updatedCurve = isWrap ? {
        ...targetCurve,
        fragments: targetCurve.fragments.map(f => f.id === hit.fragmentId ? { ...f, anchors: newAnchors } : f)
      } : { ...targetCurve, anchors: newAnchors };

      if (newAnchors.length >= 2) {
        setCurves(prev => prev.map(c => c.id !== hit.curveId ? c : updatedCurve));
        handleTrackCurve(updatedCurve);
      } else {
        const clearedCurve = isWrap ? {
          ...updatedCurve, fragments: updatedCurve.fragments.map(f => f.id === hit.fragmentId ? { ...f, points: [], segments: [], confidence: null } : f)
        } : { ...updatedCurve, points: [], segments: [], confidence: null };
        setCurves(prev => prev.map(c => c.id !== hit.curveId ? c : clearedCurve));
      }
      return;
    }
    if (hit) { 
      setDragging(hit); 
      setActiveId(hit.curveId);
      if (hit.fragmentId) {
        setCurves(prev => prev.map(c => c.id === hit.curveId ? { ...c, activeFragmentId: hit.fragmentId } : c));
      }
      return; 
    }

    // Neither panMode nor hit -> potential click OR pan!
    panStart.current = { x: e.clientX, y: e.clientY, isPotentialClick: true };
  };

  const handleMouseMove = (e) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = canvasRef.current.width / rect.width;
    const sy = canvasRef.current.height / rect.height;
    setMousePos({
      x: (e.clientX - rect.left) * sx,
      y: (e.clientY - rect.top) * sy,
      rawX: e.clientX, rawY: e.clientY,
    });

    if (panStart.current?.isPotentialClick) {
      const dx = e.clientX - panStart.current.x;
      const dy = e.clientY - panStart.current.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        setIsPanning(true);
        panStart.current = { x: e.clientX, y: e.clientY, isPotentialClick: false };
      }
    }

    // Pan mode or dynamically triggered pan from Place mode
    if (panStart.current && !panStart.current.isPotentialClick) {
      const dx = e.clientX - panStart.current.x;
      const dy = e.clientY - panStart.current.y;
      setPanOffset(p => ({ x: p.x + dx, y: p.y + dy }));
      panStart.current = { x: e.clientX, y: e.clientY, isPotentialClick: false };
      return;
    }

    if (dragging) {
      const [x, y] = canvasCoords(e);
      setCurves(prev => prev.map(c => {
        if (c.id !== dragging.curveId) return c;
        if (c.wrapMode) {
          return {
            ...c,
            fragments: c.fragments.map(f => f.id !== dragging.fragmentId ? f : {
              ...f,
              anchors: f.anchors.map((p, i) => i === dragging.anchorIdx ? [x, y] : p)
            })
          };
        }
        return {
          ...c,
          anchors: c.anchors.map((p, i) => (i === dragging.anchorIdx ? [x, y] : p)),
        };
      }));
    }
  };

  const handleMouseUp = (e) => {
    if (isPanning) {
      setIsPanning(false);
      panStart.current = null;
      return;
    }

    if (panStart.current?.isPotentialClick) {
      // It was a pure click! Place the anchor!
      const [x, y] = canvasCoords({ clientX: panStart.current.x, clientY: panStart.current.y });
      pushHistory();
      // Dashed curves: use a wider snap radius so a click in a gap still finds the nearest dash.
      const snapRadius = (activeCurve.style === 'dashed' || activeCurve.style === 'dotted') ? 30 : 15;
      const [sx, sy] = snapToDarkestPixel(x, y, snapRadius);
      
      let updatedCurve;
      if (activeCurve.wrapMode) {
        updatedCurve = {
          ...activeCurve,
          fragments: activeCurve.fragments.map(f => (f.id === activeCurve.activeFragmentId && !f.locked) ? {
            ...f, anchors: [...f.anchors, [sx, sy]]
          } : f)
        };
      } else {
        updatedCurve = { ...activeCurve, anchors: [...activeCurve.anchors, [sx, sy]] };
      }
      
      updateActive(() => updatedCurve);
      
      const tAnchors = activeCurve.wrapMode ? updatedCurve.fragments.find(f => f.id === updatedCurve.activeFragmentId).anchors : updatedCurve.anchors;
      if (tAnchors.length >= 2) handleTrackCurve(updatedCurve);

      panStart.current = null;
      return;
    }

    if (dragging) {
      const targetCurve = curves.find(c => c.id === dragging.curveId);
      if (targetCurve) {
        const dragSnapRadius = (targetCurve.style === 'dashed' || targetCurve.style === 'dotted') ? 30 : 15;
        let updatedCurve;
        if (targetCurve.wrapMode) {
          updatedCurve = {
            ...targetCurve,
            fragments: targetCurve.fragments.map(f => f.id !== dragging.fragmentId ? f : {
              ...f,
              anchors: f.anchors.map((p, i) => i === dragging.anchorIdx ? snapToDarkestPixel(p[0], p[1], dragSnapRadius) : p)
            })
          };
        } else {
          updatedCurve = {
            ...targetCurve,
            anchors: targetCurve.anchors.map((p, i) =>
              i === dragging.anchorIdx ? snapToDarkestPixel(p[0], p[1], dragSnapRadius) : p),
          };
        }
        setCurves(prev => prev.map(c => c.id !== dragging.curveId ? c : updatedCurve));
        const tAnchors = updatedCurve.wrapMode ? updatedCurve.fragments.find(f => f.id === dragging.fragmentId).anchors : updatedCurve.anchors;
        if (tAnchors.length >= 2) handleTrackCurve(updatedCurve);
      }
      setDragging(null);
    }
  };

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (panMode || e.ctrlKey || e.metaKey) {
      const delta = e.deltaY < 0 ? 0.12 : -0.12;
      setZoom(z => Math.max(0.15, Math.min(5, z + delta)));
    } else {
      setPanOffset(p => ({ x: p.x - e.deltaX, y: p.y - e.deltaY }));
    }
  }, [setZoom, setPanOffset, panMode]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

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
      if (c.hidden) continue;
      const isActive = c.id === activeId;
      const displayStyle = c.detected_style || c.style;
      const dash = displayStyle === 'dashed' ? [6, 4] : displayStyle === 'dotted' ? [2, 4] : [];

      // 1) NORMAL traced path — ALWAYS draw if present.
      if (c.points && c.points.length > 1) {
        ctx.beginPath();
        let penDown = false;
        for (let i = 0; i < c.points.length; i++) {
          const p = c.points[i];
          if (!p) { penDown = false; continue; }  // null = wrap-jump break
          if (!penDown) { ctx.moveTo(p[0], p[1]); penDown = true; }
          else ctx.lineTo(p[0], p[1]);
        }
        ctx.strokeStyle = c.color;
        ctx.setLineDash(dash);
        ctx.globalAlpha = c.wrapMode ? 0.25 : 1.0;
        ctx.lineWidth = isActive ? 5.0 : 3.0;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
      }

      // Highlight needs_anchor segments if NOT in wrapMode
      if (!c.wrapMode && c.segments && c.id === activeId) {
        c.segments.forEach(seg => {
          if (seg.needs_anchor) {
            const segPts = (c.points || []).filter(p => p && p[1] >= seg.from[1] && p[1] <= seg.to[1]);
            if (segPts.length > 1) {
              ctx.beginPath();
              ctx.moveTo(segPts[0][0], segPts[0][1]);
              for (let i = 1; i < segPts.length; i++) ctx.lineTo(segPts[i][0], segPts[i][1]);
              ctx.strokeStyle = '#f59e0b';
              ctx.setLineDash([4, 4]);
              ctx.lineWidth = 3.0;
              ctx.stroke();
              ctx.setLineDash([]);
            }
            if (seg.worst_pt) {
              const [wx, wy] = seg.worst_pt;
              ctx.beginPath();
              ctx.arc(wx, wy, 6, 0, 2 * Math.PI);
              ctx.fillStyle = '#fef3c7';
              ctx.fill();
              ctx.strokeStyle = '#f59e0b';
              ctx.lineWidth = 2;
              ctx.stroke();
              ctx.beginPath();
              ctx.moveTo(wx - 3, wy); ctx.lineTo(wx + 3, wy);
              ctx.moveTo(wx, wy - 3); ctx.lineTo(wx, wy + 3);
              ctx.strokeStyle = '#d97706';
              ctx.lineWidth = 1.5;
              ctx.stroke();
            }
          }
        });
      }

      // 2) FRAGMENTS — ALWAYS draw if present, each as its OWN path
      if (c.fragments && c.fragments.length) {
        c.fragments.forEach(frag => {
          const realPts = (frag.points || []).filter(p => p !== null);
          if (!frag.points || realPts.length < 2) return;
          const isActiveFrag = isActive && frag.id === c.activeFragmentId;
          ctx.beginPath();
          let penDown = false;
          for (let i = 0; i < frag.points.length; i++) {
            const p = frag.points[i];
            if (!p) { penDown = false; continue; }  // null = wrap-jump break
            if (!penDown) { ctx.moveTo(p[0], p[1]); penDown = true; }
            else ctx.lineTo(p[0], p[1]);
          }
          ctx.strokeStyle = c.color;
          ctx.setLineDash(dash);
          ctx.globalAlpha = frag.locked ? 0.8 : 1.0;
          ctx.lineWidth = isActiveFrag ? 5.5 : (isActive ? 4.0 : 3.0);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1.0;
          // wrap-level tag (use first non-null point)
          const firstPt = frag.points.find(p => p !== null);
          if (firstPt) {
            const [tx, ty] = firstPt;
            ctx.fillStyle = c.color;
            ctx.font = 'bold 11px sans-serif';
            ctx.fillText(`L${frag.wrapLevel > 0 ? '+' : ''}${frag.wrapLevel}`, tx + 6, ty + 4);
          }
        });
      }

      // 3) ANCHORS — show the anchors of the thing currently being edited.
      const strokeAnchor = (p, idx, big) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], big ? 5 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = c.color; ctx.fill();
        ctx.strokeStyle = 'white'; ctx.lineWidth = 1.5; ctx.stroke();
        if (big) {
          ctx.fillStyle = '#111'; ctx.font = '10px sans-serif';
          ctx.fillText(String(idx + 1), p[0] + 7, p[1] - 7);
        }
      };
      
      if (c.wrapMode && c.fragments && c.fragments.length) {
        c.fragments.forEach(frag => {
          (frag.anchors || []).forEach((p, idx) =>
            strokeAnchor(p, idx, isActive && frag.id === c.activeFragmentId)
          );
        });
      } else {
        (c.anchors || []).forEach((p, idx) => strokeAnchor(p, idx, isActive));
      }
    }
  }, [curves, activeId]);
  redrawRef.current = redraw;
  useEffect(() => { redraw(); }, [redraw]);


  /* ---------------- AI tracking ---------------- */
  const debounceRef = useRef(null);
  const abortControllerRef = useRef(null);
  const cachedImageIdRef = useRef(null);

  useEffect(() => {
    cachedImageIdRef.current = null;
  }, [imageUrl]);

  const handleTrackCurve = (curveOverride) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(async () => {
    const c = curveOverride && curveOverride.id ? curveOverride : activeCurve;
    const isWrap = c.wrapMode;
      const activeFrag = isWrap ? c.fragments.find(f => f.id === c.activeFragmentId) : null;
      if (isWrap && (!activeFrag || activeFrag.locked)) {
        setTracking(false);
        return;
      }
      const targetAnchors = isWrap ? activeFrag?.anchors : c.anchors;
    
    if (!targetAnchors || targetAnchors.length < 2) return;
    setTracking(true);
    setError(null);
    try {
      const form = new FormData();
      if (cachedImageIdRef.current) {
        form.append('image_id', cachedImageIdRef.current);
      } else {
        const blob = await fetch(imageUrl).then(r => r.blob());
        const name = blob.type.includes('tiff') ? 'image.tif'
          : blob.type.includes('jpeg') ? 'image.jpg' : 'image.png';
        form.append('file', blob, name);
      }

      form.append('points', JSON.stringify(targetAnchors));
      form.append('curve_style', c.style || 'auto');
      
      const [trackLeft, trackRight] = getActiveTrackBounds(targetAnchors);
      if (isWrap) {
        form.append('corridor_pad', 40);
        if (trackLeft >= 0) form.append('x_min', Math.round(trackLeft));
        if (trackRight >= 0) form.append('x_max', Math.round(trackRight));
      }

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      let res = await fetch(endpoint, { 
        method: 'POST', 
        body: form,
        signal: abortControllerRef.current.signal 
      });

      if (res.status === 409 && cachedImageIdRef.current) {
        // Cache expired, retry with full blob
        cachedImageIdRef.current = null;
        form.delete('image_id');
        const blob = await fetch(imageUrl).then(r => r.blob());
        const name = blob.type.includes('tiff') ? 'image.tif' : 'image.png';
        form.append('file', blob, name);
        if (abortControllerRef.current) abortControllerRef.current.abort();
        abortControllerRef.current = new AbortController();
        res = await fetch(endpoint, { 
          method: 'POST', body: form, signal: abortControllerRef.current.signal 
        });
      }

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server returned ${res.status}`);
      }
      const data = await res.json();
      
      if (data.image_id) {
        cachedImageIdRef.current = data.image_id;
      }

      if (isWrap) {
        const updatedFrag = {
          ...activeFrag,
          points: data.points,
          segments: data.segments,
          confidence: data.confidence,
          anchors: data.snapped_anchors || activeFrag.anchors,
          edgeIn: data.edge_in,
          edgeOut: data.edge_out
        };
        const updatedC = recomputeWrapLevels({
          ...c,
          fragments: c.fragments.map(f => f.id === activeFrag.id ? updatedFrag : f),
          source: 'ai',
          isSaved: false
        }, trackLeft, trackRight);
        setCurves(prev => prev.map(cu => cu.id === c.id ? updatedC : cu));
      } else {
        const tracked = {
          ...c,
          points: data.points,
          segments: data.segments,
          confidence: data.confidence,
          anchors: data.snapped_anchors || c.anchors,
          source: 'ai',
          isSaved: false,
          detected_style: data.detected_style
        };
        setCurves(prev => prev.map(cu => (cu.id === c.id ? tracked : cu)));
      }
    } catch (err) {
      if (err.name === 'AbortError') return; // Ignore aborted requests
      if (isWrap) {
        setCurves(prev => prev.map(cu => cu.id === c.id ? {
          ...cu, fragments: cu.fragments.map(f => f.id === activeFrag.id ? { ...f, points: [], confidence: null } : f), source: 'fallback'
        } : cu));
      } else {
        setCurves(prev => prev.map(cu => (cu.id === c.id ? {
          ...cu, points: [], segments: [], confidence: null, source: 'fallback'
        } : cu)));
      }
      setError(`AI tracking unavailable (${err.message}).`);
    } finally {
      setTracking(false);
    }
    }, 150);
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
    updateActive(c => ({ ...c, anchors: [], points: [], segments: [], confidence: null, fragments: [] }));
  };

  const lowConfCount = (activeCurve?.segments || []).filter(s => s.confidence < 0.55).length;

  /* close color picker on outside click */
  useEffect(() => {
    const close = () => setColorPickerOpen(null);
    window.addEventListener('click', close);
    return () => window.removeEventListener('click', close);
  }, []);



  return (
    <div className="flex flex-col flex-1 min-h-0 bg-gray-100">

      {/* ── TOP TOOLBAR (3-column grid for centered title) ─────────────────── */}
      <div className="h-10 bg-white border-b border-gray-200 grid grid-cols-3 items-center px-3 shrink-0">

        {/* LEFT: Actions */}
        <div className="flex items-center gap-1.5 justify-start">
          <button onClick={clearActive}
            className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-700 font-medium transition-colors">
            🗑️ Clear
          </button>
          <button
            onClick={toggleWrapMode}
            title="Track a curve that runs off one edge and continues on the opposite edge"
            className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded border transition-all ${
              activeCurve?.wrapMode
                ? 'bg-amber-50 border-amber-300 text-amber-700'
                : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
            }`}>
            <span>↩️</span> {activeCurve?.wrapMode ? 'Wrapping: ON' : 'Wrapping'}
          </button>
          
          <button
            onClick={handleTrackCurve}
            disabled={tracking || (activeCurve?.wrapMode ? (activeCurve.fragments.find(f => f.id === activeCurve.activeFragmentId)?.anchors.length || 0) < 2 : (activeCurve?.anchors.length || 0) < 2)}
            className={`px-2.5 py-1 text-xs rounded font-medium flex items-center gap-1 transition-colors ${
              !tracking && (activeCurve?.wrapMode ? (activeCurve.fragments.find(f => f.id === activeCurve.activeFragmentId)?.anchors.length || 0) >= 2 : (activeCurve?.anchors.length || 0) >= 2)
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}>
            {tracking ? '⏳' : '🪄'} {activeCurve?.wrapMode ? (tracking ? 'Tracing Fragment…' : 'Track Fragment') : (tracking ? 'Tracing…' : 'AI Track')}
          </button>
          
          {activeCurve?.wrapMode && (
            <button
              onClick={() => {
                const newFragId = `frag-${Date.now()}`;
                updateActive(c => {
                  const lastLevel = c.fragments.length ? c.fragments[c.fragments.length - 1].wrapLevel : 0;
                  return {
                    ...c,
                    fragments: [...c.fragments, {
                      id: newFragId,
                      points: [], anchors: [], segments: [], confidence: null,
                      wrapLevel: lastLevel, edgeIn: null, edgeOut: null, wrapLevelManual: false,
                      locked: false,
                    }],
                    activeFragmentId: newFragId
                  };
                });
              }}
              className="px-2.5 py-1 text-xs rounded font-medium bg-amber-100 hover:bg-amber-200 text-amber-800 transition-colors">
              + Next Piece
            </button>
          )}

          <button
            onClick={() => {
              if (onSave) {
                const isWrap = c => c.wrapMode && c.fragments.some(f => f.points && f.points.length > 0);
                const isNormal = c => !c.wrapMode && c.points && c.points.length > 0;
                const unsavedCurves = curves.filter(c => (isWrap(c) || isNormal(c)) && !c.isSaved);
                if (unsavedCurves.length > 0) {
                  onSave(unsavedCurves);
                  setCurves(prev => prev.map(c =>
                    (isWrap(c) || isNormal(c)) ? { ...c, isSaved: true } : c
                  ));
                } else {
                  onSave([]);
                }
              }
            }}
            disabled={!curves.some(c => c.wrapMode ? c.fragments.some(f => f.points && f.points.length > 0) : c.points && c.points.length > 0)}
            className={`px-2.5 py-1 text-xs rounded font-medium flex items-center gap-1 transition-colors ${
              curves.some(c => c.wrapMode ? c.fragments.some(f => f.points && f.points.length > 0) : c.points && c.points.length > 0)
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}>
            💾 Save &amp; Return
          </button>
        </div>

        {/* CENTER: Title */}
        <div className="flex items-center justify-center">
          <span className="text-xs font-bold text-gray-800 tracking-wide uppercase">🎯 Guided Tracking</span>
        </div>

        {/* RIGHT: Pan & Zoom */}
        <div className="flex items-center gap-2 justify-end">
          <button onClick={() => setPanMode(p => !p)}
            title={panMode ? 'Switch to Place Anchor mode' : 'Switch to Pan mode'}
            className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded border transition-colors ${
              panMode ? 'bg-blue-50 border-blue-300 text-blue-700' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
            }`}>
            {panMode ? '🖐️ Pan' : '📍 Place'}
          </button>
          <div className="flex items-center bg-gray-50 border border-gray-200 rounded divide-x divide-gray-200 text-xs">
            <button onClick={() => setZoom(z => Math.max(0.2, z - 0.2))} className="px-1.5 py-1 text-gray-600 hover:bg-gray-100">🔍−</button>
            <span className="px-2 py-1 font-semibold text-gray-700 w-12 text-center">{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom(z => Math.min(5, z + 0.2))} className="px-1.5 py-1 text-gray-600 hover:bg-gray-100">🔍+</button>
            <button onClick={() => setZoom(1)} className="px-1.5 py-1 text-gray-600 hover:bg-gray-100">🔄</button>
          </div>
        </div>
      </div>

      {/* ── CURVE TABS (Sub-header) ────────────────────────────────────────── */}
      <div className="flex items-center px-3 py-2 bg-white border-b border-gray-200 gap-2 shrink-0 flex-wrap">
        {curves.map(c => (
          <div key={c.id}
            onClick={() => setActiveId(c.id)}
            className={`relative flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs cursor-pointer border transition-all shrink-0 ${
              c.id === activeId ? 'border-blue-400 bg-blue-50 font-medium' : 'border-gray-200 text-gray-500 hover:border-gray-300'
            }`}>
            {/* color dot */}
            <button
              onClick={e => { e.stopPropagation(); setColorPickerOpen(prev => prev === c.id ? null : c.id); }}
              className="w-3.5 h-3.5 rounded-full border border-white shadow ring-1 ring-gray-300 hover:scale-125 transition-transform flex-shrink-0"
              style={{ background: c.color }} title="Change color" />
            {/* Color Picker */}
            {colorPickerOpen === c.id && (
              <div onClick={e => e.stopPropagation()}
                className="absolute top-full left-0 mt-2 z-50 bg-white border border-gray-200 rounded-xl shadow-xl p-3 min-w-[200px]">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">Choose Color</p>
                <div className="grid grid-cols-5 gap-2">
                  {CURVE_COLORS.map((col, i) => (
                    <button key={col} title={COLOR_NAMES[i]}
                      onClick={() => { setCurves(prev => prev.map(cu => cu.id === c.id ? { ...cu, color: col } : cu)); setColorPickerOpen(null); }}
                      className={`w-7 h-7 rounded-full border-2 transition-transform hover:scale-110 ${c.color === col ? 'border-gray-800 scale-110' : 'border-white shadow'}`}
                      style={{ background: col }} />
                  ))}
                </div>
              </div>
            )}
            {/* Name */}
            {editingCurveId === c.id ? (
              <input autoFocus value={c.name}
                onChange={e => setCurves(prev => prev.map(cu => cu.id === c.id ? { ...cu, name: e.target.value } : cu))}
                onBlur={() => setEditingCurveId(null)}
                onKeyDown={e => { if (e.key === 'Enter' || e.key === 'Escape') setEditingCurveId(null); }}
                onClick={e => e.stopPropagation()}
                className="w-14 bg-transparent border-b border-blue-400 outline-none text-xs font-medium text-gray-800 px-0.5" />
            ) : (
              <span onDoubleClick={e => { e.stopPropagation(); setEditingCurveId(c.id); }}
                title="Double-click to rename"
                className="text-xs font-medium text-gray-700 cursor-text select-none">{c.name}</span>
            )}
            <select value={c.style || 'auto'}
              onChange={e => {
                e.stopPropagation();
                const newStyle = e.target.value;
                const updatedC = { ...c, style: newStyle };
                setCurves(prev => prev.map(cu => cu.id === c.id ? updatedC : cu));
                if (updatedC.anchors.length >= 2) handleTrackCurve(updatedC);
              }}
              className="bg-white border border-gray-200 text-gray-500 text-[10px] rounded px-0.5 outline-none cursor-pointer">
              <option value="auto">Auto</option>
              <option value="solid">Solid</option>
              <option value="dashed">Dashed</option>
              <option value="dotted">Dotted</option>
            </select>
            <span className="text-[10px] text-gray-400">{c.wrapMode ? `${c.fragments.length} frags` : `${c.anchors.length}pt`}</span>
            {c.confidence != null && (
              <span className={`text-[10px] font-semibold ${c.confidence > 0.7 ? 'text-green-600' : 'text-amber-600'}`}>
                {(c.confidence * 100).toFixed(0)}%
              </span>
            )}
            <button onClick={e => {
              e.stopPropagation();
              setCurves(prev => prev.map(cu => cu.id === c.id ? { ...cu, hidden: !cu.hidden } : cu));
            }} className="text-gray-400 hover:text-gray-600 ml-1" title={c.hidden ? "Show curve" : "Hide curve"}>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {c.hidden ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l18 18" />
                ) : (
                  <><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></>
                )}
              </svg>
            </button>
            {curves.length > 1 && (
              <button onClick={e => { e.stopPropagation(); removeCurve(c.id); }}
                className="text-gray-300 hover:text-red-500 font-bold ml-0.5">×</button>
            )}
          </div>
        ))}
        <button onClick={addCurve}
          className="px-2 py-1 text-[10px] rounded-full border border-dashed border-gray-300 text-gray-400 hover:border-blue-400 hover:text-blue-500 transition-colors shrink-0">
          + New
        </button>

      </div>

      {/* ── WRAP MODE BANNER & FRAGMENTS ────────────────────────────────────── */}
      {activeCurve?.wrapMode && (
        <div className="bg-amber-50 border-b border-amber-200 px-3 py-2 flex flex-col gap-2 shrink-0">
          <p className="text-amber-800 text-[10px] font-medium leading-tight">
            <strong>Wrapping Mode ON:</strong> Click anchors along ONE visible piece of the curve, then press <strong>Track Fragment</strong>. Then click <strong>+ Next Piece</strong> and repeat for each wrapped piece.
          </p>
          <div className="flex items-center gap-2 flex-wrap">
            {activeCurve.fragments.map((f, i) => (
              <div key={f.id} 
                onClick={() => updateActive(c => ({ ...c, activeFragmentId: f.id }))}
                className={`flex items-center gap-1.5 px-2 py-1 rounded text-[10px] border cursor-pointer transition-colors ${f.id === activeCurve.activeFragmentId ? 'bg-amber-200 border-amber-400 font-bold' : (f.wrapAmbiguous && !f.wrapLevelManual) ? 'bg-red-50 border-red-400' : 'bg-white border-amber-200 hover:bg-amber-100'}`}>
                <span>Piece {i+1} (L{f.wrapLevel > 0 ? '+' : ''}{f.wrapLevel}){f.wrapAmbiguous && !f.wrapLevelManual ? ' ⚠️' : ''}</span>
                <div className="flex items-center gap-0.5 bg-white/80 rounded px-0.5 border border-amber-300">
                  <button onClick={e => {
                    e.stopPropagation();
                    updateActive(c => {
                      const [tL, tR] = getActiveTrackBounds(
                        c.fragments.flatMap(fr => fr.points || []).filter(Boolean)
                      );
                      const updated = {
                        ...c,
                        fragments: c.fragments.map(fr => fr.id === f.id ? { ...fr, wrapLevel: fr.wrapLevel - 1, wrapLevelManual: true } : fr)
                      };
                      return recomputeWrapLevels(updated, tL, tR);
                    });
                  }} className="hover:text-amber-700 px-0.5 font-bold">−</button>
                  <button onClick={e => {
                    e.stopPropagation();
                    updateActive(c => {
                      const [tL, tR] = getActiveTrackBounds(
                        c.fragments.flatMap(fr => fr.points || []).filter(Boolean)
                      );
                      const updated = {
                        ...c,
                        fragments: c.fragments.map(fr => fr.id === f.id ? { ...fr, wrapLevel: fr.wrapLevel + 1, wrapLevelManual: true } : fr)
                      };
                      return recomputeWrapLevels(updated, tL, tR);
                    });
                  }} className="hover:text-amber-700 px-0.5 font-bold">+</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── CANVAS (fills remaining space like graph view) ──────────────────── */}
      <div ref={containerRef}
        className={`flex-1 min-h-0 relative overflow-hidden bg-gray-100 select-none ${
          panMode ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-crosshair'
        }`}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { setMousePos(null); handleMouseUp(); }}>
        <div style={{ transform: `translate(${panOffset.x}px, ${panOffset.y}px)`, width: '100%', height: '100%' }}>
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { setMousePos(null); handleMouseUp(); }}
            onContextMenu={e => e.preventDefault()}
            className="max-w-none shadow-sm block origin-top-left"
            style={{ transform: `scale(${zoom})`, transformOrigin: 'top left' }}
          />
        </div>
        {/* loupe */}
        {mousePos && imageUrl && !dragging && !panMode && (
          <div className="fixed pointer-events-none border-2 border-blue-500 rounded-full overflow-hidden shadow-2xl z-50 bg-white"
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

      {/* ── BOTTOM STATUS BAR ───────────────────────────────────────────────── */}
      {(error || (activeCurve?.source === 'ai' && activeCurve.points.length > 0)) && (
        <div className="shrink-0 px-3 py-1.5 border-t border-gray-200 bg-white">
          {error && <p className="text-[10px] text-amber-700">⚠ {error}</p>}
          {activeCurve?.source === 'ai' && activeCurve.points.length > 0 && (
            <p className="text-[10px] text-green-700">
              ✓ AI traced <b>{activeCurve.points.length}</b> pts · confidence <b>{(activeCurve.confidence * 100).toFixed(1)}%</b>
              {lowConfCount > 0 && <span className="text-amber-600"> · {lowConfCount} low-confidence segment{lowConfCount > 1 ? 's' : ''} (add anchors)</span>}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default HumanGuidedCurveTracker;
