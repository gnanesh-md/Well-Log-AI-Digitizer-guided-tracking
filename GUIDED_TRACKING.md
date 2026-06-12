# Human-Guided AI Curve Tracking

This feature lets a user click a handful of anchor points on ONE curve in a
well-log image (TIFF/PNG/JPG), and the AI traces the actual curve pixels
between every pair of anchors — learning the curve's color/darkness from the
points the user clicked, crossing other curves and gridlines correctly, and
bridging dashed segments.

## What was added

| File | What it does |
|---|---|
| `python-multiline/guided_curve_tracker.py` | New backend module: appearance learning + cost map + anchored second-order DP tracer. Exposes `POST /guided-curve-track`. |
| `python-multiline/main.py` | One-line change: mounts the new router into the existing FastAPI app. |
| `python-multiline/test_guided_tracker.py` | Regression test on a synthetic log (two crossing black curves, dashed section, gridlines, noisy clicks). |
| `frontend/src/Components/DashboardV2/HumanGuidedCurveTracker.jsx` | Rewritten: real AI tracking, multi-curve support, draggable/deletable anchors, undo, confidence display, spline fallback. |
| `frontend/.env` | Added `VITE_GRAPH_GUIDED_TRACK`. |

## How the algorithm works

1. **Appearance learning** — small windows are sampled around each user click.
   Pixels that clearly stand out from the local paper background are kept and
   their median LAB color becomes the curve's appearance model. This is what
   makes the tracker follow *the curve the user pointed at* (black, red,
   dashed, faded — whatever was clicked), not just "any dark pixel".

2. **Cost map** — every pixel is scored by how unlike the learned curve it
   looks (chroma-weighted LAB distance). Long straight horizontal/vertical
   strokes that don't match the learned appearance (gridlines) get an extra
   penalty.

3. **Anchored optimal path** — between each consecutive pair of anchors a
   dynamic program finds the globally cheapest path that:
   - passes exactly through both anchors,
   - produces exactly **one X per depth row** (well-log curves are functions
     of depth),
   - penalizes **slope changes** (second-order model). This "curve nature"
     prior makes the path continue in its current direction at a crossing
     instead of switching onto the other curve.

   Because the path is globally optimal (not a greedy tracker), it cannot
   wander off, and dashed gaps are bridged by the cheapest straight
   continuation.

4. **Confidence** — each segment returns `confidence = 1 − mean cost`. The
   frontend draws low-confidence segments in amber dashes so the user knows
   exactly where to click one more anchor.

## About "100% accuracy"

No algorithm can guarantee 100% on scanned logs — two identical black curves
running tangent for a stretch are ambiguous even to a human. What this design
guarantees instead is **convergence**: every anchor the user adds is a hard
constraint the trace must pass through, so any wrong section is fixed with one
click in that section. On the regression test (crossing curves + dashes +
gridlines + noisy clicks), 6 anchors give ~2.4 px mean error and 3 more
corrective clicks give **0.93 px mean error / 100% of points within 4 px**.

## API

`POST /guided-curve-track` (multipart form):

| field | type | default | meaning |
|---|---|---|---|
| `file` | file | — | the graph image (tif/tiff/png/jpg/...) |
| `points` | JSON string | — | `[[x,y],[x,y],...]` user anchors in image pixels |
| `snap_radius` | int | 10 | how far a click may snap to the curve |
| `max_slope_px` | int | 14 | max horizontal px the curve may move per row |
| `move_penalty` | float | 0.030 | cost per px of horizontal movement |
| `curvature_penalty` | float | 0.060 | cost per px of slope *change* (curve-nature prior) |
| `corridor_pad` | int | 120 | search corridor width around the anchors |
| `smooth_window` | int | 5 | moving-average smoothing of the result |
| `suppress_gridlines` | bool | true | penalize non-matching grid strokes |

Response:

```json
{
  "points": [[x, y], ...],            // one per depth row, through all anchors
  "snapped_anchors": [[x, y], ...],
  "segments": [{"from": [x,y], "to": [x,y], "confidence": 0.97, "mean_cost": 0.03}],
  "confidence": 0.96,
  "curve_color_lab": [16.0, 128.0, 128.0],
  "num_points": 1135
}
```

## Running it

```bash
# backend (same as before — the new endpoint is mounted automatically)
cd python-multiline
uvicorn main:app --host 0.0.0.0 --port 8000

# regression test
python test_guided_tracker.py

# frontend
cd frontend && npm run dev
```

Use the component anywhere you already display the graph image:

```jsx
import HumanGuidedCurveTracker from './Components/DashboardV2/HumanGuidedCurveTracker';

<HumanGuidedCurveTracker
  imageUrl={graphImageUrl}                 // blob/object URL of the graph
  onCurveTracked={(curves) => {
    // curves: [{ id, name, color, anchors, points, confidence }]
    // feed curve.points into your existing pixel->physical conversion
    // (pixel_points_to_physical / create-las-from-coords)
  }}
/>
```

## Workflow for the user

1. Pick / create a curve tab, click 3–8 points along one curve (the loupe
   helps precision; clicks auto-snap to the line).
2. Press **AI Track** — the green path appears in under a couple of seconds.
3. If any section is wrong or shown in amber, click one extra anchor on the
   correct line in that section and press **AI Track** again.
4. Add a **New Curve** tab and repeat for the next curve in the TIFF.

## Tuning tips

- Steep curves getting cut off → raise `max_slope_px` (e.g. 20).
- Trace jumping onto a neighboring parallel curve → raise `curvature_penalty`
  to 0.08–0.10, or just add an anchor between the curves.
- Very wide tracks / anchors far apart horizontally → raise `corridor_pad`.
- Heavily dashed curves → nothing needed; gaps are bridged automatically, but
  an anchor on each dash run improves the bridge direction.
