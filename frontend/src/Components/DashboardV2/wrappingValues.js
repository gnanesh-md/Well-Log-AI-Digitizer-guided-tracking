export const NULL_VALUE = -999.25;

export function lerp(x, x0, x1, v0, v1) {
  if (x1 === x0) return v0;
  return v0 + ((x - x0) / (x1 - x0)) * (v1 - v0);
}

export function applyWrapOffset(baseValue, wrapLevel, vLeft, vRight, scaleType) {
  if (!wrapLevel) return baseValue;
  if (scaleType === 'log') {
    return (vLeft > 0 && vRight > 0) ? baseValue * Math.pow(vRight / vLeft, wrapLevel) : baseValue;
  }
  return baseValue + wrapLevel * (vRight - vLeft);
}

export function pointToValue(pt, lap, scale, depth) {
  const x = Array.isArray(pt) ? Number(pt[0]) : Number(pt.x);
  const y = Array.isArray(pt) ? Number(pt[1]) : Number(pt.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

  const base = (scale.type === 'log' && Number(scale.valueLeft) > 0 && Number(scale.valueRight) > 0)
    ? Math.pow(10, lerp(x, Number(scale.pxLeft), Number(scale.pxRight), Math.log10(Number(scale.valueLeft)), Math.log10(Number(scale.valueRight))))
    : lerp(x, Number(scale.pxLeft), Number(scale.pxRight), Number(scale.valueLeft), Number(scale.valueRight));
    
  const value = applyWrapOffset(base, Number(lap || 0), Number(scale.valueLeft), Number(scale.valueRight), scale.type);

  const d = lerp(y, Number(depth.pxTop), Number(depth.pxBottom),
                 Number(depth.depthTop), Number(depth.depthBottom));

  return { depth: d, value };
}

export function resampleWrappedCurve(pieces, scale, depth) {
  const step = Math.abs(Number(depth.step)) || 0.5;

  const pts = [];
  for (const piece of pieces || []) {
    const lap = Number(piece.lap || 0);
    for (const p of piece.points || []) {
      const v = pointToValue(p, lap, scale, depth);
      if (v) pts.push(v);
    }
  }
  
  pts.sort((a, b) => a.depth - b.depth);
  if (!pts.length) return [];

  const out = [];
  const top = Number(depth.depthTop);
  const bottom = Number(depth.depthBottom);
  const steps = Math.round((bottom - top) / step);
  for (let i = 0; i <= steps; i += 1) {
    const target = Number((top + i * step).toFixed(4));
    let before = null;
    let after = null;
    for (const p of pts) {
      if (p.depth <= target) before = p;
      if (p.depth >= target) { after = p; break; }
    }
    let value = NULL_VALUE;
    if (before && after) {
      if (Math.abs(before.depth - after.depth) < 1e-4) {
        value = before.value;
      } else if (after.depth - before.depth <= 2.0) {
        const r = (target - before.depth) / (after.depth - before.depth);
        value = before.value + r * (after.value - before.value);
      }
    }
    out.push({ depth: target, value: value === NULL_VALUE ? NULL_VALUE : Number(value.toFixed(5)) });
  }
  return out;
}

export function resampleCurvesOnSharedScale(curves, scale, depth) {
  return (curves || []).map((c, i) => ({
    name: c.name || `CURVE_${i + 1}`,
    color: c.color,
    unit: c.unit,
    description: c.description,
    data: resampleWrappedCurve(c.pieces, scale, depth),
  }));
}
