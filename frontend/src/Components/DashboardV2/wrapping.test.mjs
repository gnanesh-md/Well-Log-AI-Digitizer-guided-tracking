import {
  resampleWrappedCurve,
  resampleCurvesOnSharedScale,
  NULL_VALUE,
} from "./wrappingValues.js";

const scale = { pxLeft: 100, pxRight: 500, valueLeft: 0, valueRight: 100 };
const depth = { pxTop: 0, pxBottom: 1000, depthTop: 250, depthBottom: 750, step: 0.5 };
const S = scale.valueRight - scale.valueLeft;

function trueValueToPixel(v) {
  const frac = (v - scale.valueLeft) / S;
  const lap = Math.floor(frac);
  const baseFrac = frac - lap;
  const x = scale.pxLeft + baseFrac * (scale.pxRight - scale.pxLeft);
  return { x, lap };
}
function depthToPy(d) {
  return depth.pxTop + ((d - depth.depthTop) / (depth.depthBottom - depth.depthTop)) * (depth.pxBottom - depth.pxTop);
}

function buildPieces(valueFn) {
  const samples = [];
  for (let d = depth.depthTop; d <= depth.depthBottom + 1e-9; d += 0.25) {
    const dd = Number(d.toFixed(4));
    const v = valueFn(dd);
    const { x, lap } = trueValueToPixel(v);
    samples.push({ d: dd, x, y: depthToPy(dd), lap, v });
  }
  const pieces = [];
  let cur = null;
  for (const s of samples) {
    if (!cur || cur.lap !== s.lap) {
      cur = { lap: s.lap, points: [] };
      pieces.push(cur);
    }
    cur.points.push([s.x, s.y]);
  }
  return { pieces, samples };
}

const valueA = (d) => {
  const t = (d - depth.depthTop) / (depth.depthBottom - depth.depthTop);
  return t <= 0.5 ? 500 * t : 500 * (1 - t);
};
const valueB = (d) => {
  const t = (d - depth.depthTop) / (depth.depthBottom - depth.depthTop);
  return t <= 0.5 ? 40 - 160 * t : 40 - 160 * (1 - t);
};

function scramble(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(((i * 9301 + 49297) % 233280) / 233280 * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function checkCurve(label, valueFn) {
  const { pieces } = buildPieces(valueFn);
  const laps = [...new Set(pieces.map(p => p.lap))].sort((a, b) => a - b);
  const out = resampleWrappedCurve(scramble(pieces), scale, depth);

  let maxErr = 0;
  let nulls = 0;
  for (const row of out) {
    if (row.value === NULL_VALUE) { nulls += 1; continue; }
    maxErr = Math.max(maxErr, Math.abs(row.value - valueFn(row.depth)));
  }
  const ok = maxErr < 1e-3 && nulls === 0;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${label.padEnd(8)} pieces=${pieces.length} laps=[${laps.join(",")}] ` +
    `rows=${out.length} nulls=${nulls} maxErr=${maxErr.toExponential(2)}`
  );
  return ok;
}

console.log("=== single-curve wrapping ===");
let allOk = true;
allOk &= checkCurve("Curve A", valueA);
allOk &= checkCurve("Curve B", valueB);

console.log("\n=== TWO curves on the SAME scale (the reported case) ===");
const both = resampleCurvesOnSharedScale(
  [
    { name: "CURVE_A", pieces: scramble(buildPieces(valueA).pieces) },
    { name: "CURVE_B", pieces: scramble(buildPieces(valueB).pieces) },
  ],
  scale,
  depth
);
for (const c of both) {
  let maxErr = 0;
  const fn = c.name === "CURVE_A" ? valueA : valueB;
  for (const row of c.data) if (row.value !== NULL_VALUE) maxErr = Math.max(maxErr, Math.abs(row.value - fn(row.depth)));
  const ok = maxErr < 1e-3;
  allOk &= ok;
  console.log(`${ok ? "PASS" : "FAIL"}  ${c.name} independent on shared scale, maxErr=${maxErr.toExponential(2)}`);
}

const seam = resampleWrappedCurve(buildPieces(valueA).pieces, scale, depth);
const atPeak = seam.reduce((m, r) => (r.value > m.value ? r : m), seam[0]);
console.log(`\nseam check: Curve A peak value = ${atPeak.value} at depth ${atPeak.depth} (true peak 250)`);

console.log(allOk ? "\nALL TESTS PASSED ✅" : "\nSOME TESTS FAILED ❌");
process.exit(allOk ? 0 : 1);
