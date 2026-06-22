import re

with open('frontend/src/Components/DashboardV2/HumanGuidedCurveTracker.jsx', 'r') as f:
    content = f.read()

# 1. Add trackBounds prop
content = content.replace('  apiUrl,\n  zoom: parentZoom,', '  apiUrl,\n  trackBounds = [],\n  zoom: parentZoom,')
content = content.replace('  const endpoint = apiUrl || TRACK_API_URL;', '  const endpoint = apiUrl || TRACK_API_URL;\n\n  const getActiveTrackBounds = (points) => {\n    if (!points || !points.length || !trackBounds.length) return [-1, -1];\n    const [x, y] = points[0];\n    for (const b of trackBounds) {\n      if (x >= b.left && x <= b.right) return [b.left, b.right];\n    }\n    return [-1, -1];\n  };')

# 2. Add recomputeWrapLevels
wrap_levels_func = """
  const recomputeWrapLevels = (curve) => {
    if (!curve.fragments || !curve.fragments.length) return curve;
    const step = (outEdge, inEdge) => {
      if (outEdge === 'right' && inEdge === 'left') return 1;
      if (outEdge === 'left' && inEdge === 'right') return -1;
      return 0;
    };
    curve.fragments[0].wrapLevel = curve.fragments[0].wrapLevelManual ? curve.fragments[0].wrapLevel : 0;
    for (let i = 1; i < curve.fragments.length; i++) {
      const prev = curve.fragments[i - 1];
      const curr = curve.fragments[i];
      if (!curr.wrapLevelManual) {
        curr.wrapLevel = prev.wrapLevel + step(prev.edgeOut, curr.edgeIn);
      }
    }
    return curve;
  };
"""
content = content.replace('  /* ---------------- coordinate helpers ---------------- */', wrap_levels_func + '\n  /* ---------------- coordinate helpers ---------------- */')

with open('frontend/src/Components/DashboardV2/HumanGuidedCurveTracker.jsx', 'w') as f:
    f.write(content)
