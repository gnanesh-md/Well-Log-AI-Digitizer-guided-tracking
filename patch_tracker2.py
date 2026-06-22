import re

with open('frontend/src/Components/DashboardV2/HumanGuidedCurveTracker.jsx', 'r') as f:
    content = f.read()

# 3. toggleWrapMode
toggle_func = """
  const toggleWrapMode = async () => {
    const c = activeCurve;
    if (!c) return;
    
    if (c.wrapMode) {
      updateActive(() => ({ ...c, wrapMode: false }));
      return;
    }
    
    if (c.points && c.points.length > 0) {
      if (window.confirm("Detect wraps from existing path and split into fragments?")) {
        try {
          const form = new FormData();
          form.append('points', JSON.stringify(c.points));
          const [trackLeft, trackRight] = getActiveTrackBounds(c.points);
          form.append('x_min', trackLeft);
          form.append('x_max', trackRight);
          
          const res = await fetch(endpoint.replace('/guided-curve-track', '/detect-wraps'), { method: 'POST', body: form });
          const data = await res.json();
          if (data.fragments && data.fragments.length > 0) {
            updateActive(() => ({
              ...c,
              wrapMode: true,
              fragments: data.fragments.map((f, i) => ({
                id: `frag-${Date.now()}-${i}`,
                points: f.points,
                anchors: [],
                segments: [],
                confidence: 1.0,
                wrapLevel: f.wrapLevel,
                edgeIn: f.edgeIn,
                edgeOut: f.edgeOut,
                wrapLevelManual: false
              })),
              activeFragmentId: `frag-${Date.now()}-${data.fragments.length - 1}`
            }));
            return;
          }
        } catch (e) {
          console.error("Auto detect wraps failed", e);
        }
      }
    }
    
    const newFragId = `frag-${Date.now()}`;
    updateActive(() => ({
      ...c,
      wrapMode: true,
      fragments: [{
        id: newFragId,
        points: [], anchors: [], segments: [], confidence: null,
        wrapLevel: 0, edgeIn: null, edgeOut: null, wrapLevelManual: false
      }],
      activeFragmentId: newFragId
    }));
  };

"""
content = content.replace('  /* ---------------- mouse interaction ---------------- */', toggle_func + '\n  /* ---------------- mouse interaction ---------------- */')

# 4. Update handleTrackCurve
track_curve = """
  /* ---------------- AI tracking ---------------- */
  const handleTrackCurve = async (curveOverride) => {
    const c = curveOverride && curveOverride.id ? curveOverride : activeCurve;
    const isWrap = c.wrapMode;
    const activeFrag = isWrap ? c.fragments.find(f => f.id === c.activeFragmentId) : null;
    const targetAnchors = isWrap ? activeFrag?.anchors : c.anchors;
    
    if (!targetAnchors || targetAnchors.length < 2) return;
    setTracking(true);
    setError(null);
    try {
      const blob = await fetch(imageUrl).then(r => r.blob());
      const form = new FormData();
      const name = blob.type.includes('tiff') ? 'image.tif'
        : blob.type.includes('jpeg') ? 'image.jpg' : 'image.png';
      form.append('file', blob, name);
      form.append('points', JSON.stringify(targetAnchors));
      form.append('curve_style', c.style || 'auto');
      
      const [trackLeft, trackRight] = getActiveTrackBounds(targetAnchors);
      if (isWrap) {
        form.append('corridor_pad', 40);
        if (trackLeft >= 0) form.append('x_min', trackLeft);
        if (trackRight >= 0) form.append('x_max', trackRight);
      }

      const res = await fetch(endpoint, { method: 'POST', body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server returned ${res.status}`);
      }
      const data = await res.json();
      
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
        });
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
  };
"""

content = re.sub(r'  /\* ---------------- AI tracking ---------------- \*/.*?  };', track_curve, content, flags=re.DOTALL)

# 5. Anchor logic replacements
content = content.replace('const targetCurve = curves.find(c => c.id === hit.curveId);', """
      const targetCurve = curves.find(c => c.id === hit.curveId);
      const isWrap = targetCurve.wrapMode;
      const tAnchors = isWrap ? targetCurve.fragments.find(f => f.id === hit.fragmentId).anchors : targetCurve.anchors;
""")

content = content.replace('anchors: targetCurve.anchors.filter((_, i) => i !== hit.anchorIdx)', 'anchors: tAnchors.filter((_, i) => i !== hit.anchorIdx)')

# Need to fix hit logic and anchor storage

with open('frontend/src/Components/DashboardV2/HumanGuidedCurveTracker.jsx', 'w') as f:
    f.write(content)
