import re

with open('frontend/src/Components/DashboardV2/HumanGuidedCurveTracker.jsx', 'r') as f:
    content = f.read()

# 1. Update toolbar
toolbar_old = """            💾 Save & Return
          </button>
        </div>"""
toolbar_new = """            💾 Save & Return
          </button>

          {/* Manual LAS grid controls */}
          <button
            onClick={() => { setLasMode?.('manual'); setGridEditMode(m => !m); }}
            className={`px-2.5 py-1 text-xs rounded font-medium border transition-colors ${
              gridEditMode ? 'bg-amber-100 border-amber-300 text-amber-800' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
            }`}>
            {gridEditMode ? '✏️ Adjusting Grid' : '✏️ Adjust Grid'}
          </button>
          <button
            onClick={() => { setLasMode?.('manual'); onAddLasRegion?.('column'); setGridEditMode(true); }}
            className="px-2.5 py-1 text-xs rounded font-medium bg-white border border-gray-200 text-gray-600 hover:bg-gray-50">
            ➕ Add Column
          </button>
          <button
            onClick={() => { setLasMode?.('manual'); onAddLasRegion?.('row'); setGridEditMode(true); }}
            className="px-2.5 py-1 text-xs rounded font-medium bg-white border border-gray-200 text-gray-600 hover:bg-gray-50">
            ➕ Add Row
          </button>
          {manualGridInfo && (
            <span className="px-2 py-1 text-[11px] font-semibold text-gray-500">
              {manualGridInfo.rows || 0} rows × {manualGridInfo.cols || 0} cols
            </span>
          )}
        </div>"""
content = content.replace(toolbar_old, toolbar_new)

# 2. Add redraw logic
redraw_old = """      } else {
        (c.anchors || []).forEach((p, idx) => strokeAnchor(p, idx, isActive));
      }
    }

  }, [curves, activeId]);"""

redraw_new = """      } else {
        (c.anchors || []).forEach((p, idx) => strokeAnchor(p, idx, isActive));
      }
    }

    // --- manual LAS row/column boxes (image-space; canvas is CSS-scaled) ---
    if (Array.isArray(manualSelections)) {
      for (const sel of manualSelections) {
        const b = sel.bounds; if (!b) continue;
        const active = sel.id === activeSelectionId;
        ctx.save();
        ctx.strokeStyle = active ? '#2563EB' : (sel.type === 'column' ? '#10B981' : '#F59E0B');
        ctx.fillStyle = active ? 'rgba(37,99,235,0.08)' : 'rgba(16,185,129,0.05)';
        ctx.lineWidth = active ? 2 : 1.5;
        ctx.setLineDash(active ? [8, 4] : [6, 4]);
        ctx.fillRect(b.left, b.top, b.right - b.left, b.bottom - b.top);
        ctx.strokeRect(b.left, b.top, b.right - b.left, b.bottom - b.top);
        ctx.setLineDash([]);
        // label
        ctx.fillStyle = active ? '#2563EB' : (sel.type === 'column' ? '#10B981' : '#B45309');
        ctx.fillRect(b.left, b.top - 16, 130, 16);
        ctx.fillStyle = '#fff';
        ctx.font = '11px sans-serif';
        ctx.fillText(`${sel.label || sel.type} (${sel.type})`, b.left + 4, b.top - 4);
        // edge handles when active
        if (active) {
          const midX = (b.left + b.right) / 2, midY = (b.top + b.bottom) / 2;
          const hs = [[b.left, midY], [b.right, midY], [midX, b.top], [midX, b.bottom]];
          ctx.fillStyle = '#2563EB'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
          for (const [hx, hy] of hs) { ctx.beginPath(); ctx.arc(hx, hy, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
        }
        ctx.restore();
      }
    }

  }, [curves, activeId, manualSelections, activeSelectionId]);"""
content = content.replace(redraw_old, redraw_new)

# 3. Add hitTestBox helper
mouse_down_old = """  const findAnchorAt = (x, y) => {"""
mouse_down_new = """  const hitTestBox = (x, y) => {
    // topmost first
    for (let i = manualSelections.length - 1; i >= 0; i--) {
      const s = manualSelections[i]; const b = s.bounds; if (!b) continue;
      const nearL = Math.abs(x - b.left) <= BOX_EDGE_HIT;
      const nearR = Math.abs(x - b.right) <= BOX_EDGE_HIT;
      const nearT = Math.abs(y - b.top) <= BOX_EDGE_HIT;
      const nearB = Math.abs(y - b.bottom) <= BOX_EDGE_HIT;
      const insideX = x >= b.left - BOX_EDGE_HIT && x <= b.right + BOX_EDGE_HIT;
      const insideY = y >= b.top - BOX_EDGE_HIT && y <= b.bottom + BOX_EDGE_HIT;
      if (insideY && nearL) return { selId: s.id, edge: 'left' };
      if (insideY && nearR) return { selId: s.id, edge: 'right' };
      if (insideX && nearT) return { selId: s.id, edge: 'top' };
      if (insideX && nearB) return { selId: s.id, edge: 'bottom' };
      if (x > b.left && x < b.right && y > b.top && y < b.bottom) return { selId: s.id, edge: 'move' };
    }
    return null;
  };

  const findAnchorAt = (x, y) => {"""
content = content.replace(mouse_down_old, mouse_down_new)

# 4. Handle grid mode in mousedown
handle_md_old = """  const handleMouseDown = (e) => {
    if (e.button === 2) { e.preventDefault(); return; }

    if (panMode) {"""

handle_md_new = """  const handleMouseDown = (e) => {
    if (e.button === 2) { e.preventDefault(); return; }

    if (gridEditMode && !panMode) {
      const [x, y] = canvasCoords(e);
      const hit = hitTestBox(x, y);
      if (hit) {
        const sel = manualSelections.find(s => s.id === hit.selId);
        setActiveSelectionId?.(hit.selId);
        setBoxDrag({ selId: hit.selId, edge: hit.edge, startImg: [x, y], startBounds: { ...sel.bounds } });
        return; // do NOT place an anchor
      }
      // clicking empty space in grid mode: just deselect, don't place an anchor
      setActiveSelectionId?.(null);
      return;
    }

    if (panMode) {"""
content = content.replace(handle_md_old, handle_md_new)

# 5. Handle box drag in mousemove
handle_mm_old = """    // Pan mode or dynamically triggered pan from Place mode
    if (panStart.current && !panStart.current.isPotentialClick) {"""

handle_mm_new = """    if (boxDrag) {
      const [x, y] = canvasCoords(e);
      const dx = x - boxDrag.startImg[0];
      const dy = y - boxDrag.startImg[1];
      const W = imageDimensions?.width || Infinity;
      const H = imageDimensions?.height || Infinity;
      setManualSelections?.(prev => prev.map(s => {
        if (s.id !== boxDrag.selId) return s;
        const bb = { ...boxDrag.startBounds };
        if (boxDrag.edge === 'move') {
          bb.left += dx; bb.right += dx; bb.top += dy; bb.bottom += dy;
        } else if (boxDrag.edge === 'left') bb.left = Math.min(bb.right - 4, boxDrag.startBounds.left + dx);
        else if (boxDrag.edge === 'right') bb.right = Math.max(bb.left + 4, boxDrag.startBounds.right + dx);
        else if (boxDrag.edge === 'top') bb.top = Math.min(bb.bottom - 4, boxDrag.startBounds.top + dy);
        else if (boxDrag.edge === 'bottom') bb.bottom = Math.max(bb.top + 4, boxDrag.startBounds.bottom + dy);
        // clamp to image
        bb.left = Math.max(0, Math.min(bb.left, W)); bb.right = Math.max(0, Math.min(bb.right, W));
        bb.top = Math.max(0, Math.min(bb.top, H)); bb.bottom = Math.max(0, Math.min(bb.bottom, H));
        return { ...s, bounds: bb };
      }));
      return; // don't fall through to panning / anchor drag
    }

    // Pan mode or dynamically triggered pan from Place mode
    if (panStart.current && !panStart.current.isPotentialClick) {"""
content = content.replace(handle_mm_old, handle_mm_new)

# 6. Handle box drag in mouseup
handle_mu_old = """  const handleMouseUp = (e) => {
    if (isPanning) {"""

handle_mu_new = """  const handleMouseUp = (e) => {
    if (boxDrag) {
      setBoxDrag(null);
      onManualGridRecompute?.();
      return;
    }

    if (isPanning) {"""
content = content.replace(handle_mu_old, handle_mu_new)

# 7. Add container cursor and loupe guard
container_old = """    <div className="flex flex-col flex-1 min-h-0 bg-gray-100">"""
container_new = """    <div className={`flex flex-col flex-1 min-h-0 bg-gray-100 ${gridEditMode ? 'cursor-move' : ''}`}>"""
content = content.replace(container_old, container_new)

loupe_old = """      {/* ── CANVAS & LOUPE ───────────────────────────────────────────────── */}
      <div className="flex-1 relative overflow-hidden flex items-center justify-center bg-gray-200" ref={containerRef}>

        {/* Dynamic Loupe (Only active when NOT in panMode) */}
        {!panMode && imageRef.current && mousePos && !dragging && (() => {"""
loupe_new = """      {/* ── CANVAS & LOUPE ───────────────────────────────────────────────── */}
      <div className="flex-1 relative overflow-hidden flex items-center justify-center bg-gray-200" ref={containerRef}>

        {/* Dynamic Loupe (Only active when NOT in panMode) */}
        {!panMode && !gridEditMode && imageRef.current && mousePos && !dragging && (() => {"""
content = content.replace(loupe_old, loupe_new)

with open('frontend/src/Components/DashboardV2/HumanGuidedCurveTracker.jsx', 'w') as f:
    f.write(content)
