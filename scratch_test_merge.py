import numpy as np

def merge_curve_pieces(lines_dict, depth_step=0.5):
    """
    Groups pieces by base mnemonic (stripping trailing _X suffix) and merges
    them into a single (depths, values) array per group with a priority rule:
    Higher piece index (e.g. _2 > _1) overwrites lower piece at same depth.
    """
    groups = {}
    for line_name, (depths, values) in lines_dict.items():
        base_name = line_name.split('_')[0]
        try:
            piece_idx = int(line_name.split('_')[-1]) if '_' in line_name else 0
        except ValueError:
            piece_idx = 0
            
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append((piece_idx, line_name, np.array(depths, dtype=float), np.array(values, dtype=float)))
        
    merged_dict = {}
    for base_name, pieces in groups.items():
        if len(pieces) == 1:
            # Single piece: just pass it through
            _, line_name, depths, values = pieces[0]
            merged_dict[line_name] = (depths.tolist(), values.tolist())
            continue
            
        # Multiple pieces: we need to resample them onto a common grid FIRST
        # Sort pieces by piece index descending
        pieces.sort(key=lambda x: x[0], reverse=True)
        
        # Find global depth range
        all_depths = np.concatenate([p[2] for p in pieces])
        start_depth = float(np.nanmin(all_depths))
        stop_depth = float(np.nanmax(all_depths))
        
        # We need a reference depth array
        ref_depth_array = np.arange(start_depth, stop_depth + depth_step * 0.5, depth_step, dtype=float)
        merged_values = np.full_like(ref_depth_array, -999.25)
        
        from scipy.interpolate import interp1d
        
        # We process from lowest priority (piece 0) to highest priority (piece N)
        # Wait, the rule says: "where piece 2 exists, piece 2 wins"
        # We process from lowest to highest, overwriting
        pieces.sort(key=lambda x: x[0])
        for piece_idx, line_name, depths, values in pieces:
            valid = np.isfinite(depths) & np.isfinite(values)
            if not np.any(valid): continue
            
            d = depths[valid]
            v = values[valid]
            order = np.argsort(d)
            d, v = d[order], v[order]
            
            unique_d, inverse = np.unique(np.round(d, 6), return_inverse=True)
            mean_v = np.zeros_like(unique_d, dtype=float)
            for uidx in range(len(unique_d)):
                mean_v[uidx] = float(np.nanmedian(v[inverse == uidx]))
                
            if len(unique_d) == 1:
                # Find closest ref depth
                diff = np.abs(ref_depth_array - unique_d[0])
                if np.min(diff) <= depth_step:
                    idx = np.argmin(diff)
                    merged_values[idx] = mean_v[0]
            else:
                f_interp = interp1d(unique_d, mean_v, bounds_error=False, fill_value=np.nan)
                interp_v = f_interp(ref_depth_array)
                
                # Replace where interp_v is not nan
                mask = ~np.isnan(interp_v)
                merged_values[mask] = interp_v[mask]
                
        # Keep the base piece name
        merged_dict[base_name] = (ref_depth_array.tolist(), merged_values.tolist())
        
    return merged_dict

# Test it
lines = {
    "GR": ([10, 20, 30], [50, 60, 70]),
    "GR_2": ([20, 30, 40], [600, 700, 800])
}
m = merge_curve_pieces(lines, 10.0)
print(m)

