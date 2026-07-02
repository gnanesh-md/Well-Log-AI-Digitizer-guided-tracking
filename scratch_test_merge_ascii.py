import numpy as np

def merge_curve_pieces(lines_dict, depth_step=0.5):
    import numpy as np
    from scipy.interpolate import interp1d
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
            _, line_name, depths, values = pieces[0]
            merged_dict[line_name] = (depths.tolist(), values.tolist())
            continue
            
        pieces.sort(key=lambda x: x[0], reverse=True)
        all_depths = np.concatenate([p[2] for p in pieces])
        start_depth = float(np.nanmin(all_depths))
        stop_depth = float(np.nanmax(all_depths))
        
        ref_depth_array = np.arange(start_depth, stop_depth + depth_step * 0.5, depth_step, dtype=float)
        merged_values = np.full_like(ref_depth_array, -999.25)
        
        pieces.sort(key=lambda x: x[0])
        for piece_idx, line_name, depths, values in pieces:
            valid = np.isfinite(depths) & np.isfinite(values)
            if not np.any(valid): continue
            
            d, v = depths[valid], values[valid]
            order = np.argsort(d)
            d, v = d[order], v[order]
            
            unique_d, inverse = np.unique(np.round(d, 6), return_inverse=True)
            mean_v = np.zeros_like(unique_d, dtype=float)
            for uidx in range(len(unique_d)):
                mean_v[uidx] = float(np.nanmedian(v[inverse == uidx]))
                
            if len(unique_d) == 1:
                diff = np.abs(ref_depth_array - unique_d[0])
                if np.min(diff) <= depth_step:
                    idx = np.argmin(diff)
                    merged_values[idx] = mean_v[0]
            else:
                f_interp = interp1d(unique_d, mean_v, bounds_error=False, fill_value=np.nan)
                interp_v = f_interp(ref_depth_array)
                mask = ~np.isnan(interp_v)
                merged_values[mask] = interp_v[mask]
                
        merged_dict[base_name] = (ref_depth_array.tolist(), merged_values.tolist())
        
    return merged_dict

print("BEFORE MERGING:")
lines = {
    "RES_1": ([10, 20, 30, 40], [50, 60, 70, 80]),
    "RES_2": ([30, 40, 50], [600, 700, 800])
}
# A simple mock to show what happens before merging:
for k, v in lines.items():
    print(f"Curve {k}")
    for d, val in zip(v[0], v[1]):
        print(f"{d:6.2f} {val:6.2f}")

print("\nAFTER MERGING:")
merged = merge_curve_pieces(lines, 10.0)
for k, v in merged.items():
    print(f"Curve {k}")
    for d, val in zip(v[0], v[1]):
        print(f"{d:6.2f} {val:8.2f}")

