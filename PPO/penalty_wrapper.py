"""
penalty_wrapper.py
-------------------
Phase 1: Penalty wrapper – load board config, compile fitness core, compute penalty.
Reuses the Numba-compiled `compute_fitness_core` from the original CMA-ES code.
"""

import json
import os
import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# 1.  Global tolerance (same as in new.py)
# --------------------------------------------------------------------------
CG_TOLERANCE = 1.0   # mm

# --------------------------------------------------------------------------
# 2.  Numba‑compiled core fitness function (copied from new.py)
#     This is the exact same function used by CMA‑ES.
# --------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components,
                         flat_offsets, offset_counts, offset_indices, is_back_side,
                         clearance_dirs, panel_width, panel_height, border_spacing):
    """
    Returns total penalty (0 = perfect).  All arrays are NumPy arrays of float64/int32.
    """
    total_mass = 0.0
    cg_x = 0.0
    cg_y = 0.0
    abs_x = np.zeros(num_components)

    for i in range(num_components):
        m = masses[i]
        total_mass += m
        if is_back_side[i]:
            abs_x[i] = -cx[i]
        else:
            abs_x[i] = cx[i]
        cg_x += abs_x[i] * m
        cg_y += cy[i] * m

    if total_mass == 0.0:
        return 1e15

    cg_x /= total_mass
    cg_y /= total_mass

    PENALTY_WEIGHT = 1e8

    active_x_min = -panel_width / 2.0 + border_spacing
    active_x_max =  panel_width / 2.0 - border_spacing
    active_y_min = -panel_height / 2.0 + border_spacing
    active_y_max =  panel_height / 2.0 - border_spacing

    cg_offset = np.sqrt(cg_x**2 + cg_y**2)
    cg_penalty = 0.0
    if cg_offset > CG_TOLERANCE:
        cg_penalty = ((cg_offset - CG_TOLERANCE) ** 2) * PENALTY_WEIGHT

    overlap_penalty = 0.0
    border_penalty = 0.0

    for i in range(num_components):
        c_top = clearance_dirs[i, 0]
        c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
        c_bot = clearance_dirs[i, 2]
        c_left = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]

        i_x_min = abs_x[i] - hl[i] - c_left
        i_x_max = abs_x[i] + hl[i] + c_right
        i_y_min = cy[i] - hw[i] - c_bot
        i_y_max = cy[i] + hw[i] + c_top

        viol_left   = max(0.0, active_x_min - i_x_min)
        viol_right  = max(0.0, i_x_max - active_x_max)
        viol_bottom = max(0.0, active_y_min - i_y_min)
        viol_top    = max(0.0, i_y_max - active_y_max)
        total_border_viol = viol_left + viol_right + viol_bottom + viol_top
        if total_border_viol > 0.0:
            border_penalty += (total_border_viol ** 2) * PENALTY_WEIGHT

        for j in range(i + 1, num_components):
            if is_back_side[i] == is_back_side[j]:
                cj_top = clearance_dirs[j, 0]
                cj_right = clearance_dirs[j, 3] if is_back_side[j] else clearance_dirs[j, 1]
                cj_bot = clearance_dirs[j, 2]
                cj_left = clearance_dirs[j, 1] if is_back_side[j] else clearance_dirs[j, 3]

                j_x_min = abs_x[j] - hl[j] - cj_left
                j_x_max = abs_x[j] + hl[j] + cj_right
                j_y_min = cy[j] - hw[j] - cj_bot
                j_y_max = cy[j] + hw[j] + cj_top

                overlap_x = max(0.0, min(i_x_max, j_x_max) - max(i_x_min, j_x_min))
                overlap_y = max(0.0, min(i_y_max, j_y_max) - max(i_y_min, j_y_min))
                if overlap_x > 0.0 and overlap_y > 0.0:
                    overlap_area = overlap_x * overlap_y
                    overlap_penalty += (overlap_area ** 2) * PENALTY_WEIGHT

    insert_penalty = 0.0
    for i in range(num_components):
        i_start = offset_indices[i]
        i_count = offset_counts[i]
        for j in range(i + 1, num_components):
            j_start = offset_indices[j]
            j_count = offset_counts[j]
            req_dist = req_d_matrix[i, j]
            for ii in range(i_count):
                idx_i = i_start + ii
                xi_abs = abs_x[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                yi_abs = cy[i] + flat_offsets[idx_i, 1]
                for jj in range(j_count):
                    idx_j = j_start + jj
                    xj_abs = abs_x[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                    yj_abs = cy[j] + flat_offsets[idx_j, 1]
                    dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                    if req_dist > dist:
                        insert_penalty += ((req_dist - dist) ** 2) * PENALTY_WEIGHT

    return cg_penalty + overlap_penalty + border_penalty + insert_penalty


# --------------------------------------------------------------------------
# 3.  Board configuration loader (mimics the parsing in new.py)
# --------------------------------------------------------------------------
def load_board_config(json_path):
    """
    Parse the JSON board file and return a dictionary with all static attributes.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Board config not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    panel_width = float(data['BOARD']['Length (mm)'])
    panel_height = float(data['BOARD']['Breadth (mm)'])
    border_spacing = float(data['BOARD']['Border (mm)'])
    clearance = float(data['BOARD']['Clearance (mm)'])

    # Flatten components (duplicate if quantity > 1)
    raw = []
    for comp in data['COMPONENTS'].get('FRONT', []):
        raw.append((comp, False))
    for comp in data['COMPONENTS'].get('BACK', []):
        raw.append((comp, True))

    components_to_process = []
    for comp, is_back in raw:
        qty = int(comp.get('Object qty', 1))
        name_base = comp['Object name']
        for q in range(qty):
            cloned = comp.copy()
            cloned['Unique Name'] = f"{name_base}{q}" if qty > 1 else name_base
            components_to_process.append((cloned, is_back))

    num_elements = len(components_to_process)

    # Pre‑allocate arrays
    masses = np.zeros(num_elements, dtype=np.float64)
    lengths = np.zeros(num_elements, dtype=np.float64)
    widths = np.zeros(num_elements, dtype=np.float64)
    insert_diams = np.zeros(num_elements, dtype=np.float64)
    is_back_side = np.zeros(num_elements, dtype=np.int32)
    clearance_dirs = np.zeros((num_elements, 4), dtype=np.float64)
    offset_counts = np.zeros(num_elements, dtype=np.int32)
    offset_indices = np.zeros(num_elements, dtype=np.int32)
    element_names = []
    element_shapes = []

    flat_offsets_list = []
    current_idx = 0

    for idx, (component, is_back) in enumerate(components_to_process):
        element_names.append(component['Unique Name'])
        shape = component.get('Shape', 'rectangle')
        element_shapes.append(shape)
        is_back_side[idx] = 1 if is_back else 0

        # Clearance directions
        cf_faces = component.get('CF', [])
        cf_len = float(component.get('CFLen (mm)', clearance))
        c_dirs = [clearance, clearance, clearance, clearance]
        for face in cf_faces:
            if face == 1:
                c_dirs[0] = cf_len   # Top
            elif face == 2:
                c_dirs[1] = cf_len   # Right
            elif face == 3:
                c_dirs[2] = cf_len   # Bottom
            elif face == 4:
                c_dirs[3] = cf_len   # Left
        clearance_dirs[idx] = c_dirs

        # Dimensions and insert
        if shape == 'rectangle':
            length = float(component['Length (mm)'])
            width = float(component['Breadth (mm)'])
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 4))
            half_l, half_w = length/2.0, width/2.0
            if insert_qty == 2:
                if length >= width:
                    offsets = [(half_l, 0.0), (-half_l, 0.0)]
                else:
                    offsets = [(0.0, half_w), (0.0, -half_w)]
            elif insert_qty == 6:
                offsets = [(half_l, half_w), (half_l, -half_w),
                           (-half_l, half_w), (-half_l, -half_w)]
                if length >= width:
                    offsets.extend([(0.0, half_w), (0.0, -half_w)])
                else:
                    offsets.extend([(half_l, 0.0), (-half_l, 0.0)])
            else:   # default 4 corners
                offsets = [(half_l, half_w), (half_l, -half_w),
                           (-half_l, half_w), (-half_l, -half_w)]
        else:   # circle
            diameter = float(component['Diameter (mm)'])
            length, width = diameter, diameter
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 3))
            inner_radius = max(0.0, (diameter/2.0) - 1.0)
            offsets = [(inner_radius * np.cos(2*np.pi*k/insert_qty),
                        inner_radius * np.sin(2*np.pi*k/insert_qty))
                       for k in range(insert_qty)]

        masses[idx] = float(component['Weight (kg)'])
        lengths[idx] = length
        widths[idx] = width
        insert_diams[idx] = insert_diam
        offset_counts[idx] = len(offsets)
        offset_indices[idx] = current_idx
        flat_offsets_list.extend(offsets)
        current_idx += len(offsets)

    flat_offsets = np.array(flat_offsets_list, dtype=np.float64)

    # Required pin‑to‑pin distance matrix (same heuristic as new.py)
    req_d_matrix = np.full((num_elements, num_elements), 30.0, dtype=np.float64)
    for i in range(num_elements):
        for j in range(num_elements):
            if insert_diams[i] == 4.0 and insert_diams[j] == 4.0:
                req_d_matrix[i, j] = 24.0
            elif insert_diams[i] == 6.0 and insert_diams[j] == 6.0:
                req_d_matrix[i, j] = 36.0

    # Half dimensions
    hl = lengths / 2.0
    hw = widths / 2.0

    # Boundary limits (for reference, not used by penalty core)
    # (computed similarly to new.py, but we keep them for environment)

    config = {
        'panel_width': panel_width,
        'panel_height': panel_height,
        'border_spacing': border_spacing,
        'clearance': clearance,
        'num_components': num_elements,
        'masses': masses,
        'lengths': lengths,
        'widths': widths,
        'hl': hl,
        'hw': hw,
        'insert_diams': insert_diams,
        'is_back_side': is_back_side,
        'clearance_dirs': clearance_dirs,
        'flat_offsets': flat_offsets,
        'offset_counts': offset_counts,
        'offset_indices': offset_indices,
        'req_d_matrix': req_d_matrix,
        'element_names': element_names,
        'element_shapes': element_shapes,
        # additionally, store the raw data for possible later use
        'components_raw': components_to_process,
    }
    return config


# --------------------------------------------------------------------------
# 4.  Penalty computation wrapper
# --------------------------------------------------------------------------
def compute_penalty(x, y, config):
    """
    Compute total penalty given arrays x, y (length = num_components)
    and a configuration dictionary from load_board_config().
    """
    return compute_fitness_core(
        x, y,
        config['masses'],
        config['hl'],
        config['hw'],
        config['clearance'],
        config['req_d_matrix'],
        config['num_components'],
        config['flat_offsets'],
        config['offset_counts'],
        config['offset_indices'],
        config['is_back_side'],
        config['clearance_dirs'],
        config['panel_width'],
        config['panel_height'],
        config['border_spacing']
    )


# --------------------------------------------------------------------------
# 5.  Quick test (if run as main)
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Example: load newobj.json (assumed in same directory)
    cfg = load_board_config('newobj.json')
    n = cfg['num_components']
    # random positions within bounds (just for testing)
    x = np.random.uniform(-200, 200, n)
    y = np.random.uniform(-150, 150, n)
    penalty = compute_penalty(x, y, cfg)
    print(f"Computed penalty for random layout: {penalty:.2f}")