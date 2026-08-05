#!/usr/bin/env python3
"""
Generate 10,000 valid PCB layouts using CMA-ES.
Each valid layout is saved as a .npz file in the 'dataset/' folder.
"""

import os
import json
import time
import random
import warnings
import multiprocessing as mp
from functools import partial

import numpy as np
import cma
from numba import njit

# =============================================================================
# CORE PENALTY FUNCTION (JIT-compiled, same as original)
# =============================================================================
CG_TOLERANCE = 1.0  # mm

@njit(fastmath=True, cache=True)
def compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components,
                         flat_offsets, offset_counts, offset_indices, is_back_side,
                         clearance_dirs, panel_width, panel_height, border_spacing):
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

def evaluate(individual, masses, lengths, widths, clearance, req_d_matrix,
             flat_offsets, offset_counts, offset_indices, is_back_side,
             clearance_dirs, panel_width, panel_height, border_spacing):
    cx = individual[::2]
    cy = individual[1::2]
    num_components = len(masses)
    hl = lengths / 2.0
    hw = widths / 2.0
    return compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix,
                                num_components, flat_offsets, offset_counts,
                                offset_indices, is_back_side, clearance_dirs,
                                panel_width, panel_height, border_spacing)

# =============================================================================
# PREPARE PROBLEM (converts JSON config to arrays)
# =============================================================================
def prepare_problem(config):
    panel_width = float(config['BOARD']['Length (mm)'])
    panel_height = float(config['BOARD']['Breadth (mm)'])
    border_spacing = float(config['BOARD']['Border (mm)'])
    clearance = float(config['BOARD']['Clearance (mm)'])

    raw_components = []
    for comp in config['COMPONENTS'].get('FRONT', []):
        raw_components.append((comp, False))
    for comp in config['COMPONENTS'].get('BACK', []):
        raw_components.append((comp, True))

    components_to_process = []
    for comp, is_back in raw_components:
        qty = int(comp.get('Object qty', 1))
        if qty > 1:
            for q in range(qty):
                cloned = comp.copy()
                cloned['Unique Name'] = f"{comp['Object name']}{q}"
                components_to_process.append((cloned, is_back))
        else:
            cloned = comp.copy()
            cloned['Unique Name'] = comp['Object name']
            components_to_process.append((cloned, is_back))

    num_elements = len(components_to_process)
    element_names = []
    element_shapes = []
    is_back_side_list = []
    flat_offsets_list = []
    clearance_dirs_list = []
    element_data = []

    offset_counts = np.zeros(num_elements, dtype=np.int32)
    offset_indices = np.zeros(num_elements, dtype=np.int32)
    current_idx = 0

    for idx, (component, is_back) in enumerate(components_to_process):
        element_name = component['Unique Name']
        shape = component.get('Shape', 'rectangle')
        element_shapes.append(shape)
        is_back_side_list.append(1 if is_back else 0)

        cf_faces = component.get('CF', [])
        cf_len = float(component.get('CFLen (mm)', clearance))
        c_dirs = [clearance, clearance, clearance, clearance]
        for face in cf_faces:
            if face == 1:
                c_dirs[0] = cf_len
            elif face == 2:
                c_dirs[1] = cf_len
            elif face == 3:
                c_dirs[2] = cf_len
            elif face == 4:
                c_dirs[3] = cf_len
        clearance_dirs_list.append(c_dirs)

        if shape == 'rectangle':
            length = float(component['Length (mm)'])
            width = float(component['Breadth (mm)'])
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 4))
            half_l, half_w = length / 2.0, width / 2.0
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
            else:
                offsets = [(half_l, half_w), (half_l, -half_w),
                           (-half_l, half_w), (-half_l, -half_w)]
        else:  # circle
            diameter = float(component['Diameter (mm)'])
            length, width = diameter, diameter
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 3))
            inner_radius = max(0.0, (diameter / 2.0) - 1.0)
            offsets = [(inner_radius * np.cos(2 * np.pi * k / insert_qty),
                        inner_radius * np.sin(2 * np.pi * k / insert_qty))
                       for k in range(insert_qty)]

        flat_offsets_list.extend(offsets)
        offset_counts[idx] = len(offsets)
        offset_indices[idx] = current_idx
        current_idx += len(offsets)

        element_data.append([float(component['Weight (kg)']), length, width, insert_diam])
        element_names.append(element_name)

    flat_offsets = np.array(flat_offsets_list, dtype=np.float64)
    element_data = np.array(element_data, dtype=float)
    masses = element_data[:, 0]
    lengths = element_data[:, 1]
    widths = element_data[:, 2]
    insert_diams = element_data[:, 3]
    is_back_side = np.array(is_back_side_list, dtype=np.int32)
    clearance_dirs = np.array(clearance_dirs_list, dtype=np.float64)
    n = len(element_data)

    # Compute bounds
    lower_bounds = []
    upper_bounds = []
    for i in range(n):
        i_start = offset_indices[i]
        i_count = offset_counts[i]
        comp_offsets = flat_offsets[i_start : i_start + i_count]
        max_off_x = max(abs(x) for x, _ in comp_offsets) if i_count > 0 else 0.0
        max_off_y = max(abs(y) for _, y in comp_offsets) if i_count > 0 else 0.0

        c_top = clearance_dirs[i, 0]
        c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
        c_bot = clearance_dirs[i, 2]
        c_left = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]

        req_margin_left = max(lengths[i]/2.0 + c_left, max_off_x + insert_diams[i]/2.0)
        req_margin_right = max(lengths[i]/2.0 + c_right, max_off_x + insert_diams[i]/2.0)
        req_margin_bottom = max(widths[i]/2.0 + c_bot, max_off_y + insert_diams[i]/2.0)
        req_margin_top = max(widths[i]/2.0 + c_top, max_off_y + insert_diams[i]/2.0)

        if not is_back_side[i]:
            x_min = -panel_width / 2.0 + border_spacing + req_margin_left
            x_max =  panel_width / 2.0 - border_spacing - req_margin_right
        else:
            x_min = -panel_width / 2.0 + border_spacing + req_margin_right
            x_max =  panel_width / 2.0 - border_spacing - req_margin_left

        y_min = -panel_height / 2.0 + border_spacing + req_margin_bottom
        y_max =  panel_height / 2.0 - border_spacing - req_margin_top

        lower_bounds.extend([x_min, y_min])
        upper_bounds.extend([x_max, y_max])

    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # ====== VALIDATION: ensure lb < ub for all coordinates ======
    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Impossible bounds: component cannot fit on the board.")

    req_d_matrix = np.full((n, n), 30.0, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if insert_diams[i] == 4.0 and insert_diams[j] == 4.0:
                req_d_matrix[i, j] = 24.0
            elif insert_diams[i] == 6.0 and insert_diams[j] == 6.0:
                req_d_matrix[i, j] = 36.0

    return (masses, lengths, widths, insert_diams,
            flat_offsets, offset_counts, offset_indices,
            is_back_side, clearance_dirs, req_d_matrix,
            panel_width, panel_height, border_spacing, clearance,
            element_names, element_shapes,
            lower_bounds, upper_bounds,
            components_to_process)

# =============================================================================
# RANDOM CONFIG GENERATOR (with 1‑15 components per side)
# =============================================================================
def generate_random_config():
    width = random.uniform(400, 1200)
    height = random.uniform(300, 800)
    border = random.uniform(10, 30)
    clearance = random.uniform(10, 30)
    board = {
        "Object name": "BOARD",
        "Length (mm)": width,
        "Breadth (mm)": height,
        "Weight (kg)": 2.0,
        "Border (mm)": border,
        "Clearance (mm)": clearance
    }

    def random_component(side):
        shape = random.choice(['rectangle', 'circle'])
        comp = {
            "Object name": f"Comp_{random.randint(1000,9999)}",
            "Weight (kg)": round(random.uniform(0.05, 3.0), 3),
            "Insert (mm)": random.choice([4.0, 6.0]),
            "Shape": shape,
            "Object qty": random.randint(1, 3),
            "CF": random.sample([1,2,3,4], k=random.randint(0,4)),
            "CFLen (mm)": round(random.uniform(10, 80), 1)
        }
        if shape == 'rectangle':
            comp["Length (mm)"] = round(random.uniform(30, 300), 1)
            comp["Breadth (mm)"] = round(random.uniform(15, 250), 1)
            comp["Insert qty"] = random.choice([2,4,6])
        else:
            comp["Diameter (mm)"] = round(random.uniform(20, 200), 1)
            comp["Insert qty"] = random.choice([3,4,6])
        # ensure CFLen not too small
        if comp["CFLen (mm)"] < comp.get("Length (mm)", 50) * 0.3:
            comp["CFLen (mm)"] = comp.get("Length (mm)", 50) * 0.5
        return comp

    n_front = random.randint(1, 15)
    n_back = random.randint(1, 15)
    front_comps = [random_component('front') for _ in range(n_front)]
    back_comps = [random_component('back') for _ in range(n_back)]

    config = {
        "BOARD": board,
        "COMPONENTS": {
            "FRONT": front_comps,
            "BACK": back_comps
        }
    }

    # Quick area check: reject if total component area exceeds 60% of board
    total_area = 0.0
    for side in [front_comps, back_comps]:
        for comp in side:
            if comp['Shape'] == 'rectangle':
                area = comp['Length (mm)'] * comp['Breadth (mm)']
            else:
                r = comp['Diameter (mm)'] / 2.0
                area = np.pi * r * r
            total_area += area * comp.get('Object qty', 1)
    board_area = width * height
    if total_area > 0.6 * board_area:
        return generate_random_config()  # retry
    return config

# =============================================================================
# CMA-ES SOLVER FOR ONE CONFIG
# =============================================================================
def solve_one_config(config, max_attempts=10, maxiter=800):
    try:
        (masses, lengths, widths, insert_diams,
         flat_offsets, offset_counts, offset_indices,
         is_back_side, clearance_dirs, req_d_matrix,
         panel_width, panel_height, border_spacing, clearance,
         element_names, element_shapes,
         lower_bounds, upper_bounds,
         components_to_process) = prepare_problem(config)
    except ValueError:
        # Config cannot fit on board – skip it
        return None
    except Exception:
        return None

    n = len(masses)
    dummy_ind = np.zeros(2 * n)
    evaluate(dummy_ind, masses, lengths, widths, clearance, req_d_matrix,
             flat_offsets, offset_counts, offset_indices, is_back_side,
             clearance_dirs, panel_width, panel_height, border_spacing)

    best_sol = None
    best_score = np.inf
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        x0 = np.random.uniform(lower_bounds, upper_bounds)
        sigma0 = min(15.0, np.max(upper_bounds - lower_bounds) / 10.0)

        cma_opts = {
            'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
            'maxiter': maxiter,
            'popsize': 32,
            'tolfun': 1e-6,
            'verbose': -9,
            'seed': int(np.random.default_rng().integers(0, 2**31 - 1))
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
        while not es.stop():
            solutions = es.ask()
            scores = [evaluate(sol, masses, lengths, widths, clearance, req_d_matrix,
                               flat_offsets, offset_counts, offset_indices, is_back_side,
                               clearance_dirs, panel_width, panel_height, border_spacing)
                      for sol in solutions]
            es.tell(solutions, scores)

        cur_best = np.array(es.result.xbest)
        cur_score = es.result.fbest

        if cur_score < best_score:
            best_score = cur_score
            best_sol = cur_best

        if best_score < 1e-3:
            break

    if best_score >= 1e-3:
        return None

    # Verify CG
    cx_v = best_sol[::2]
    cy_v = best_sol[1::2]
    total_mass = np.sum(masses)
    abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
    cg_x_v = np.sum(abs_x_positions * masses) / total_mass
    cg_y_v = np.sum(cy_v * masses) / total_mass

    if abs(cg_x_v) > CG_TOLERANCE or abs(cg_y_v) > CG_TOLERANCE:
        return None

    result = {
        'success': True,
        'config': config,
        'best_sol': best_sol,
        'penalty': best_score,
        'cg_x': cg_x_v,
        'cg_y': cg_y_v,
        'num_components': n,
        'element_names': element_names,
        'masses': masses,
        'lengths': lengths,
        'widths': widths,
        'insert_diams': insert_diams,
        'is_back_side': is_back_side,
        'panel_width': panel_width,
        'panel_height': panel_height,
        'border_spacing': border_spacing,
        'clearance': clearance,
        'flat_offsets': flat_offsets,
        'offset_counts': offset_counts,
        'offset_indices': offset_indices,
        'clearance_dirs': clearance_dirs,
        'req_d_matrix': req_d_matrix,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
    }
    return result

# =============================================================================
# MAIN: GENERATE 10,000 LAYOUTS IN PARALLEL
# =============================================================================
def main():
    warnings.filterwarnings("ignore")
    total_needed = 10000
    save_dir = "dataset"
    os.makedirs(save_dir, exist_ok=True)

    max_attempts = 10
    maxiter = 800

    print(f"Generating {total_needed} valid layouts using {mp.cpu_count()} cores...")
    pool = mp.Pool(processes=mp.cpu_count())
    solve_func = partial(solve_one_config, max_attempts=max_attempts, maxiter=maxiter)

    def task_generator():
        while True:
            yield generate_random_config()

    collected = 0
    start = time.time()
    for result in pool.imap_unordered(solve_func, task_generator()):
        if result is None:
            continue
        # Save
        filename = f"layout_{collected+1:06d}.npz"
        filepath = os.path.join(save_dir, filename)
        config_json = json.dumps(result['config'])
        save_dict = {
            'config_json': config_json,
            'best_sol': result['best_sol'],
            'penalty': result['penalty'],
            'cg_x': result['cg_x'],
            'cg_y': result['cg_y'],
            'num_components': result['num_components'],
            'element_names': np.array(result['element_names'], dtype=object),
            'masses': result['masses'],
            'lengths': result['lengths'],
            'widths': result['widths'],
            'insert_diams': result['insert_diams'],
            'is_back_side': result['is_back_side'],
            'panel_width': result['panel_width'],
            'panel_height': result['panel_height'],
            'border_spacing': result['border_spacing'],
            'clearance': result['clearance'],
            'flat_offsets': result['flat_offsets'],
            'offset_counts': result['offset_counts'],
            'offset_indices': result['offset_indices'],
            'clearance_dirs': result['clearance_dirs'],
            'req_d_matrix': result['req_d_matrix'],
            'lower_bounds': result['lower_bounds'],
            'upper_bounds': result['upper_bounds'],
        }
        np.savez_compressed(filepath, **save_dict)

        collected += 1
        elapsed = time.time() - start
        rate = collected / elapsed if elapsed > 0 else 0
        print(f"Saved {collected}/{total_needed} ({rate:.2f} instances/s)")

        if collected >= total_needed:
            break

    pool.terminate()
    pool.join()
    print(f"\n✅ Done. {collected} layouts saved in '{save_dir}'")
    print(f"Total time: {(time.time()-start)/60:.2f} minutes.")

if __name__ == '__main__':
    main()