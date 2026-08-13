#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative study of PCB placement algorithms:
  - PPO (pre‑trained, inference only)
  - Differential Evolution (SciPy)
  - CMA‑ES (cma package)

IMPORTANT: Make sure there is NO local file named 'cma.py' in this directory.
           Rename or delete it if present.
Usage: python comp.py --runs 30 --episodes 1000
"""

import os
import sys
import time
import json
import warnings
import argparse
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from scipy.stats import wilcoxon
from numba import njit
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# ---------- CMA-ES import with fallback ----------
try:
    from cma import CMAEvolutionStrategy
except ImportError:
    # If the import fails (e.g., local cma.py exists), try to import the module
    import cma
    if hasattr(cma, 'CMAEvolutionStrategy'):
        CMAEvolutionStrategy = cma.CMAEvolutionStrategy
    else:
        raise ImportError(
            "Could not import CMAEvolutionStrategy. "
            "Please rename or remove any local 'cma.py' file in this directory."
        )

warnings.filterwarnings("ignore")

# -------------------- Configuration --------------------
CG_TOLERANCE = 1.0
MAX_STEP_MM = 12.0
MIN_STEP_MM = 0.25
MAX_STEPS = 250
STEP_COST = 0.01
SUCCESS_BONUS = 50.0
SCORE_WEIGHTS = {"border": 3.0, "overlap": 3.0, "insert": 10.0, "cg": 0.5}
VALID_EPS = 1e-6
PENALTY_WEIGHT = 1e8   # used in DE/CMA-ES

# -------------------- Command line arguments --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Comparative study of PCB placement algorithms")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of independent runs for DE and CMA-ES (default 30)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to run PPO inference (default 1000)")
    parser.add_argument("--output_dir", default="comparison_results",
                        help="Directory to save all results and figures")
    parser.add_argument("--json", default="newobj.json", help="Path to newobj.json")
    parser.add_argument("--model_dir", default=".",
                        help="Directory containing ppo_pcb_placement.zip and vecnormalize.pkl")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    return parser.parse_args()

# -------------------- Load config and build component arrays --------------------
def load_config(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"[✓] Config loaded: '{filepath}'")
    return data

def build_arrays(data):
    panel_width = float(data['BOARD']['Length (mm)'])
    panel_height = float(data['BOARD']['Breadth (mm)'])
    border_spacing = float(data['BOARD']['Border (mm)'])
    clearance = float(data['BOARD']['Clearance (mm)'])

    raw_components_to_process = []
    for comp in data['COMPONENTS'].get('FRONT', []):
        raw_components_to_process.append((comp, False))
    for comp in data['COMPONENTS'].get('BACK', []):
        raw_components_to_process.append((comp, True))

    components_to_process = []
    for comp, is_back in raw_components_to_process:
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
    element_data = []  # [mass, length, width, insert_diam]
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
        c_dirs = [clearance, clearance, clearance, clearance]  # [Top, Right, Bottom, Left]
        for face in cf_faces:
            if face == 1: c_dirs[0] = cf_len
            elif face == 2: c_dirs[1] = cf_len
            elif face == 3: c_dirs[2] = cf_len
            elif face == 4: c_dirs[3] = cf_len
        clearance_dirs_list.append(c_dirs)

        if shape == 'rectangle':
            length = float(component['Length (mm)'])
            width = float(component['Breadth (mm)'])
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 4))
            half_l, half_w = length / 2.0, width / 2.0
            if insert_qty == 2:
                offsets = [(half_l, 0.0), (-half_l, 0.0)] if length >= width else [(0.0, half_w), (0.0, -half_w)]
            elif insert_qty == 6:
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
                offsets.extend([(0.0, half_w), (0.0, -half_w)] if length >= width else [(half_l, 0.0), (-half_l, 0.0)])
            else:
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
        else:
            diameter = float(component['Diameter (mm)'])
            length, width = diameter, diameter
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 3))
            inner_radius = max(0.0, (diameter / 2.0) - 1.0)
            offsets = [(inner_radius * np.cos(2*np.pi*k/insert_qty),
                        inner_radius * np.sin(2*np.pi*k/insert_qty))
                       for k in range(insert_qty)]

        flat_offsets_list.extend(offsets)
        offset_counts[idx] = len(offsets)
        offset_indices[idx] = current_idx
        current_idx += len(offsets)
        element_data.append([float(component['Weight (kg)']), length, width, insert_diam])
        element_names.append(element_name)

    flat_offsets = np.array(flat_offsets_list, dtype=np.float64)
    element_data = np.array(element_data, dtype=np.float64)
    masses = element_data[:, 0]
    lengths = element_data[:, 1]
    widths = element_data[:, 2]
    insert_diams = element_data[:, 3]
    is_back_side = np.array(is_back_side_list, dtype=np.int32)
    clearance_dirs = np.array(clearance_dirs_list, dtype=np.float64)
    n = num_elements

    req_d_matrix = np.full((n, n), 30.0)
    for i in range(n):
        for j in range(n):
            if insert_diams[i] == 4.0 and insert_diams[j] == 4.0:
                req_d_matrix[i, j] = 24.0
            elif insert_diams[i] == 6.0 and insert_diams[j] == 6.0:
                req_d_matrix[i, j] = 36.0

    print(f"[✓] {n} components parsed ({int(is_back_side.sum())} back-side, {n - int(is_back_side.sum())} front-side)")
    return (panel_width, panel_height, border_spacing, clearance,
            n, element_names, element_shapes, is_back_side, flat_offsets,
            offset_counts, offset_indices, masses, lengths, widths,
            insert_diams, clearance_dirs, req_d_matrix)

# -------------------- Numba penalty engine --------------------
@njit(fastmath=True, cache=True)
def compute_penalty_breakdown(cx, cy, masses, hl, hw, req_d_matrix, num_components,
                               flat_offsets, offset_counts, offset_indices, is_back_side,
                               clearance_dirs, panel_width, panel_height, border_spacing,
                               CG_TOLERANCE=1.0):
    total_mass = 0.0
    cg_x = 0.0
    cg_y = 0.0
    abs_x = np.zeros(num_components)
    for i in range(num_components):
        m = masses[i]
        total_mass += m
        abs_x[i] = -cx[i] if is_back_side[i] else cx[i]
        cg_x += abs_x[i] * m
        cg_y += cy[i] * m
    if total_mass == 0.0:
        return 1e15, 1e15, 1e15, 1e15
    cg_x /= total_mass
    cg_y /= total_mass

    active_x_min = -panel_width/2.0 + border_spacing
    active_x_max =  panel_width/2.0 - border_spacing
    active_y_min = -panel_height/2.0 + border_spacing
    active_y_max =  panel_height/2.0 - border_spacing

    cg_offset = np.sqrt(cg_x**2 + cg_y**2)
    cg_penalty = 0.0
    if cg_offset > CG_TOLERANCE:
        cg_penalty = (cg_offset - CG_TOLERANCE) ** 2

    overlap_penalty = 0.0
    border_penalty = 0.0

    for i in range(num_components):
        c_top   = clearance_dirs[i, 0]
        c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
        c_bot   = clearance_dirs[i, 2]
        c_left  = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]

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
            border_penalty += total_border_viol ** 2

        for j in range(i + 1, num_components):
            if is_back_side[i] == is_back_side[j]:
                cj_top   = clearance_dirs[j, 0]
                cj_right = clearance_dirs[j, 3] if is_back_side[j] else clearance_dirs[j, 1]
                cj_bot   = clearance_dirs[j, 2]
                cj_left  = clearance_dirs[j, 1] if is_back_side[j] else clearance_dirs[j, 3]

                j_x_min = abs_x[j] - hl[j] - cj_left
                j_x_max = abs_x[j] + hl[j] + cj_right
                j_y_min = cy[j] - hw[j] - cj_bot
                j_y_max = cy[j] + hw[j] + cj_top

                overlap_x = max(0.0, min(i_x_max, j_x_max) - max(i_x_min, j_x_min))
                overlap_y = max(0.0, min(i_y_max, j_y_max) - max(i_y_min, j_y_min))

                if overlap_x > 0.0 and overlap_y > 0.0:
                    overlap_area = overlap_x * overlap_y
                    overlap_penalty += overlap_area ** 2

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
                        insert_penalty += (req_dist - dist) ** 2

    return border_penalty, overlap_penalty, insert_penalty, cg_penalty

def evaluate_layout(cx, cy, params):
    """Return breakdown dict and total penalty."""
    (panel_width, panel_height, border_spacing, clearance,
     n, element_names, element_shapes, is_back_side, flat_offsets,
     offset_counts, offset_indices, masses, lengths, widths,
     insert_diams, clearance_dirs, req_d_matrix) = params
    hl = lengths / 2.0
    hw = widths / 2.0
    b, o, ins, cg = compute_penalty_breakdown(
        cx, cy, masses, hl, hw, req_d_matrix, n,
        flat_offsets, offset_counts, offset_indices, is_back_side,
        clearance_dirs, panel_width, panel_height, border_spacing,
        CG_TOLERANCE=1.0
    )
    weighted = (SCORE_WEIGHTS["border"] * np.log1p(b) +
                SCORE_WEIGHTS["overlap"] * np.log1p(o) +
                SCORE_WEIGHTS["insert"] * np.log1p(ins) +
                SCORE_WEIGHTS["cg"] * np.log1p(cg))
    raw_penalty = b + o + ins + cg
    breakdown = {"border": b, "overlap": o, "insert": ins, "cg": cg,
                 "weighted_score": weighted, "raw_penalty": raw_penalty}
    return breakdown

def is_fully_valid(breakdown):
    return (breakdown["border"] < VALID_EPS and
            breakdown["overlap"] < VALID_EPS and
            breakdown["insert"] < VALID_EPS and
            breakdown["cg"] < VALID_EPS)

# -------------------- Fitness functions (module level for pickling) --------------------
def de_fitness(x, params):
    cx = x[::2]
    cy = x[1::2]
    bd = evaluate_layout(cx, cy, params)
    return bd["raw_penalty"]

def cma_fitness(x, params):
    cx = x[::2]
    cy = x[1::2]
    bd = evaluate_layout(cx, cy, params)
    return bd["raw_penalty"]

# -------------------- DE optimizer --------------------
def run_de_single(seed, params, bounds, maxiter=600, popsize=25, tol=1e-6):
    fitness = partial(de_fitness, params=params)
    history = []

    def callback(xk, convergence):
        history.append(fitness(xk))

    t0 = time.time()
    result = differential_evolution(
        fitness, bounds,
        maxiter=maxiter, popsize=popsize, tol=tol,
        strategy='best1bin', seed=seed,
        workers=-1, updating='deferred',
        callback=callback
    )
    elapsed = time.time() - t0

    best_x = result.x
    best_fitness = result.fun
    cx_best = best_x[::2]
    cy_best = best_x[1::2]
    breakdown = evaluate_layout(cx_best, cy_best, params)

    return {
        "best_x": best_x,
        "best_fitness": best_fitness,
        "breakdown": breakdown,
        "history": history,
        "time": elapsed,
        "success": is_fully_valid(breakdown)
    }

# -------------------- CMA-ES optimizer --------------------
def run_cma_single(seed, params, bounds, sigma0=15.0, maxiter=600, popsize=32):
    (panel_width, panel_height, border_spacing, clearance,
     n, element_names, element_shapes, is_back_side, flat_offsets,
     offset_counts, offset_indices, masses, lengths, widths,
     insert_diams, clearance_dirs, req_d_matrix) = params

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    x0 = np.random.RandomState(seed).uniform(lower, upper)

    fitness = partial(cma_fitness, params=params)

    cma_opts = {
        'bounds': [lower.tolist(), upper.tolist()],
        'maxiter': maxiter,
        'popsize': popsize,
        'tolfun': 1e-6,
        'verbose': -9,
        'seed': seed
    }

    es = CMAEvolutionStrategy(x0, sigma0, cma_opts)   # now defined globally
    history = []
    t0 = time.time()
    while not es.stop():
        solutions = es.ask()
        scores = [fitness(sol) for sol in solutions]
        es.tell(solutions, scores)
        history.append(es.result.fbest)
    elapsed = time.time() - t0

    best_x = np.array(es.result.xbest)
    best_fitness = es.result.fbest
    cx_best = best_x[::2]
    cy_best = best_x[1::2]
    breakdown = evaluate_layout(cx_best, cy_best, params)

    return {
        "best_x": best_x,
        "best_fitness": best_fitness,
        "breakdown": breakdown,
        "history": history,
        "time": elapsed,
        "success": is_fully_valid(breakdown)
    }

# -------------------- PPO inference runner --------------------
def run_ppo_inference(episodes, params, model_path, vecnorm_path, parallel=8, deterministic=True, seed=42):
    try:
        from ppo import PCBPlacementEnv
    except ImportError:
        print("Error: ppo.py not found. Please ensure it's in the same directory.")
        sys.exit(1)

    def make_env():
        env = PCBPlacementEnv(params)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([make_env for _ in range(parallel)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    venv.seed(seed + 1000)

    model = PPO.load(model_path, env=venv, device="auto")
    print(f"[✓] PPO model loaded from {model_path}")

    obs = venv.reset()
    candidates = []
    episodes_done = 0
    t0 = time.time()

    while episodes_done < episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = venv.step(action)
        for i, info in enumerate(infos):
            if done[i]:
                episodes_done += 1
                cx, cy, breakdown = info.get("cx"), info.get("cy"), info.get("breakdown")
                if cx is not None and is_fully_valid(breakdown):
                    verify = evaluate_layout(cx, cy, params)
                    if is_fully_valid(verify):
                        candidates.append({
                            "cx": cx, "cy": cy,
                            "breakdown": verify,
                            "episode": episodes_done
                        })

    total_time = time.time() - t0
    success_rate = len(candidates) / episodes if episodes > 0 else 0

    return {
        "candidates": candidates,
        "success_rate": success_rate,
        "total_time": total_time,
        "episodes": episodes,
        "num_valid": len(candidates)
    }

# -------------------- Main comparison driver --------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_config(args.json)
    params = build_arrays(data)
    (panel_width, panel_height, border_spacing, clearance,
     n, element_names, element_shapes, is_back_side, flat_offsets,
     offset_counts, offset_indices, masses, lengths, widths,
     insert_diams, clearance_dirs, req_d_matrix) = params

    # Build bounds
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
            x_min = -panel_width/2.0 + border_spacing + req_margin_left
            x_max =  panel_width/2.0 - border_spacing - req_margin_right
        else:
            x_min = -panel_width/2.0 + border_spacing + req_margin_right
            x_max =  panel_width/2.0 - border_spacing - req_margin_left
        y_min = -panel_height/2.0 + border_spacing + req_margin_bottom
        y_max =  panel_height/2.0 - border_spacing - req_margin_top
        lower_bounds.append(x_min)
        lower_bounds.append(y_min)
        upper_bounds.append(x_max)
        upper_bounds.append(y_max)
    bounds = list(zip(lower_bounds, upper_bounds))

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. DE
    print(f"\n🔬 Running DE with {args.runs} seeds...")
    de_results = []
    for seed in range(args.runs):
        print(f"  DE seed {seed+1}/{args.runs}")
        res = run_de_single(seed, params, bounds)
        de_results.append(res)

    # 2. CMA-ES
    print(f"\n🔬 Running CMA-ES with {args.runs} seeds...")
    cma_results = []
    for seed in range(args.runs):
        print(f"  CMA-ES seed {seed+1}/{args.runs}")
        res = run_cma_single(seed, params, bounds)
        cma_results.append(res)

    # 3. PPO
    print(f"\n🔬 Running PPO inference for {args.episodes} episodes...")
    model_path = os.path.join(args.model_dir, "ppo_pcb_placement")
    vecnorm_path = os.path.join(args.model_dir, "vecnormalize.pkl")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"PPO model not found: {model_path}.zip")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize stats not found: {vecnorm_path}")

    ppo_result = run_ppo_inference(
        episodes=args.episodes,
        params=params,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        parallel=8,
        deterministic=True,
        seed=args.seed
    )

    # -------------------- Collect statistics --------------------
    def gather_breakdowns(results):
        return [r["breakdown"] for r in results]

    def success_rate(results):
        return np.mean([r["success"] for r in results])

    de_breakdowns = gather_breakdowns(de_results)
    de_success = success_rate(de_results)
    de_times = [r["time"] for r in de_results]
    de_best_fitness = [r["best_fitness"] for r in de_results]

    cma_breakdowns = gather_breakdowns(cma_results)
    cma_success = success_rate(cma_results)
    cma_times = [r["time"] for r in cma_results]
    cma_best_fitness = [r["best_fitness"] for r in cma_results]

    ppo_breakdowns = [c["breakdown"] for c in ppo_result["candidates"]]
    ppo_success = ppo_result["success_rate"]
    ppo_best_fitness = [bd["raw_penalty"] for bd in ppo_breakdowns]

    # DataFrames for plotting
    def breakdown_to_df(breakdowns, alg_name):
        df = pd.DataFrame(breakdowns)
        df['algorithm'] = alg_name
        return df

    df_de = breakdown_to_df(de_breakdowns, 'DE')
    df_cma = breakdown_to_df(cma_breakdowns, 'CMA-ES')
    df_ppo = breakdown_to_df(ppo_breakdowns, 'PPO')
    all_df = pd.concat([df_de, df_cma, df_ppo], ignore_index=True)

    # -------------------- Generate plots --------------------
    print("\n📊 Generating plots...")

    # 1. Box plots of penalty components
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    components = ['border', 'overlap', 'insert', 'cg']
    for ax, comp in zip(axes.ravel(), components):
        sns.boxplot(data=all_df, x='algorithm', y=comp, ax=ax)
        ax.set_title(f'{comp} penalty (log scale)' if comp != 'cg' else f'{comp} penalty')
        if comp != 'cg':
            ax.set_yscale('log')
        ax.set_ylabel(comp)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'penalty_components_boxplot.png'), dpi=200)
    plt.close()

    # 2. Success rate bar chart
    fig, ax = plt.subplots(figsize=(8,5))
    algorithms = ['DE', 'CMA-ES', 'PPO']
    rates = [de_success, cma_success, ppo_success]
    ax.bar(algorithms, rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Success rate')
    ax.set_ylim(0, 1.05)
    ax.set_title('Success rate comparison')
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')
    plt.savefig(os.path.join(args.output_dir, 'success_rate.png'), dpi=200)
    plt.close()

    # 3. Runtime comparison
    fig, ax = plt.subplots(figsize=(8,5))
    mean_times = [np.mean(de_times), np.mean(cma_times), ppo_result["total_time"]]
    std_times = [np.std(de_times), np.std(cma_times), 0]
    ax.bar(algorithms, mean_times, yerr=std_times, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Wall-clock time (seconds)')
    ax.set_title('Runtime comparison (DE/CMA: per run; PPO: total inference)')
    for i, v in enumerate(mean_times):
        ax.text(i, v + 0.5*std_times[i] + 0.5, f'{v:.1f}s', ha='center')
    plt.savefig(os.path.join(args.output_dir, 'runtime_comparison.png'), dpi=200)
    plt.close()

    # 4. Convergence curves (DE vs CMA-ES)
    def interpolate_history(hist, num_points=100):
        if len(hist) == 0:
            return np.zeros(num_points)
        x_old = np.linspace(0, 1, len(hist))
        x_new = np.linspace(0, 1, num_points)
        return np.interp(x_new, x_old, np.array(hist))

    de_hist = np.array([interpolate_history(r["history"]) for r in de_results])
    cma_hist = np.array([interpolate_history(r["history"]) for r in cma_results])

    fig, ax = plt.subplots(figsize=(10,6))
    gen = np.linspace(0, 100, 100)
    mean_de = de_hist.mean(axis=0)
    std_de = de_hist.std(axis=0)
    mean_cma = cma_hist.mean(axis=0)
    std_cma = cma_hist.std(axis=0)

    ax.plot(gen, mean_de, label='DE', color='#1f77b4')
    ax.fill_between(gen, mean_de - std_de, mean_de + std_de, alpha=0.2, color='#1f77b4')
    ax.plot(gen, mean_cma, label='CMA-ES', color='#ff7f0e')
    ax.fill_between(gen, mean_cma - std_cma, mean_cma + std_cma, alpha=0.2, color='#ff7f0e')
    ax.set_xlabel('Generation (normalized)')
    ax.set_ylabel('Best fitness (raw penalty)')
    ax.set_yscale('log')
    ax.set_title('Convergence: DE vs CMA-ES')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'convergence_curves.png'), dpi=200)
    plt.close()

    # 5. Statistical test
    if len(de_best_fitness) > 1 and len(cma_best_fitness) > 1:
        stat, p_val = wilcoxon(de_best_fitness, cma_best_fitness)
        print(f"\nWilcoxon signed-rank test DE vs CMA-ES (best fitness): p = {p_val:.4f}")

    # 6. Position diversity scatter plots
    def get_positions_from_result(result):
        x = result["best_x"][::2]
        y = result["best_x"][1::2]
        return x, y

    de_positions = [get_positions_from_result(r) for r in de_results if r["success"]]
    cma_positions = [get_positions_from_result(r) for r in cma_results if r["success"]]
    ppo_candidates = ppo_result["candidates"]
    if len(ppo_candidates) > 50:
        idx = np.random.choice(len(ppo_candidates), 50, replace=False)
        ppo_positions = [(c["cx"], c["cy"]) for c in np.array(ppo_candidates)[idx]]
    else:
        ppo_positions = [(c["cx"], c["cy"]) for c in ppo_candidates]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, positions, title in zip(axes, [de_positions, cma_positions, ppo_positions],
                                    ['DE (valid runs)', 'CMA-ES (valid runs)', 'PPO (sampled episodes)']):
        ax.set_title(title)
        ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height,
                                   fill=False, edgecolor='black', linewidth=2))
        ax.add_patch(plt.Rectangle((-panel_width/2 + border_spacing, -panel_height/2 + border_spacing),
                                   panel_width - 2*border_spacing, panel_height - 2*border_spacing,
                                   fill=False, edgecolor='red', linestyle='--', alpha=0.5))
        for i in range(n):
            xs = []
            ys = []
            for cx, cy in positions:
                abs_x = -cx[i] if is_back_side[i] else cx[i]
                xs.append(abs_x)
                ys.append(cy[i])
            ax.scatter(xs, ys, s=10, alpha=0.5, label=element_names[i] if i==0 else "")
        ax.set_aspect('equal')
        ax.set_xlim(-panel_width/2 - 20, panel_width/2 + 20)
        ax.set_ylim(-panel_height/2 - 20, panel_height/2 + 20)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'position_diversity.png'), dpi=200)
    plt.close()

    # 7. Save summary CSV
    summary_data = {
        'Algorithm': ['DE', 'CMA-ES', 'PPO'],
        'Success Rate': [de_success, cma_success, ppo_success],
        'Mean Raw Penalty (best)': [np.mean(de_best_fitness), np.mean(cma_best_fitness),
                                    np.mean(ppo_best_fitness) if ppo_best_fitness else np.nan],
        'Std Raw Penalty': [np.std(de_best_fitness), np.std(cma_best_fitness),
                            np.std(ppo_best_fitness) if ppo_best_fitness else np.nan],
        'Mean Time (s)': [np.mean(de_times), np.mean(cma_times), ppo_result["total_time"]],
        'Std Time': [np.std(de_times), np.std(cma_times), 0],
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(args.output_dir, 'summary_statistics.csv'), index=False)

    # Detailed breakdowns per run/episode
    de_df = pd.DataFrame([r["breakdown"] for r in de_results])
    de_df['run'] = range(len(de_df))
    de_df['algorithm'] = 'DE'
    cma_df = pd.DataFrame([r["breakdown"] for r in cma_results])
    cma_df['run'] = range(len(cma_df))
    cma_df['algorithm'] = 'CMA-ES'
    ppo_df = pd.DataFrame(ppo_breakdowns)
    ppo_df['run'] = range(len(ppo_df))
    ppo_df['algorithm'] = 'PPO'
    all_details = pd.concat([de_df, cma_df, ppo_df], ignore_index=True)
    all_details.to_csv(os.path.join(args.output_dir, 'all_breakdowns.csv'), index=False)

    print(f"\n✅ All results saved to {os.path.abspath(args.output_dir)}/")
    print("Comparison complete.")

if __name__ == "__main__":
    main()