#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate PCB layouts using a trained PPO model and newobj.json.
Usage: python ppo.py [--json JSON_FILE] [--model_dir MODEL_DIR] [--output_dir OUTPUT_DIR] [--episodes EPISODES]
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import torch

# Suppress Gym deprecation warnings BEFORE SB3 loads gym internally
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# -------------------- Configuration --------------------
CG_TOLERANCE = 1.0
MAX_STEP_MM = 12.0
MIN_STEP_MM = 0.25
MAX_STEPS = 250
STEP_COST = 0.01
SUCCESS_BONUS = 50.0
SCORE_WEIGHTS = {"border": 3.0, "overlap": 3.0, "insert": 10.0, "cg": 0.5}
VALID_EPS = 1e-6

# -------------------- Command Line Arguments --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate PCB layouts using PPO model.")
    parser.add_argument("--json", default="newobj.json", help="Path to newobj.json")
    parser.add_argument("--model_dir", default="optimized_layouts_rl", help="Directory containing ppo_pcb_placement.zip and vecnormalize.pkl")
    parser.add_argument("--output_dir", default="generated_layouts", help="Output directory for images and CSV")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes to run")
    parser.add_argument("--parallel", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--max_layouts", type=int, default=5, help="Maximum distinct layouts to keep")
    parser.add_argument("--dedup_dist", type=float, default=25.0, help="Shift threshold for deduplication (mm)")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

# -------------------- Load Config & Process Inserts --------------------
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
            
            if insert_qty % 2 != 0:
                insert_qty += 1
                
            half_l, half_w = length / 2.0, width / 2.0
            
            if insert_qty == 2:
                offsets = [(half_l, 0.0), (-half_l, 0.0)] if length < width else [(0.0, half_w), (0.0, -half_w)]
            elif insert_qty == 4:
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
            else:
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
                extra_per_side = (insert_qty - 4) // 2
                
                if length >= width:
                    x_coords = np.linspace(-half_l, half_l, extra_per_side + 2)[1:-1]
                    for x in x_coords:
                        offsets.append((x, half_w))
                        offsets.append((x, -half_w))
                else:
                    y_coords = np.linspace(-half_w, half_w, extra_per_side + 2)[1:-1]
                    for y in y_coords:
                        offsets.append((half_l, y))
                        offsets.append((-half_l, y))
        else:
            diameter = float(component['Diameter (mm)'])
            length, width = diameter, diameter
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 3))
            inner_radius = max(0.0, (diameter / 2.0) - 1.0)
            offsets = [(inner_radius * np.cos(2*np.pi*k/insert_qty), inner_radius * np.sin(2*np.pi*k/insert_qty))
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

# -------------------- Penalty Engine --------------------
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
    return {"border": b, "overlap": o, "insert": ins, "cg": cg}

def composite_score(breakdown):
    return (SCORE_WEIGHTS["border"]  * np.log1p(breakdown["border"]) +
            SCORE_WEIGHTS["overlap"] * np.log1p(breakdown["overlap"]) +
            SCORE_WEIGHTS["insert"]  * np.log1p(breakdown["insert"]) +
            SCORE_WEIGHTS["cg"]      * np.log1p(breakdown["cg"]))

def is_fully_valid(breakdown):
    return (breakdown["border"] < VALID_EPS and
            breakdown["overlap"] < VALID_EPS and
            breakdown["insert"] < VALID_EPS)

# -------------------- Gymnasium Environment --------------------
class PCBPlacementEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, params, target_obs_dim=None, seed=None):
        super().__init__()
        (panel_width, panel_height, border_spacing, clearance,
         n, element_names, element_shapes, is_back_side, flat_offsets,
         offset_counts, offset_indices, masses, lengths, widths,
         insert_diams, clearance_dirs, req_d_matrix) = params
        self.params = params
        self.n = n
        self.target_obs_dim = target_obs_dim
        
        lower_bounds = []
        upper_bounds = []
        for i in range(n):
            i_start = offset_indices[i]
            i_count = offset_counts[i]
            comp_offsets = flat_offsets[i_start : i_start + i_count]
            max_off_x = max(abs(x) for x, _ in comp_offsets) if i_count > 0 else 0.0
            max_off_y = max(abs(y) for _, y in comp_offsets) if i_count > 0 else 0.0
            c_top   = clearance_dirs[i, 0]
            c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
            c_bot   = clearance_dirs[i, 2]
            c_left  = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]
            req_margin_left   = max(lengths[i]/2.0 + c_left,  max_off_x + insert_diams[i]/2.0)
            req_margin_right  = max(lengths[i]/2.0 + c_right, max_off_x + insert_diams[i]/2.0)
            req_margin_bottom = max(widths[i]/2.0 + c_bot,    max_off_y + insert_diams[i]/2.0)
            req_margin_top    = max(widths[i]/2.0 + c_top,    max_off_y + insert_diams[i]/2.0)
            if not is_back_side[i]:
                x_min = -panel_width/2.0 + border_spacing + req_margin_left
                x_max =  panel_width/2.0 - border_spacing - req_margin_right
            else:
                x_min = -panel_width/2.0 + border_spacing + req_margin_right
                x_max =  panel_width/2.0 - border_spacing - req_margin_left
            y_min = -panel_height/2.0 + border_spacing + req_margin_bottom
            y_max =  panel_height/2.0 - border_spacing - req_margin_top
            lower_bounds.extend([x_min, y_min])
            upper_bounds.extend([x_max, y_max])
        self.lower = np.array(lower_bounds, dtype=np.float32)
        self.upper = np.array(upper_bounds, dtype=np.float32)
        self.range_ = (self.upper - self.lower)
        self.range_[self.range_ == 0] = 1.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * self.n,), dtype=np.float32)
        
        obs_dim = self.target_obs_dim if self.target_obs_dim is not None else (2 * self.n + 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.cx = None
        self.cy = None
        self.step_count = 0

    def _normalize_pos(self):
        nx = 2.0 * (self.cx - self.lower[0::2]) / self.range_[0::2] - 1.0
        ny = 2.0 * (self.cy - self.lower[1::2]) / self.range_[1::2] - 1.0
        return np.stack([nx, ny], axis=1).ravel().astype(np.float32)

    def _get_breakdown(self):
        (panel_width, panel_height, border_spacing, clearance,
         n, element_names, element_shapes, is_back_side, flat_offsets,
         offset_counts, offset_indices, masses, lengths, widths,
         insert_diams, clearance_dirs, req_d_matrix) = self.params
        hl = lengths / 2.0
        hw = widths / 2.0
        b, o, ins, cg = compute_penalty_breakdown(
            self.cx, self.cy, masses, hl, hw, req_d_matrix, n,
            flat_offsets, offset_counts, offset_indices, is_back_side,
            clearance_dirs, panel_width, panel_height, border_spacing,
            CG_TOLERANCE=1.0
        )
        return {"border": b, "overlap": o, "insert": ins, "cg": cg}

    def _get_obs(self, breakdown):
        pos = self._normalize_pos()
        pen = np.array([
            np.log1p(breakdown["border"]),
            np.log1p(breakdown["overlap"]),
            np.log1p(breakdown["insert"]),
            np.log1p(breakdown["cg"]),
        ], dtype=np.float32)
        step_frac = np.array([self.step_count / MAX_STEPS], dtype=np.float32)
        raw_obs = np.concatenate([pos, pen, step_frac])
        
        target_len = self.observation_space.shape[0]
        if len(raw_obs) == target_len:
            return raw_obs
        elif len(raw_obs) > target_len:
            return raw_obs[:target_len]
        else:
            return np.pad(raw_obs, (0, target_len - len(raw_obs)), 'constant')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.cx = rng.uniform(self.lower[0::2], self.upper[0::2]).astype(np.float32)
        self.cy = rng.uniform(self.lower[1::2], self.upper[1::2]).astype(np.float32)
        self.step_count = 0
        breakdown = self._get_breakdown()
        self.prev_score = composite_score(breakdown)
        obs = self._get_obs(breakdown)
        info = {"breakdown": breakdown}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        progress = self.step_count / MAX_STEPS
        current_max_step = MAX_STEP_MM * (1.0 - progress) + MIN_STEP_MM * progress
        
        # Action vector dimensions check
        act_x = action[0::2]
        act_y = action[1::2]
        
        # Pad or trim actions to match active component count (self.n)
        if len(act_x) < self.n:
            dx = np.pad(act_x, (0, self.n - len(act_x))) * current_max_step
            dy = np.pad(act_y, (0, self.n - len(act_y))) * current_max_step
        else:
            dx = act_x[:self.n] * current_max_step
            dy = act_y[:self.n] * current_max_step

        self.cx = np.clip(self.cx + dx, self.lower[0::2], self.upper[0::2])
        self.cy = np.clip(self.cy + dy, self.lower[1::2], self.upper[1::2])
        self.step_count += 1
        
        breakdown = self._get_breakdown()
        score = composite_score(breakdown)
        reward = (self.prev_score - score) - STEP_COST
        self.prev_score = score
        
        terminated = False
        if is_fully_valid(breakdown):
            reward += SUCCESS_BONUS
            terminated = True
            
        truncated = self.step_count >= MAX_STEPS
        obs = self._get_obs(breakdown)
        info = {
            "breakdown": breakdown, 
            "score": score,
            "cx": self.cx.copy(), 
            "cy": self.cy.copy()
        }
        return obs, float(reward), terminated, truncated, info

# -------------------- Rendering --------------------
def get_abs_x_positions(cx, is_back_side):
    return np.where(is_back_side == 1, -cx, cx)

def render_layout(layout_idx, cx, cy, cg_x, cg_y, params, output_dir):
    (panel_width, panel_height, border_spacing, clearance,
     n, element_names, element_shapes, is_back_side, flat_offsets,
     offset_counts, offset_indices, masses, lengths, widths,
     insert_diams, clearance_dirs, req_d_matrix) = params
    num_elements = n
    abs_x_positions = get_abs_x_positions(cx, is_back_side)

    # FIGURE 1: MAIN VIEWS
    fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Layout Alternative {layout_idx} (PPO)\nCombined Assembly CG Offset: ({cg_x:.4f}, {cg_y:.4f}) mm",
                 fontsize=14, fontweight='bold')

    for ax in [ax_f, ax_b]:
        ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height,
                                   color='lightgray', alpha=0.3))
        ax.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing),
                                   panel_width, border_spacing, color='red', alpha=0.15))
        ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2),
                                   panel_width, border_spacing, color='red', alpha=0.15))
        ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing),
                                   border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))
        ax.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing),
                                   border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))

    for i in range(num_elements):
        ins_rad = insert_diams[i] / 2.0
        i_start = offset_indices[i]
        c_top, c_right, c_bot, c_left = clearance_dirs[i]
        if not is_back_side[i]:
            ex, ey = cx[i], cy[i]
            if element_shapes[i] == 'rectangle':
                c_x = ex - lengths[i]/2.0 - c_left
                c_y = ey - widths[i]/2.0 - c_bot
                c_w = lengths[i] + c_left + c_right
                c_h = widths[i] + c_top + c_bot
                ax_f.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h,
                                             facecolor='#A9C7EB', edgecolor='crimson',
                                             linestyle='--', linewidth=1.2, alpha=0.5))
                ax_f.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2),
                                             lengths[i], widths[i],
                                             color='royalblue', alpha=0.9,
                                             edgecolor='navy', linewidth=1.5))
            else:
                ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance,
                                          facecolor='#A9C7EB', edgecolor='crimson',
                                          linestyle='--', linewidth=1.2, alpha=0.5))
                ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2,
                                          color='royalblue', alpha=0.9,
                                          edgecolor='navy', linewidth=1.5))
            for ii in range(offset_counts[i]):
                dx, dy = flat_offsets[i_start + ii]
                ax_f.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
                ax_b.add_patch(plt.Circle((-(ex + dx), ey + dy), ins_rad, facecolor='crimson', edgecolor='#5C0612', linewidth=1.5, alpha=0.6, zorder=3))
            ax_f.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
        else:
            ex_b, ey_b = cx[i], cy[i]
            if element_shapes[i] == 'rectangle':
                c_x = ex_b - lengths[i]/2.0 - c_left
                c_y = ey_b - widths[i]/2.0 - c_bot
                c_w = lengths[i] + c_left + c_right
                c_h = widths[i] + c_top + c_bot
                ax_b.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle='--', linewidth=1.2, alpha=0.5))
                ax_b.add_patch(plt.Rectangle((ex_b - lengths[i]/2, ey_b - widths[i]/2), lengths[i], widths[i], color='darkgreen', alpha=0.9, edgecolor='darkslategrey', linewidth=1.5))
            else:
                ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2 + clearance, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle='--', linewidth=1.2, alpha=0.5))
                ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2, color='darkgreen', alpha=0.9, edgecolor='darkslategrey', linewidth=1.5))
            for ii in range(offset_counts[i]):
                dx, dy = flat_offsets[i_start + ii]
                ax_b.add_patch(plt.Circle((ex_b + dx, ey_b + dy), ins_rad, color='orange', zorder=4))
                ax_f.add_patch(plt.Circle((-(ex_b + dx), ey_b + dy), ins_rad, facecolor='orange', edgecolor='#733D00', linewidth=1.5, alpha=0.6, zorder=3))
            ax_b.text(ex_b, ey_b, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

    for name, ax in [("FRONT SIDE VIEW", ax_f), ("BACK SIDE VIEW (FLIPPED)", ax_b)]:
        ax.set_title(name, fontweight='bold', fontsize=11)
        ax.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
        ax.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    main_path = os.path.join(output_dir, f"layout_{layout_idx}_main.png")
    plt.savefig(main_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # FIGURE 2: DISTANCE MAP WITH TABLE
    fig_c = plt.figure(figsize=(16, 12))
    gs_c = fig_c.add_gridspec(2, 1, height_ratios=[6.0, 4.0])
    ax_c = fig_c.add_subplot(gs_c[0, 0])
    ax_table = fig_c.add_subplot(gs_c[1, 0])
    ax_table.axis('off')

    fig_c.suptitle(f"Layout Alternative {layout_idx} (PPO) — Inter-Insert Proximity Verification (< 40mm)\n(Front Side Projection View)",
                   fontsize=13, fontweight='bold')

    ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.2))

    table_data = []
    for i in range(num_elements):
        i_start = offset_indices[i]
        ins_rad = insert_diams[i] / 2.0
        ey = cy[i]
        c_top, c_right, c_bot, c_left = clearance_dirs[i]
        side_str = "BACK" if is_back_side[i] else "FRONT"
        front_view_x = abs_x_positions[i]
        front_view_y = cy[i]
        table_data.append([
            element_names[i], side_str, f"{masses[i]:.3f}",
            f"{lengths[i]:.1f} x {widths[i]:.1f}" if element_shapes[i] == 'rectangle' else f"Ø {lengths[i]:.1f}",
            f"{front_view_x:.2f}", f"{front_view_y:.2f}"
        ])

        if not is_back_side[i]:
            ex = cx[i]
            if element_shapes[i] == 'rectangle':
                ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.75, edgecolor='navy', linewidth=1.5))
            else:
                ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.75, edgecolor='navy', linewidth=1.5))
            for ii in range(offset_counts[i]):
                dx, dy = flat_offsets[i_start + ii]
                ax_c.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
            ax_c.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

    close_pairs_labels = []
    label_counter = 0
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i in range(num_elements):
        i_start = offset_indices[i]
        for j in range(i + 1, num_elements):
            j_start = offset_indices[j]
            for ii in range(offset_counts[i]):
                idx_i = i_start + ii
                xi_abs = abs_x_positions[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                yi_abs = cy[i] + flat_offsets[idx_i, 1]
                for jj in range(offset_counts[j]):
                    idx_j = j_start + jj
                    xj_abs = abs_x_positions[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                    yj_abs = cy[j] + flat_offsets[idx_j, 1]
                    dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                    if dist < 40.0:
                        xf1 = cx[i] + flat_offsets[idx_i, 0] if not is_back_side[i] else -(cx[i] + flat_offsets[idx_i, 0])
                        yf1 = cy[i] + flat_offsets[idx_i, 1]
                        xf2 = cx[j] + flat_offsets[idx_j, 0] if not is_back_side[j] else -(cx[j] + flat_offsets[idx_j, 0])
                        yf2 = cy[j] + flat_offsets[idx_j, 1]
                        letter = alphabet[label_counter % len(alphabet)]
                        label_counter += 1
                        ax_c.plot([xf1, xf2], [yf1, yf2], color='purple', linestyle='--', linewidth=1.5, alpha=0.85, zorder=10)
                        ax_c.text((xf1 + xf2)/2, (yf1 + yf2)/2, letter, color='white', fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle="circle,pad=0.2", fc="darkmagenta", ec="none", alpha=0.9), zorder=11)
                        close_pairs_labels.append(f"{letter} — {dist:.2f} mm ({element_names[i]} ↔ {element_names[j]})")

    legend_text = "Pin Distances Summary:\n\n" + ("\n".join(close_pairs_labels) if close_pairs_labels else "No insert violations\nor pins found under 40mm.")
    ax_c.text(1.02, 0.95, legend_text, transform=ax_c.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))

    front_patch = mpatches.Patch(color='royalblue', alpha=0.75, label='Front Layer Components')
    back_patch = mpatches.Patch(facecolor='none', edgecolor='darkgreen', linestyle='--', label='Back Layer Components (Projected Outline)')
    ax_c.legend(handles=[front_patch, back_patch], loc='upper left', bbox_to_anchor=(1.02, 0.45), frameon=True)

    ax_c.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
    ax_c.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
    ax_c.set_aspect('equal')
    ax_c.grid(True, linestyle=':', alpha=0.4)

    headers = ["Component Name", "Layer Placement", "Mass (kg)", "Dimensions (mm)", "CoG X (mm)", "CoG Y (mm)"]
    ui_table = ax_table.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    ui_table.auto_set_font_size(False)
    ui_table.set_fontsize(9)
    ui_table.scale(1.0, 1.3)
    for col_idx in range(len(headers)):
        cell = ui_table[0, col_idx]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C3E50')

    plt.subplots_adjust(hspace=0.4)
    dist_path = os.path.join(output_dir, f"layout_{layout_idx}_distance_map.png")
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    plt.close(fig_c)

    return main_path, dist_path

# -------------------- Main --------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Using device: {device}")

    data = load_config(args.json)
    params = build_arrays(data)
    (panel_width, panel_height, border_spacing, clearance,
     n, element_names, element_shapes, is_back_side, flat_offsets,
     offset_counts, offset_indices, masses, lengths, widths,
     insert_diams, clearance_dirs, req_d_matrix) = params

    model_path = os.path.join(args.model_dir, "ppo_pcb_placement")
    vecnorm_path = os.path.join(args.model_dir, "vecnormalize.pkl")
    
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    # Read observation dimension expected by vecnormalize or model
    target_dim = 41
    if os.path.exists(vecnorm_path):
        try:
            import pickle
            with open(vecnorm_path, "rb") as f:
                stats = pickle.load(f)
                if hasattr(stats, "obs_rms") and stats.obs_rms is not None:
                    target_dim = stats.obs_rms.mean.shape[0]
        except Exception:
            pass

    def make_env():
        env = PCBPlacementEnv(params, target_obs_dim=target_dim)
        return Monitor(env)

    venv = DummyVecEnv([make_env for _ in range(args.parallel)])
    if os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(model_path, env=venv, device=device)
    print(f"[✓] Loaded model from {model_path}.zip")

    obs = venv.reset()
    candidates = []
    episodes_done = 0
    t0 = time.time()

    while episodes_done < args.episodes:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, infos = venv.step(action)
        for i, info in enumerate(infos):
            if done[i]:
                episodes_done += 1
                cx, cy, breakdown = info.get("cx"), info.get("cy"), info.get("breakdown")
                if cx is not None and is_fully_valid(breakdown):
                    verify = evaluate_layout(cx, cy, params)
                    if is_fully_valid(verify):
                        candidates.append({"cx": cx, "cy": cy, "breakdown": verify, "cg": verify["cg"]})

    print(f"[✓] {episodes_done} episodes completed in {time.time()-t0:.1f}s.")
    print(f"[✓] {len(candidates)} / {episodes_done} rollouts produced a valid layout.")

    if len(candidates) == 0:
        print("[!] No valid layouts found. Increasing --episodes is recommended.")
        sys.exit(0)

    candidates.sort(key=lambda c: c["cg"])
    def mean_shift_mm(a, b):
        return np.sqrt((a["cx"] - b["cx"])**2 + (a["cy"] - b["cy"])**2).mean()

    distinct_layouts = []
    for cand in candidates:
        if all(mean_shift_mm(cand, kept) >= args.dedup_dist for kept in distinct_layouts):
            distinct_layouts.append(cand)
        if len(distinct_layouts) >= args.max_layouts:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []
    total_mass = masses.sum()

    for layout_idx, lay in enumerate(distinct_layouts):
        cx, cy = lay["cx"], lay["cy"]
        cg_x = sum(get_abs_x_positions(cx, is_back_side)[i] * masses[i] for i in range(n)) / total_mass
        cg_y = sum(cy[i] * masses[i] for i in range(n)) / total_mass
        render_layout(layout_idx+1, cx, cy, cg_x, cg_y, params, args.output_dir)

        abs_x_pos = get_abs_x_positions(cx, is_back_side)
        for i in range(n):
            dims = f"{lengths[i]:.1f} x {widths[i]:.1f}" if element_shapes[i] == 'rectangle' else f"Ø {lengths[i]:.1f}"
            all_rows.append({
                "Layout": layout_idx + 1,
                "Layout CG Penalty": round(lay["cg"], 4),
                "Component Name": element_names[i],
                "Layer Placement": "BACK" if is_back_side[i] else "FRONT",
                "Mass (kg)": round(masses[i], 3),
                "Dimensions (mm)": dims,
                "CoG X (mm)": round(abs_x_pos[i], 2),
                "CoG Y (mm)": round(cy[i], 2),
            })

    df_all = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, "all_layouts_table.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"[✓] Saved combined table: {csv_path}")
    print(f"[✓] Done — {len(distinct_layouts)} layouts rendered in {args.output_dir}/")

if __name__ == "__main__":
    main()