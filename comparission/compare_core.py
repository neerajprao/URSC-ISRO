import os
import sys
import time
import json
import warnings
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from scipy.stats import wilcoxon, mannwhitneyu
from numba import njit
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Setup
warnings.filterwarnings("ignore")
os.makedirs("comparission/comparison_results", exist_ok=True)

CG_TOLERANCE = 1.0
VALID_EPS = 1e-6
SCORE_WEIGHTS = {"border": 3.0, "overlap": 3.0, "insert": 10.0, "cg": 0.5}

# Try importing cma-es
try:
    import cma
    CMAEvolutionStrategy = cma.CMAEvolutionStrategy
except ImportError:
    print("Could not import CMA-ES library.")
    sys.exit(1)

# Numba Penalty Core (Common across all three)
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
                    dist = np.sqrt((xi_abs - xj_abs)**2 + (yj_abs - yj_abs)**2)
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
    weighted = (SCORE_WEIGHTS["border"] * np.log1p(b) +
                SCORE_WEIGHTS["overlap"] * np.log1p(o) +
                SCORE_WEIGHTS["insert"] * np.log1p(ins) +
                SCORE_WEIGHTS["cg"] * np.log1p(cg))
    raw_penalty = b + o + ins + cg
    return {"border": b, "overlap": o, "insert": ins, "cg": cg,
            "weighted_score": weighted, "raw_penalty": raw_penalty}

def is_fully_valid(breakdown):
    return (breakdown["border"] < VALID_EPS and
            breakdown["overlap"] < VALID_EPS and
            breakdown["insert"] < VALID_EPS)

def load_config(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

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

    return (panel_width, panel_height, border_spacing, clearance,
            n, element_names, element_shapes, is_back_side, flat_offsets,
            offset_counts, offset_indices, masses, lengths, widths,
            insert_diams, clearance_dirs, req_d_matrix)

# PCB Placement Gymnasium Env for PPO inference
class PCBPlacementEnv(gym.Env):
    def __init__(self, params):
        super().__init__()
        (panel_width, panel_height, border_spacing, clearance,
         n, element_names, element_shapes, is_back_side, flat_offsets,
         offset_counts, offset_indices, masses, lengths, widths,
         insert_diams, clearance_dirs, req_d_matrix) = params
        self.params = params
        self.n = n
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
        obs_dim = 2 * self.n + 4 + 1
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
        step_frac = np.array([self.step_count / 250], dtype=np.float32)
        return np.concatenate([pos, pen, step_frac])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cx = np.random.uniform(self.lower[0::2], self.upper[0::2]).astype(np.float32)
        self.cy = np.random.uniform(self.lower[1::2], self.upper[1::2]).astype(np.float32)
        self.step_count = 0
        breakdown = self._get_breakdown()
        self.prev_score = composite_score(breakdown)
        obs = self._get_obs(breakdown)
        info = {"breakdown": breakdown}
        return obs, info

    def step(self, action):
        from PPO.gen import MAX_STEP_MM, MIN_STEP_MM, MAX_STEPS, STEP_COST, SUCCESS_BONUS
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        progress = self.step_count / MAX_STEPS
        current_max_step = MAX_STEP_MM * (1.0 - progress) + MIN_STEP_MM * progress
        dx = action[0::2] * current_max_step
        dy = action[1::2] * current_max_step
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
        info = {"breakdown": breakdown, "score": score,
                "cx": self.cx.copy(), "cy": self.cy.copy()}
        return obs, float(reward), terminated, truncated, info

def composite_score(breakdown):
    return (SCORE_WEIGHTS["border"]  * np.log1p(breakdown["border"]) +
            SCORE_WEIGHTS["overlap"] * np.log1p(breakdown["overlap"]) +
            SCORE_WEIGHTS["insert"]  * np.log1p(breakdown["insert"]) +
            SCORE_WEIGHTS["cg"]      * np.log1p(breakdown["cg"]))
from gymnasium import spaces
