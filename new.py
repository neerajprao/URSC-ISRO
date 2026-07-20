import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from numba import njit
import warnings
import os

warnings.filterwarnings("ignore")

# ================= STRUCTURAL CONFIGURATION ==================
CG_TOLERANCE =  0.01  # mm: Target combined center of gravity
OUTPUT_DIR = "optimized_layouts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= NUMBA JIT ACCELERATED CORE MATHEMATICS =================
@njit(fastmath=True, cache=True)
def compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, 
                         flat_offsets, offset_counts, offset_indices, is_back_side):
    
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
    
    # 1. SCREENING STEP: Center of Gravity Check
    cg_offset = np.sqrt(cg_x**2 + cg_y**2)
    cg_penalty = 0.0
    if cg_offset > CG_TOLERANCE:
        cg_penalty = (cg_offset - CG_TOLERANCE)**2 * 1e9
        if cg_penalty > 1e11:
            return cg_penalty

    # 2. SCREENING STEP: GLOBAL 2D Clearance Overlaps
    overlap_penalty = 0.0
    for i in range(num_components):
        for j in range(i + 1, num_components):
            diff_x = abs(abs_x[i] - abs_x[j])
            diff_y = abs(cy[i] - cy[j])
            req_dx = hl[i] + hl[j] + clearance
            req_dy = hw[i] + hw[j] + clearance
            
            if req_dx > diff_x and req_dy > diff_y:
                overlap_penalty += (req_dx - diff_x) * (req_dy - diff_y) * 1e6
                
    if overlap_penalty > 1e10:
        return cg_penalty + overlap_penalty

    # 3. DETAILED STEP: Pin Insert Safe Distances
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
                xi_abs = abs_x[i] + ( -flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0] )
                yi_abs = cy[i] + flat_offsets[idx_i, 1]
                
                for jj in range(j_count):
                    idx_j = j_start + jj
                    xj_abs = abs_x[j] + ( -flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0] )
                    yj_abs = cy[j] + flat_offsets[idx_j, 1]
                    
                    dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                    if req_dist > dist:
                        insert_penalty += (req_dist - dist) * 1e6
                        
    return cg_penalty + overlap_penalty + insert_penalty


def evaluate(individual, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side):
    cx = individual[::2]
    cy = individual[1::2]
    num_components = len(masses)
    hl = lengths / 2.0
    hw = widths / 2.0
    
    return compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, flat_offsets, offset_counts, offset_indices, is_back_side)


def run_optimization():
    filepath = 'newobj.json'
    if not os.path.exists(filepath):
        filepath = 'C:\\Users\\shravanz\\Desktop\\New folder\\newobj.json'
        
    with open(filepath, 'r') as f:
        data = json.load(f)

    panel_width = float(data['BOARD']['Length (mm)'])
    panel_height = float(data['BOARD']['Breadth (mm)'])
    border_spacing = float(data['BOARD']['Border (mm)'])
    clearance = float(data['BOARD']['Clearance (mm)'])

    element_data = []
    element_names = []
    element_shapes = []
    is_back_side_list = []
    
    flat_offsets_list = []
    
    raw_components_to_process = []
    for comp in data['COMPONENTS'].get('FRONT', []):
        raw_components_to_process.append((comp, False))
    for comp in data['COMPONENTS'].get('BACK', []):
        raw_components_to_process.append((comp, True))

    # --- NEW: Quantity Suffix Handling Expansion Logic ---
    components_to_process = []
    for comp, is_back in raw_components_to_process:
        qty = int(comp.get('Object qty', 1))
        if qty > 1:
            for q in range(qty):
                cloned_comp = comp.copy()
                cloned_comp['Unique Name'] = f"{comp['Object name']}{q}"
                components_to_process.append((cloned_comp, is_back))
        else:
            cloned_comp = comp.copy()
            cloned_comp['Unique Name'] = comp['Object name']
            components_to_process.append((cloned_comp, is_back))

    num_elements = len(components_to_process)
    offset_counts = np.zeros(num_elements, dtype=np.int32)
    offset_indices = np.zeros(num_elements, dtype=np.int32)
    
    current_idx = 0
    for idx, (component, is_back) in enumerate(components_to_process):
        element_name = component['Unique Name']
        shape = component.get('Shape', 'rectangle')
        element_shapes.append(shape)
        is_back_side_list.append(1 if is_back else 0)
        
        if shape == 'rectangle':
            length = float(component['Length (mm)'])
            width = float(component['Breadth (mm)'])
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 4))
            half_l, half_w = length / 2.0, width / 2.0
            
            if insert_qty == 6:
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
                if length >= width:
                    offsets.extend([(0.0, half_w), (0.0, -half_w)])
                else:
                    offsets.extend([(half_l, 0.0), (-half_l, 0.0)])
            else:
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
        else:
            diameter = float(component['Diameter (mm)'])
            length, width = diameter, diameter
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 3))
            inner_radius = max(0.0, (diameter / 2.0) - 1.0)
            offsets = [(inner_radius * np.cos(2 * np.pi * k / insert_qty), 
                        inner_radius * np.sin(2 * np.pi * k / insert_qty)) for k in range(insert_qty)]
                        
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
    n = len(element_data)

    bounds = []
    possible = True
    for i in range(num_elements):
        i_start = offset_indices[i]
        i_count = offset_counts[i]
        comp_offsets = flat_offsets[i_start : i_start + i_count]
        
        max_off_x = max(abs(x) for x, _ in comp_offsets) if i_count > 0 else 0
        max_off_y = max(abs(y) for _, y in comp_offsets) if i_count > 0 else 0
        
        x_min = -panel_width / 2.0 + border_spacing + max_off_x + insert_diams[i] / 2.0
        x_max = panel_width / 2.0 - border_spacing - max_off_x - insert_diams[i] / 2.0
        y_min = -panel_height / 2.0 + border_spacing + max_off_y + insert_diams[i] / 2.0
        y_max = panel_height / 2.0 - border_spacing - max_off_y - insert_diams[i] / 2.0
        
        bounds.append((x_min, x_max))
        bounds.append((y_min, y_max))

    if possible:
        req_d_matrix = np.full((n, n), 30.0)
        for i in range(n):
            for j in range(n):
                if insert_diams[i] == 4.0 and insert_diams[j] == 4.0:
                    req_d_matrix[i, j] = 24.0
                elif insert_diams[i] == 6.0 and insert_diams[j] == 6.0:
                    req_d_matrix[i, j] = 36.0

        distinct_layouts = []
        max_attempts = 150
        attempt = 0

        print(f"\n🚀 Launching parallel spatial distribution across available execution threads...")

        while len(distinct_layouts) < 5 and attempt < max_attempts:
            attempt += 1
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))

            try:
                result = differential_evolution(
                    evaluate, bounds,
                    args=(masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side),
                    maxiter=450, popsize=25, tol=1e-5,
                    strategy='best1bin', seed=seed,
                    workers=-1, updating='deferred'
                )
                
                if np.any(np.isinf(result.x)) or np.any(np.isnan(result.x)):
                    continue

                cx_v = result.x[::2]
                cy_v = result.x[1::2]
                
                total_mass = np.sum(masses)
                abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
                cg_x_v = np.sum(abs_x_positions * masses) / total_mass
                cg_y_v = np.sum(cy_v * masses) / total_mass

                if abs(cg_x_v) > CG_TOLERANCE or abs(cg_y_v) > CG_TOLERANCE:
                    continue

                is_distinct = True
                for existing in distinct_layouts:
                    if np.max(np.abs(result.x - existing)) < 6.0:
                        is_distinct = False
                        break

                if is_distinct:
                    distinct_layouts.append(result.x)
                    print(f"[√] Solution Saved {len(distinct_layouts)}/5. Combined CG: ({cg_x_v:.4f}, {cg_y_v:.4f}) mm")

            except Exception as e:
                continue

        # ================= PLOTTING ENGINE (ISOLATED POST-PROCESSING) =================
        if len(distinct_layouts) > 0:
            print("\n📊 Array generation finished. Initializing engineering report matrices...")
            total_mass = np.sum(masses)
            
            for idx, layout in enumerate(distinct_layouts):
                layout_idx = idx + 1
                cx_v = layout[::2]
                cy_v = layout[1::2]
                abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
                
                final_cg_x = np.sum(abs_x_positions * masses) / total_mass
                final_cg_y = np.sum(cy_v * masses) / total_mass
                
                fig = plt.figure(figsize=(18, 10))
                gs = fig.add_gridspec(2, 2, height_ratios=[7, 2.2], hspace=0.25)
                
                ax_f = fig.add_subplot(gs[0, 0])
                ax_b = fig.add_subplot(gs[0, 1])
                ax_table = fig.add_subplot(gs[1, :])
                ax_table.axis('off')
                
                fig.suptitle(f"Layout Alternative {layout_idx}\nCombined Assembly CG Offset: ({final_cg_x:.4f}, {final_cg_y:.4f}) mm", 
                             fontsize=14, fontweight='bold')
                
                for ax in [ax_f, ax_b]:
                    ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.3))
                    ax.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing), panel_width, border_spacing, color='red', alpha=0.15))
                    ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, border_spacing, color='red', alpha=0.15))
                    ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))
                    ax.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))

                table_data = []

                for i in range(num_elements):
                    ins_rad = insert_diams[i] / 2.0
                    i_start = offset_indices[i]
                    side_str = "BACK" if is_back_side[i] else "FRONT"
                    
                    table_data.append([
                        element_names[i],
                        side_str,
                        f"{masses[i]:.3f}",
                        f"{lengths[i]:.1f} x {widths[i]:.1f}" if element_shapes[i] == 'rectangle' else f"Ø {lengths[i]:.1f}",
                        f"({cx_v[i]:.2f}, {cy_v[i]:.2f})"
                    ])
                    
                    if not is_back_side[i]: 
                        ex, ey = cx_v[i], cy_v[i]
                        if element_shapes[i] == 'rectangle':
                            ax_f.add_patch(plt.Rectangle((ex - lengths[i]/2 - clearance, ey - widths[i]/2 - clearance), lengths[i] + 2*clearance, widths[i] + 2*clearance, fill=False, edgecolor='green', alpha=0.4, linestyle='--'))
                            ax_f.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.8, edgecolor='navy'))
                        else:
                            ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, fill=False, edgecolor='green', alpha=0.4, linestyle='--'))
                            ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.8, edgecolor='navy'))
                        
                        for ii in range(offset_counts[i]):
                            dx, dy = flat_offsets[i_start + ii]
                            ax_f.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson'))
                        ax_f.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold')
                        
                        ex_trace_b = -cx_v[i]
                        if element_shapes[i] == 'rectangle':
                            ax_b.add_patch(plt.Rectangle((ex_trace_b - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle=':', edgecolor='gray', alpha=0.5))
                        else:
                            ax_b.add_patch(plt.Circle((ex_trace_b, cy_v[i]), lengths[i]/2, fill=False, linestyle=':', edgecolor='gray', alpha=0.5))
                    
                    else:
                        ex_b, ey_b = cx_v[i], cy_v[i]
                        if element_shapes[i] == 'rectangle':
                            ax_b.add_patch(plt.Rectangle((ex_b - lengths[i]/2 - clearance, ey_b - widths[i]/2 - clearance), lengths[i] + 2*clearance, widths[i] + 2*clearance, fill=False, edgecolor='green', alpha=0.4, linestyle='--'))
                            ax_b.add_patch(plt.Rectangle((ex_b - lengths[i]/2, ey_b - widths[i]/2), lengths[i], widths[i], color='darkgreen', alpha=0.8, edgecolor='darkslategrey'))
                        else:
                            ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2 + clearance, fill=False, edgecolor='green', alpha=0.4, linestyle='--'))
                            ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2, color='darkgreen', alpha=0.8, edgecolor='darkslategrey'))
                        
                        for ii in range(offset_counts[i]):
                            dx, dy = flat_offsets[i_start + ii]
                            ax_b.add_patch(plt.Circle((ex_b + dx, ey_b + dy), ins_rad, color='orange'))
                        ax_b.text(ex_b, ey_b, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold')
                        
                        ex_trace_f = -cx_v[i]
                        if element_shapes[i] == 'rectangle':
                            ax_f.add_patch(plt.Rectangle((ex_trace_f - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle=':', edgecolor='gray', alpha=0.5))
                        else:
                            ax_f.add_patch(plt.Circle((ex_trace_f, cy_v[i]), lengths[i]/2, fill=False, linestyle=':', edgecolor='gray', alpha=0.5))

                for name, ax in [("FRONT SIDE VIEW", ax_f), ("BACK SIDE VIEW (FLIPPED)", ax_b)]:
                    ax.set_title(name, fontweight='bold', fontsize=11)
                    ax.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
                    ax.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
                    ax.set_aspect('equal')
                    ax.grid(True, linestyle=':', alpha=0.5)
                
                headers = ["Component Name", "Layer Placement", "Mass (kg)", "Dimensions (mm)", "CoG Coordinates (X, Y)"]
                ui_table = ax_table.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
                ui_table.auto_set_font_size(False)
                ui_table.set_fontsize(9)
                ui_table.scale(1.0, 1.3)
                
                for col_idx in range(len(headers)):
                    cell = ui_table[0, col_idx]
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2C3E50')
                
                plt.tight_layout()
                save_path = os.path.join(OUTPUT_DIR, f"layout_alternative_{layout_idx}.png")
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                print(f"[P] Exported engineering layout layout_{layout_idx}.png")
                plt.close(fig)

if __name__ == '__main__':
    run_optimization()