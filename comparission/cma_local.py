# ==============================================================================
# SECTION 1: IMPORTING EXTERNAL LIBRARIES AND MODULES
# ==============================================================================
# Python relies on built-in and external libraries to perform complex tasks.
# Think of these as importing toolkits into our workspace.

# Import the 'os' (Operating System) module to navigate computer directories and create files.
import os

# Import the 'time' module to measure how long the program takes to execute.
import time

# Import the 'json' module to read data structured in JSON format (a human-readable data file format).
import json

# Import the 'warnings' module to hide non-critical warning messages from clogging the computer terminal.
import warnings

# Import 'numpy' (Numerical Python), a fundamental library for fast numerical and mathematical matrix calculations.
import numpy as np

# Import 'matplotlib', a plotting library used for generating graphs and drawing visuals.
import matplotlib

# Set matplotlib backend to 'Agg' so it can generate image files silently in the background without needing a visual screen window.
matplotlib.use('Agg')  # Headless backend: disables GUI rendering completely

# Import 'pyplot' from matplotlib to draw 2D layout shapes (rectangles, circles, grid lines).
import matplotlib.pyplot as plt

# Import 'patches' from matplotlib to draw geometric shapes like legend rectangles/boxes.
import matplotlib.patches as mpatches

# Import 'cma' (CMA-ES), an advanced evolutionary optimization algorithm used to find optimal geometric positions.
import cma

# Import special optimization features from 'numba':
# 'njit' compiles Python code into fast machine code.
# 'prange' allows code loops to run across multiple CPU computer cores simultaneously.
from numba import njit, prange

# Instruct Python to ignore harmless warning messages during execution.
warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 2: GLOBAL STRUCTURAL CONFIGURATION
# ==============================================================================

# Define the maximum allowed offset tolerance for the Center of Gravity (CG) in millimeters.
# Target: keep the combined balance center within 1.0 mm of the origin (0,0).
CG_TOLERANCE = 1.0  # mm: Target combined center of gravity tolerance

# Define the directory folder name where output plot images will be saved.
OUTPUT_DIR = "optimized_layouts"

# Check if the folder exists on the computer; if it does not, create it.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# SECTION 3: CORE MATHEMATICAL EVALUATION ENGINE (COMPILED FOR ULTRA-FAST SPEED)
# ==============================================================================

# The @njit decorator tells Numba to convert this entire mathematical function directly into binary machine code.
# 'fastmath=True' enables floating-point mathematical shortcuts for speed.
# 'cache=True' saves the compiled code so it loads faster on future runs.
@njit(fastmath=True, cache=True)
def compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, 
                         flat_offsets, offset_counts, offset_indices, is_back_side,
                         clearance_dirs, panel_width, panel_height, border_spacing):
    """
    This function acts as a "scoring judge" (Fitness Function).
    It takes a proposed layout arrangement of physical components on a board and calculates 
    a total 'penalty score'. 
    
    Perfect layout = score of 0.
    Invalid layout (overlapping parts, off-balance, going off the board) = massive mathematical penalty.
    """
    
    # Initialize variables to accumulate total weight and center of gravity sums.
    total_mass = 0.0
    cg_x = 0.0
    cg_y = 0.0
    
    # Create an array of zeroes to store the true X-coordinates of components from a single reference view.
    abs_x = np.zeros(num_components)
    
    # Loop over every component to calculate total mass and combined Center of Gravity (CG).
    for i in range(num_components):
        m = masses[i]            # Retrieve mass of the i-th component
        total_mass += m          # Add mass to the running total mass
        
        # If a component is placed on the BACK side of the board, its X-position is flipped (mirrored).
        if is_back_side[i]:
            abs_x[i] = -cx[i]
        else:
            abs_x[i] = cx[i]
        
        # Calculate weighted mass moment for Center of Gravity: Sum(Mass * Distance)
        cg_x += abs_x[i] * m
        cg_y += cy[i] * m
        
    # Guard clause: If total mass is zero (impossible physical state), return an extremely large penalty.
    if total_mass == 0.0:
        return 1e15
        
    # Finalize Center of Gravity coordinates by dividing moment sum by total mass: (Sum(m*x) / Sum(m))
    cg_x /= total_mass
    cg_y /= total_mass
    
    # Define a huge numerical multiplier used to heavily punish layout rule violations.
    PENALTY_WEIGHT = 1e8

    # Calculate the inner boundary edges (usable layout area) of the board, considering border margin.
    active_x_min = -panel_width / 2.0 + border_spacing
    active_x_max =  panel_width / 2.0 - border_spacing
    active_y_min = -panel_height / 2.0 + border_spacing
    active_y_max =  panel_height / 2.0 - border_spacing

    # Calculate distance of combined Center of Gravity from the ideal board center (0,0) using Euclidean formula: sqrt(x^2 + y^2)
    cg_offset = np.sqrt(cg_x**2 + cg_y**2)
    cg_penalty = 0.0
    
    # If the balance offset exceeds the allowed tolerance (1.0 mm), apply a squared penalty.
    if cg_offset > CG_TOLERANCE:
        cg_penalty = ((cg_offset - CG_TOLERANCE) ** 2) * PENALTY_WEIGHT

    # Initialize penalty buckets for physical overlaps and boundary violations.
    overlap_penalty = 0.0
    border_penalty = 0.0

    # Evaluate border boundaries and physical component-to-component overlaps.
    for i in range(num_components):
        # Extract clearance margins around component 'i' in 4 directions [Top, Right, Bottom, Left]
        c_top = clearance_dirs[i, 0]
        # Swap left/right clearances if component is mounted on the BACK side (mirrored view)
        c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
        c_bot = clearance_dirs[i, 2]
        c_left = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]

        # Calculate bounding box edges (Left, Right, Bottom, Top) including safety clearances.
        i_x_min = abs_x[i] - hl[i] - c_left
        i_x_max = abs_x[i] + hl[i] + c_right
        i_y_min = cy[i] - hw[i] - c_bot
        i_y_max = cy[i] + hw[i] + c_top

        # Check if component extends past the allowed board edge boundaries.
        viol_left   = max(0.0, active_x_min - i_x_min)  # Left edge violation
        viol_right  = max(0.0, i_x_max - active_x_max)  # Right edge violation
        viol_bottom = max(0.0, active_y_min - i_y_min)  # Bottom edge violation
        viol_top    = max(0.0, i_y_max - active_y_max)  # Top edge violation

        # Add squared penalty if any border was violated.
        total_border_viol = viol_left + viol_right + viol_bottom + viol_top
        if total_border_viol > 0.0:
            border_penalty += (total_border_viol ** 2) * PENALTY_WEIGHT

        # Compare component 'i' against every other component 'j' to check for physical collision.
        for j in range(i + 1, num_components):
            # Only check overlap if both components are mounted on the SAME side of the board!
            if is_back_side[i] == is_back_side[j]:
                cj_top = clearance_dirs[j, 0]
                cj_right = clearance_dirs[j, 3] if is_back_side[j] else clearance_dirs[j, 1]
                cj_bot = clearance_dirs[j, 2]
                cj_left = clearance_dirs[j, 1] if is_back_side[j] else clearance_dirs[j, 3]

                j_x_min = abs_x[j] - hl[j] - cj_left
                j_x_max = abs_x[j] + hl[j] + cj_right
                j_y_min = cy[j] - hw[j] - cj_bot
                j_y_max = cy[j] + hw[j] + cj_top

                # Calculate width and height of 2D overlap region between component 'i' and component 'j'
                overlap_x = max(0.0, min(i_x_max, j_x_max) - max(i_x_min, j_x_min))
                overlap_y = max(0.0, min(i_y_max, j_y_max) - max(i_y_min, j_y_min))

                # If overlap width AND height are both positive, the two components are physically colliding!
                if overlap_x > 0.0 and overlap_y > 0.0:
                    overlap_area = overlap_x * overlap_y
                    overlap_penalty += (overlap_area ** 2) * PENALTY_WEIGHT

    # Initialize penalty bucket for mounting insert/pin distance violations.
    insert_penalty = 0.0
    
    # Loop over all pairs of components to ensure mounting pins maintain required clearance distances.
    for i in range(num_components):
        i_start = offset_indices[i]    # Index start in flat pin array for component i
        i_count = offset_counts[i]      # Number of mounting pins in component i
        
        for j in range(i + 1, num_components):
            j_start = offset_indices[j]
            j_count = offset_counts[j]
            req_dist = req_d_matrix[i, j]  # Required physical separation distance between pins
            
            # Loop through all pins on component i
            for ii in range(i_count):
                idx_i = i_start + ii
                # Compute absolute position of pin 'ii' on board
                xi_abs = abs_x[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                yi_abs = cy[i] + flat_offsets[idx_i, 1]
                
                # Loop through all pins on component j
                for jj in range(j_count):
                    idx_j = j_start + jj
                    # Compute absolute position of pin 'jj' on board
                    xj_abs = abs_x[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                    yj_abs = cy[j] + flat_offsets[idx_j, 1]
                    
                    # Compute straight-line Euclidean distance between the two mounting pins
                    dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                    
                    # If pins are closer than required distance, penalize layout heavily
                    if req_dist > dist:
                        insert_penalty += ((req_dist - dist) ** 2) * PENALTY_WEIGHT

    # Return total aggregated penalty score. (0.0 represents a flawless, valid layout).
    return cg_penalty + overlap_penalty + border_penalty + insert_penalty

# Helper function to unpack continuous algorithm input values into component (x, y) coordinates.
def evaluate(individual, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing):
    cx = individual[::2]  # Extract every EVEN index element (X coordinates of components)
    cy = individual[1::2] # Extract every ODD index element (Y coordinates of components)
    num_components = len(masses)
    hl = lengths / 2.0    # Half-lengths of components (radius along X axis)
    hw = widths / 2.0     # Half-widths of components (radius along Y axis)
    
    # Run core penalty computation function
    return compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing)

# ==============================================================================
# SECTION 4: MAIN OPTIMIZATION EXECUTION CONTROLLER
# ==============================================================================

def run_optimization():
    # File name of input configuration file containing component specifications.
    filepath = 'newobj.json'
    
    # If file isn't directly found, check the current working folder path.
    if not os.path.exists(filepath):
        filepath = os.path.join(os.getcwd(), 'newobj.json')

    # If configuration file still cannot be found, abort execution and display error.
    if not os.path.exists(filepath):
        raise FileNotFoundError("❌ Critical Error: Unable to locate 'newobj.json' in working directory.")

    print(f"\n[✓] Config loaded successfully: '{filepath}'")

    # Read and parse the JSON input file into a Python dictionary data structure.
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Read physical board boundaries and margin settings from JSON file.
    panel_width = float(data['BOARD']['Length (mm)'])      # Board overall width
    panel_height = float(data['BOARD']['Breadth (mm)'])    # Board overall height
    border_spacing = float(data['BOARD']['Border (mm)'])  # Margin edge off-limits to components
    clearance = float(data['BOARD']['Clearance (mm)'])    # Minimum separation distance

    # Initialize empty lists to store structured component attributes.
    element_data = []
    element_names = []
    element_shapes = []
    is_back_side_list = []
    flat_offsets_list = []
    clearance_dirs_list = []
    
    # Unpack components mounted on FRONT side and BACK side into a single processing list.
    raw_components_to_process = []
    for comp in data['COMPONENTS'].get('FRONT', []):
        raw_components_to_process.append((comp, False))
    for comp in data['COMPONENTS'].get('BACK', []):
        raw_components_to_process.append((comp, True))

    # Duplicate component definitions if quantity ("Object qty") is greater than 1.
    components_to_process = []
    for comp, is_back in raw_components_to_process:
        qty = int(comp.get('Object qty', 1))
        if qty > 1:
            for q in range(qty):
                cloned_comp = comp.copy()
                cloned_comp['Unique Name'] = f"{comp['Object name']}{q}" # Append duplicate index to name
                components_to_process.append((cloned_comp, is_back))
        else:
            cloned_comp = comp.copy()
            cloned_comp['Unique Name'] = comp['Object name']
            components_to_process.append((cloned_comp, is_back))

    num_elements = len(components_to_process)
    
    # Pre-allocate zero arrays to track mounting pin indexing for each component.
    offset_counts = np.zeros(num_elements, dtype=np.int32)
    offset_indices = np.zeros(num_elements, dtype=np.int32)
    
    current_idx = 0
    
    # Loop over every processed component to construct geometric properties and mounting pin positions.
    for idx, (component, is_back) in enumerate(components_to_process):
        element_name = component['Unique Name']
        shape = component.get('Shape', 'rectangle')
        element_shapes.append(shape)
        is_back_side_list.append(1 if is_back else 0)
        
        # Read customized face clearances if specified.
        cf_faces = component.get('CF', [])
        cf_len = float(component.get('CFLen (mm)', clearance))
        
        c_dirs = [clearance, clearance, clearance, clearance] # Default [Top, Right, Bottom, Left]
        for face in cf_faces:
            if face == 1:
                c_dirs[0] = cf_len # Top
            elif face == 2:
                c_dirs[1] = cf_len # Right
            elif face == 3:
                c_dirs[2] = cf_len # Bottom
            elif face == 4:
                c_dirs[3] = cf_len # Left
                
        clearance_dirs_list.append(c_dirs)

        # Process geometry for RECTANGULAR components.
        if shape == 'rectangle':
            length = float(component['Length (mm)'])
            width = float(component['Breadth (mm)'])
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 4))
            half_l, half_w = length / 2.0, width / 2.0
            
            # Generate pin relative offset positions relative to component center (0,0) based on pin count.
            if insert_qty == 2:
                if length >= width:
                    offsets = [(half_l, 0.0), (-half_l, 0.0)]
                else:
                    offsets = [(0.0, half_w), (0.0, -half_w)]
            elif insert_qty == 6:
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
                if length >= width:
                    offsets.extend([(0.0, half_w), (0.0, -half_w)])
                else:
                    offsets.extend([(half_l, 0.0), (-half_l, 0.0)])
            else: # Default 4 corners
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
        
        # Process geometry for CIRCULAR components using trigonometry.
        else:
            diameter = float(component['Diameter (mm)'])
            length, width = diameter, diameter
            insert_diam = float(component['Insert (mm)'])
            insert_qty = int(component.get('Insert qty', 3))
            inner_radius = max(0.0, (diameter / 2.0) - 1.0)
            
            # Place pins in a circular radial pattern using polar coordinates: x = r*cos(theta), y = r*sin(theta)
            offsets = [(inner_radius * np.cos(2 * np.pi * k / insert_qty), 
                        inner_radius * np.sin(2 * np.pi * k / insert_qty)) for k in range(insert_qty)]
                        
        flat_offsets_list.extend(offsets)
        offset_counts[idx] = len(offsets)
        offset_indices[idx] = current_idx
        current_idx += len(offsets)
        
        # Save mass, dimensions, and insert hole size into master array.
        element_data.append([float(component['Weight (kg)']), length, width, insert_diam])
        element_names.append(element_name)

    # Convert Python lists to high-speed Numpy numerical arrays.
    flat_offsets = np.array(flat_offsets_list, dtype=np.float64)
    element_data = np.array(element_data, dtype=float)
    masses = element_data[:, 0]
    lengths = element_data[:, 1]
    widths = element_data[:, 2]
    insert_diams = element_data[:, 3]
    is_back_side = np.array(is_back_side_list, dtype=np.int32)
    clearance_dirs = np.array(clearance_dirs_list, dtype=np.float64)
    n = len(element_data)

    # Calculate strict lower and upper coordinate boundaries (x_min, x_max, y_min, y_max) for each component
    # to guarantee components stay inside board physical limits during optimization search.
    lower_bounds = []
    upper_bounds = []
    for i in range(num_elements):
        i_start = offset_indices[i]
        i_count = offset_counts[i]
        comp_offsets = flat_offsets[i_start : i_start + i_count]
        
        max_off_x = max(abs(x) for x, _ in comp_offsets) if i_count > 0 else 0
        max_off_y = max(abs(y) for _, y in comp_offsets) if i_count > 0 else 0
        
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

    # Initialize matrix defining minimum required separation distance between pin pairs.
    req_d_matrix = np.full((n, n), 30.0) # Default distance 30 mm
    for i in range(n):
        for j in range(n):
            if insert_diams[i] == 4.0 and insert_diams[j] == 4.0:
                req_d_matrix[i, j] = 24.0 # Reduced distance for smaller 4mm pins
            elif insert_diams[i] == 6.0 and insert_diams[j] == 6.0:
                req_d_matrix[i, j] = 36.0 # Increased distance for larger 6mm pins

    distinct_layouts = []     # List to collect 5 completely distinct valid layout solutions
    max_attempts = 150        # Maximum algorithm restart attempts allowed
    attempt = 0

    # Perform a "dummy" run of the compiled evaluation function to trigger Numba compilation upfront.
    dummy_ind = np.zeros(2 * num_elements)
    evaluate(dummy_ind, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing)

    # Detect computer CPU hardware cores to display system status.
    cpu_cores = os.cpu_count() or 8
    print(f"🚀 CMA-ES Optimization Active | Available Cores: {cpu_cores}")
    print("⚡ Executing CMA-ES with Covariance Adaptation & Boundary Constraints...\n")
    start_time = time.time()

    # Loop until 5 distinct valid layouts are found or max attempts reached.
    while len(distinct_layouts) < 5 and attempt < max_attempts:
        attempt += 1
        
        # Pick a random starting point for layout positions within allowed boundary ranges.
        x0 = np.random.uniform(lower_bounds, upper_bounds)
        sigma0 = 15.0 # Initial search step size (std deviation)

        # Set configuration options for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
        cma_opts = {
            'bounds': [lower_bounds.tolist(), upper_bounds.tolist()], # Hard spatial limits
            'maxiter': 600,                                           # Max search generations
            'popsize': 32,                                            # Candidate solutions generated per generation
            'tolfun': 1e-6,                                           # Precision convergence stop criteria
            'verbose': -9,                                            # Mute verbose output logs
            'seed': int(np.random.default_rng().integers(0, 2**31 - 1)) # Randomization seed
        }

        try:
            # Initialize CMA-ES Evolutionary Optimizer instance.
            es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
            
            # Run evolutionary algorithm until stopping criterion met.
            while not es.stop():
                solutions = es.ask() # Generate population of candidate layouts
                # Evaluate penalty score for every candidate layout in population
                scores = [
                    evaluate(
                        sol, masses, lengths, widths, clearance, req_d_matrix, 
                        flat_offsets, offset_counts, offset_indices, is_back_side, 
                        clearance_dirs, panel_width, panel_height, border_spacing
                    ) for sol in solutions
                ]
                es.tell(solutions, scores) # Update evolutionary probability distribution based on scores

            best_sol = np.array(es.result.xbest) # Retrieve best layout solution coordinates
            best_score = es.result.fbest         # Retrieve lowest penalty score

            # If solution contains rule violations (score > 0.001) or invalid numbers (Inf/NaN), reject it.
            if best_score > 1e-3 or np.any(np.isinf(best_sol)) or np.any(np.isnan(best_sol)):
                continue

            # Extract final component positions
            cx_v = best_sol[::2]
            cy_v = best_sol[1::2]
            
            # Recalculate and verify Center of Gravity for this best solution.
            total_mass = np.sum(masses)
            abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
            cg_x_v = np.sum(abs_x_positions * masses) / total_mass
            cg_y_v = np.sum(cy_v * masses) / total_mass

            # Double check Center of Gravity tolerance requirement (<= 1.0 mm offset).
            if abs(cg_x_v) > CG_TOLERANCE or abs(cg_y_v) > CG_TOLERANCE:
                continue

            # Check that this newly found layout is geometrically unique compared to previously saved layouts.
            is_distinct = True
            for existing in distinct_layouts:
                if np.max(np.abs(best_sol - existing)) < 6.0: # Must differ by at least 6mm somewhere
                    is_distinct = False
                    break

            # If unique, append to layout results list.
            if is_distinct:
                distinct_layouts.append(best_sol)
                print(f"  [✓] CMA-ES Solution {len(distinct_layouts)}/5 calculated | CG Offset: ({cg_x_v:.2f}, {cg_y_v:.2f}) mm")

        except Exception as e:
            # Ignore algorithm convergence crashes and try again on next attempt loop
            print(f"Attempt {attempt} failed with error: {e}")
            continue

    print(f"\n✨ Execution completed in {time.time() - start_time:.2f}s. Saving plots...\n")

    # ==============================================================================
    # SECTION 5: DRAWING VISUAL DIAGRAMS AND GRAPHICAL EXPORTS
    # ==============================================================================
    
    # If valid optimized layouts were computed, render and export layout diagrams.
    if len(distinct_layouts) > 0:
        total_mass = np.sum(masses)
        
        # Process each layout solution individually.
        for idx, layout in enumerate(distinct_layouts):
            layout_idx = idx + 1
            cx_v = layout[::2]
            cy_v = layout[1::2]
            
            # Correct X-coordinates for back-mounted components to project into front-view alignment:
            # Front components use cx_v directly; Back components are negated (-cx_v)
            abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
            
            # Compute final board center of gravity.
            final_cg_x = np.sum(abs_x_positions * masses) / total_mass
            final_cg_y = np.sum(cy_v * masses) / total_mass
            
            # ---------------- IMAGE 1: FRONT & BACK SIDE VIEWS ONLY ----------------
            # Create a figure with 2 side-by-side plot subpanels.
            fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle(f"Layout Alternative {layout_idx} (CMA-ES)\nCombined Assembly CG Offset: ({final_cg_x:.4f}, {final_cg_y:.4f}) mm", 
                         fontsize=14, fontweight='bold')
            
            # Draw board outline and red border margins on both subplot panels.
            for ax in [ax_f, ax_b]:
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.3))
                ax.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing), panel_width, border_spacing, color='red', alpha=0.15))
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, border_spacing, color='red', alpha=0.15))
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))
                ax.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))

            # Draw components on Front and Back visual panels.
            for i in range(num_elements):
                ins_rad = insert_diams[i] / 2.0
                i_start = offset_indices[i]
                c_top, c_right, c_bot, c_left = clearance_dirs[i]

                # If component is mounted on FRONT:
                if not is_back_side[i]: 
                    ex, ey = cx_v[i], cy_v[i]
                    if element_shapes[i] == 'rectangle':
                        c_x = ex - lengths[i]/2.0 - c_left
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left + c_right
                        c_h = widths[i] + c_top + c_bot
                        # Draw clearance envelope (dashed rectangle)
                        ax_f.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.5))
                        # Draw solid component body (blue rectangle)
                        ax_f.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.9, edgecolor='navy', linewidth=1.5))
                    else:
                        # Draw clearance envelope (dashed circle)
                        ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.5))
                        # Draw solid component body (blue circle)
                        ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.9, edgecolor='navy', linewidth=1.5))
                    
                    # Draw mounting pins (red circles)
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_f.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
                        ax_b.add_patch(plt.Circle((-(ex + dx), ey + dy), ins_rad, facecolor='crimson', edgecolor='#5C0612', linewidth=1.5, alpha=0.6, zorder=3))
                        
                    # Add text name label on center of component
                    ax_f.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
                    
                    # Draw trace outline on Back panel
                    ex_trace_b = -cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        ax_b.add_patch(plt.Rectangle((ex_trace_b - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='navy', linewidth=1.2, alpha=0.7))
                    else:
                        ax_b.add_patch(plt.Circle((ex_trace_b, cy_v[i]), lengths[i]/2, fill=False, linestyle='--', edgecolor='navy', linewidth=1.2, alpha=0.7))
                
                # If component is mounted on BACK:
                else:
                    ex_b, ey_b = cx_v[i], cy_v[i]
                    if element_shapes[i] == 'rectangle':
                        c_x = ex_b - lengths[i]/2.0 - c_left
                        c_y = ey_b - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left + c_right
                        c_h = widths[i] + c_top + c_bot
                        # Draw clearance envelope (dashed green rectangle)
                        ax_b.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle='--', linewidth=1.2, alpha=0.5))
                        # Draw solid component body (green rectangle)
                        ax_b.add_patch(plt.Rectangle((ex_b - lengths[i]/2, ey_b - widths[i]/2), lengths[i], widths[i], color='darkgreen', alpha=0.9, edgecolor='darkslategrey', linewidth=1.5))
                    else:
                        # Draw clearance envelope (dashed green circle)
                        ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2 + clearance, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle='--', linewidth=1.2, alpha=0.5))
                        # Draw solid component body (green circle)
                        ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2, color='darkgreen', alpha=0.9, edgecolor='darkslategrey', linewidth=1.5))
                    
                    # Draw mounting pins (orange circles)
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_b.add_patch(plt.Circle((ex_b + dx, ey_b + dy), ins_rad, color='orange', zorder=4))
                        ax_f.add_patch(plt.Circle((-(ex_b + dx), ey_b + dy), ins_rad, facecolor='orange', edgecolor='#733D00', linewidth=1.5, alpha=0.6, zorder=3))
                        
                    # Add text name label
                    ax_b.text(ex_b, ey_b, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
                    
                    # Draw trace outline on Front panel
                    ex_trace_f = -cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        ax_f.add_patch(plt.Rectangle((ex_trace_f - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='darkslategrey', linewidth=1.2, alpha=0.7))
                    else:
                        ax_f.add_patch(plt.Circle((ex_trace_f, cy_v[i]), lengths[i]/2, fill=False, linestyle='--', edgecolor='darkslategrey', linewidth=1.2, alpha=0.7))

            # Format axis titles, zoom limits, aspect ratios, and background grids.
            for name, ax in [("FRONT SIDE VIEW", ax_f), ("BACK SIDE VIEW (FLIPPED)", ax_b)]:
                ax.set_title(name, fontweight='bold', fontsize=11)
                ax.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
                ax.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
                ax.set_aspect('equal') # Keep 1:1 aspect ratio so rectangles don't look stretched
                ax.grid(True, linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            # Save visual image file to disk
            main_layout_path = os.path.join(OUTPUT_DIR, f"layout_{layout_idx}_main.png")
            plt.savefig(main_layout_path, dpi=200, bbox_inches='tight')
            plt.close(fig) # Close figure plot from memory

            # ---------------- IMAGE 2: DISTANCE PROXIMITY MAPPING WITH TABLE BELOW ----------------
            # Create a second figure layout containing a top plot map and a bottom data summary table.
            fig_c = plt.figure(figsize=(16, 11))
            gs_c = fig_c.add_gridspec(2, 1, height_ratios=[6.5, 3.5], hspace=0.25)
            
            ax_c = fig_c.add_subplot(gs_c[0, 0])      # Top subpanel: Proximity plot map
            ax_table = fig_c.add_subplot(gs_c[1, 0])  # Bottom subpanel: Data table
            ax_table.axis('off')                      # Hide axis lines for table subpanel

            fig_c.suptitle(f"Layout Alternative {layout_idx} (CMA-ES) — Inter-Insert Proximity Verification (< 40mm)\n(Front Side Projection View)", 
                           fontsize=13, fontweight='bold')
            
            # Draw board outline on proximity map
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.2))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing), panel_width, border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.08))

            # Build list of rows to display in the lower data table.
            table_data = []

            for i in range(num_elements):
                i_start = offset_indices[i]
                ins_rad = insert_diams[i] / 2.0
                ey = cy_v[i]
                c_top, c_right, c_bot, c_left = clearance_dirs[i]
                side_str = "BACK" if is_back_side[i] else "FRONT"
                
                # Compute projection coordinates
                front_view_x = abs_x_positions[i]
                front_view_y = cy_v[i]

                # Append component data row [Name, Side, Weight, Dimensions, CoG X, CoG Y]
                table_data.append([
                    element_names[i],
                    side_str,
                    f"{masses[i]:.3f}",
                    f"{lengths[i]:.1f} x {widths[i]:.1f}" if element_shapes[i] == 'rectangle' else f"Ø {lengths[i]:.1f}",
                    f"{front_view_x:.2f}",
                    f"{front_view_y:.2f}"
                ])

                # Draw components onto proximity map projection panel
                if not is_back_side[i]:
                    ex = cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        c_x = ex - lengths[i]/2.0 - c_left
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left + c_right
                        c_h = widths[i] + c_top + c_bot
                        ax_c.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.35))
                        ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.75, edgecolor='navy', linewidth=1.5))
                    else:
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.35))
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.75, edgecolor='navy', linewidth=1.5))
                    
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_c.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
                        
                    ax_c.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

                else:
                    ex = -cx_v[i]
                    c_left_proj, c_right_proj = c_right, c_left
                    
                    if element_shapes[i] == 'rectangle':
                        c_x = ex - lengths[i]/2.0 - c_left_proj
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left_proj + c_right_proj
                        c_h = widths[i] + c_top + c_bot
                        ax_c.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle=':', linewidth=1.2, alpha=0.35))
                        ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='darkgreen', linewidth=1.5, zorder=3))
                    else:
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle=':', linewidth=1.2, alpha=0.35))
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, fill=False, linestyle='--', edgecolor='darkgreen', linewidth=1.5, zorder=3))
                    
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_c.add_patch(plt.Circle((ex - dx, ey + dy), ins_rad, facecolor='orange', edgecolor='#733D00', linewidth=1.2, alpha=0.85, zorder=4))
                        
                    ax_c.text(ex, ey, element_names[i], color='darkgreen', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

            close_pairs_labels = []
            label_counter = 0
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

            # Measure distance between every pair of mounting pins across different components.
            for i in range(num_elements):
                i_start = offset_indices[i]
                for j in range(i + 1, num_elements):
                    j_start = offset_indices[j]
                    
                    for ii in range(offset_counts[i]):
                        idx_i = i_start + ii
                        xi_abs = abs_x_positions[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                        yi_abs = cy_v[i] + flat_offsets[idx_i, 1]
                        
                        for jj in range(offset_counts[j]):
                            idx_j = j_start + jj
                            xj_abs = abs_x_positions[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                            yj_abs = cy_v[j] + flat_offsets[idx_j, 1]
                            
                            dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                            
                            # If pins are closer than 40 mm, draw a dashed connecting line and label it on plot!
                            if dist < 40.0:
                                xf1 = cx_v[i] + flat_offsets[idx_i, 0] if not is_back_side[i] else -(cx_v[i] + flat_offsets[idx_i, 0])
                                yf1 = cy_v[i] + flat_offsets[idx_i, 1]
                                xf2 = cx_v[j] + flat_offsets[idx_j, 0] if not is_back_side[j] else -(cx_v[j] + flat_offsets[idx_j, 0])
                                yf2 = cy_v[j] + flat_offsets[idx_j, 1]
                                
                                current_letter = alphabet[label_counter % len(alphabet)]
                                label_counter += 1
                                
                                # Draw purple line connecting close pins
                                ax_c.plot([xf1, xf2], [yf1, yf2], color='purple', linestyle='--', linewidth=1.5, alpha=0.85, zorder=10)
                                # Add letter bubble at center of line
                                ax_c.text((xf1 + xf2)/2, (yf1 + yf2)/2, current_letter, color='white', 
                                          fontsize=8, fontweight='bold', ha='center', va='center',
                                          bbox=dict(boxstyle="circle,pad=0.2", fc="darkmagenta", ec="none", alpha=0.9), zorder=11)
                                
                                close_pairs_labels.append(f"{current_letter} — {dist:.2f} mm ({element_names[i]} ↔ {element_names[j]})")

            # Display side legend listing pin distance measurements.
            if close_pairs_labels:
                legend_box_text = "\n".join(close_pairs_labels)
                ax_c.text(1.02, 0.95, f"Pin Distances Summary:\n\n{legend_box_text}", 
                          transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))
            else:
                ax_c.text(1.02, 0.95, "Pin Distances Summary:\n\nNo insert violations\nor pins found under 40mm.", 
                          transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))

            # Add color legends for Front and Back component layers
            front_patch = mpatches.Patch(color='royalblue', alpha=0.75, label='Front Layer Components')
            back_patch = mpatches.Patch(facecolor='none', edgecolor='darkgreen', linestyle='--', label='Back Layer Components (Projected Outline)')
            ax_c.legend(handles=[front_patch, back_patch], loc='lower left')

            ax_c.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
            ax_c.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
            ax_c.set_aspect('equal')
            ax_c.grid(True, linestyle=':', alpha=0.4)
            
            # --- TABLE GENERATION UNDER MAPPING IMAGE ---
            headers = ["Component Name", "Layer Placement", "Mass (kg)", "Dimensions (mm)", "CoG X (mm)", "CoG Y (mm)"]
            ui_table = ax_table.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
            ui_table.auto_set_font_size(False)
            ui_table.set_fontsize(9)
            ui_table.scale(1.0, 1.3)
            
            # Format table header row styling (Dark blue background with bold white text)
            for col_idx in range(len(headers)):
                cell = ui_table[0, col_idx]
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2C3E50')
            
            plt.tight_layout()
            # Save distance map layout image file to disk
            distance_layout_path = os.path.join(OUTPUT_DIR, f"layout_{layout_idx}_distance_map.png")
            plt.savefig(distance_layout_path, dpi=200, bbox_inches='tight')
            plt.close(fig_c) # Close figure plot from memory

        print(f"📁 All images generated and saved to: {os.path.abspath(OUTPUT_DIR)}/")

# Special Python safeguard: Ensures the optimization script runs only when directly executed by the user.
if __name__ == '__main__':
    run_optimization()