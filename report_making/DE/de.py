# Import the 'os' module to interact with the computer's operating system (e.g., file paths, directory creation)
import os
# Import the 'time' module to measure program execution runtime
import time
# Import the 'json' module to load and parse structural configuration files written in JSON format
import json
# Import the 'warnings' module to manage and suppress non-critical system warnings
import warnings
# Import 'numpy' (aliased as 'np'), a fundamental library for fast matrix calculations and numerical operations
import numpy as np
# Import the main 'matplotlib' plotting library used to construct figures and graphics
import matplotlib
# Force matplotlib to use the 'Agg' non-interactive backend to render graphics without spawning GUI pop-up windows
matplotlib.use('Agg')  # Headless backend: disables GUI rendering completely
# Import 'pyplot' from matplotlib to construct and format custom multi-panel graphics and plots
import matplotlib.pyplot as plt
# Import drawing patches from matplotlib to render geometric shapes such as rectangles and circles
import matplotlib.patches as mpatches
# Import the 'differential_evolution' global optimizer from SciPy to search for ideal component placements
from scipy.optimize import differential_evolution
# Import Numba tools ('njit' for fast compilation, 'prange' for multithreaded CPU loops)
from numba import njit, prange

# Mute all non-critical program warning messages to ensure a clean console output
warnings.filterwarnings("ignore")

# ================= STRUCTURAL CONFIGURATION ==================
# Define the maximum allowed physical offset (in mm) between the combined assembly Center of Gravity and board center (0,0)
CG_TOLERANCE = 1.0  # mm: Target combined center of gravity
# Set the target folder path where rendered diagram images will be generated and stored
OUTPUT_DIR = "optimized_layouts"
# Automatically create the output directory on the filesystem if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= NUMBA PARALLEL ACCELERATED CORE MATHEMATICS =================
# Compile the mathematical scoring core into ultra-fast machine code using Numba's JIT compiler
@njit(fastmath=True, cache=True)
# Define the core fitness function that calculates and returns penalty scores for candidate layouts
def compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, 
                         flat_offsets, offset_counts, offset_indices, is_back_side,
                         clearance_dirs, panel_width, panel_height, border_spacing):
    # Initialize the total assembly weight accumulator variable to zero
    total_mass = 0.0
    # Initialize the horizontal center-of-gravity moment accumulator variable to zero
    cg_x = 0.0
    # Initialize the vertical center-of-gravity moment accumulator variable to zero
    cg_y = 0.0
    
    # Allocate a double-precision floating-point array to hold true physical horizontal coordinates of components
    abs_x = np.zeros(num_components)
    # Loop over every component in the assembly to evaluate mass properties and spatial position
    for i in range(num_components):
        # Retrieve the individual mass value of the current component
        m = masses[i]
        # Add the component's mass to the running assembly mass sum
        total_mass += m
        # Determine whether the component is mounted on the back side layer
        if is_back_side[i]:
            # Negate the X-coordinate for back-side components to reflect physical horizontal mirroring
            abs_x[i] = -cx[i]
        # Otherwise, the component is mounted on the front side layer
        else:
            # Maintain the standard X-coordinate for front-side components
            abs_x[i] = cx[i]
        
        # Calculate horizontal moment contribution (mass multiplied by absolute horizontal position)
        cg_x += abs_x[i] * m
        # Calculate vertical moment contribution (mass multiplied by vertical position)
        cg_y += cy[i] * m
        
    # Guard against division by zero in edge cases where all component masses evaluate to zero
    if total_mass == 0.0:
        # Return an arbitrarily huge penalty value to reject physically invalid component models
        return 1e15
        
    # Calculate the overall assembly X-axis Center of Gravity by dividing total horizontal moment by total mass
    cg_x /= total_mass
    # Calculate the overall assembly Y-axis Center of Gravity by dividing total vertical moment by total mass
    cg_y /= total_mass
    
    # EQUAL TOP PRIORITY PENALTY SCALE
    # Define a high-order penalty multiplier (100 million) to penalize rule violations heavily
    PENALTY_WEIGHT = 1e8

    # Calculate the active left boundary limit of the board by applying outer border keepout margin
    active_x_min = -panel_width / 2.0 + border_spacing
    # Calculate the active right boundary limit of the board by applying outer border keepout margin
    active_x_max =  panel_width / 2.0 - border_spacing
    # Calculate the active bottom boundary limit of the board by applying outer border keepout margin
    active_y_min = -panel_height / 2.0 + border_spacing
    # Calculate the active top boundary limit of the board by applying outer border keepout margin
    active_y_max =  panel_height / 2.0 - border_spacing

    # 1. CENTER OF GRAVITY PENALTY
    # Compute the Euclidean offset distance of the Center of Gravity from the physical origin (0, 0)
    cg_offset = np.sqrt(cg_x**2 + cg_y**2)
    # Initialize Center of Gravity penalty value to zero
    cg_penalty = 0.0
    # Check whether the total Center of Gravity drift exceeds the maximum allowable tolerance limit (1.0 mm)
    if cg_offset > CG_TOLERANCE:
        # Calculate a quadratic penalty scaled to the magnitude of the Center of Gravity displacement violation
        cg_penalty = ((cg_offset - CG_TOLERANCE) ** 2) * PENALTY_WEIGHT

    # 2. STRICT CLEARANCE OVERLAP & BORDER KEEPOUT PENALTY
    # Initialize inter-component physical collision penalty accumulator to zero
    overlap_penalty = 0.0
    # Initialize outer border encroachment penalty accumulator to zero
    border_penalty = 0.0

    # Iterate over all components to verify geometric keepout and clearance boundaries
    for i in range(num_components):
        # Extract top clearance margin requirement for component 'i'
        c_top = clearance_dirs[i, 0]
        # Extract right clearance margin requirement for component 'i' (swapping sides if back-mounted)
        c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
        # Extract bottom clearance margin requirement for component 'i'
        c_bot = clearance_dirs[i, 2]
        # Extract left clearance margin requirement for component 'i' (swapping sides if back-mounted)
        c_left = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]

        # Calculate minimum horizontal extent (left boundary) including clearance zone
        i_x_min = abs_x[i] - hl[i] - c_left
        # Calculate maximum horizontal extent (right boundary) including clearance zone
        i_x_max = abs_x[i] + hl[i] + c_right
        # Calculate minimum vertical extent (bottom boundary) including clearance zone
        i_y_min = cy[i] - hw[i] - c_bot
        # Calculate maximum vertical extent (top boundary) including clearance zone
        i_y_max = cy[i] + hw[i] + c_top

        # --- RULE 1: STRICT ZERO OVERLAP WITH BOARD BORDER ---
        # Quantify left boundary border violation distance
        viol_left   = max(0.0, active_x_min - i_x_min)
        # Quantify right boundary border violation distance
        viol_right  = max(0.0, i_x_max - active_x_max)
        # Quantify bottom boundary border violation distance
        viol_bottom = max(0.0, active_y_min - i_y_min)
        # Quantify top boundary border violation distance
        viol_top    = max(0.0, i_y_max - active_y_max)

        # Sum total border encroachment distances across all four edges
        total_border_viol = viol_left + viol_right + viol_bottom + viol_top
        # If any portion of component 'i' breaches the forbidden border zone
        if total_border_viol > 0.0:
            # Apply quadratic penalty proportional to the border encroachment magnitude
            border_penalty += (total_border_viol ** 2) * PENALTY_WEIGHT

        # --- RULE 2 & 3: STRICT ZERO OVERLAP WITH OTHER SAME-SIDE COMPONENTS/CLEARANCES ---
        # Compare component 'i' against all subsequent components 'j' to evaluate potential overlaps
        for j in range(i + 1, num_components):
            # Evaluate collisions only if both components reside on the same board layer (FRONT vs BACK)
            if is_back_side[i] == is_back_side[j]:
                # Extract top clearance margin requirement for component 'j'
                cj_top = clearance_dirs[j, 0]
                # Extract right clearance margin requirement for component 'j' (swapping sides if back-mounted)
                cj_right = clearance_dirs[j, 3] if is_back_side[j] else clearance_dirs[j, 1]
                # Extract bottom clearance margin requirement for component 'j'
                cj_bot = clearance_dirs[j, 2]
                # Extract left clearance margin requirement for component 'j' (swapping sides if back-mounted)
                cj_left = clearance_dirs[j, 1] if is_back_side[j] else clearance_dirs[j, 3]

                # Calculate component 'j' left boundary edge including clearance
                j_x_min = abs_x[j] - hl[j] - cj_left
                # Calculate component 'j' right boundary edge including clearance
                j_x_max = abs_x[j] + hl[j] + cj_right
                # Calculate component 'j' bottom boundary edge including clearance
                j_y_min = cy[j] - hw[j] - cj_bot
                # Calculate component 'j' top boundary edge including clearance
                j_y_max = cy[j] + hw[j] + cj_top

                # Calculate the horizontal overlap length between clearance bounding boxes
                overlap_x = max(0.0, min(i_x_max, j_x_max) - max(i_x_min, j_x_min))
                # Calculate the vertical overlap height between clearance bounding boxes
                overlap_y = max(0.0, min(i_y_max, j_y_max) - max(i_y_min, j_y_min))

                # Check if non-zero overlap exists along both horizontal and vertical axes simultaneously
                if overlap_x > 0.0 and overlap_y > 0.0:
                    # Calculate total 2D bounding box intersection area
                    overlap_area = overlap_x * overlap_y
                    # Apply quadratic penalty proportional to the square of the intersection area
                    overlap_penalty += (overlap_area ** 2) * PENALTY_WEIGHT

    # 3. PIN INSERT SAFE DISTANCE PENALTY
    # Initialize pin clearance violation penalty accumulator to zero
    insert_penalty = 0.0
    # Loop over all components to verify physical mounting pin separation distances
    for i in range(num_components):
        # Obtain start memory offset for component 'i' mounting pin array
        i_start = offset_indices[i]
        # Retrieve total mounting pin count for component 'i'
        i_count = offset_counts[i]
        
        # Compare mounting pins of component 'i' against mounting pins of component 'j'
        for j in range(i + 1, num_components):
            # Obtain start memory offset for component 'j' mounting pin array
            j_start = offset_indices[j]
            # Retrieve total mounting pin count for component 'j'
            j_count = offset_counts[j]
            # Lookup precomputed required pin separation threshold distance between component pair (i, j)
            req_dist = req_d_matrix[i, j]
            
            # Loop over all individual mounting pins associated with component 'i'
            for ii in range(i_count):
                # Compute flat array index for pin 'ii' of component 'i'
                idx_i = i_start + ii
                # Compute absolute horizontal coordinate of pin 'ii' considering board side flip
                xi_abs = abs_x[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                # Compute absolute vertical coordinate of pin 'ii'
                yi_abs = cy[i] + flat_offsets[idx_i, 1]
                
                # Loop over all individual mounting pins associated with component 'j'
                for jj in range(j_count):
                    # Compute flat array index for pin 'jj' of component 'j'
                    idx_j = j_start + jj
                    # Compute absolute horizontal coordinate of pin 'jj' considering board side flip
                    xj_abs = abs_x[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                    # Compute absolute vertical coordinate of pin 'jj'
                    yj_abs = cy[j] + flat_offsets[idx_j, 1]
                    
                    # Compute straight-line 2D Euclidean distance between pin 'ii' and pin 'jj'
                    dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                    # Check if actual pin separation distance is less than required separation distance
                    if req_dist > dist:
                        # Apply quadratic penalty proportional to the magnitude of pin distance violation
                        insert_penalty += ((req_dist - dist) ** 2) * PENALTY_WEIGHT
                        
    # Return total aggregated penalty score (a value of 0.0 indicates full physical compliance)
    return cg_penalty + overlap_penalty + border_penalty + insert_penalty


# Interface function linking SciPy optimization routines to Numba accelerated mathematical core
def evaluate(individual, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing):
    # Slice out component center X-coordinates from odd positions in optimizer guess vector
    cx = individual[::2]
    # Slice out component center Y-coordinates from even positions in optimizer guess vector
    cy = individual[1::2]
    # Determine the total number of components contained in parameter list
    num_components = len(masses)
    # Calculate half-lengths (center-to-edge radius along horizontal axis)
    hl = lengths / 2.0
    # Calculate half-widths (center-to-edge radius along vertical axis)
    hw = widths / 2.0
    
    # Execute Numba accelerated core calculation and return composite fitness score
    return compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing)


# Main execution routine that loads specifications, executes optimization, and generates visual reports
def run_optimization():
    # Define primary target JSON configuration path containing board and component data
    filepath = 'newobj.json'
    # Check if target JSON file exists in current execution context directory
    if not os.path.exists(filepath):
        # Re-construct path relative to system working directory as fallback
        filepath = os.path.join(os.getcwd(), 'newobj.json')

    # Terminate execution if configuration file cannot be found in system path
    if not os.path.exists(filepath):
        # Raise explicit FileNotFoundError exception alerting user to missing input file
        raise FileNotFoundError("❌ Critical Error: Unable to locate 'newobj.json' in working directory.")

    # Log configuration confirmation to system console
    print(f"\n[✓] Config loaded successfully: '{filepath}'")

    # Open target configuration file in read-only mode
    with open(filepath, 'r') as f:
        # Deserialize JSON data stream into structured Python dictionary
        data = json.load(f)

    # Extract board total horizontal length in millimeters
    panel_width = float(data['BOARD']['Length (mm)'])
    # Extract board total vertical height in millimeters
    panel_height = float(data['BOARD']['Breadth (mm)'])
    # Extract board outer border keepout width in millimeters
    border_spacing = float(data['BOARD']['Border (mm)'])
    # Extract baseline inter-component keepout clearance distance in millimeters
    clearance = float(data['BOARD']['Clearance (mm)'])

    # Initialize empty list to accumulate component mass and geometric parameters
    element_data = []
    # Initialize empty list to accumulate component string names
    element_names = []
    # Initialize empty list to accumulate component geometry types (rectangle/circle)
    element_shapes = []
    # Initialize empty list to accumulate component layer placement flags
    is_back_side_list = []
    # Initialize empty list to accumulate component pin offset vector sequences
    flat_offsets_list = []
    # Initialize empty list to accumulate component directional clearance configurations
    clearance_dirs_list = []
    
    # Initialize temporary sequence to store unrolled raw JSON component entries
    raw_components_to_process = []
    # Process component specifications defined under FRONT board layer
    for comp in data['COMPONENTS'].get('FRONT', []):
        # Append component payload coupled with layer indicator False (FRONT)
        raw_components_to_process.append((comp, False))
    # Process component specifications defined under BACK board layer
    for comp in data['COMPONENTS'].get('BACK', []):
        # Append component payload coupled with layer indicator True (BACK)
        raw_components_to_process.append((comp, True))

    # Initialize list to contain expanded individual instances of components
    components_to_process = []
    # Unpack components to handle multiplicity based on requested item quantity
    for comp, is_back in raw_components_to_process:
        # Extract requested quantity count, defaulting to 1 instance if unspecified
        qty = int(comp.get('Object qty', 1))
        # If item quantity exceeds 1, create distinct uniquely-named instances
        if qty > 1:
            # Loop through requested instance count
            for q in range(qty):
                # Duplicate component property dictionary
                cloned_comp = comp.copy()
                # Append sequence index to establish unique instance identifier
                cloned_comp['Unique Name'] = f"{comp['Object name']}{q}"
                # Append cloned instance tuple to master processing queue
                components_to_process.append((cloned_comp, is_back))
        # Handle single instance component specification
        else:
            # Duplicate component property dictionary
            cloned_comp = comp.copy()
            # Retain primary name string as unique identifier
            cloned_comp['Unique Name'] = comp['Object name']
            # Append instance tuple to master processing queue
            components_to_process.append((cloned_comp, is_back))

    # Quantify total count of discrete elements requiring optimization placement
    num_elements = len(components_to_process)
    # Initialize zero-filled integer array to track mounting pin count per component
    offset_counts = np.zeros(num_elements, dtype=np.int32)
    # Initialize zero-filled integer array to track pin offset memory indices
    offset_indices = np.zeros(num_elements, dtype=np.int32)
    
    # Track cumulative pin offset memory index pointer
    current_idx = 0
    # Process individual expanded component entries
    for idx, (component, is_back) in enumerate(components_to_process):
        # Retrieve unique string identifier for current component
        element_name = component['Unique Name']
        # Read geometric shape property, defaulting to 'rectangle'
        shape = component.get('Shape', 'rectangle')
        # Append shape identifier to master geometry tracking list
        element_shapes.append(shape)
        # Store integer layer flag (1 = BACK layer, 0 = FRONT layer)
        is_back_side_list.append(1 if is_back else 0)
        
        # Read custom face clearance declarations if present in JSON definition
        cf_faces = component.get('CF', [])
        # Read custom face clearance distance, falling back to global clearance if absent
        cf_len = float(component.get('CFLen (mm)', clearance))
        
        # Initialize default directional clearances [Top, Right, Bottom, Left]
        c_dirs = [clearance, clearance, clearance, clearance]
        # Assign custom face clearance distances based on target face keys
        for face in cf_faces:
            # Override Top clearance if face key 1 is present
            if face == 1:
                c_dirs[0] = cf_len
            # Override Right clearance if face key 2 is present
            elif face == 2:
                c_dirs[1] = cf_len
            # Override Bottom clearance if face key 3 is present
            elif face == 3:
                c_dirs[2] = cf_len
            # Override Left clearance if face key 4 is present
            elif face == 4:
                c_dirs[3] = cf_len
                
        # Append directional clearance configuration to tracking master list
        clearance_dirs_list.append(c_dirs)

        # Parse geometric properties and generate pin offset layouts for rectangular components
        if shape == 'rectangle':
            # Extract component horizontal length dimension in mm
            length = float(component['Length (mm)'])
            # Extract component vertical width dimension in mm
            width = float(component['Breadth (mm)'])
            # Extract mounting pin diameter in mm
            insert_diam = float(component['Insert (mm)'])
            # Extract mounting pin count, defaulting to 4 corner pins
            insert_qty = int(component.get('Insert qty', 4))
            # Calculate half-length and half-width values for relative pin displacement
            half_l, half_w = length / 2.0, width / 2.0
            
            # Construct pin offset geometry for 2-pin components
            if insert_qty == 2:
                # Align pins along the major axis depending on aspect ratio
                if length >= width:
                    # Place pins on left and right center edges
                    offsets = [(half_l, 0.0), (-half_l, 0.0)]
                else:
                    # Place pins on top and bottom center edges
                    offsets = [(0.0, half_w), (0.0, -half_w)]
            # Construct pin offset geometry for 6-pin components
            elif insert_qty == 6:
                # Place 4 pins on component corners
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
                # Add 2 additional mid-edge pins along major axis
                if length >= width:
                    # Append top-center and bottom-center pin positions
                    offsets.extend([(0.0, half_w), (0.0, -half_w)])
                else:
                    # Append left-center and right-center pin positions
                    offsets.extend([(half_l, 0.0), (-half_l, 0.0)])
            # Construct pin offset geometry for default 4-pin components
            else:
                # Place 4 pins directly at component bounding box corners
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
        # Parse geometric properties and generate pin offset layouts for circular components
        else:
            # Extract component diameter dimension in mm
            diameter = float(component['Diameter (mm)'])
            # Set length and width equivalent to circular diameter
            length, width = diameter, diameter
            # Extract mounting pin diameter in mm
            insert_diam = float(component['Insert (mm)'])
            # Extract mounting pin count, defaulting to 3 symmetric pins
            insert_qty = int(component.get('Insert qty', 3))
            # Determine radial placement distance for mounting pins near outer boundary
            inner_radius = max(0.0, (diameter / 2.0) - 1.0)
            # Compute radially symmetric pin locations using polar coordinate transformation
            offsets = [(inner_radius * np.cos(2 * np.pi * k / insert_qty), 
                        inner_radius * np.sin(2 * np.pi * k / insert_qty)) for k in range(insert_qty)]
                        
        # Append pin offset locations to flat offset accumulator list
        flat_offsets_list.extend(offsets)
        # Store total pin count assigned to current component
        offset_counts[idx] = len(offsets)
        # Store initial flat array memory index pointer for current component
        offset_indices[idx] = current_idx
        # Increment flat array memory index pointer by pin count
        current_idx += len(offsets)
        
        # Append component weight, dimensions, and pin diameter parameters to element data matrix
        element_data.append([float(component['Weight (kg)']), length, width, insert_diam])
        # Append component string name to name master list
        element_names.append(element_name)

    # Convert flat offset list into contiguous 2D double-precision NumPy array
    flat_offsets = np.array(flat_offsets_list, dtype=np.float64)
    # Convert element parameter list into 2D floating-point NumPy array
    element_data = np.array(element_data, dtype=float)
    # Extract component masses into standalone 1D array
    masses = element_data[:, 0]
    # Extract component lengths into standalone 1D array
    lengths = element_data[:, 1]
    # Extract component widths into standalone 1D array
    widths = element_data[:, 2]
    # Extract mounting pin diameters into standalone 1D array
    insert_diams = element_data[:, 3]
    # Convert back-side placement list into 1D integer NumPy array
    is_back_side = np.array(is_back_side_list, dtype=np.int32)
    # Convert directional clearances into 2D double-precision NumPy array
    clearance_dirs = np.array(clearance_dirs_list, dtype=np.float64)
    # Store total component count
    n = len(element_data)

    # Tight Variable Bounds Calculation (Flipped for back components)
    # Initialize list to hold variable optimization coordinate boundaries (min/max X, Y)
    bounds = []
    # Calculate feasible placement bounds for each individual component
    for i in range(num_elements):
        # Read pin array start memory index
        i_start = offset_indices[i]
        # Read pin array count
        i_count = offset_counts[i]
        # Extract component relative pin offsets
        comp_offsets = flat_offsets[i_start : i_start + i_count]
        
        # Determine maximum relative horizontal pin offset from origin
        max_off_x = max(abs(x) for x, _ in comp_offsets) if i_count > 0 else 0
        # Determine maximum relative vertical pin offset from origin
        max_off_y = max(abs(y) for _, y in comp_offsets) if i_count > 0 else 0
        
        # Read component Top clearance margin
        c_top = clearance_dirs[i, 0]
        # Read component Right clearance margin (swapping sides if back-mounted)
        c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
        # Read component Bottom clearance margin
        c_bot = clearance_dirs[i, 2]
        # Read component Left clearance margin (swapping sides if back-mounted)
        c_left = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]
        
        # Determine total required left border margin accounting for body and pins
        req_margin_left = max(lengths[i]/2.0 + c_left, max_off_x + insert_diams[i]/2.0)
        # Determine total required right border margin accounting for body and pins
        req_margin_right = max(lengths[i]/2.0 + c_right, max_off_x + insert_diams[i]/2.0)
        # Determine total required bottom border margin accounting for body and pins
        req_margin_bottom = max(widths[i]/2.0 + c_bot, max_off_y + insert_diams[i]/2.0)
        # Determine total required top border margin accounting for body and pins
        req_margin_top = max(widths[i]/2.0 + c_top, max_off_y + insert_diams[i]/2.0)
        
        # Set spatial horizontal movement limits for FRONT-mounted components
        if not is_back_side[i]:
            # Calculate lower horizontal coordinate bound (X-min)
            x_min = -panel_width / 2.0 + border_spacing + req_margin_left
            # Calculate upper horizontal coordinate bound (X-max)
            x_max =  panel_width / 2.0 - border_spacing - req_margin_right
        # Set spatial horizontal movement limits for BACK-mounted components (swapping left/right bounds)
        else:
            # Calculate lower horizontal coordinate bound with reversed margins
            x_min = -panel_width / 2.0 + border_spacing + req_margin_right
            # Calculate upper horizontal coordinate bound with reversed margins
            x_max =  panel_width / 2.0 - border_spacing - req_margin_left

        # Calculate lower vertical coordinate bound (Y-min)
        y_min = -panel_height / 2.0 + border_spacing + req_margin_bottom
        # Calculate upper vertical coordinate bound (Y-max)
        y_max =  panel_height / 2.0 - border_spacing - req_margin_top
        
        # Append horizontal coordinate bounds tuple (min, max)
        bounds.append((x_min, x_max))
        # Append vertical coordinate bounds tuple (min, max)
        bounds.append((y_min, y_max))

    # Initialize symmetric matrix for pairwise required pin separation distances, default 30mm
    req_d_matrix = np.full((n, n), 30.0)
    # Populate required pin separation matrix based on component pin diameter combinations
    for i in range(n):
        for j in range(n):
            # Reduce required separation to 24mm if both components feature 4mm pins
            if insert_diams[i] == 4.0 and insert_diams[j] == 4.0:
                req_d_matrix[i, j] = 24.0
            # Increase required separation to 36mm if both components feature 6mm pins
            elif insert_diams[i] == 6.0 and insert_diams[j] == 6.0:
                req_d_matrix[i, j] = 36.0

    # Initialize empty list to collect distinct valid layout solution vectors
    distinct_layouts = []
    # Set maximum allowable optimization attempt iterations
    max_attempts = 150
    # Initialize attempt iteration counter to zero
    attempt = 0

    # Warmup Numba JIT compiler by passing dummy vector through evaluation entry point
    dummy_ind = np.zeros(2 * num_elements)
    # Force Numba to compile functions before starting timed optimization loops
    evaluate(dummy_ind, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing)

    # Detect active system CPU core count for parallel worker allocation
    cpu_cores = os.cpu_count() or 8
    # Print system execution parameters to console
    print(f"🚀 M3 Pro Acceleration Active | Available Cores: {cpu_cores}")
    # Print task execution status message
    print("⚡ Executing Parallel Differential Evolution with Border & Clearance Keepouts...\n")
    # Record current timestamp to measure total execution runtime
    start_time = time.time()

    # Execute optimization attempts until 5 distinct compliant solutions are found or max attempts reached
    while len(distinct_layouts) < 5 and attempt < max_attempts:
        # Increment attempt counter
        attempt += 1
        # Generate random integer seed for Differential Evolution stochastic optimization
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))

        # Enclose optimizer invocation in exception handling block to handle math errors safely
        try:
            # Execute Differential Evolution global optimization process across CPU cores
            result = differential_evolution(
                evaluate, bounds,
                args=(masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing),
                maxiter=600, popsize=25, tol=1e-6,
                strategy='best1bin', seed=seed,
                workers=-1, updating='deferred'
            )
            
            # STRICT FILTER: Reject layout solutions containing non-zero penalty scores or numerical anomalies
            if result.fun > 1e-3 or np.any(np.isinf(result.x)) or np.any(np.isnan(result.x)):
                # Abandon non-compliant solution candidate
                continue

            # Extract component center horizontal positions from solution vector
            cx_v = result.x[::2]
            # Extract component center vertical positions from solution vector
            cy_v = result.x[1::2]
            
            # Compute total assembly mass
            total_mass = np.sum(masses)
            # Compute absolute horizontal component positions considering back layer flip
            abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
            # Compute evaluated layout horizontal Center of Gravity
            cg_x_v = np.sum(abs_x_positions * masses) / total_mass
            # Compute evaluated layout vertical Center of Gravity
            cg_y_v = np.sum(cy_v * masses) / total_mass

            # Reject solution candidate if Center of Gravity drift breaches tolerance threshold (1.0 mm)
            if abs(cg_x_v) > CG_TOLERANCE or abs(cg_y_v) > CG_TOLERANCE:
                # Abandon non-compliant solution candidate
                continue

            # Initialize flag to verify solution diversity relative to previously accepted solutions
            is_distinct = True
            # Compare candidate vector against existing distinct layout vectors
            for existing in distinct_layouts:
                # Calculate maximum component position shift relative to existing solutions
                if np.max(np.abs(result.x - existing)) < 6.0:
                    # Mark solution as non-distinct duplicate
                    is_distinct = False
                    # Terminate further comparative loop
                    break

            # Process candidate solution if unique, compliant, and verified
            if is_distinct:
                # Append accepted solution vector to distinct layout store
                distinct_layouts.append(result.x)
                # Print progress message detailing collected solution count and Center of Gravity offset
                print(f"  [✓] Layout Solution {len(distinct_layouts)}/5 calculated | CG Offset: ({cg_x_v:.2f}, {cg_y_v:.2f}) mm")

        # Catch operational numerical exceptions to prevent script termination
        except Exception:
            # Continue to subsequent optimization attempt
            continue

    # Log completion time summary to console
    print(f"\n✨ Execution completed in {time.time() - start_time:.2f}s. Saving plots...\n")

    # ================= SILENT IMAGE GENERATION & EXPORT =================
    # Check if at least one valid layout solution was successfully found
    if len(distinct_layouts) > 0:
        # Compute combined assembly mass sum for display reporting
        total_mass = np.sum(masses)
        
        # Iterate through distinct accepted layouts to generate visual reports
        for idx, layout in enumerate(distinct_layouts):
            # Establish 1-indexed solution layout number
            layout_idx = idx + 1
            # Extract component horizontal center positions for layout instance
            cx_v = layout[::2]
            # Extract component vertical center positions for layout instance
            cy_v = layout[1::2]
            # Compute absolute horizontal coordinates (normalized to FRONT-view coordinate system)
            abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
            
            # Compute exact evaluated X-axis Center of Gravity coordinate
            final_cg_x = np.sum(abs_x_positions * masses) / total_mass
            # Compute exact evaluated Y-axis Center of Gravity coordinate
            final_cg_y = np.sum(cy_v * masses) / total_mass
            
            # ---------------- IMAGE 1: STANDARD LAYOUT REPORT (CLEANED - NO TABLE) ----------------
            # Create a high-resolution Matplotlib figure canvas
            fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Set figure main title displaying layout option index and exact Center of Gravity offset
            fig.suptitle(f"Layout Alternative {layout_idx}\nCombined Assembly CG Offset: ({final_cg_x:.4f}, {final_cg_y:.4f}) mm", 
                         fontsize=14, fontweight='bold')
            
            # Draw board substrate and border keepout zones across FRONT and BACK view subplots
            for ax in [ax_f, ax_b]:
                # Render gray rectangle representing substrate body
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.3))
                # Render top red shaded strip representing forbidden border keepout zone
                ax.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing), panel_width, border_spacing, color='red', alpha=0.15))
                # Render bottom red shaded strip representing forbidden border keepout zone
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, border_spacing, color='red', alpha=0.15))
                # Render left red shaded strip representing forbidden border keepout zone
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))
                # Render right red shaded strip representing forbidden border keepout zone
                ax.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))

            # Initialize empty list to accumulate data rows for summary table
            table_data = []

            # Render component graphics and gather metrics for display table
            for i in range(num_elements):
                # Calculate radius of component mounting pins
                ins_rad = insert_diams[i] / 2.0
                # Retrieve starting pin index pointer
                i_start = offset_indices[i]
                # Determine human-readable layer string indicator
                side_str = "BACK" if is_back_side[i] else "FRONT"
                
                # APPEND DATA WITH SEPARATED X, Y AND FRONT-VIEW NORMALIZED X COORDINATE
                table_data.append([
                    element_names[i],
                    side_str,
                    f"{masses[i]:.3f}",
                    f"{lengths[i]:.1f} x {widths[i]:.1f}" if element_shapes[i] == 'rectangle' else f"Ø {lengths[i]:.1f}",
                    f"{abs_x_positions[i]:.2f}",
                    f"{cy_v[i]:.2f}"
                ])
                
                # Extract directional clearance values for component 'i'
                c_top, c_right, c_bot, c_left = clearance_dirs[i]

                # Draw graphics for FRONT-mounted components
                if not is_back_side[i]: 
                    # Set component coordinates
                    ex, ey = cx_v[i], cy_v[i]
                    # Render rectangular component geometry on FRONT view
                    if element_shapes[i] == 'rectangle':
                        # Calculate clearance boundary origin coordinates
                        c_x = ex - lengths[i]/2.0 - c_left
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left + c_right
                        c_h = widths[i] + c_top + c_bot
                        # Draw light blue dashed keepout boundary box
                        ax_f.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.5))
                        # Draw solid blue rectangle representing physical component body
                        ax_f.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.9, edgecolor='navy', linewidth=1.5))
                    # Render circular component geometry on FRONT view
                    else:
                        # Draw light blue dashed keepout boundary circle
                        ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.5))
                        # Draw solid blue circle representing physical component body
                        ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.9, edgecolor='navy', linewidth=1.5))
                    
                    # Render mounting pins for FRONT component
                    for ii in range(offset_counts[i]):
                        # Extract relative pin displacement
                        dx, dy = flat_offsets[i_start + ii]
                        # Render crimson pin circle on FRONT subplot
                        ax_f.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
                        # Render mirrored reference pin trace on BACK subplot for visual alignment
                        ax_b.add_patch(plt.Circle((-(ex + dx), ey + dy), ins_rad, facecolor='crimson', edgecolor='#5C0612', linewidth=1.5, alpha=0.6, zorder=3))
                        
                    # Annotate component name text string at physical body center
                    ax_f.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
                    
                    # Compute mirrored X-coordinate for reference trace on BACK subplot
                    ex_trace_b = -cx_v[i]
                    # Render dashed component body reference outline on BACK subplot
                    if element_shapes[i] == 'rectangle':
                        ax_b.add_patch(plt.Rectangle((ex_trace_b - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='navy', linewidth=1.2, alpha=0.7))
                    else:
                        ax_b.add_patch(plt.Circle((ex_trace_b, cy_v[i]), lengths[i]/2, fill=False, linestyle='--', edgecolor='navy', linewidth=1.2, alpha=0.7))
                # Draw graphics for BACK-mounted components
                else:
                    # Set component coordinates
                    ex_b, ey_b = cx_v[i], cy_v[i]
                    # Render rectangular component geometry on BACK view
                    if element_shapes[i] == 'rectangle':
                        # Calculate clearance boundary origin coordinates
                        c_x = ex_b - lengths[i]/2.0 - c_left
                        c_y = ey_b - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left + c_right
                        c_h = widths[i] + c_top + c_bot
                        # Draw light green dashed keepout boundary box
                        ax_b.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle='--', linewidth=1.2, alpha=0.5))
                        # Draw solid dark green rectangle representing physical component body
                        ax_b.add_patch(plt.Rectangle((ex_b - lengths[i]/2, ey_b - widths[i]/2), lengths[i], widths[i], color='darkgreen', alpha=0.9, edgecolor='darkslategrey', linewidth=1.5))
                    # Render circular component geometry on BACK view
                    else:
                        # Draw light green dashed keepout boundary circle
                        ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2 + clearance, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle='--', linewidth=1.2, alpha=0.5))
                        # Draw solid dark green circle representing physical component body
                        ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2, color='darkgreen', alpha=0.9, edgecolor='darkslategrey', linewidth=1.5))
                    
                    # Render mounting pins for BACK component
                    for ii in range(offset_counts[i]):
                        # Extract relative pin displacement
                        dx, dy = flat_offsets[i_start + ii]
                        # Render orange pin circle on BACK subplot
                        ax_b.add_patch(plt.Circle((ex_b + dx, ey_b + dy), ins_rad, color='orange', zorder=4))
                        # Render mirrored reference pin trace on FRONT subplot for visual alignment
                        ax_f.add_patch(plt.Circle((-(ex_b + dx), ey_b + dy), ins_rad, facecolor='orange', edgecolor='#733D00', linewidth=1.5, alpha=0.6, zorder=3))
                        
                    # Annotate component name text string at physical body center
                    ax_b.text(ex_b, ey_b, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
                    
                    # Compute mirrored X-coordinate for reference trace on FRONT subplot
                    ex_trace_f = -cx_v[i]
                    # Render dashed component body reference outline on FRONT subplot
                    if element_shapes[i] == 'rectangle':
                        ax_f.add_patch(plt.Rectangle((ex_trace_f - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='darkslategrey', linewidth=1.2, alpha=0.7))
                    else:
                        ax_f.add_patch(plt.Circle((ex_trace_f, cy_v[i]), lengths[i]/2, fill=False, linestyle='--', edgecolor='darkslategrey', linewidth=1.2, alpha=0.7))

            # Apply title, axis limits, aspect ratio, and grid formatting to subplots
            for name, ax in [("FRONT SIDE VIEW", ax_f), ("BACK SIDE VIEW (FLIPPED)", ax_b)]:
                # Set subplot heading title
                ax.set_title(name, fontweight='bold', fontsize=11)
                # Set horizontal X axis view limits with padding
                ax.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
                # Set vertical Y axis view limits with padding
                ax.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
                # Force equal 1:1 scaling across coordinate axes
                ax.set_aspect('equal')
                # Draw grid lines on subplot background
                ax.grid(True, linestyle=':', alpha=0.5)
            
            # Adjust figure layout spacing automatically
            plt.tight_layout()
            # Define target path string for main report image
            main_layout_path = os.path.join(OUTPUT_DIR, f"layout_{layout_idx}_main.png")
            # Save figure to file as high-resolution PNG image
            plt.savefig(main_layout_path, dpi=200, bbox_inches='tight')
            # Release figure canvas memory allocation
            plt.close(fig)

            # ---------------- IMAGE 2: DISTANCE PROXIMITY MAPPING + COMPONENT TABLE AT BOTTOM ----------------
            # Create high-resolution figure canvas (2 rows: top for mapping diagram, bottom for summary table)
            fig_c = plt.figure(figsize=(16, 11))
            gs_c = fig_c.add_gridspec(2, 1, height_ratios=[6.5, 2.5], hspace=0.20)
            
            # Assign top subplot for proximity distance map
            ax_c = fig_c.add_subplot(gs_c[0, 0])
            # Assign bottom subplot for table display
            ax_table = fig_c.add_subplot(gs_c[1, 0])
            ax_table.axis('off')

            # Set main title for distance verification plot
            fig_c.suptitle(f"Layout Alternative {layout_idx} — Inter-Insert Proximity Verification (< 40mm)\n(Front Side Projection View)", 
                           fontsize=13, fontweight='bold')
            
            # Render substrate outline and border keepout region on proximity map
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.2))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing), panel_width, border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.08))

            # Render all components and pins projected onto FRONT side view space
            for i in range(num_elements):
                # Retrieve starting pin index pointer
                i_start = offset_indices[i]
                # Compute pin radius
                ins_rad = insert_diams[i] / 2.0
                ey = cy_v[i]
                c_top, c_right, c_bot, c_left = clearance_dirs[i]

                # Render FRONT-layer component graphics
                if not is_back_side[i]:
                    ex = cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        c_x = ex - lengths[i]/2.0 - c_left
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left + c_right
                        c_h = widths[i] + c_top + c_bot
                        # Render clearance box
                        ax_c.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.35))
                        # Render solid body
                        ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.75, edgecolor='navy', linewidth=1.5))
                    else:
                        # Render clearance circle and solid body
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.35))
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.75, edgecolor='navy', linewidth=1.5))
                    
                    # Render mounting pins for front component
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_c.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
                        
                    # Annotate component name
                    ax_c.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

                # Render BACK-layer component graphics projected onto FRONT view space
                else:
                    # Mirror X-coordinate for projection onto FRONT perspective
                    ex = -cx_v[i]
                    # Swap left and right clearances for projected orientation
                    c_left_proj, c_right_proj = c_right, c_left
                    
                    if element_shapes[i] == 'rectangle':
                        c_x = ex - lengths[i]/2.0 - c_left_proj
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left_proj + c_right_proj
                        c_h = widths[i] + c_top + c_bot
                        # Render projected clearance box
                        ax_c.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle=':', linewidth=1.2, alpha=0.35))
                        # Render projected body outline
                        ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='darkgreen', linewidth=1.5, zorder=3))
                    else:
                        # Render projected clearance circle and outline
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle=':', linewidth=1.2, alpha=0.35))
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, fill=False, linestyle='--', edgecolor='darkgreen', linewidth=1.5, zorder=3))
                    
                    # Render projected mounting pins for back component
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_c.add_patch(plt.Circle((ex - dx, ey + dy), ins_rad, facecolor='orange', edgecolor='#733D00', linewidth=1.2, alpha=0.85, zorder=4))
                        
                    # Annotate component name
                    ax_c.text(ex, ey, element_names[i], color='darkgreen', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

            # Initialize tracking list for close pin pair label text strings
            close_pairs_labels = []
            # Reset alphabetic index counter to zero
            label_counter = 0
            # Define lookup string containing alphabetic characters for line annotations
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

            # Compute pairwise 3D physical distances between all pins to highlight close pairs (< 40mm)
            for i in range(num_elements):
                i_start = offset_indices[i]
                for j in range(i + 1, num_elements):
                    j_start = offset_indices[j]
                    
                    # Iterate through pins of component 'i'
                    for ii in range(offset_counts[i]):
                        idx_i = i_start + ii
                        # Compute absolute physical coordinate of pin 'i'
                        xi_abs = abs_x_positions[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                        yi_abs = cy_v[i] + flat_offsets[idx_i, 1]
                        
                        # Iterate through pins of component 'j'
                        for jj in range(offset_counts[j]):
                            idx_j = j_start + jj
                            # Compute absolute physical coordinate of pin 'j'
                            xj_abs = abs_x_positions[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                            yj_abs = cy_v[j] + flat_offsets[idx_j, 1]
                            
                            # Calculate Euclidean distance between pin pair
                            dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                            
                            # Check if pin separation distance is less than 40.0mm threshold
                            if dist < 40.0:
                                # Determine projected plot X coordinate for pin 'i'
                                xf1 = cx_v[i] + flat_offsets[idx_i, 0] if not is_back_side[i] else -(cx_v[i] + flat_offsets[idx_i, 0])
                                yf1 = cy_v[i] + flat_offsets[idx_i, 1]
                                # Determine projected plot X coordinate for pin 'j'
                                xf2 = cx_v[j] + flat_offsets[idx_j, 0] if not is_back_side[j] else -(cx_v[j] + flat_offsets[idx_j, 0])
                                yf2 = cy_v[j] + flat_offsets[idx_j, 1]
                                
                                # Select current letter label from alphabet lookup string
                                current_letter = alphabet[label_counter % len(alphabet)]
                                # Increment label index counter
                                label_counter += 1
                                
                                # Render purple dashed line connecting close pin pair
                                ax_c.plot([xf1, xf2], [yf1, yf2], color='purple', linestyle='--', linewidth=1.5, alpha=0.85, zorder=10)
                                # Render letter badge at line midpoint
                                ax_c.text((xf1 + xf2)/2, (yf1 + yf2)/2, current_letter, color='white', 
                                          fontsize=8, fontweight='bold', ha='center', va='center',
                                          bbox=dict(boxstyle="circle,pad=0.2", fc="darkmagenta", ec="none", alpha=0.9), zorder=11)
                                
                                # Append entry string to proximity summary log list
                                close_pairs_labels.append(f"{current_letter} — {dist:.2f} mm ({element_names[i]} ↔ {element_names[j]})")

            # Render text legend box detailing close pin distances if any are detected
            if close_pairs_labels:
                # Format array of proximity strings into single line-separated block
                legend_box_text = "\n".join(close_pairs_labels)
                # Render text box on right side of figure panel
                ax_c.text(1.02, 0.95, f"Pin Distances Summary:\n\n{legend_box_text}", 
                          transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))
            # Render text box indicating full pin spacing compliance if no close pairs exist
            else:
                # Render clear compliance text box
                ax_c.text(1.02, 0.95, "Pin Distances Summary:\n\nNo insert violations\nor pins found under 40mm.", 
                          transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))

            # Create visual color patches for diagram key legend
            front_patch = mpatches.Patch(color='royalblue', alpha=0.75, label='Front Layer Components')
            back_patch = mpatches.Patch(facecolor='none', edgecolor='darkgreen', linestyle='--', label='Back Layer Components (Projected Outline)')
            # Attach legend key box to lower left corner of plot
            ax_c.legend(handles=[front_patch, back_patch], loc='lower left')

            # Set plot axis limits, equal aspect ratio, and background grid
            ax_c.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
            ax_c.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
            ax_c.set_aspect('equal')
            ax_c.grid(True, linestyle=':', alpha=0.4)

            # RENDER THE COMPONENT SUMMARY TABLE IN THE MAPPING IMAGE (IMAGE 2)
            headers = ["Component Name", "Layer Placement", "Mass (kg)", "Dimensions (mm)", "CoG X (mm)", "CoG Y (mm)"]
            ui_table = ax_table.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
            ui_table.auto_set_font_size(False)
            ui_table.set_fontsize(9)
            ui_table.scale(1.0, 1.3)
            
            # Style table column headers with dark background and white bold text
            for col_idx in range(len(headers)):
                cell = ui_table[0, col_idx]
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2C3E50')
            
            # Construct output file path for distance map image
            distance_layout_path = os.path.join(OUTPUT_DIR, f"layout_{layout_idx}_distance_map.png")
            # Save distance map image as high-resolution PNG file
            plt.savefig(distance_layout_path, dpi=200, bbox_inches='tight')
            # Release figure canvas memory allocation
            plt.close(fig_c)

        # Print export path confirmation log to console
        print(f"📁 All images generated and saved to: {os.path.abspath(OUTPUT_DIR)}/")

# Check if script is being executed as main entry point
if __name__ == '__main__':
    # Launch optimization workflow execution
    run_optimization()