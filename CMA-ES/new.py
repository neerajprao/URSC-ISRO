# Import the 'os' module to interact with the computer's operating system (like creating folders)
import os
# Import the 'time' module to measure how long the program takes to run
import time
# Import the 'json' module to read and process configuration files written in JSON format
import json
# Import the 'warnings' module to silence non-critical program warnings from cluttering the screen
import warnings
# Import 'numpy' (aliased as 'np'), a fundamental library for fast numerical and mathematical calculations
import numpy as np
# Import the core 'matplotlib' plotting library used to draw graphs and visual layouts
import matplotlib
# Tell matplotlib to run in 'headless' mode so it creates images without popping up windows on screen
matplotlib.use('Agg')  # Headless backend: disables GUI rendering completely
# Import 'pyplot' from matplotlib to create custom diagrams and plots
import matplotlib.pyplot as plt
# Import shape drawing helpers (patches) from matplotlib to draw geometric shapes like rectangles
import matplotlib.patches as mpatches
# Import CMA-ES optimizer
import cma
# Import Numba tools ('njit' for high-speed compilation, 'prange' for running tasks in parallel across CPU cores)
from numba import njit, prange

# Mute all non-critical warning messages so the output screen remains clean
warnings.filterwarnings("ignore")

# ================= STRUCTURAL CONFIGURATION ==================
# Set the maximum acceptable distance (in millimeters) that the physical center of weight can drift from the exact center
CG_TOLERANCE = 1.0  # mm: Target combined center of gravity
# Define the folder name where all generated layout pictures will be saved
OUTPUT_DIR = "optimized_layouts"
# Create the output folder on the computer if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= NUMBA PARALLEL ACCELERATED CORE MATHEMATICS =================
# Use Numba's JIT compiler to translate this math function directly into ultra-fast machine code
@njit(fastmath=True, cache=True)
# Define the mathematical function that scores how good or bad a specific layout arrangement is
def compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, 
                         flat_offsets, offset_counts, offset_indices, is_back_side,
                         clearance_dirs, panel_width, panel_height, border_spacing):
    # Initialize the total mass variable to zero before summing up all component weights
    total_mass = 0.0
    # Initialize the overall center-of-gravity X coordinate to zero
    cg_x = 0.0
    # Initialize the overall center-of-gravity Y coordinate to zero
    cg_y = 0.0
    
    # Create an empty numerical array to store the true horizontal positions of all components
    abs_x = np.zeros(num_components)
    # Loop through every component on the board one by one
    for i in range(num_components):
        # Retrieve the individual weight of the current component
        m = masses[i]
        # Add the current component's weight to the total weight accumulator
        total_mass += m
        # Check if the component is mounted on the back side of the board
        if is_back_side[i]:
            # Flip the horizontal coordinate for back-side components because the back side is a mirror view
            abs_x[i] = -cx[i]
        # Otherwise, the component is on the front side
        else:
            # Keep the original horizontal coordinate for front-side components
            abs_x[i] = cx[i]
        
        # Multiply position by mass to calculate the horizontal weight balance contribution
        cg_x += abs_x[i] * m
        # Multiply position by mass to calculate the vertical weight balance contribution
        cg_y += cy[i] * m
        
    # Safety check: if all components weigh zero in total, stop and return a massive error penalty score
    if total_mass == 0.0:
        # Return an extremely high penalty score signaling an impossible layout
        return 1e15
        
    # Calculate the true final horizontal center of gravity by dividing total weighted balance by total mass
    cg_x /= total_mass
    # Calculate the true final vertical center of gravity by dividing total weighted balance by total mass
    cg_y /= total_mass
    
    # EQUAL TOP PRIORITY PENALTY SCALE
    # Define a huge penalty multiplier (100 million) so any mistake heavily punishes bad layout designs
    PENALTY_WEIGHT = 1e8

    # Calculate the usable left boundary limit of the board by subtracting the outer border zone
    active_x_min = -panel_width / 2.0 + border_spacing
    # Calculate the usable right boundary limit of the board by subtracting the outer border zone
    active_x_max =  panel_width / 2.0 - border_spacing
    # Calculate the usable bottom boundary limit of the board by subtracting the outer border zone
    active_y_min = -panel_height / 2.0 + border_spacing
    # Calculate the usable top boundary limit of the board by subtracting the outer border zone
    active_y_max =  panel_height / 2.0 - border_spacing

    # 1. CENTER OF GRAVITY PENALTY
    # Calculate the straight-line distance from the exact center (0,0) to the current center of gravity
    cg_offset = np.sqrt(cg_x**2 + cg_y**2)
    # Start the center of gravity penalty score at zero
    cg_penalty = 0.0
    # Check if the center of gravity offset exceeds our allowed tolerance (1.0 mm)
    if cg_offset > CG_TOLERANCE:
        # Calculate a heavy penalty proportional to how far the center of gravity drifted past the limit
        cg_penalty = ((cg_offset - CG_TOLERANCE) ** 2) * PENALTY_WEIGHT

    # 2. STRICT CLEARANCE OVERLAP & BORDER KEEPOUT PENALTY
    # Start the physical overlap penalty score at zero
    overlap_penalty = 0.0
    # Start the border encroachment penalty score at zero
    border_penalty = 0.0

    # Loop through each component to verify physical boundary and collision safety rules
    for i in range(num_components):
        # Read the top required safety clearance distance for component 'i'
        c_top = clearance_dirs[i, 0]
        # Read the right safety clearance distance (swapping left/right if on the back side)
        c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
        # Read the bottom required safety clearance distance for component 'i'
        c_bot = clearance_dirs[i, 2]
        # Read the left safety clearance distance (swapping left/right if on the back side)
        c_left = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]

        # Calculate the leftmost boundary edge including safety clearance area
        i_x_min = abs_x[i] - hl[i] - c_left
        # Calculate the rightmost boundary edge including safety clearance area
        i_x_max = abs_x[i] + hl[i] + c_right
        # Calculate the bottommost boundary edge including safety clearance area
        i_y_min = cy[i] - hw[i] - c_bot
        # Calculate the topmost boundary edge including safety clearance area
        i_y_max = cy[i] + hw[i] + c_top

        # --- RULE 1: STRICT ZERO OVERLAP WITH BOARD BORDER ---
        # Measure how far the component spills past the allowed left board boundary
        viol_left   = max(0.0, active_x_min - i_x_min)
        # Measure how far the component spills past the allowed right board boundary
        viol_right  = max(0.0, i_x_max - active_x_max)
        # Measure how far the component spills past the allowed bottom board boundary
        viol_bottom = max(0.0, active_y_min - i_y_min)
        # Measure how far the component spills past the allowed top board boundary
        viol_top    = max(0.0, i_y_max - active_y_max)

        # Sum up all border boundary violations for this component
        total_border_viol = viol_left + viol_right + viol_bottom + viol_top
        # If any part of the component spills outside the allowed board area
        if total_border_viol > 0.0:
            # Apply a massive squared penalty score for violating the board boundaries
            border_penalty += (total_border_viol ** 2) * PENALTY_WEIGHT

        # --- RULE 2 & 3: STRICT ZERO OVERLAP WITH OTHER SAME-SIDE COMPONENTS/CLEARANCES ---
        # Loop through all remaining components to test if component 'i' collides with component 'j'
        for j in range(i + 1, num_components):
            # Collisions only matter if both components are mounted on the same side of the board
            if is_back_side[i] == is_back_side[j]:
                # Read component j's top safety clearance
                cj_top = clearance_dirs[j, 0]
                # Read component j's right safety clearance (adjusting for back side)
                cj_right = clearance_dirs[j, 3] if is_back_side[j] else clearance_dirs[j, 1]
                # Read component j's bottom safety clearance
                cj_bot = clearance_dirs[j, 2]
                # Read component j's left safety clearance (adjusting for back side)
                cj_left = clearance_dirs[j, 1] if is_back_side[j] else clearance_dirs[j, 3]

                # Calculate component j's leftmost edge including clearance
                j_x_min = abs_x[j] - hl[j] - cj_left
                # Calculate component j's rightmost edge including clearance
                j_x_max = abs_x[j] + hl[j] + cj_right
                # Calculate component j's bottom edge including clearance
                j_y_min = cy[j] - hw[j] - cj_bot
                # Calculate component j's top edge including clearance
                j_y_max = cy[j] + hw[j] + cj_top

                # Calculate the horizontal length of the overlapping rectangular region (if any)
                overlap_x = max(0.0, min(i_x_max, j_x_max) - max(i_x_min, j_x_min))
                # Calculate the vertical height of the overlapping rectangular region (if any)
                overlap_y = max(0.0, min(i_y_max, j_y_max) - max(i_y_min, j_y_min))

                # If there is both horizontal and vertical overlap, the two components are physically crashing
                if overlap_x > 0.0 and overlap_y > 0.0:
                    # Calculate the surface area of the physical collision region
                    overlap_area = overlap_x * overlap_y
                    # Apply a massive penalty score proportional to the area of collision
                    overlap_penalty += (overlap_area ** 2) * PENALTY_WEIGHT

    # 3. PIN INSERT SAFE DISTANCE PENALTY
    # Initialize the pin insertion safety distance penalty score to zero
    insert_penalty = 0.0
    # Loop over all components to inspect mounting pin positions
    for i in range(num_components):
        # Look up where component i's pin list starts in memory
        i_start = offset_indices[i]
        # Look up how many mounting pins component i has
        i_count = offset_counts[i]
        
        # Compare component i's pins against every other component j's pins
        for j in range(i + 1, num_components):
            # Look up where component j's pin list starts in memory
            j_start = offset_indices[j]
            # Look up how many mounting pins component j has
            j_count = offset_counts[j]
            # Look up the mandatory minimum separation distance required between pins of component i and j
            req_dist = req_d_matrix[i, j]
            
            # Loop over every individual pin belonging to component i
            for ii in range(i_count):
                # Calculate memory index for pin 'ii' of component i
                idx_i = i_start + ii
                # Calculate the absolute horizontal coordinate of pin 'ii' in 2D space
                xi_abs = abs_x[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                # Calculate the absolute vertical coordinate of pin 'ii' in 2D space
                yi_abs = cy[i] + flat_offsets[idx_i, 1]
                
                # Loop over every individual pin belonging to component j
                for jj in range(j_count):
                    # Calculate memory index for pin 'jj' of component j
                    idx_j = j_start + jj
                    # Calculate the absolute horizontal coordinate of pin 'jj' in 2D space
                    xj_abs = abs_x[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                    # Calculate the absolute vertical coordinate of pin 'jj' in 2D space
                    yj_abs = cy[j] + flat_offsets[idx_j, 1]
                    
                    # Calculate the straight-line distance between pin 'ii' and pin 'jj' using the Pythagorean formula
                    dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                    # Check if the actual distance is smaller than the required safe separation distance
                    if req_dist > dist:
                        # Apply a penalty score for placing pins too close to one another
                        insert_penalty += ((req_dist - dist) ** 2) * PENALTY_WEIGHT
                        
    # Return the total combined penalty score (a perfect layout yields a score of zero)
    return cg_penalty + overlap_penalty + border_penalty + insert_penalty


# Define a bridge function so the optimizer can talk to our fast Numba math function
def evaluate(individual, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing):
    # Extract all horizontal (X) component coordinates from the optimizer's guess vector
    cx = individual[::2]
    # Extract all vertical (Y) component coordinates from the optimizer's guess vector
    cy = individual[1::2]
    # Count the total number of components being placed
    num_components = len(masses)
    # Convert full component lengths into half-lengths (distance from center to edge)
    hl = lengths / 2.0
    # Convert full component widths into half-widths (distance from center to edge)
    hw = widths / 2.0
    
    # Run the Numba scoring calculation and return the total penalty score
    return compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing)


# Define the main function that loads data, runs the optimization, and creates report images
def run_optimization():
    # Set the target filename containing board and component specifications
    filepath = 'newobj.json'
    # Check if the JSON file exists in the current folder path
    if not os.path.exists(filepath):
        # Look in the full current working folder path as a fallback
        filepath = os.path.join(os.getcwd(), 'newobj.json')

    # If the specification file cannot be found anywhere, stop execution and print an error
    if not os.path.exists(filepath):
        # Raise an explicit file error to inform the user
        raise FileNotFoundError("❌ Critical Error: Unable to locate 'newobj.json' in working directory.")

    # Print a success message confirming the configuration file was located
    print(f"\n[✓] Config loaded successfully: '{filepath}'")

    # Open the JSON specification file for reading
    with open(filepath, 'r') as f:
        # Load the structured text content into a Python dictionary object
        data = json.load(f)

    # Read the overall board length (in millimeters) from the configuration dictionary
    panel_width = float(data['BOARD']['Length (mm)'])
    # Read the overall board height/breadth (in millimeters) from the configuration dictionary
    panel_height = float(data['BOARD']['Breadth (mm)'])
    # Read the outer border keepout width (in millimeters) from the configuration dictionary
    border_spacing = float(data['BOARD']['Border (mm)'])
    # Read the default component-to-component clearance gap from the configuration dictionary
    clearance = float(data['BOARD']['Clearance (mm)'])

    # Initialize an empty list to store component dimensions and weights
    element_data = []
    # Initialize an empty list to store human-readable component names
    element_names = []
    # Initialize an empty list to store component geometric shapes (e.g., rectangle, circle)
    element_shapes = []
    # Initialize an empty list to track whether components belong to front or back sides
    is_back_side_list = []
    # Initialize an empty list to hold relative pin offset coordinates for all components
    flat_offsets_list = []
    # Initialize an empty list to hold directional clearance settings for each component
    clearance_dirs_list = []
    
    # Initialize a temporary list to hold all raw JSON component structures
    raw_components_to_process = []
    # Loop over every component listed under the FRONT board layer in JSON
    for comp in data['COMPONENTS'].get('FRONT', []):
        # Add the component to our list marked with 'False' (meaning front layer)
        raw_components_to_process.append((comp, False))
    # Loop over every component listed under the BACK board layer in JSON
    for comp in data['COMPONENTS'].get('BACK', []):
        # Add the component to our list marked with 'True' (meaning back layer)
        raw_components_to_process.append((comp, True))

    # Initialize a list to hold expanded component entries (handling quantity multipliers)
    components_to_process = []
    # Unpack each raw component and its side marker
    for comp, is_back in raw_components_to_process:
        # Read the quantity count specified for this component type
        qty = int(comp.get('Object qty', 1))
        # If the quantity is greater than 1, expand it into individual unique objects
        if qty > 1:
            # Create duplicate entries numbered 0, 1, 2... for each instance
            for q in range(qty):
                # Make a clean copy of the component properties
                cloned_comp = comp.copy()
                # Append an index number to create a unique identifier name
                cloned_comp['Unique Name'] = f"{comp['Object name']}{q}"
                # Add the duplicated component object to our processing list
                components_to_process.append((cloned_comp, is_back))
        # Otherwise, process the single component as-is
        else:
            # Make a copy of the component dictionary
            cloned_comp = comp.copy()
            # Set its unique name to the given object name
            cloned_comp['Unique Name'] = comp['Object name']
            # Add the single component object to our processing list
            components_to_process.append((cloned_comp, is_back))

    # Count the total number of individual components to be placed on the board
    num_elements = len(components_to_process)
    # Create an array of zeros to track how many mounting pins each component owns
    offset_counts = np.zeros(num_elements, dtype=np.int32)
    # Create an array of zeros to record where each component's pin data starts in memory
    offset_indices = np.zeros(num_elements, dtype=np.int32)
    
    # Track the running memory offset index position
    current_idx = 0
    # Loop through each component and index pair in our expanded list
    for idx, (component, is_back) in enumerate(components_to_process):
        # Store the unique name of the current component
        element_name = component['Unique Name']
        # Read the geometric shape, defaulting to 'rectangle' if not specified
        shape = component.get('Shape', 'rectangle')
        # Record the shape in our master shape list
        element_shapes.append(shape)
        # Store a binary flag (1 for back side, 0 for front side)
        is_back_side_list.append(1 if is_back else 0)
        
        # Read custom face clearance definitions if present in JSON
        cf_faces = component.get('CF', [])
        # Read the custom clearance length (in mm), defaulting to global clearance if absent
        cf_len = float(component.get('CFLen (mm)', clearance))
        
        # Initialize default clearances [Top, Right, Bottom, Left] to the global default clearance value
        c_dirs = [clearance, clearance, clearance, clearance]
        # Override clearance values for specific custom-labeled faces
        for face in cf_faces:
            # If face 1 is specified, update top clearance
            if face == 1:
                c_dirs[0] = cf_len
            # If face 2 is specified, update right clearance
            elif face == 2:
                c_dirs[1] = cf_len
            # If face 3 is specified, update bottom clearance
            elif face == 3:
                c_dirs[2] = cf_len
            # If face 4 is specified, update left clearance
            elif face == 4:
                c_dirs[3] = cf_len
                
        # Append this component's directional clearance settings to the main list
        clearance_dirs_list.append(c_dirs)

        # Process sizing and pin locations for rectangular components
        if shape == 'rectangle':
            # Read component length from JSON
            length = float(component['Length (mm)'])
            # Read component width/breadth from JSON
            width = float(component['Breadth (mm)'])
            # Read mounting insert pin diameter from JSON
            insert_diam = float(component['Insert (mm)'])
            # Read mounting pin quantity (defaulting to 4 corner pins if missing)
            insert_qty = int(component.get('Insert qty', 4))
            # Calculate half-length and half-width for relative offset calculations
            half_l, half_w = length / 2.0, width / 2.0
            
            # Position pins for a 2-pin component configuration
            if insert_qty == 2:
                # Place inserts at the midpoints of the shorter sides
                if length >= width:
                    # Place pins on the left and right outer center edges
                    offsets = [(half_l, 0.0), (-half_l, 0.0)]
                else:
                    # Place pins on the top and bottom outer center edges
                    offsets = [(0.0, half_w), (0.0, -half_w)]
            # Position pins for a 6-pin component configuration
            elif insert_qty == 6:
                # Place 4 pins at the four corners of the rectangle
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
                # Add two additional midpoint pins along the longer sides
                if length >= width:
                    # Add top-middle and bottom-middle pins
                    offsets.extend([(0.0, half_w), (0.0, -half_w)])
                else:
                    # Add left-middle and right-middle pins
                    offsets.extend([(half_l, 0.0), (-half_l, 0.0)])
            # Default fallback: 4-pin configuration placed at the four corners
            else:
                # Place one pin at each corner relative to component center
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
        # Process sizing and pin locations for circular/round components
        else:
            # Read component diameter from JSON
            diameter = float(component['Diameter (mm)'])
            # Set length and width equal to the diameter for boundary calculations
            length, width = diameter, diameter
            # Read mounting insert pin diameter
            insert_diam = float(component['Insert (mm)'])
            # Read pin count (defaulting to 3 pins arranged in a ring)
            insert_qty = int(component.get('Insert qty', 3))
            # Calculate inner radius ring position for mounting pins
            inner_radius = max(0.0, (diameter / 2.0) - 1.0)
            # Evenly space pins in a circular pattern using sine and cosine trigonometry
            offsets = [(inner_radius * np.cos(2 * np.pi * k / insert_qty), 
                        inner_radius * np.sin(2 * np.pi * k / insert_qty)) for k in range(insert_qty)]
                        
        # Append the calculated pin offsets to the flat offsets master list
        flat_offsets_list.extend(offsets)
        # Record how many pins this specific component possesses
        offset_counts[idx] = len(offsets)
        # Record the starting memory index position for this component's pins
        offset_indices[idx] = current_idx
        # Increment running memory index counter by the pin count
        current_idx += len(offsets)
        
        # Save weight, length, width, and pin diameter parameters into element data table
        element_data.append([float(component['Weight (kg)']), length, width, insert_diam])
        # Save name string to name list
        element_names.append(element_name)

    # Convert pin offsets list into a fast high-performance NumPy numerical array
    flat_offsets = np.array(flat_offsets_list, dtype=np.float64)
    # Convert element data into a standard floating-point NumPy array
    element_data = np.array(element_data, dtype=float)
    # Extract component masses column into a dedicated array
    masses = element_data[:, 0]
    # Extract component lengths column into a dedicated array
    lengths = element_data[:, 1]
    # Extract component widths column into a dedicated array
    widths = element_data[:, 2]
    # Extract insert pin diameters column into a dedicated array
    insert_diams = element_data[:, 3]
    # Convert layer flags list into a fast integer NumPy array
    is_back_side = np.array(is_back_side_list, dtype=np.int32)
    # Convert clearance settings list into a fast double-precision NumPy array
    clearance_dirs = np.array(clearance_dirs_list, dtype=np.float64)
    # Count the total number of components
    n = len(element_data)

    # Tight Variable Bounds Calculation (Flipped for back components)
    lower_bounds = []
    upper_bounds = []
    # Loop over every component to calculate its specific allowed movement box on the board
    for i in range(num_elements):
        # Retrieve pin list start memory index for component i
        i_start = offset_indices[i]
        # Retrieve pin count for component i
        i_count = offset_counts[i]
        # Slice out pin offset values for component i
        comp_offsets = flat_offsets[i_start : i_start + i_count]
        
        # Calculate maximum horizontal distance from center to any mounting pin
        max_off_x = max(abs(x) for x, _ in comp_offsets) if i_count > 0 else 0
        # Calculate maximum vertical distance from center to any mounting pin
        max_off_y = max(abs(y) for _, y in comp_offsets) if i_count > 0 else 0
        
        # Read top clearance requirement
        c_top = clearance_dirs[i, 0]
        # Read right clearance requirement (swap left/right if component is on back side)
        c_right = clearance_dirs[i, 3] if is_back_side[i] else clearance_dirs[i, 1]
        # Read bottom clearance requirement
        c_bot = clearance_dirs[i, 2]
        # Read left clearance requirement (swap left/right if component is on back side)
        c_left = clearance_dirs[i, 1] if is_back_side[i] else clearance_dirs[i, 3]
        
        # Calculate left edge buffer required (considering both body width and pin clearance)
        req_margin_left = max(lengths[i]/2.0 + c_left, max_off_x + insert_diams[i]/2.0)
        # Calculate right edge buffer required
        req_margin_right = max(lengths[i]/2.0 + c_right, max_off_x + insert_diams[i]/2.0)
        # Calculate bottom edge buffer required
        req_margin_bottom = max(widths[i]/2.0 + c_bot, max_off_y + insert_diams[i]/2.0)
        # Calculate top edge buffer required
        req_margin_top = max(widths[i]/2.0 + c_top, max_off_y + insert_diams[i]/2.0)
        
        # Set movement limits for front-side components
        if not is_back_side[i]:
            # Calculate minimum allowed horizontal coordinate (X-min)
            x_min = -panel_width / 2.0 + border_spacing + req_margin_left
            # Calculate maximum allowed horizontal coordinate (X-max)
            x_max =  panel_width / 2.0 - border_spacing - req_margin_right
        # Set movement limits for back-side components (swapping left and right margins)
        else:
            # Back components are negated in abs_x, swap left/right bound margins
            x_min = -panel_width / 2.0 + border_spacing + req_margin_right
            # Calculate maximum horizontal bound for back component
            x_max =  panel_width / 2.0 - border_spacing - req_margin_left

        # Calculate minimum allowed vertical coordinate (Y-min)
        y_min = -panel_height / 2.0 + border_spacing + req_margin_bottom
        # Calculate maximum allowed vertical coordinate (Y-max)
        y_max =  panel_height / 2.0 - border_spacing - req_margin_top
        
        # Append bounds to lower and upper boundary vectors
        lower_bounds.extend([x_min, y_min])
        upper_bounds.extend([x_max, y_max])

    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Initialize a square matrix filled with default 30mm safe pin separation distance values
    req_d_matrix = np.full((n, n), 30.0)
    # Loop over every pair of components to apply specialized pin clearance rules
    for i in range(n):
        for j in range(n):
            # If both components use small 4.0mm pins, lower required distance to 24.0mm
            if insert_diams[i] == 4.0 and insert_diams[j] == 4.0:
                req_d_matrix[i, j] = 24.0
            # If both components use large 6.0mm pins, raise required distance to 36.0mm
            elif insert_diams[i] == 6.0 and insert_diams[j] == 6.0:
                req_d_matrix[i, j] = 36.0

    # Initialize a list to hold unique, successful layout solutions found by the optimizer
    distinct_layouts = []
    # Set a maximum number of optimization retry attempts
    max_attempts = 150
    # Initialize attempt counter to zero
    attempt = 0

    # Warmup Numba JIT compiler
    dummy_ind = np.zeros(2 * num_elements)
    evaluate(dummy_ind, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs, panel_width, panel_height, border_spacing)

    cpu_cores = os.cpu_count() or 8
    print(f"🚀 CMA-ES Optimization Active | Available Cores: {cpu_cores}")
    print("⚡ Executing CMA-ES with Covariance Adaptation & Boundary Constraints...\n")
    start_time = time.time()

    # Continue running CMA-ES restarts until we collect 5 distinct valid layouts or hit attempt limit
    while len(distinct_layouts) < 5 and attempt < max_attempts:
        attempt += 1
        
        # Generate random initial solution mean within upper and lower bounds
        x0 = np.random.uniform(lower_bounds, upper_bounds)
        # Set initial standard deviation search step size (sigma)
        sigma0 = 15.0

        # Pass bounds cleanly as a two-element list of vectors
        cma_opts = {
            'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
            'maxiter': 600,
            'popsize': 32,
            'tolfun': 1e-6,
            'verbose': -9,  # Mute internal verbose printing
            'seed': int(np.random.default_rng().integers(0, 2**31 - 1))
        }

        try:
            # Instantiate CMA-ES Evolution Strategy
            es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
            
            # Run evolution loop
            while not es.stop():
                solutions = es.ask()
                scores = [
                    evaluate(
                        sol, masses, lengths, widths, clearance, req_d_matrix, 
                        flat_offsets, offset_counts, offset_indices, is_back_side, 
                        clearance_dirs, panel_width, panel_height, border_spacing
                    ) for sol in solutions
                ]
                es.tell(solutions, scores)

            # Get best solution found in this run
            best_sol = np.array(es.result.xbest)
            best_score = es.result.fbest

            # STRICT FILTER: Require 0.0 total penalty (Perfect physical compliance)
            if best_score > 1e-3 or np.any(np.isinf(best_sol)) or np.any(np.isnan(best_sol)):
                continue

            cx_v = best_sol[::2]
            cy_v = best_sol[1::2]
            
            total_mass = np.sum(masses)
            abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
            cg_x_v = np.sum(abs_x_positions * masses) / total_mass
            cg_y_v = np.sum(cy_v * masses) / total_mass

            # Reject solution if center of gravity drift exceeds tolerance
            if abs(cg_x_v) > CG_TOLERANCE or abs(cg_y_v) > CG_TOLERANCE:
                continue

            is_distinct = True
            for existing in distinct_layouts:
                if np.max(np.abs(best_sol - existing)) < 6.0:
                    is_distinct = False
                    break

            if is_distinct:
                distinct_layouts.append(best_sol)
                print(f"  [✓] CMA-ES Solution {len(distinct_layouts)}/5 calculated | CG Offset: ({cg_x_v:.2f}, {cg_y_v:.2f}) mm")

        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            continue

    print(f"\n✨ Execution completed in {time.time() - start_time:.2f}s. Saving plots...\n")

    # ================= SILENT IMAGE GENERATION & EXPORT =================
    if len(distinct_layouts) > 0:
        total_mass = np.sum(masses)
        
        for idx, layout in enumerate(distinct_layouts):
            layout_idx = idx + 1
            cx_v = layout[::2]
            cy_v = layout[1::2]
            abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
            
            final_cg_x = np.sum(abs_x_positions * masses) / total_mass
            final_cg_y = np.sum(cy_v * masses) / total_mass
            
            # ---------------- IMAGE 1: STANDARD LAYOUT REPORT (UNTOUCHED) ----------------
            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(2, 2, height_ratios=[7, 2.2], hspace=0.25)
            
            ax_f = fig.add_subplot(gs[0, 0])
            ax_b = fig.add_subplot(gs[0, 1])
            ax_table = fig.add_subplot(gs[1, :])
            ax_table.axis('off')
            
            fig.suptitle(f"Layout Alternative {layout_idx} (CMA-ES)\nCombined Assembly CG Offset: ({final_cg_x:.4f}, {final_cg_y:.4f}) mm", 
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
                
                c_top, c_right, c_bot, c_left = clearance_dirs[i]

                if not is_back_side[i]: 
                    ex, ey = cx_v[i], cy_v[i]
                    if element_shapes[i] == 'rectangle':
                        c_x = ex - lengths[i]/2.0 - c_left
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left + c_right
                        c_h = widths[i] + c_top + c_bot
                        ax_f.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.5))
                        ax_f.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.9, edgecolor='navy', linewidth=1.5))
                    else:
                        ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.5))
                        ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.9, edgecolor='navy', linewidth=1.5))
                    
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_f.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
                        ax_b.add_patch(plt.Circle((-(ex + dx), ey + dy), ins_rad, facecolor='crimson', edgecolor='#5C0612', linewidth=1.5, alpha=0.6, zorder=3))
                        
                    ax_f.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
                    
                    ex_trace_b = -cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        ax_b.add_patch(plt.Rectangle((ex_trace_b - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='navy', linewidth=1.2, alpha=0.7))
                    else:
                        ax_b.add_patch(plt.Circle((ex_trace_b, cy_v[i]), lengths[i]/2, fill=False, linestyle='--', edgecolor='navy', linewidth=1.2, alpha=0.7))
                else:
                    ex_b, ey_b = cx_v[i], cy_v[i]
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
                    
                    ex_trace_f = -cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        ax_f.add_patch(plt.Rectangle((ex_trace_f - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='darkslategrey', linewidth=1.2, alpha=0.7))
                    else:
                        ax_f.add_patch(plt.Circle((ex_trace_f, cy_v[i]), lengths[i]/2, fill=False, linestyle='--', edgecolor='darkslategrey', linewidth=1.2, alpha=0.7))

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
            main_layout_path = os.path.join(OUTPUT_DIR, f"layout_{layout_idx}_main.png")
            plt.savefig(main_layout_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            # ---------------- IMAGE 2: DISTANCE PROXIMITY MAPPING (FRONT SIDE PROJECTION) ----------------
            fig_c, ax_c = plt.subplots(figsize=(13, 9))
            fig_c.suptitle(f"Layout Alternative {layout_idx} (CMA-ES) — Inter-Insert Proximity Verification (< 40mm)\n(Front Side Projection View)", 
                           fontsize=13, fontweight='bold')
            
            # Draw board outline and border keepout region
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.2))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing), panel_width, border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.08))
            ax_c.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.08))

            # Plot components, clearance boxes, and insert hole pins projected onto Front Side Perspective
            for i in range(num_elements):
                i_start = offset_indices[i]
                ins_rad = insert_diams[i] / 2.0
                ey = cy_v[i]
                c_top, c_right, c_bot, c_left = clearance_dirs[i]

                # Draw FRONT-SIDE components (Solid Blue Body + Light Blue Dashed Clearance Box + Crimson Pins)
                if not is_back_side[i]:
                    ex = cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        c_x = ex - lengths[i]/2.0 - c_left
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left + c_right
                        c_h = widths[i] + c_top + c_bot
                        # Dashed clearance box
                        ax_c.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.35))
                        # Solid body
                        ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.75, edgecolor='navy', linewidth=1.5))
                    else:
                        # Circular clearance and body
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A9C7EB', edgecolor='crimson', linestyle='--', linewidth=1.2, alpha=0.35))
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.75, edgecolor='navy', linewidth=1.5))
                    
                    # Draw front mounting pin insert holes
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_c.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
                        
                    # Component Name label
                    ax_c.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

                # Draw BACK-SIDE components projected onto Front View (Dotted Outline + Light Green Clearance + Orange Pins)
                else:
                    # In Front view projection, Back component center is mirrored (-cx_v[i])
                    ex = -cx_v[i]
                    # Swap left and right clearances for front projection
                    c_left_proj, c_right_proj = c_right, c_left
                    
                    if element_shapes[i] == 'rectangle':
                        c_x = ex - lengths[i]/2.0 - c_left_proj
                        c_y = ey - widths[i]/2.0 - c_bot
                        c_w = lengths[i] + c_left_proj + c_right_proj
                        c_h = widths[i] + c_top + c_bot
                        # Dashed clearance box for back component
                        ax_c.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle=':', linewidth=1.2, alpha=0.35))
                        # Dotted outline for back component body
                        ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='darkgreen', linewidth=1.5, zorder=3))
                    else:
                        # Circular clearance and body for back component
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A3D1A3', edgecolor='darkgreen', linestyle=':', linewidth=1.2, alpha=0.35))
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, fill=False, linestyle='--', edgecolor='darkgreen', linewidth=1.5, zorder=3))
                    
                    # Draw back mounting pin insert holes mirrored to Front View projection space
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_c.add_patch(plt.Circle((ex - dx, ey + dy), ins_rad, facecolor='orange', edgecolor='#733D00', linewidth=1.2, alpha=0.85, zorder=4))
                        
                    # Component Name label for back component
                    ax_c.text(ex, ey, element_names[i], color='darkgreen', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

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
                        yi_abs = cy_v[i] + flat_offsets[idx_i, 1]
                        
                        for jj in range(offset_counts[j]):
                            idx_j = j_start + jj
                            xj_abs = abs_x_positions[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                            yj_abs = cy_v[j] + flat_offsets[idx_j, 1]
                            
                            dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                            
                            if dist < 40.0:
                                xf1 = cx_v[i] + flat_offsets[idx_i, 0] if not is_back_side[i] else -(cx_v[i] + flat_offsets[idx_i, 0])
                                yf1 = cy_v[i] + flat_offsets[idx_i, 1]
                                xf2 = cx_v[j] + flat_offsets[idx_j, 0] if not is_back_side[j] else -(cx_v[j] + flat_offsets[idx_j, 0])
                                yf2 = cy_v[j] + flat_offsets[idx_j, 1]
                                
                                current_letter = alphabet[label_counter % len(alphabet)]
                                label_counter += 1
                                
                                ax_c.plot([xf1, xf2], [yf1, yf2], color='purple', linestyle='--', linewidth=1.5, alpha=0.85, zorder=10)
                                ax_c.text((xf1 + xf2)/2, (yf1 + yf2)/2, current_letter, color='white', 
                                          fontsize=8, fontweight='bold', ha='center', va='center',
                                          bbox=dict(boxstyle="circle,pad=0.2", fc="darkmagenta", ec="none", alpha=0.9), zorder=11)
                                
                                close_pairs_labels.append(f"{current_letter} — {dist:.2f} mm ({element_names[i]} ↔ {element_names[j]})")

            if close_pairs_labels:
                legend_box_text = "\n".join(close_pairs_labels)
                ax_c.text(1.02, 0.95, f"Pin Distances Summary:\n\n{legend_box_text}", 
                          transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))
            else:
                ax_c.text(1.02, 0.95, "Pin Distances Summary:\n\nNo insert violations\nor pins found under 40mm.", 
                          transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))

            # Visual color legend key
            front_patch = mpatches.Patch(color='royalblue', alpha=0.75, label='Front Layer Components')
            back_patch = mpatches.Patch(facecolor='none', edgecolor='darkgreen', linestyle='--', label='Back Layer Components (Projected Outline)')
            ax_c.legend(handles=[front_patch, back_patch], loc='lower left')

            ax_c.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
            ax_c.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
            ax_c.set_aspect('equal')
            ax_c.grid(True, linestyle=':', alpha=0.4)
            
            distance_layout_path = os.path.join(OUTPUT_DIR, f"layout_{layout_idx}_distance_map.png")
            plt.savefig(distance_layout_path, dpi=200, bbox_inches='tight')
            plt.close(fig_c)

        print(f"📁 All images generated and saved to: {os.path.abspath(OUTPUT_DIR)}/")

if __name__ == '__main__':
    run_optimization()