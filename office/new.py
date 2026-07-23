# Import the 'os' module to interact with your computer's operating system (like creating folders or finding files).
import os
# Import the 'time' module to measure how long the program takes to run.
import time
# Import the 'json' module to read data stored in JSON file formats (a common way to structure configuration text).
import json
# Import the 'warnings' module so we can hide non-critical alert messages that Python might show.
import warnings
# Import 'numpy' (shortened to 'np'), which is a powerful math library for handling grids, matrices, and lists of numbers fast.
import numpy as np
# Import 'matplotlib', a plotting library used to draw graphs and pictures in Python.
import matplotlib
# Tell Matplotlib to run in 'Agg' mode (headless backend), which creates images in the background without opening popup windows.
matplotlib.use('Agg')  # Headless backend: disables GUI rendering completely
# Import the main drawing tool inside Matplotlib, called 'pyplot' (shortened to 'plt').
import matplotlib.pyplot as plt
# Import 'patches' from Matplotlib, which are basic geometric shapes like rectangles and circles used for drawing on graph plots.
import matplotlib.patches as mpatches
# Import 'differential_evolution' from SciPy, an intelligent algorithm that tries thousands of combinations to find the best layout.
from scipy.optimize import differential_evolution
# Import 'njit' (compiles Python code into super-fast machine code) and 'prange' (runs loops in parallel) from Numba.
from numba import njit, prange

# Tell Python to ignore any harmless warning messages so the console output stays clean and readable.
warnings.filterwarnings("ignore")

# ================= STRUCTURAL CONFIGURATION ==================
# Define the maximum allowed distance (in millimeters) that the physical balance point (Center of Gravity) can drift from the board center.
CG_TOLERANCE = 20.0  # mm: Target combined center of gravity
# Name the target folder where all output layout images will be saved on your computer.
OUTPUT_DIR = "optimized_layouts"
# Automatically create the folder named above if it does not already exist on your computer.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= NUMBA PARALLEL ACCELERATED CORE MATHEMATICS =================
# Tell Numba to convert this function into high-speed raw machine language, speeding up heavy calculations significantly.
@njit(fastmath=True, cache=True)
# Define the master math function that calculates how "bad" (penalty score/fitness) a specific layout arrangement is.
def compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, 
                         flat_offsets, offset_counts, offset_indices, is_back_side,
                         clearance_dirs):
    # Start tracking the combined weight of all installed electrical components, initialized to zero.
    total_mass = 0.0
    # Start tracking the horizontal (X-axis) balance point weight total, initialized to zero.
    cg_x = 0.0
    # Start tracking the vertical (Y-axis) balance point weight total, initialized to zero.
    cg_y = 0.0
    
    # Create a list of zeros to hold the true left-to-right (X) position of each component on the physical board.
    abs_x = np.zeros(num_components)
    # Loop through every component one by one to calculate the overall weight balance (Center of Gravity).
    for i in range(num_components):
        # Retrieve the weight of the current component 'i'.
        m = masses[i]
        # Add the current component's weight to the total weight of the board assembly.
        total_mass += m
        # Check if this component is attached to the back side of the board.
        if is_back_side[i]:
            # Backside components are flipped horizontally, so negate their X position coordinate.
            abs_x[i] = -cx[i]
        # Otherwise, the component is on the front side.
        else:
            # Frontside components keep their normal X coordinate.
            abs_x[i] = cx[i]
        
        # Multiply X position by mass and add to the running total X-balance sum.
        cg_x += abs_x[i] * m
        # Multiply Y position by mass and add to the running total Y-balance sum.
        cg_y += cy[i] * m
        
    # If the total weight is zero (meaning no valid parts were provided), stop and give an extremely high penalty score.
    if total_mass == 0.0:
        # Return a massive penalty number signaling an impossible layout.
        return 1e15
        
    # Calculate the true overall horizontal balance point by dividing total weighted X position by total mass.
    cg_x /= total_mass
    cg_y /= total_mass
    
    # 1. SCREENING STEP: Center of Gravity Check
    # Use the Pythagorean theorem to calculate the straight-line distance from the center of the board (0,0) to the balance point.
    cg_offset = np.sqrt(cg_x**2 + cg_y**2)
    # Start the weight balance penalty score at zero.
    cg_penalty = 0.0
    # Check if the calculated balance point lies further away from center than our maximum allowed tolerance (20mm).
    if cg_offset > CG_TOLERANCE:
        # Square the excess distance and multiply by 1 billion to penalize the solver severely for bad weight distribution.
        cg_penalty = (cg_offset - CG_TOLERANCE)**2 * 1e9
        # If the balance penalty is overwhelmingly huge, stop evaluating early to save computing time.
        if cg_penalty > 1e11:
            # Return the bad score immediately.
            return cg_penalty

    # 2. SCREENING STEP: STRICT ZERO-OVERLAP CLEARANCE & WIRING KEEPOUT CHECK
    # clearance_dirs: [Top(1), Right(2), Bottom(3), Left(4)]
    # Initialize the total overlap penalty score to zero.
    overlap_penalty = 0.0
    # Loop through every component 'i' on the board to compare it against every other component.
    for i in range(num_components):
        # Calculate the leftmost boundary edge including required safety clearance for component 'i'.
        i_x_min = abs_x[i] - hl[i] - clearance_dirs[i, 3]  # Left boundary
        # Calculate the rightmost boundary edge including required safety clearance for component 'i'.
        i_x_max = abs_x[i] + hl[i] + clearance_dirs[i, 1]  # Right boundary
        # Calculate the bottommost boundary edge including required safety clearance for component 'i'.
        i_y_min = cy[i] - hw[i] - clearance_dirs[i, 2]      # Bottom boundary
        # Calculate the topmost boundary edge including required safety clearance for component 'i'.
        i_y_max = cy[i] + hw[i] + clearance_dirs[i, 0]      # Top boundary

        # Compare component 'i' against all remaining components 'j' coming after it.
        for j in range(i + 1, num_components):
            # Only check for physical collisions if both components reside on the exact same side of the board.
            if is_back_side[i] == is_back_side[j]:
                # Calculate component 'j' left safety boundary.
                j_x_min = abs_x[j] - hl[j] - clearance_dirs[j, 3]
                # Calculate component 'j' right safety boundary.
                j_x_max = abs_x[j] + hl[j] + clearance_dirs[j, 1]
                # Calculate component 'j' bottom safety boundary.
                j_y_min = cy[j] - hw[j] - clearance_dirs[j, 2]
                # Calculate component 'j' top safety boundary.
                j_y_max = cy[j] + hw[j] + clearance_dirs[j, 0]

                # Measure how much two components overlap horizontally across their clearance boundary boxes.
                overlap_x = max(0.0, min(i_x_max, j_x_max) - max(i_x_min, j_x_min))
                # Measure how much two components overlap vertically across their clearance boundary boxes.
                overlap_y = max(0.0, min(i_y_max, j_y_max) - max(i_y_min, j_y_min))

                # If both horizontal and vertical overlap measurements are greater than zero, they are colliding!
                if overlap_x > 0.0 and overlap_y > 0.0:
                    # Add a heavy penalty score proportional to the area of collision (10 million factor).
                    overlap_penalty += (overlap_x * overlap_y) * 1e7
                
    # If overlap penalties exceed a huge threshold (10 billion), abort early to avoid useless calculations.
    if overlap_penalty > 1e10:
        # Return the sum of weight penalties and physical overlap penalties.
        return cg_penalty + overlap_penalty

    # 3. DETAILED STEP: Pin Insert Safe Distances
    # Start tracking penalties for mounting pins/screws being installed too close to each other.
    insert_penalty = 0.0
    # Loop through all components on the board.
    for i in range(num_components):
        # Look up where component 'i''s mounting pin positions start inside the big list of pins.
        i_start = offset_indices[i]
        # Look up how many total mounting pins component 'i' has.
        i_count = offset_counts[i]
        
        # Compare component 'i''s pins against every other component 'j' on the board.
        for j in range(i + 1, num_components):
            # Look up where component 'j''s mounting pin positions start in the list.
            j_start = offset_indices[j]
            # Look up how many mounting pins component 'j' has.
            j_count = offset_counts[j]
            # Retrieve the minimum allowed safety distance required between component 'i' and component 'j' pins.
            req_dist = req_d_matrix[i, j]
            
            # Loop through each individual pin belonging to component 'i'.
            for ii in range(i_count):
                # Calculate the exact list index for component 'i''s current pin.
                idx_i = i_start + ii
                # Compute the absolute world X position of component 'i''s pin (accounting for front/back flipping).
                xi_abs = abs_x[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                # Compute the absolute world Y position of component 'i''s pin.
                yi_abs = cy[i] + flat_offsets[idx_i, 1]
                
                # Loop through each individual pin belonging to component 'j'.
                for jj in range(j_count):
                    # Calculate the exact list index for component 'j''s current pin.
                    idx_j = j_start + jj
                    # Compute the absolute world X position of component 'j''s pin (accounting for front/back flipping).
                    xj_abs = abs_x[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                    # Compute the absolute world Y position of component 'j''s pin.
                    yj_abs = cy[j] + flat_offsets[idx_j, 1]
                    
                    # Calculate the exact straight-line distance between pin 'i' and pin 'j' using standard distance math.
                    dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                    # Check if the actual distance between pins is smaller than the required safety clearance.
                    if req_dist > dist:
                        # Add a penalty proportional to how severely the safety distance was violated.
                        insert_penalty += (req_dist - dist) * 1e6
                        
    # Return the grand total score combining Center of Gravity, Overlap, and Pin clearance penalties.
    return cg_penalty + overlap_penalty + insert_penalty


# Helper wrapper function that unpacks raw layout guess coordinates and passes them into the fast Numba core.
def evaluate(individual, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs):
    # Extract every even-indexed number from the algorithm's guess list to get all X coordinates.
    cx = individual[::2]
    # Extract every odd-indexed number from the algorithm's guess list to get all Y coordinates.
    cy = individual[1::2]
    # Count how many total physical components are being arranged.
    num_components = len(masses)
    # Divide component lengths by 2 to get distance from center to side edges (half-length).
    hl = lengths / 2.0
    # Divide component widths by 2 to get distance from center to top/bottom edges (half-width).
    hw = widths / 2.0
    
    # Run the Numba mathematical evaluation engine with extracted data and return the fitness score.
    return compute_fitness_core(cx, cy, masses, hl, hw, clearance, req_d_matrix, num_components, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs)


# Main function that orchestrates loading data, running optimization attempts, and creating visual diagrams.
def run_optimization():
    # Specify the local filename containing all board and component configuration details.
    filepath = 'newobj.json'
    # Check if the JSON configuration file exists in the current folder location.
    if not os.path.exists(filepath):
        # Fallback to creating a full absolute directory path to locate 'newobj.json'.
        filepath = os.path.join(os.getcwd(), 'newobj.json')

    # If the configuration file still cannot be found anywhere, stop execution and show an error.
    if not os.path.exists(filepath):
        # Raise an error stopping the program so the user knows to provide 'newobj.json'.
        raise FileNotFoundError("❌ Critical Error: Unable to locate 'newobj.json' in working directory.")

    # Print a success message confirming the configuration file was located.
    print(f"\n[✓] Config loaded successfully: '{filepath}'")

    # Open the JSON text file for reading.
    with open(filepath, 'r') as f:
        # Parse the JSON text file and load it into a Python data dictionary called 'data'.
        data = json.load(f)

    # Convert and extract the physical length of the circuit board from JSON into millimeters.
    panel_width = float(data['BOARD']['Length (mm)'])
    # Convert and extract the physical breadth (height) of the circuit board from JSON into millimeters.
    panel_height = float(data['BOARD']['Breadth (mm)'])
    # Convert and extract the un-buildable border edge width around the board parameter.
    border_spacing = float(data['BOARD']['Border (mm)'])
    # Convert and extract the default clearance distance buffer parameter.
    clearance = float(data['BOARD']['Clearance (mm)'])

    # Initialize an empty list to hold calculated numerical properties (mass, dimensions) of components.
    element_data = []
    # Initialize an empty list to store the names of components.
    element_names = []
    # Initialize an empty list to store component geometric shapes ('rectangle' or 'circle').
    element_shapes = []
    # Initialize an empty list storing flags indicating whether a component is on the back layer (1) or front layer (0).
    is_back_side_list = []
    # Initialize an empty list storing relative pin placement offsets from component centers.
    flat_offsets_list = []
    # Initialize an empty list storing directional clearance distances for each component [top, right, bottom, left].
    clearance_dirs_list = []
    
    # Create an empty list to collect components directly parsed from the JSON file.
    raw_components_to_process = []
    # Loop over all components listed under the 'FRONT' section in the JSON file.
    for comp in data['COMPONENTS'].get('FRONT', []):
        # Add the component item along with a boolean 'False' flag (meaning front side).
        raw_components_to_process.append((comp, False))
    # Loop over all components listed under the 'BACK' section in the JSON file.
    for comp in data['COMPONENTS'].get('BACK', []):
        # Add the component item along with a boolean 'True' flag (meaning back side).
        raw_components_to_process.append((comp, True))

    # Initialize an empty list to hold expanded components (expanding quantities greater than 1 into separate items).
    components_to_process = []
    # Loop through each collected component item and its side flag.
    for comp, is_back in raw_components_to_process:
        # Read the quantity multiplier for this object (defaults to 1 if unspecified).
        qty = int(comp.get('Object qty', 1))
        # If quantity is greater than 1, break them down into separate uniquely named components.
        if qty > 1:
            # Loop as many times as specified by quantity.
            for q in range(qty):
                # Make an exact duplicate copy of the component details dictionary.
                cloned_comp = comp.copy()
                # Assign a distinct index name, e.g., "Resistor0", "Resistor1".
                cloned_comp['Unique Name'] = f"{comp['Object name']}{q}"
                # Add the cloned component to our expanded list.
                components_to_process.append((cloned_comp, is_back))
        # Otherwise, if quantity is just 1.
        else:
            # Make a copy of the component details.
            cloned_comp = comp.copy()
            # Assign its original name as its unique name.
            cloned_comp['Unique Name'] = comp['Object name']
            # Add the component to our list.
            components_to_process.append((cloned_comp, is_back))

    # Calculate the total number of individual physical components that must be placed on the board.
    num_elements = len(components_to_process)
    # Create a list of zeros to record how many mounting pins each component possesses.
    offset_counts = np.zeros(num_elements, dtype=np.int32)
    # Create a list of zeros to track the starting index position of each component's pins inside the global list.
    offset_indices = np.zeros(num_elements, dtype=np.int32)
    
    # Initialize a counter tracking how many pins have been processed in total across all components.
    current_idx = 0
    # Loop through each component using 'enumerate' to get both index number and component details.
    for idx, (component, is_back) in enumerate(components_to_process):
        # Retrieve the component's unique name label.
        element_name = component['Unique Name']
        # Read the geometric shape type (defaults to 'rectangle' if not specified).
        shape = component.get('Shape', 'rectangle')
        # Store the shape in our shapes tracking list.
        element_shapes.append(shape)
        # Store a 1 if on the back layer, or 0 if on the front layer.
        is_back_side_list.append(1 if is_back else 0)
        
        # Read special cable/wiring clearance faces configured in JSON (if any).
        cf_faces = component.get('CF', [])
        # Read the extra clearance distance extension value for specified cable faces.
        cf_len = float(component.get('CFLen (mm)', clearance))
        
        # Directions: [Top (1), Right (2), Bottom (3), Left (4)]
        # Default all four directional clearances around the component to the standard clearance value.
        c_dirs = [clearance, clearance, clearance, clearance]
        # Check through every designated extra clearance side face requested in JSON.
        for face in cf_faces:
            # If side face 1 is specified, apply extra clearance to the Top side.
            if face == 1:
                c_dirs[0] = cf_len
            # If side face 2 is specified, apply extra clearance to the Right side.
            elif face == 2:
                c_dirs[1] = cf_len
            # If side face 3 is specified, apply extra clearance to the Bottom side.
            elif face == 3:
                c_dirs[2] = cf_len
            # If side face 4 is specified, apply extra clearance to the Left side.
            elif face == 4:
                c_dirs[3] = cf_len
                
        # Append this component's directional clearance values to our overall list.
        clearance_dirs_list.append(c_dirs)

        # Handle pin calculation if the component shape is rectangular.
        if shape == 'rectangle':
            # Extract component length in millimeters.
            length = float(component['Length (mm)'])
            # Extract component width in millimeters.
            width = float(component['Breadth (mm)'])
            # Extract the diameter size of the mounting pins/holes.
            insert_diam = float(component['Insert (mm)'])
            # Read how many mounting pins/holes this part has (defaults to 4 at corners).
            insert_qty = int(component.get('Insert qty', 4))
            # Calculate half-length and half-width distances from component center to outer edges.
            half_l, half_w = length / 2.0, width / 2.0
            
            # If the part requires 6 mounting pins.
            if insert_qty == 6:
                # Place 4 pins at the four corners of the rectangular component.
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
                # If length is longer than or equal to width, place two extra pins in the middle of top/bottom edges.
                if length >= width:
                    offsets.extend([(0.0, half_w), (0.0, -half_w)])
                # Otherwise, place two extra pins in the middle of left/right edges.
                else:
                    offsets.extend([(half_l, 0.0), (-half_l, 0.0)])
            # Default case for 4 pins (placed at the four corners).
            else:
                offsets = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
        # Handle pin calculation if the component shape is circular.
        else:
            # For circular parts, both length and width equal the circle's diameter.
            diameter = float(component['Diameter (mm)'])
            length, width = diameter, diameter
            # Extract mounting pin diameter.
            insert_diam = float(component['Insert (mm)'])
            # Read how many mounting pins this circular part has (defaults to 3 arranged in a circle).
            insert_qty = int(component.get('Insert qty', 3))
            # Calculate the inner radial distance on which pins sit inside the circle.
            inner_radius = max(0.0, (diameter / 2.0) - 1.0)
            # Calculate pin (x, y) offset positions evenly spread in a radial wheel pattern using trigonometry.
            offsets = [(inner_radius * np.cos(2 * np.pi * k / insert_qty), 
                        inner_radius * np.sin(2 * np.pi * k / insert_qty)) for k in range(insert_qty)]
                        
        # Append calculated pin offsets to our running global offsets list.
        flat_offsets_list.extend(offsets)
        # Record how many pins belong to this specific component.
        offset_counts[idx] = len(offsets)
        # Record where this component's pins begin in the master offsets list.
        offset_indices[idx] = current_idx
        # Advance the global index counter by the number of pins added.
        current_idx += len(offsets)
        
        # Store numerical mass, length, width, and pin diameter values into element data list.
        element_data.append([float(component['Weight (kg)']), length, width, insert_diam])
        # Append the unique name of the component.
        element_names.append(element_name)

    # Convert flat list of all pin offsets into a structured, fast NumPy array.
    flat_offsets = np.array(flat_offsets_list, dtype=np.float64)
    # Convert numerical element data into a fast NumPy floating point array.
    element_data = np.array(element_data, dtype=float)
    # Extract component masses into a dedicated 1D array.
    masses = element_data[:, 0]
    # Extract component lengths into a dedicated 1D array.
    lengths = element_data[:, 1]
    # Extract component widths into a dedicated 1D array.
    widths = element_data[:, 2]
    # Extract component pin diameters into a dedicated 1D array.
    insert_diams = element_data[:, 3]
    # Convert back-side flags list into a fast NumPy integer array.
    is_back_side = np.array(is_back_side_list, dtype=np.int32)
    # Convert directional clearance values into a fast NumPy array.
    clearance_dirs = np.array(clearance_dirs_list, dtype=np.float64)
    # Count total number of components.
    n = len(element_data)

    # Calculate variable bounds so that clearance regions never violate board borders
    # Initialize an empty list to store valid (min, max) placement range boundaries for each component.
    bounds = []
    # Loop over each component to calculate its allowable spatial region on the board.
    for i in range(num_elements):
        # Look up the starting index of pins for component 'i'.
        i_start = offset_indices[i]
        # Look up how many pins component 'i' has.
        i_count = offset_counts[i]
        # Slice out pin offsets belonging solely to component 'i'.
        comp_offsets = flat_offsets[i_start : i_start + i_count]
        
        # Calculate maximum horizontal pin displacement relative to component center.
        max_off_x = max(abs(x) for x, _ in comp_offsets) if i_count > 0 else 0
        # Calculate maximum vertical pin displacement relative to component center.
        max_off_y = max(abs(y) for _, y in comp_offsets) if i_count > 0 else 0
        
        # Determine total required left side margin considering both component body clearance and pin positions.
        req_margin_left = max(lengths[i]/2.0 + clearance_dirs[i, 3], max_off_x + insert_diams[i]/2.0)
        # Determine total required right side margin considering both component body clearance and pin positions.
        req_margin_right = max(lengths[i]/2.0 + clearance_dirs[i, 1], max_off_x + insert_diams[i]/2.0)
        # Determine total required bottom side margin considering both component body clearance and pin positions.
        req_margin_bottom = max(widths[i]/2.0 + clearance_dirs[i, 2], max_off_y + insert_diams[i]/2.0)
        # Determine total required top side margin considering both component body clearance and pin positions.
        req_margin_top = max(widths[i]/2.0 + clearance_dirs[i, 0], max_off_y + insert_diams[i]/2.0)
        
        # Calculate maximum allowable leftward position (X-min) without crossing board border limits.
        x_min = -panel_width / 2.0 + border_spacing + req_margin_left
        # Calculate maximum allowable rightward position (X-max) without crossing board border limits.
        x_max = panel_width / 2.0 - border_spacing - req_margin_right
        # Calculate maximum allowable bottomward position (Y-min) without crossing board border limits.
        y_min = -panel_height / 2.0 + border_spacing + req_margin_bottom
        # Calculate maximum allowable upward position (Y-max) without crossing board border limits.
        y_max = panel_height / 2.0 - border_spacing - req_margin_top
        
        # Append allowable X coordinate boundaries (min, max) for component 'i'.
        bounds.append((x_min, x_max))
        # Append allowable Y coordinate boundaries (min, max) for component 'i'.
        bounds.append((y_min, y_max))

    # Pre-fill an n x n matrix with default 30mm required pin-to-pin distance safety limits.
    req_d_matrix = np.full((n, n), 30.0)
    # Loop over every pair of components 'i' and 'j' to customize pin distance rules based on pin sizes.
    for i in range(n):
        for j in range(n):
            # If both components use small 4mm pins, lower required safe distance to 24mm.
            if insert_diams[i] == 4.0 and insert_diams[j] == 4.0:
                req_d_matrix[i, j] = 24.0
            # If both components use medium 6mm pins, raise required safe distance to 36mm.
            elif insert_diams[i] == 6.0 and insert_diams[j] == 6.0:
                req_d_matrix[i, j] = 36.0

    # Initialize a list to hold uniquely different valid board layout results found by solver.
    distinct_layouts = []
    # Set the maximum number of optimization attempts allowed before stopping.
    max_attempts = 150
    # Start attempt counter at zero.
    attempt = 0

    # Warmup Numba JIT compiler
    # Create dummy coordinate list of zeros to force Numba to compile mathematical functions in advance.
    dummy_ind = np.zeros(2 * num_elements)
    # Execute dummy calculation so compiler warm-up overhead doesn't slow down real optimization timer.
    evaluate(dummy_ind, masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs)

    # Detect the number of processor CPU cores available on this computer (defaulting to 8 if detection fails).
    cpu_cores = os.cpu_count() or 8
    # Print status message notifying the user about active CPU core acceleration.
    print(f"🚀 M3 Pro Acceleration Active | Available Cores: {cpu_cores}")
    # Print status message indicating parallel solver process start.
    print("⚡ Executing Parallel Differential Evolution with Strict Keepouts...\n")
    # Record current clock time in seconds to calculate total script runtime later.
    start_time = time.time()

    # Loop until we successfully find 5 distinct valid layouts or exhaust maximum allowed attempt limit (150).
    while len(distinct_layouts) < 5 and attempt < max_attempts:
        # Increment attempt counter by 1.
        attempt += 1
        # Generate a random integer seed value to ensure each optimization run searches differently.
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))

        # Enclose solver attempt inside a try-block to prevent sudden crashes if a run fails.
        try:
            # Execute Differential Evolution global optimizer algorithm from SciPy.
            result = differential_evolution(
                evaluate, bounds,
                args=(masses, lengths, widths, clearance, req_d_matrix, flat_offsets, offset_counts, offset_indices, is_back_side, clearance_dirs),
                maxiter=350, popsize=15, tol=1e-4,
                strategy='best1bin', seed=seed,
                workers=-1, updating='deferred'
            )
            
            # If solver output contains broken math values (Infinity or NaN), skip this attempt.
            if np.any(np.isinf(result.x)) or np.any(np.isnan(result.x)):
                continue

            # Unpack X coordinates of components from optimizer result vector.
            cx_v = result.x[::2]
            # Unpack Y coordinates of components from optimizer result vector.
            cy_v = result.x[1::2]
            
            # Sum up total weight of all components on board.
            total_mass = np.sum(masses)
            # Adjust X coordinates depending on whether components sit on front or back side.
            abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
            # Compute exact final balance point X coordinate for candidate layout.
            cg_x_v = np.sum(abs_x_positions * masses) / total_mass
            # Compute exact final balance point Y coordinate for candidate layout.
            cg_y_v = np.sum(cy_v * masses) / total_mass

            # Reject layout if overall X or Y Center of Gravity balance falls outside allowable tolerance range.
            if abs(cg_x_v) > CG_TOLERANCE or abs(cg_y_v) > CG_TOLERANCE:
                continue

            # Flag to track whether candidate layout is distinct from previously stored solutions.
            is_distinct = True
            # Compare candidate layout against already accepted layouts in list.
            for existing in distinct_layouts:
                # If component position differences are under 6mm, treat it as a duplicate and reject it.
                if np.max(np.abs(result.x - existing)) < 6.0:
                    is_distinct = False
                    break

            # If layout is unique and valid, store it!
            if is_distinct:
                # Add successful layout coordinate array to accepted solutions list.
                distinct_layouts.append(result.x)
                # Print progress update to console showing successful layout count and its balance offset.
                print(f"  [✓] Layout Solution {len(distinct_layouts)}/5 calculated | CG Offset: ({cg_x_v:.2f}, {cg_y_v:.2f}) mm")

        # Catch any unexpected runtime errors silently and allow the loop to try again.
        except Exception:
            continue

    # Print total execution runtime in seconds upon finishing optimization attempts.
    print(f"\n✨ Execution completed in {time.time() - start_time:.2f}s. Saving plots silently...\n")

    # ================= SILENT IMAGE GENERATION & EXPORT =================
    # Check if at least one valid distinct layout solution was discovered.
    if len(distinct_layouts) > 0:
        # Calculate total assembly weight.
        total_mass = np.sum(masses)
        
        # Loop over each found distinct layout solution to plot and save visual report images.
        for idx, layout in enumerate(distinct_layouts):
            # Create a 1-based index number for layout naming (1, 2, 3...).
            layout_idx = idx + 1
            # Extract component X coordinates for current solution.
            cx_v = layout[::2]
            # Extract component Y coordinates for current solution.
            cy_v = layout[1::2]
            # Convert X coordinates to absolute positions based on front/back layer assignment.
            abs_x_positions = np.where(is_back_side == 1, -cx_v, cx_v)
            
            # Calculate precise X Center of Gravity for current layout visualization header.
            final_cg_x = np.sum(abs_x_positions * masses) / total_mass
            # Calculate precise Y Center of Gravity for current layout visualization header.
            final_cg_y = np.sum(cy_v * masses) / total_mass
            
            # ---------------- IMAGE 1: STANDARD LAYOUT REPORT ----------------
            # Create a new image figure canvas with dimensions 16x9 inches.
            fig = plt.figure(figsize=(16, 9))
            # Define a 2-row layout grid split into front view, back view, and summary table bottom row.
            gs = fig.add_gridspec(2, 2, height_ratios=[7, 2.2], hspace=0.25)
            
            # Create sub-plot for Front side layout view (top-left panel).
            ax_f = fig.add_subplot(gs[0, 0])
            # Create sub-plot for Back side layout view (top-right panel).
            ax_b = fig.add_subplot(gs[0, 1])
            # Create sub-plot for bottom table display spanning full width.
            ax_table = fig.add_subplot(gs[1, :])
            # Turn off coordinate grid axes for table panel area.
            ax_table.axis('off')
            
            # Set top title text showing alternative layout number and overall Center of Gravity offset values.
            fig.suptitle(f"Layout Alternative {layout_idx}\nCombined Assembly CG Offset: ({final_cg_x:.4f}, {final_cg_y:.4f}) mm", 
                         fontsize=14, fontweight='bold')
            
            # Draw gray board background and red border keepout zone rectangles on both front and back views.
            for ax in [ax_f, ax_b]:
                # Draw light gray rectangle representing physical board body.
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.3))
                # Draw translucent red border zone along top edge.
                ax.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing), panel_width, border_spacing, color='red', alpha=0.15))
                # Draw translucent red border zone along bottom edge.
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, border_spacing, color='red', alpha=0.15))
                # Draw translucent red border zone along left edge.
                ax.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))
                # Draw translucent red border zone along right edge.
                ax.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.15))

            # Initialize empty list to accumulate rows of text data for bottom report table.
            table_data = []

            # Loop over all components to draw their shapes, clearances, and mounting pins on plots.
            for i in range(num_elements):
                # Calculate radius of mounting pin (diameter divided by 2).
                ins_rad = insert_diams[i] / 2.0
                # Retrieve starting pin list index for component 'i'.
                i_start = offset_indices[i]
                # Label string indicating whether part sits on "BACK" or "FRONT".
                side_str = "BACK" if is_back_side[i] else "FRONT"
                
                # Append row detailing component metadata to summary table data list.
                table_data.append([
                    element_names[i],
                    side_str,
                    f"{masses[i]:.3f}",
                    f"{lengths[i]:.1f} x {widths[i]:.1f}" if element_shapes[i] == 'rectangle' else f"Ø {lengths[i]:.1f}",
                    f"({cx_v[i]:.2f}, {cy_v[i]:.2f})"
                ])
                
                # Unpack directional clearances [top, right, bottom, left].
                c_top, c_right, c_bot, c_left = clearance_dirs[i]

                # Check if component resides on Front side.
                if not is_back_side[i]: 
                    # Store center X and Y position coordinates.
                    ex, ey = cx_v[i], cy_v[i]
                    # Handle drawing for rectangular component shapes on Front view.
                    if element_shapes[i] == 'rectangle':
                        # Compute outer clearance boundary rectangle left edge coordinate.
                        c_x = ex - lengths[i]/2.0 - c_left
                        # Compute outer clearance boundary rectangle bottom edge coordinate.
                        c_y = ey - widths[i]/2.0 - c_bot
                        # Compute total width of clearance region (body width + left/right clearances).
                        c_w = lengths[i] + c_left + c_right
                        # Compute total height of clearance region (body height + top/bottom clearances).
                        c_h = widths[i] + c_top + c_bot
                        # Draw soft blue rectangle showing component extended clearance boundary area.
                        ax_f.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A9C7EB', edgecolor='none', alpha=0.5))
                        # Draw solid royal blue rectangle representing physical component body.
                        ax_f.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.9, edgecolor='navy', linewidth=1.5))
                    # Handle drawing for circular component shapes on Front view.
                    else:
                        # Draw soft blue circle showing clearance region around circular component.
                        ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2 + clearance, facecolor='#A9C7EB', edgecolor='none', alpha=0.5))
                        # Draw solid royal blue circle representing physical component body.
                        ax_f.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.9, edgecolor='navy', linewidth=1.5))
                    
                    # Draw mounting pins/holes for front-side component.
                    for ii in range(offset_counts[i]):
                        # Get relative pin offset distances (dx, dy).
                        dx, dy = flat_offsets[i_start + ii]
                        # Draw solid red pin circle on Front view.
                        ax_f.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', zorder=4))
                        # Draw ghosted red pin circle on Back view (flipped horizontally) showing drill hole penetration.
                        ax_b.add_patch(plt.Circle((-(ex + dx), ey + dy), ins_rad, facecolor='crimson', edgecolor='#5C0612', linewidth=1.5, alpha=0.6, zorder=3))
                        
                    # Write component name label text centered on component body.
                    ax_f.text(ex, ey, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
                    
                    # Draw dashed trace outline on Back view showing component's presence on opposite side.
                    ex_trace_b = -cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        ax_b.add_patch(plt.Rectangle((ex_trace_b - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='navy', linewidth=1.2, alpha=0.7))
                    else:
                        ax_b.add_patch(plt.Circle((ex_trace_b, cy_v[i]), lengths[i]/2, fill=False, linestyle='--', edgecolor='navy', linewidth=1.2, alpha=0.7))
                # Otherwise, component resides on Back side.
                else:
                    # Store center X and Y position coordinates.
                    ex_b, ey_b = cx_v[i], cy_v[i]
                    # Handle drawing for rectangular component shapes on Back view.
                    if element_shapes[i] == 'rectangle':
                        # Compute clearance boundary rectangle left edge coordinate.
                        c_x = ex_b - lengths[i]/2.0 - c_left
                        # Compute clearance boundary rectangle bottom edge coordinate.
                        c_y = ey_b - widths[i]/2.0 - c_bot
                        # Compute total width of clearance region.
                        c_w = lengths[i] + c_left + c_right
                        # Compute total height of clearance region.
                        c_h = widths[i] + c_top + c_bot
                        # Draw soft green rectangle showing clearance region for back-side component.
                        ax_b.add_patch(plt.Rectangle((c_x, c_y), c_w, c_h, facecolor='#A3D1A3', edgecolor='none', alpha=0.5))
                        # Draw solid dark green rectangle representing physical component body.
                        ax_b.add_patch(plt.Rectangle((ex_b - lengths[i]/2, ey_b - widths[i]/2), lengths[i], widths[i], color='darkgreen', alpha=0.9, edgecolor='darkslategrey', linewidth=1.5))
                    # Handle drawing for circular component shapes on Back view.
                    else:
                        # Draw soft green circle showing clearance region.
                        ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2 + clearance, facecolor='#A3D1A3', edgecolor='none', alpha=0.5))
                        # Draw solid dark green circle representing physical component body.
                        ax_b.add_patch(plt.Circle((ex_b, ey_b), lengths[i]/2, color='darkgreen', alpha=0.9, edgecolor='darkslategrey', linewidth=1.5))
                    
                    # Draw mounting pins for back-side component.
                    for ii in range(offset_counts[i]):
                        # Get relative pin offset distances (dx, dy).
                        dx, dy = flat_offsets[i_start + ii]
                        # Draw solid orange pin circle on Back view.
                        ax_b.add_patch(plt.Circle((ex_b + dx, ey_b + dy), ins_rad, color='orange', zorder=4))
                        # Draw ghosted orange pin circle on Front view showing drill hole penetration.
                        ax_f.add_patch(plt.Circle((-(ex_b + dx), ey_b + dy), ins_rad, facecolor='orange', edgecolor='#733D00', linewidth=1.5, alpha=0.6, zorder=3))
                        
                    # Write component name label text centered on component body.
                    ax_b.text(ex_b, ey_b, element_names[i], color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
                    
                    # Draw dashed trace outline on Front view showing back component's ghosted footprint.
                    ex_trace_f = -cx_v[i]
                    if element_shapes[i] == 'rectangle':
                        ax_f.add_patch(plt.Rectangle((ex_trace_f - lengths[i]/2, cy_v[i] - widths[i]/2), lengths[i], widths[i], fill=False, linestyle='--', edgecolor='darkslategrey', linewidth=1.2, alpha=0.7))
                    else:
                        ax_f.add_patch(plt.Circle((ex_trace_f, cy_v[i]), lengths[i]/2, fill=False, linestyle='--', edgecolor='darkslategrey', linewidth=1.2, alpha=0.7))

            # Apply title names, plot boundaries, aspect ratios, and grid lines to Front and Back view panels.
            for name, ax in [("FRONT SIDE VIEW", ax_f), ("BACK SIDE VIEW (FLIPPED)", ax_b)]:
                # Assign sub-plot title text.
                ax.set_title(name, fontweight='bold', fontsize=11)
                # Set horizontal axis limits extending 10mm past physical board edges.
                ax.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
                # Set vertical axis limits extending 10mm past physical board edges.
                ax.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
                # Force equal 1:1 scale proportions so rectangles look true to real-world dimensions.
                ax.set_aspect('equal')
                # Draw light dotted grid background lines.
                ax.grid(True, linestyle=':', alpha=0.5)
            
            # Define column header names for bottom summary report table.
            headers = ["Component Name", "Layer Placement", "Mass (kg)", "Dimensions (mm)", "CoG Coordinates (X, Y)"]
            # Render formatted table widget into lower subplot area.
            ui_table = ax_table.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
            # Disable automatic font sizing so we can set fixed readable size manually.
            ui_table.auto_set_font_size(False)
            # Set table text font size to 9pt.
            ui_table.set_fontsize(9)
            # Adjust horizontal and vertical scaling padding of table cells.
            ui_table.scale(1.0, 1.3)
            
            # Style header row cells with dark blue background and bold white text.
            for col_idx in range(len(headers)):
                # Access top row header cell at column index.
                cell = ui_table[0, col_idx]
                # Set bold weight and white text color.
                cell.set_text_props(weight='bold', color='white')
                # Set cell background color to dark slate blue (#2C3E50).
                cell.set_facecolor('#2C3E50')
            
            # Adjust layout spacing tightly so elements don't overlap edges.
            plt.tight_layout()
            # Formulate output file path name for primary layout plot image.
            main_layout_path = os.path.join(OUTPUT_DIR, f"layout_{layout_idx}_main.png")
            # Save generated diagram to disk silently at high 200 DPI resolution.
            plt.savefig(main_layout_path, dpi=200, bbox_inches='tight')
            # Close figure window from memory to keep memory clean.
            plt.close(fig)

            # ---------------- IMAGE 2: CLEARANCE & DISTANCE PROXIMITY ENGINE ----------------
            # Create a second figure canvas (13x9 inches) dedicated to checking close pin-to-pin distances.
            fig_c, ax_c = plt.subplots(figsize=(13, 9))
            # Assign overall figure title explaining proximity verification map.
            fig_c.suptitle(f"Layout Alternative {layout_idx} — Inter-Insert Proximity Verification (< 40mm)\n(Combined Layer Projection Space)", 
                           fontsize=13, fontweight='bold')
            
            # Draw gray background rectangle for board body on proximity view plot.
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, panel_height, color='lightgray', alpha=0.2))
            # Draw translucent red border zone along top edge.
            ax_c.add_patch(plt.Rectangle((-panel_width/2, panel_height/2 - border_spacing), panel_width, border_spacing, color='red', alpha=0.08))
            # Draw translucent red border zone along bottom edge.
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2), panel_width, border_spacing, color='red', alpha=0.08))
            # Draw translucent red border zone along left edge.
            ax_c.add_patch(plt.Rectangle((-panel_width/2, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.08))
            # Draw translucent red border zone along right edge.
            ax_c.add_patch(plt.Rectangle((panel_width/2 - border_spacing, -panel_height/2 + border_spacing), border_spacing, panel_height - 2*border_spacing, color='red', alpha=0.08))

            # Draw translucent component body footprints and pins for combined top-down projection.
            for i in range(num_elements):
                # Retrieve starting index of pins for component 'i'.
                i_start = offset_indices[i]
                # Calculate pin radius.
                ins_rad = insert_diams[i] / 2.0
                
                # Determine absolute X coordinate based on front or back layer assignment.
                ex = cx_v[i] if not is_back_side[i] else -cx_v[i]
                # Determine Y coordinate.
                ey = cy_v[i]
                
                # If component is on Front layer.
                if not is_back_side[i]:
                    # Draw blue rectangle footprint if shape is rectangular.
                    if element_shapes[i] == 'rectangle':
                        ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='royalblue', alpha=0.25, edgecolor='navy', linewidth=1.2))
                    # Draw blue circle footprint if shape is circular.
                    else:
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='royalblue', alpha=0.25, edgecolor='navy', linewidth=1.2))
                    
                    # Draw red pins for front component.
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_c.add_patch(plt.Circle((ex + dx, ey + dy), ins_rad, color='crimson', alpha=0.7, zorder=4))
                # Otherwise, component is on Back layer.
                else:
                    # Draw green rectangle footprint if shape is rectangular.
                    if element_shapes[i] == 'rectangle':
                        ax_c.add_patch(plt.Rectangle((ex - lengths[i]/2, ey - widths[i]/2), lengths[i], widths[i], color='darkgreen', alpha=0.25, edgecolor='darkslategrey', linewidth=1.2))
                    # Draw green circle footprint if shape is circular.
                    else:
                        ax_c.add_patch(plt.Circle((ex, ey), lengths[i]/2, color='darkgreen', alpha=0.25, edgecolor='darkslategrey', linewidth=1.2))
                    
                    # Draw orange pins for back component.
                    for ii in range(offset_counts[i]):
                        dx, dy = flat_offsets[i_start + ii]
                        ax_c.add_patch(plt.Circle((ex - dx, ey + dy), ins_rad, color='orange', alpha=0.7, zorder=4))
                
                # Write transparent component name text over footprint.
                ax_c.text(ex, ey, element_names[i], color='black', alpha=0.6, ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

            # Initialize list to collect textual distance summary descriptions for pins closer than 40mm.
            close_pairs_labels = []
            # Start callout letter counter at zero.
            label_counter = 0
            # Define alphabet string used for lettering callout bubbles ('a', 'b', 'c'...).
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

            # Check every pair of pins between components to discover and measure close distances.
            for i in range(num_elements):
                # Retrieve pin starting index for component 'i'.
                i_start = offset_indices[i]
                # Compare against all subsequent components 'j'.
                for j in range(i + 1, num_elements):
                    # Retrieve pin starting index for component 'j'.
                    j_start = offset_indices[j]
                    
                    # Loop over every pin belonging to component 'i'.
                    for ii in range(offset_counts[i]):
                        # Get index offset for pin 'i'.
                        idx_i = i_start + ii
                        # Compute absolute world X position of pin 'i'.
                        xi_abs = abs_x_positions[i] + (-flat_offsets[idx_i, 0] if is_back_side[i] else flat_offsets[idx_i, 0])
                        # Compute absolute world Y position of pin 'i'.
                        yi_abs = cy_v[i] + flat_offsets[idx_i, 1]
                        
                        # Loop over every pin belonging to component 'j'.
                        for jj in range(offset_counts[j]):
                            # Get index offset for pin 'j'.
                            idx_j = j_start + jj
                            # Compute absolute world X position of pin 'j'.
                            xj_abs = abs_x_positions[j] + (-flat_offsets[idx_j, 0] if is_back_side[j] else flat_offsets[idx_j, 0])
                            # Compute absolute world Y position of pin 'j'.
                            yj_abs = cy_v[j] + flat_offsets[idx_j, 1]
                            
                            # Calculate straight-line physical distance between pin 'i' and pin 'j'.
                            dist = np.sqrt((xi_abs - xj_abs)**2 + (yi_abs - yj_abs)**2)
                            
                            # If pins are located within 40mm of each other, highlight them with a callout line!
                            if dist < 40.0:
                                # Get final projected X coordinate for pin 'i'.
                                xf1 = cx_v[i] + flat_offsets[idx_i, 0] if not is_back_side[i] else -(cx_v[i] + flat_offsets[idx_i, 0])
                                # Get final projected Y coordinate for pin 'i'.
                                yf1 = cy_v[i] + flat_offsets[idx_i, 1]
                                # Get final projected X coordinate for pin 'j'.
                                xf2 = cx_v[j] + flat_offsets[idx_j, 0] if not is_back_side[j] else -(cx_v[j] + flat_offsets[idx_j, 0])
                                # Get final projected Y coordinate for pin 'j'.
                                yf2 = cy_v[j] + flat_offsets[idx_j, 1]
                                
                                # Pick next letter from alphabet string for this callout pair.
                                current_letter = alphabet[label_counter % len(alphabet)]
                                # Increment callout letter counter.
                                label_counter += 1
                                
                                # Draw purple dashed line connecting the two close pins.
                                ax_c.plot([xf1, xf2], [yf1, yf2], color='purple', linestyle='--', linewidth=1.5, alpha=0.85, zorder=10)
                                # Place a purple circular letter badge ('a', 'b', 'c'...) directly at midpoint between pins.
                                ax_c.text((xf1 + xf2)/2, (yf1 + yf2)/2, current_letter, color='white', 
                                          fontsize=8, fontweight='bold', ha='center', va='center',
                                          bbox=dict(boxstyle="circle,pad=0.2", fc="darkmagenta", ec="none", alpha=0.9), zorder=11)
                                
                                # Record measurement text into summary legend list.
                                close_pairs_labels.append(f"{current_letter} — {dist:.2f} mm ({element_names[i]} ↔ {element_names[j]})")

            # Check if any close pin pairs under 40mm were flagged.
            if close_pairs_labels:
                # Join all close distance strings with newline line breaks.
                legend_box_text = "\n".join(close_pairs_labels)
                # Render summary text box on upper-right side outside plot boundaries listing all flagged pairs.
                ax_c.text(1.02, 0.95, f"Pin Distances Summary:\n\n{legend_box_text}", 
                          transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))
            # If no close pin pairs under 40mm were found.
            else:
                # Render text box declaring that no pin clearance violations under 40mm exist.
                ax_c.text(1.02, 0.95, "Pin Distances Summary:\n\nNo insert violations\nor pins found under 40mm.", 
                          transform=ax_c.transAxes, fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec="#BDC3C7", lw=1.2))

            # Create visual legend patch entry for Front Layer components (blue).
            front_patch = mpatches.Patch(color='royalblue', alpha=0.4, label='Front Layer Components')
            # Create visual legend patch entry for Back Layer components (green).
            back_patch = mpatches.Patch(color='darkgreen', alpha=0.4, label='Back Layer Components')
            # Display color legend in bottom-left corner of plot.
            ax_c.legend(handles=[front_patch, back_patch], loc='lower left')

            # Set plot bounds, equal aspect ratios, and background grid lines for proximity plot.
            ax_c.set_xlim(-panel_width/2 - 10, panel_width/2 + 10)
            ax_c.set_ylim(-panel_height/2 - 10, panel_height/2 + 10)
            ax_c.set_aspect('equal')
            ax_c.grid(True, linestyle=':', alpha=0.4)
            
            # Formulate output file path name for proximity distance layout map.
            distance_layout_path = os.path.join(OUTPUT_DIR, f"layout_{layout_idx}_distance_map.png")
            # Save distance map plot silently at 200 DPI.
            plt.savefig(distance_layout_path, dpi=200, bbox_inches='tight')
            # Close figure canvas to free memory.
            plt.close(fig_c)

        # Print final completion message providing full path to output directory containing saved images.
        print(f"📁 All 10 images silently generated and saved to: {os.path.abspath(OUTPUT_DIR)}/")

# Check if this Python script is being run directly from the command terminal.
if __name__ == '__main__':
    # Launch the master optimization workflow function.
    run_optimization()