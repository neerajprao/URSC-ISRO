"""
data_loader.py
---------------
Phase 1: Dataset integration – read .npz files, build PyG Data objects,
and provide a PyTorch Dataset for supervised pretraining.
"""

import os
import glob
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
from penalty_wrapper import load_board_config, compute_penalty

# --------------------------------------------------------------------------
# 1.  Build a PyG Data object from positions and static config
# --------------------------------------------------------------------------
def build_pyg_data(x, y, config, moved_flag=None, edge_radius=None):
    """
    Construct a torch_geometric.data.Data object from component positions
    and the static board configuration.

    Args:
        x (np.ndarray): 1D array of x coordinates (length N)
        y (np.ndarray): 1D array of y coordinates (length N)
        config (dict): from load_board_config()
        moved_flag (np.ndarray, optional): boolean array (N) indicating
            if component was moved recently. Default: all False.
        edge_radius (float, optional): if given, connect components
            within this distance; otherwise fully connected.

    Returns:
        Data object with:
            x (node features)      : [N, F_node]
            edge_index             : [2, E]
            edge_attr              : [E, F_edge]
            global_features        : [1, F_global] (stored as `globals`)
            y (target penalty)     : scalar (if provided, else 0)
            Also stores original positions for reference.
    """
    N = config['num_components']
    if moved_flag is None:
        moved_flag = np.zeros(N, dtype=bool)

    # -------- Node features --------
    # Static: mass, length, width, area, aspect_ratio, insert_diam,
    #         is_back_side, normalized? Also include half-dimensions?
    masses = config['masses']
    lengths = config['lengths']
    widths = config['widths']
    insert_diams = config['insert_diams']
    is_back = config['is_back_side'].astype(np.float32)

    # Compute area and aspect ratio
    area = lengths * widths
    aspect = np.where(widths > 0, lengths / widths, 1.0)

    # Normalise static features roughly (maybe later we can do standardisation)
    # For now, keep raw but scale to ~[0,1] range.
    # Use board dimensions for scaling.
    panel_w = config['panel_width']
    panel_h = config['panel_height']
    # Dynamic: x, y (normalised to [-1,1] relative to board centre)
    x_norm = x / (panel_w / 2.0)   # in [-1,1]
    y_norm = y / (panel_h / 2.0)

    # moved flag as float
    moved_f = moved_flag.astype(np.float32).reshape(-1, 1)

    # Combine node features
    node_feats = np.column_stack([
        masses,          # kg
        lengths,         # mm
        widths,          # mm
        area,            # mm^2
        aspect,          # ratio
        insert_diams,    # mm
        is_back,         # 0/1
        x_norm,          # [-1,1]
        y_norm,          # [-1,1]
        moved_f          # 0/1
    ])
    # Convert to torch float
    node_feats = torch.tensor(node_feats, dtype=torch.float)

    # -------- Edge features & edge_index --------
    # Compute pairwise distances (Euclidean on original coordinates)
    coords = np.column_stack([x, y])
    diff = coords[:, None, :] - coords[None, :, :]   # N x N x 2
    dist = np.linalg.norm(diff, axis=2)              # N x N

    if edge_radius is not None:
        # k‑NN or threshold: use threshold for simplicity
        mask = (dist <= edge_radius) & (dist > 0)
    else:
        # fully connected without self-loops
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)

    rows, cols = np.where(mask)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)

    # Edge attributes: dx, dy, distance, same_side, overlap_indicator (approx)
    # overlap indicator: for same side components, check if bounding boxes overlap
    # (we can compute a rough overlap flag using half dimensions + clearance)
    dx = diff[rows, cols, 0]
    dy = diff[rows, cols, 1]
    dist_edges = dist[rows, cols]
    same_side = (config['is_back_side'][rows] == config['is_back_side'][cols]).astype(np.float32)

    # Approximate overlap: if same side and |dx| < half_len_sum and |dy| < half_wid_sum
    hl = config['hl']
    hw = config['hw']
    # Include clearance? For now just half dims.
    half_len_sum = hl[rows] + hl[cols]
    half_wid_sum = hw[rows] + hw[cols]
    overlap_x = np.abs(dx) < half_len_sum
    overlap_y = np.abs(dy) < half_wid_sum
    overlap_ind = (overlap_x & overlap_y).astype(np.float32)

    edge_attr = np.column_stack([
        dx, dy, dist_edges, same_side, overlap_ind
    ])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # -------- Global features --------
    # Compute current total penalty and CG offset (for reference)
    penalty = compute_penalty(x, y, config)
    total_mass = np.sum(masses)
    abs_x = np.where(config['is_back_side'] == 1, -x, x)
    cg_x = np.sum(abs_x * masses) / total_mass
    cg_y = np.sum(y * masses) / total_mass
    cg_offset = np.sqrt(cg_x**2 + cg_y**2)

    # Also compute number of overlapping pairs? (maybe later)
    # For now: panel dimensions (normalised), border, clearance, current penalty, cg_offset
    glob_feats = np.array([
        config['panel_width'] / 1000.0,   # scale to metres or normalize
        config['panel_height'] / 1000.0,
        config['border_spacing'] / 1000.0,
        config['clearance'] / 1000.0,
        penalty / 1e6,                   # scale penalty
        cg_offset / 10.0,                # scale
    ], dtype=np.float32)
    glob_feats = torch.tensor(glob_feats, dtype=torch.float).view(1, -1)

    # Create Data object
    data = Data(
        x=node_feats,
        edge_index=edge_index,
        edge_attr=edge_attr,
        globals=glob_feats,            # custom attribute
        y=torch.tensor([penalty], dtype=torch.float),  # target for pretraining
        # store original coordinates for later use
        pos=torch.tensor(coords, dtype=torch.float),
        cg_offset=torch.tensor([cg_offset], dtype=torch.float)
    )
    return data


# --------------------------------------------------------------------------
# 2.  Dataset class for .npz files
# --------------------------------------------------------------------------
class PCBnpzDataset(Dataset):
    """
    PyTorch Geometric Dataset that reads all .npz files from a directory.
    Each .npz is expected to contain at least arrays 'x' and 'y' (length N)
    and optionally 'penalty' (scalar). If 'penalty' is missing, it is computed
    on the fly using the config.
    """
    def __init__(self, root, board_config, transform=None, pre_transform=None):
        self.root = root
        self.config = board_config
        self.transform = transform
        self.pre_transform = pre_transform
        self.files = sorted(glob.glob(os.path.join(root, '*.npz')))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {root}")

    def len(self):
        return len(self.files)

    def get(self, idx):
        filepath = self.files[idx]
        with np.load(filepath, allow_pickle=True) as npz:
            # Expect 'x' and 'y'
            if 'x' in npz and 'y' in npz:
                x = npz['x'].astype(np.float64)
                y = npz['y'].astype(np.float64)
            elif 'positions' in npz:
                pos = npz['positions']
                if pos.ndim == 1:
                    # flat alternating x,y
                    x = pos[::2]
                    y = pos[1::2]
                else:
                    x = pos[:, 0]
                    y = pos[:, 1]
            else:
                raise KeyError(f"File {filepath} does not contain 'x'/'y' or 'positions'")

            # Optional: moved_flag
            moved_flag = npz.get('moved_flag', None)
            if moved_flag is not None:
                moved_flag = moved_flag.astype(bool)
            else:
                moved_flag = np.zeros(len(x), dtype=bool)

            # Optional: penalty (if not given, compute)
            if 'penalty' in npz:
                penalty = float(npz['penalty'])
            else:
                penalty = compute_penalty(x, y, self.config)

        data = build_pyg_data(x, y, self.config, moved_flag)
        # Override penalty if computed from file (to keep consistency)
        data.y = torch.tensor([penalty], dtype=torch.float)
        return data


# --------------------------------------------------------------------------
# 3.  Helper: Load all data into a list (for quick testing)
# --------------------------------------------------------------------------
def load_all_npz_data(data_dir, board_config):
    """
    Load all .npz files and return a list of PyG Data objects.
    """
    dataset = PCBnpzDataset(data_dir, board_config)
    return [dataset[i] for i in range(len(dataset))]


# --------------------------------------------------------------------------
# 4.  Test (if run as main)
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Example usage: load config and dataset
    cfg = load_board_config('newobj.json')
    # Assume we have a directory 'data' containing .npz files
    # data_dir = 'path/to/npz_files'
    # dataset = PCBnpzDataset(data_dir, cfg)
    # print(f"Loaded {len(dataset)} samples.")
    # sample = dataset[0]
    # print(sample)
    print("data_loader ready. Uncomment and set data_dir to test.")