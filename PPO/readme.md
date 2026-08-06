# PCB Layout Optimization with GNN + PPO

## Project Overview

This project replaces the existing CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer with a **Graph Neural Network (GNN) + Proximal Policy Optimization (PPO)** reinforcement learning agent for iterative PCB component layout refinement.

**Goal**: Train an RL agent that can iteratively adjust component (x, y) positions to minimize a composite penalty function (overlaps, CG offset, boundary violations, pin-to-pin clearances) and find **5 distinct feasible layouts** for a given board configuration.

**Hardware**: Apple MacBook M3 Pro with Metal Performance Shaders (MPS) acceleration.

---

## Table of Contents

1. [Architecture Summary](#architecture-summary)
2. [What Has Been Built](#what-has-been-built)
3. [What Remains To Be Done](#what-remains-to-be-done)
4. [Technical Deep Dive](#technical-deep-dive)
   - 4.1 [Board & Component Specifications](#board--component-specifications)
   - 4.2 [Penalty Function (Fitness Core)](#penalty-function-fitness-core)
   - 4.3 [Graph Representation](#graph-representation)
   - 4.4 [GNN Architecture](#gnn-architecture)
   - 4.5 [Action Space (Hybrid Discrete + Continuous)](#action-space-hybrid-discrete--continuous)
   - 4.6 [Reward Function Design](#reward-function-design)
   - 4.7 [PPO Training Algorithm](#ppo-training-algorithm)
   - 4.8 [Pretraining Strategy](#pretraining-strategy)
   - 4.9 [Adaptive Delta Schedule](#adaptive-delta-schedule)
5. [Dataset](#dataset)
6. [Training Configuration](#training-configuration)
7. [Known Issues & Bugs](#known-issues--bugs)
8. [File Structure](#file-structure)
9. [Usage Instructions](#usage-instructions)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Future Work](#future-work)
12. [References](#references)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT: newobj.json + .npz datasets               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐   │
│  │   Geometry   │───▶│   Penalty    │───▶│   Graph Construction (PyG)   │   │
│  │    Parser    │    │   Function   │    │                              │   │
│  └──────────────┘    │  (Numba JIT) │    │  • 18 nodes (components)       │   │
│                      └──────────────┘    │  • 306 edges (fully connected) │   │
│                                          │  • Node features: 18 dims      │   │
│                                          │  • Edge features: 5 dims       │   │
│                                          │  • Global features: 5 dims     │   │
│                                          └──────────────────┬───────────────┘   │
│                                                             │                   │
│                                                             ▼                   │
│                                          ┌──────────────────────────────┐   │
│                                          │      GNN ENCODER             │   │
│                                          │  • 3 GIN layers (custom)       │   │
│                                          │  • Edge-aware message passing  │   │
│                                          │  • Residual connections        │   │
│                                          │  • LayerNorm                   │   │
│                                          │  • Hidden dim: 256             │   │
│                                          │  • Output: node_emb [18, 256]  │   │
│                                          │            global_emb [1, 256]   │   │
│                                          └──────────────────┬───────────────┘   │
│                                                             │                   │
│                                          ┌──────────────────┴───────────────┐   │
│                                          │                                  │   │
│                                          ▼                                  ▼   │
│                               ┌─────────────────┐                  ┌──────────────┐ │
│                               │   POLICY HEAD   │                  │ VALUE HEAD   │ │
│                               │                 │                  │              │ │
│                               │ • Component     │                  │ • Predicts   │ │
│                               │   selector      │                  │   V(s)       │ │
│                               │   (Categorical) │                  │ • MLP:       │ │
│                               │                 │                  │   256→128→1  │ │
│                               │ • Shift predictor│                  │              │ │
│                               │   (Normal x2)   │                  └──────────────┘ │
│                               │                 │                                   │
│                               │ • Masked to     │                                   │
│                               │   violating     │                                   │
│                               │   components    │                                   │
│                               └─────────────────┘                                   │
│                                          │                                        │
│                                          ▼                                        │
│                               ┌─────────────────┐                                 │
│                               │  PPO TRAINER    │                                 │
│                               │                 │                                 │
│                               │ • 16 parallel   │                                 │
│                               │   CPU envs      │                                 │
│                               │ • MPS for NN    │                                 │
│                               │ • GAE (γ=0.99,  │                                 │
│                               │   λ=0.95)       │                                 │
│                               │ • Clipped       │                                 │
│                               │   surrogate     │                                 │
│                               │ • Reward norm   │                                 │
│                               └─────────────────┘                                 │
│                                          │                                        │
│                                          ▼                                        │
│                               ┌─────────────────┐                                 │
│                               │  5 FEASIBLE     │                                 │
│                               │  LAYOUTS        │                                 │
│                               └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## What Has Been Built

### ✅ Completed Components

| # | Component | Status | Details |
|---|-----------|--------|---------|
| 1 | **Imports & Device Detection** | ✅ | PyTorch, PyG, Numba, Matplotlib. Auto-detects MPS/CUDA/CPU |
| 2 | **Configuration System** | ✅ | `Config` dataclass with all hyperparameters. 80/20 dataset split |
| 3 | **Data Loading** | ✅ | `newobj.json` parser + `.npz` dataset inspector (1049 files) |
| 4 | **Component Expansion** | ✅ | Handles `Object qty > 1`. 18 total components from 10 definitions |
| 5 | **Geometry Parser** | ✅ | Extracts masses, lengths, widths, pin offsets, clearances, bounds |
| 6 | **Numba Penalty Function** | ✅ | `compute_fitness_core` JIT-compiled. ~0.006 ms per call |
| 7 | **Graph Construction** | ✅ | PyG `Data` with 18 nodes, 306 edges, 18-dim node features, 5-dim edge features |
| 8 | **Violation Mask** | ✅ | Computes which components violate constraints (for action masking) |
| 9 | **GNN Encoder** | ✅ | Custom edge-aware GIN. 3 layers, 256 hidden, residuals, LayerNorm |
| 10 | **Policy Head** | ✅ | Discrete (Categorical over 18 components) + Continuous (Normal dx, dy) |
| 11 | **Value Head** | ✅ | Predicts V(s) from global embedding. 256→128→1 |
| 12 | **GNN Policy Model** | ✅ | 1,439,878 parameters. Forward + action selection methods |
| 13 | **Environment (PCBEnv)** | ✅ | `reset()`, `step()`, dataset sampling, random init, jitter |
| 14 | **Reward Function** | ✅ | Log-scale penalty delta + tiered bonuses + CG bonus + step/move penalties |
| 15 | **Adaptive Delta Max** | ✅ | 30mm → 5mm based on current penalty magnitude |
| 16 | **PPO Trainer** | ✅ | Rollout collection, GAE, clipped surrogate, entropy bonus, gradient clipping |
| 17 | **Reward Normalization** | ✅ | Running mean/std normalization for stable training |
| 18 | **Supervised Pretraining** | ✅ | Encoder pretraining on 500 dataset files. Predicts log(penalty+1) |
| 19 | **Training Loop** | ✅ | Main loop with progress printing, testing, checkpointing |
| 20 | **Checkpointing** | ✅ | Saves model + optimizer + metrics every 50 updates |

### 📊 Pretraining Results

| Metric | Value |
|--------|-------|
| Dataset files used | 500 (450 train / 50 val) |
| Epochs | 50 |
| Final train loss | 1.0243 |
| Final val loss | 2.1953 |
| Pretraining time | ~10-15 minutes |

---

## What Remains To Be Done

### 🔴 Critical (Blocking)

| # | Task | Priority | Estimated Time |
|---|------|----------|----------------|
| 1 | **Fix training convergence** | 🔴 High | Unknown |
| 2 | **Verify reward signal is non-zero** | 🔴 High | 30 min |
| 3 | **Test if policy improves over random** | 🔴 High | 1 hour |

### 🟡 Important

| # | Task | Priority | Estimated Time |
|---|------|----------|----------------|
| 4 | **Add exploration noise** (entropy annealing, random action injection) | 🟡 Medium | 30 min |
| 5 | **Better initialization from dataset** (start episodes from near-valid layouts) | 🟡 Medium | 1 hour |
| 6 | **Curriculum learning** (start with fewer components, add gradually) | 🟡 Medium | 2 hours |
| 7 | **Hyperparameter sweep** (LR, entropy coef, delta_max schedule) | 🟡 Medium | 2 hours |
| 8 | **Add WandB or TensorBoard logging** | 🟡 Medium | 30 min |

### 🟢 Nice to Have

| # | Task | Priority | Estimated Time |
|---|------|----------|----------------|
| 9 | **Visualization of final 5 layouts** (front/back views, distance maps) | 🟢 Low | 1 hour |
| 10 | **Export layouts to .npz format** | 🟢 Low | 30 min |
| 11 | **Compare with CMA-ES baseline** | 🟢 Low | 2 hours |
| 12 | **Generalization test** (different board configs from dataset) | 🟢 Low | 2 hours |

---

## Technical Deep Dive

### 4.1 Board & Component Specifications

**Board Dimensions**: 960.0 mm × 582.4 mm  
**Border Spacing**: 20.0 mm (margin around edges)  
**Default Clearance**: 20.0 mm (minimum separation between components)  
**CG Tolerance**: 1.0 mm (target center of gravity offset from origin)

**18 Components (after quantity expansion)**:

| Idx | Name | Side | Shape | Dimensions | Mass (kg) | Pins | Insert (mm) | CF Faces | CF Len |
|-----|------|------|-------|------------|-----------|------|-------------|----------|--------|
| 0 | PDS | FRONT | rectangle | 100.0 × 112.0 | 1.500 | 4 | 4.0 | [1,3] | 65.0 |
| 1 | CCRR0 | FRONT | circle | Ø 45.0 | 0.150 | 6 | 6.0 | — | 65.0 |
| 2 | CCRR1 | FRONT | circle | Ø 45.0 | 0.150 | 6 | 6.0 | — | 65.0 |
| 3 | LD0 | FRONT | circle | Ø 32.0 | 0.040 | 3 | 6.0 | — | 65.0 |
| 4 | LD1 | FRONT | circle | Ø 32.0 | 0.040 | 3 | 6.0 | — | 65.0 |
| 5 | LD2 | FRONT | circle | Ø 32.0 | 0.040 | 3 | 6.0 | — | 65.0 |
| 6 | LD3 | FRONT | circle | Ø 32.0 | 0.040 | 3 | 6.0 | — | 65.0 |
| 7 | LD4 | FRONT | circle | Ø 32.0 | 0.040 | 3 | 6.0 | — | 65.0 |
| 8 | LD5 | FRONT | circle | Ø 32.0 | 0.040 | 3 | 6.0 | — | 65.0 |
| 9 | RS | FRONT | rectangle | 110.0 × 90.5 | 1.500 | 4 | 4.0 | [1,3] | 65.0 |
| 10 | BM1 | BACK | rectangle | 100.0 × 100.0 | 1.500 | 4 | 6.0 | [1] | 65.0 |
| 11 | PB0 | BACK | rectangle | 69.0 × 16.0 | 0.100 | 4 | 4.0 | [1] | 65.0 |
| 12 | PB1 | BACK | rectangle | 69.0 × 16.0 | 0.100 | 4 | 4.0 | [1] | 65.0 |
| 13 | PB2 | BACK | rectangle | 69.0 × 16.0 | 0.100 | 4 | 4.0 | [1] | 65.0 |
| 14 | PB3 | BACK | rectangle | 69.0 × 16.0 | 0.100 | 4 | 4.0 | [1] | 65.0 |
| 15 | RPU | BACK | rectangle | 260.0 × 253.0 | 3.000 | 6 | 4.0 | [2,4] | 65.0 |
| 16 | MES | BACK | rectangle | 128.0 × 117.0 | 0.125 | 4 | 6.0 | [3,4] | 65.0 |
| 17 | MT20 | BACK | rectangle | 275.0 × 25.0 | 0.400 | 4 | 4.0 | [1] | 65.0 |

**Total Pins**: 72 (4+6+6+3+3+3+3+3+3+4+4+4+4+4+4+6+4+4)

**Custom Face Clearances (CF)**:
- Face 1 = Top, Face 2 = Right, Face 3 = Bottom, Face 4 = Left
- Components with CF have extended clearance (65.0 mm) on specified faces
- Back-side components have left/right swapped in projection view

---

### 4.2 Penalty Function (Fitness Core)

The penalty function is the **heart of the system**. It computes a scalar score where **0.0 = perfect layout**.

**Penalty Components**:

| Component | Formula | Weight |
|-----------|---------|--------|
| CG Offset | `((offset - tolerance)²) × 1e8` if `offset > 1.0 mm` | 1e8 |
| Border Violation | `(total_violation²) × 1e8` | 1e8 |
| Overlap Area | `(overlap_area²) × 1e8` (same-side only) | 1e8 |
| Pin Distance | `((req_dist - actual_dist)²) × 1e8` | 1e8 |

**Key Implementation Details**:
- Compiled with Numba `@njit(fastmath=True, cache=True)`
- Back-side components have X-coordinate mirrored: `abs_x = -cx`
- Pin positions computed with component offsets: `pin_x = abs_x + offset_x`
- Required pin distances: 24mm (4mm pins), 36mm (6mm pins), 30mm (mixed)

**Performance**: ~0.006 ms per evaluation (100 calls averaged)

---

### 4.3 Graph Representation

**Node Features (18 dimensions)**:

| Dim | Feature | Type |
|-----|---------|------|
| 0 | Mass (kg) | Static |
| 1 | Length (mm) | Static |
| 2 | Width (mm) | Static |
| 3 | Insert diameter (mm) | Static |
| 4 | Is back side (0/1) | Static |
| 5 | Is rectangle (0/1) | Static |
| 6 | Is circle (0/1) | Static |
| 7 | Number of pins | Static |
| 8 | Clearance top (mm) | Static |
| 9 | Clearance right (mm) | Static |
| 10 | Clearance bottom (mm) | Static |
| 11 | Clearance left (mm) | Static |
| 12 | Max pin offset X (mm) | Static |
| 13 | Max pin offset Y (mm) | Static |
| 14 | Current cx (mm) | Dynamic |
| 15 | Current cy (mm) | Dynamic |
| 16 | Absolute x (mm, mirrored for back) | Dynamic |
| 17 | Moved recently flag (placeholder) | Dynamic |

**Edge Features (5 dimensions)**:

| Dim | Feature |
|-----|---------|
| 0 | Euclidean distance between component centers |
| 1 | Relative dx (abs_x_j - abs_x_i) |
| 2 | Relative dy (cy_j - cy_i) |
| 3 | Required pin-to-pin distance |
| 4 | Same side flag (0/1) |

**Global Features (5 dimensions)**:
- Panel width, Panel height, Border spacing, Current penalty, Current CG offset

**Graph Statistics**:
- Nodes: 18 (one per component)
- Edges: 306 (fully connected, directed)
- Edge index: [2, 306] tensor

---

### 4.4 GNN Architecture

**Encoder**: Custom edge-aware message passing (not standard GINConv due to PyG version compatibility)

```
Input: x [N, 18], edge_index [2, E], edge_attr [E, 5]

1. Edge Encoding:
   edge_emb = MLP(edge_attr) → [E, 256]

2. Node Projection:
   h = Linear(18 → 256)(x) → [N, 256]

3. For each of 3 layers:
   a. Message: h_src + edge_emb → combiner → [E, 256]
   b. Aggregate: mean aggregation to destination nodes → [N, 256]
   c. Update: concat[h, aggr] → MLP → [N, 256]
   d. Residual: h_new = h + h_new
   e. LayerNorm + ReLU

4. Global Pooling:
   h_max = max_pool(h) → [1, 256]
   h_mean = mean_pool(h) → [1, 256]
   global_emb = concat[h_max, h_mean] → MLP → [1, 256]

Output: node_emb [N, 256], global_emb [1, 256]
```

**Policy Head**:
```
Input: node_emb [N, 256], global_emb [1, 256], violation_mask [N], delta_max

Component Selector:
  logits = Linear(256 → 1)(node_emb) → [N]
  masked_logits = logits (violation_mask=True) or -inf (False)
  comp_dist = Categorical(masked_logits)

Shift Predictor (per component):
  combined = concat[node_emb, global_emb_expanded] → [N, 512]
  h = MLP(512 → 256 → 256)(combined)
  mu_x = tanh(Linear(256 → 1)(h)) * delta_max
  log_std_x = clamp(Linear(256 → 1)(h), -5, 2)
  std_x = exp(log_std_x) * delta_max
  (same for y)

Output: comp_dist, (Normal_x, Normal_y), masked_logits
```

**Value Head**:
```
Input: global_emb [1, 256]
V(s) = MLP(256 → 128 → 128 → 1)(global_emb)
```

**Total Parameters**: 1,439,878

---

### 4.5 Action Space (Hybrid Discrete + Continuous)

**Discrete Action**: Select component to move
- Space: {0, 1, ..., 17} (18 components)
- Masking: Only components with at least one violation are selectable
- Fallback: If no violations (layout valid), all components available (episode terminates anyway)

**Continuous Action**: Shift selected component
- dx ~ Normal(μ_x, σ_x), dy ~ Normal(μ_y, σ_y)
- μ bounded by tanh: μ ∈ [-delta_max, +delta_max]
- σ = exp(log_std) * delta_max, with log_std ∈ [-5, 2]
- Actual shift clipped to board bounds after application

**Action Sampling**:
```python
comp_idx ~ Categorical(masked_logits)
dx ~ Normal(mu_x[comp_idx], std_x[comp_idx])
dy ~ Normal(mu_y[comp_idx], std_y[comp_idx])
```

---

### 4.6 Reward Function Design

**Version 2 (Current)**:

```python
reward = reward_penalty + threshold_bonus + cg_bonus + step_penalty + move_penalty
```

| Component | Formula | Notes |
|-----------|---------|-------|
| Penalty Delta | `sign(Δ) * log1p(|Δ| / 1e6)` | Log-scale handles 1e15 → 0 smoothly |
| Threshold Bonus | +5 for crossing 1e12, 1e6, 1e3; +10 for crossing 1.0 | Encourages milestone progress |
| CG Bonus | +10 (one-time) when `penalty < 100` AND `CG < 1.0mm` | Only near-valid layouts |
| Step Penalty | -0.01 per step | Living penalty |
| Move Penalty | -0.001 * (dx² + dy²) | Discourages unnecessary large moves |

**Reward Normalization**: Running mean/std over batch of 16 envs

---

### 4.7 PPO Training Algorithm

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 (with 1000-step linear warmup) |
| Discount (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| Clip Epsilon (ε) | 0.2 |
| Value Coefficient | 0.5 |
| Entropy Coefficient | 0.01 |
| Gradient Clip | 0.5 |
| Minibatch Size | 64 |
| PPO Epochs | 10 |
| Rollout Steps | 2048 (total across all envs) |

**Training Loop**:
1. Reset 16 parallel environments
2. For 128 steps per env (2048 total):
   - Sample actions from current policy
   - Step environments, collect (s, a, r, s', done)
   - Normalize rewards using running statistics
3. Compute GAE advantages and returns
4. For 10 epochs:
   - Shuffle transitions
   - Update policy with clipped surrogate loss
   - Update value function with MSE loss
   - Add entropy bonus
5. Repeat until 5 feasible layouts found or max updates reached

---

### 4.8 Pretraining Strategy

**Task**: Supervised learning to predict `log(penalty + 1)` from layout graph

**Dataset**: 500 `.npz` files (450 train / 50 val)

**Model**: GNN encoder + penalty prediction head

**Loss**: MSE between predicted and actual log(penalty + 1)

**Results**:
- Train loss: decreased from ~5.0 to ~1.0
- Val loss: ~2.2 (some overfitting after epoch 30)

**Transfer**: Pretrained encoder and value head weights loaded into PPO model

---

### 4.9 Adaptive Delta Schedule

Delta max (maximum shift per step) adapts based on current penalty:

| Penalty Range | Delta Max | Rationale |
|---------------|-----------|-----------|
| > 1e12 | 30.0 mm | Layout is catastrophic, need big moves |
| 1e6 - 1e12 | 20.0 mm | Major overlap, large adjustments |
| 1e3 - 1e6 | 10.0 mm | Moderate issues, finer control |
| 1.0 - 1e3 | 7.0 mm | Getting close, precise moves |
| < 1.0 | 5.0 mm | Near valid, minimal adjustments |

---

## Dataset

**Source**: 1049 `.npz` files generated by CMA-ES optimizer

**File Structure** (per `.npz`):
```python
{
    'config_json': str,          # Board configuration JSON
    'best_sol': ndarray [2N],    # Optimal positions [cx0, cy0, cx1, cy1, ...]
    'penalty': float64,          # Final penalty score
    'cg_x': float64,             # Center of gravity X
    'cg_y': float64,             # Center of gravity Y
    'num_components': int64,       # N (varies per file)
    'element_names': ndarray [N], # Component names
    'masses': ndarray [N],
    'lengths': ndarray [N],
    'widths': ndarray [N],
    'insert_diams': ndarray [N],
    'is_back_side': ndarray [N],
    'panel_width': float64,
    'panel_height': float64,
    'border_spacing': float64,
    'clearance': float64,
    'flat_offsets': ndarray [total_pins, 2],
    'offset_counts': ndarray [N],
    'offset_indices': ndarray [N],
    'clearance_dirs': ndarray [N, 4],
    'req_d_matrix': ndarray [N, N],
    'lower_bounds': ndarray [2N],
    'upper_bounds': ndarray [2N],
}
```

**Important**: Dataset contains layouts from **various board configurations** (different N), not just the target 18-component board. Used for generalization pretraining.

**Split**: 839 train (80%) / 210 val (20%)

---

## Training Configuration

```python
@dataclass
class Config:
    # Board
    PANEL_WIDTH = 960.0
    PANEL_HEIGHT = 582.4
    BORDER_SPACING = 20.0
    CLEARANCE = 20.0
    CG_TOLERANCE = 1.0
    
    # Dataset
    DATASET_DIR = "./datasets"
    DATASET_SPLIT = 0.8
    JITTER_STD = 15.0
    
    # Episode
    MAX_STEPS_PER_EPISODE = 50
    PENALTY_THRESHOLD = 1e-3
    TARGET_LAYOUT_COUNT = 5
    
    # Action Space
    DELTA_MAX_START = 30.0
    DELTA_MAX_END = 5.0
    
    # Reward
    CG_BONUS = 10.0
    STEP_PENALTY = 0.01
    MOVE_PENALTY_COEF = 0.001
    
    # GNN
    NODE_FEATURE_DIM = 18
    EDGE_FEATURE_DIM = 5
    GLOBAL_FEATURE_DIM = 5
    HIDDEN_DIM = 256
    GNN_LAYERS = 3
    
    # PPO
    NUM_ENVS = 16
    ROLLOUT_STEPS = 2048
    MINIBATCH_SIZE = 64
    PPO_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    LR = 3e-4
    LR_WARMUP_STEPS = 1000
    
    # Pretraining
    PRETRAIN_EPOCHS = 50
    PRETRAIN_BATCH_SIZE = 32
    PRETRAIN_LR = 1e-3
    
    # Logging
    LOG_INTERVAL = 10
    PLOT_INTERVAL = 20
    SAVE_INTERVAL = 50
    OUTPUT_DIR = "./checkpoints"
```

---

## Known Issues & Bugs

### 🔴 Critical

| # | Issue | Status | Impact |
|---|-------|--------|--------|
| 1 | **Training not converging** | 🔴 Open | Agent may not be learning; need to verify reward signal is non-zero and gradients flow |
| 2 | **Policy loss vs value loss imbalance** | 🔴 Open | Value loss dominates; may need separate learning rates |
| 3 | **No exploration mechanism** | 🔴 Open | Entropy coefficient may be too low; agent gets stuck in local optima |

### 🟡 Resolved

| # | Issue | Fix | Cell |
|---|-------|-----|------|
| 4 | `GINConv` doesn't accept `edge_dim` | Custom message passing with manual edge aggregation | 7 |
| 5 | `select_action` samples all 18 components | Index into selected component only | 7 |
| 6 | `allow_pickle=False` on `.npz` | Added `allow_pickle=True` | 3 |
| 7 | `PCBEnvFinal` has no `cfg` attribute | Added `self.config = config` in `__init__` | 10 |
| 8 | CG bonus triggers when penalty worsens | Added `new_penalty < 100` guard | 10 |
| 9 | Value loss astronomically high | Added reward normalization | 9 |

### 🟢 Minor

| # | Issue | Status |
|---|-------|--------|
| 10 | Pretraining overfitting after epoch 30 | Monitor, may need early stopping |
| 11 | Dataset layouts have varying N | Handled by padding/truncation |
| 12 | MPS may fall back to CPU for some PyG ops | Not observed yet |

---

## File Structure

```
project/
├── newobj.json                    # Board & component specifications
├── new.py                         # Original CMA-ES code (reference)
├── readme.md                      # This file
├── datasets/                      # 1049 .npz layout files
│   ├── layout_000001.npz
│   ├── layout_000002.npz
│   └── ...
├── optimized_layouts/             # Output from CMA-ES (reference)
│   ├── layout_1_main.png
│   ├── layout_1_distance_map.png
│   └── ...
├── checkpoints/                   # Saved models during training
│   ├── checkpoint_u50.pt
│   ├── checkpoint_u100.pt
│   └── ...
└── notebook.ipynb                 # Main training notebook (cells 1-10)
```

---

## Usage Instructions

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy numba matplotlib
```

### Running the Notebook

1. **Cell 1**: Imports & device detection
2. **Cell 2**: Configuration
3. **Cell 3**: Load `newobj.json` + inspect `.npz` files
4. **Cell 4**: Parse component geometry
5. **Cell 5**: Compile Numba penalty function
6. **Cell 6**: Build graph construction
7. **Cell 7**: Define GNN model (consolidated)
8. **Cell 8**: Define environment
9. **Cell 9**: Define PPO trainer
10. **Cell 9.5**: Pretrain encoder + test reward function
11. **Cell 10**: Main training loop

**After kernel restart**: Re-run cells 1-7 in order, then continue from where you left off.

### Monitoring Training

Progress prints every 5 updates:
```
[Update 5] time=2.5min | avg_penalty=1.23e+15 | viol=12.5 | reward=+0.123±0.456 | policy_loss=0.123 | value_loss=0.456 | entropy=0.789
```

Test evaluation every 20 updates:
```
--- TESTING at update 20 ---
Test summary: 0/10 successes (0.0%)
Avg final penalty: 5.67e+14
Feasible layouts so far: 0/5
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Penalty function speed | ~0.006 ms/call |
| Graph construction | ~5 ms (18 nodes, 306 edges) |
| Model forward pass | ~10 ms on MPS |
| Rollout collection (2048 steps, 16 envs) | ~30-60 seconds |
| PPO update (10 epochs) | ~60-120 seconds |
| Total time per update | ~2-3 minutes |
| Estimated time for 200 updates | ~4-6 hours |

---

## Future Work

1. **Behavior Cloning**: If dataset contains optimization trajectories, pretrain policy via imitation learning
2. **Offline RL (CQL/IQL)**: Train from dataset without online interaction
3. **Multi-objective optimization**: Separate reward components for CG, overlap, pin distance
4. **Hierarchical policy**: First select region, then component, then shift
5. **3D extension**: Add Z-axis for multi-layer boards
6. **Transfer learning**: Train on multiple board configs, fine-tune on new ones

---

## References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, 2017
- Xu et al., *How Powerful are Graph Neural Networks?*, 2019 (GIN)
- Original CMA-ES code (Hansen, 2016)
- PyTorch Geometric documentation

---

## Contact / Development Notes

- **Hardware**: Apple M3 Pro, 18GB unified memory
- **Python**: 3.13
- **PyTorch**: 2.x with MPS backend
- **PyTorch Geometric**: Latest stable
- **Numba**: 0.60+ for JIT compilation

**Last Updated**: 2026-08-05
```