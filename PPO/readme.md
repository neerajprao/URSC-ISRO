# Project Plan: GNN + PPO for PCB Layout Refinement (Iterative Improvement)

## 1. Overview
This project replaces the CMA‑ES optimizer with a **Graph Neural Network (GNN) + Proximal Policy Optimization (PPO)** agent that iteratively improves an existing PCB component layout. The agent selects a component and adjusts its (x, y) position by a small amount to reduce the overall penalty (overlaps, CG offset, boundary violations, pin‑to‑pin clearances). The approach learns to satisfy constraints efficiently and can generalise across board configurations. Training is parallelised and accelerated using Apple’s MPS (Metal Performance Shaders) on an M3 Pro chip.

We have **1049 datasets in `.npz` format** (layouts with penalties and component attributes). These will be used for **supervised pretraining** of the GNN encoder and value network, **behaviour cloning** (if actions are available), and as a **rich source of initial states** to improve exploration and generalisation. This data dramatically accelerates training and boosts final performance.

---

## 2. MDP Formulation (Iterative Improvement)

### State (Observation)
- Complete layout of all components (positions fixed).
- **Graph representation**:  
  - **Nodes** (N components): static features (mass, dimensions, shape, side, insert diameter, offset list, clearance directions) + dynamic features (current x, y, and a flag indicating if moved recently).  
  - **Edges**: fully connected (or k‑NN) with features: Euclidean distance, relative (dx, dy), required pin‑to‑pin distance, same‑side flag, and overlap indicator.  
  - **Global features**: board dimensions, border spacing, CG tolerance, current total penalty, and current CG offset.

### Action (Hybrid)
1. **Discrete**: select one component to move (categorical over N components).  
   - **Masking**: only components that violate at least one constraint (overlap, boundary, pin distance, or CG imbalance) are selectable. This focuses exploration on problematic parts and improves sample efficiency.
2. **Continuous**: choose a shift vector (Δx, Δy) ∈ [-δ_max, δ_max]², where δ_max is a hyper‑parameter (e.g., 20 mm). The shift is sampled from a squashed Gaussian (tanh scaling).  
   - **Alternative**: use a Beta distribution for bounded shifts to avoid clipping artefacts.

### Reward
- **Immediate reward**: `r = -(penalty_new - penalty_old)` (negative penalty change).
- **Shaping bonuses**:
  - `+10` if the CG offset falls below `CG_TOLERANCE` (prioritise balance).
  - `-0.01` per step (encourage minimal movement if penalty is already low).
  - `-0.001 * (Δx² + Δy²)` to penalise unnecessarily large moves.
- The total return is the sum of per‑step rewards.

### Transition
- Deterministic: update the selected component’s coordinates with the chosen shift.  
- The graph is updated accordingly.

### Episode
- Starts from an initial layout (random, heuristic, or sampled from our `.npz` dataset with jittering).  
- The agent performs a fixed number of moves (e.g., 30–50) or until the penalty falls below a threshold.  
- The total return is the sum of per‑step rewards.

---

## 3. Neural Network Architecture

All components are implemented in **PyTorch** and run on the **MPS** device (Apple M3 Pro). We use **PyTorch Geometric** for GNN layers.

### GNN Encoder (Shared)
- **Input dimensions**:  
  - Node features: ~20  
  - Edge features: ~5  
  - Global features: ~5  
- **Layers**:  
  - 3 Graph Convolutional layers (e.g., `GraphConv` or `GINConv`) with residual connections.  
  - Hidden size: 256.  
  - Activation: ReLU, with LayerNorm after each layer.  
- **Aggregation**: sum over neighbours.  
- **Output**: Node embeddings (size 256) and global graph embedding (max‑pooling over nodes, optionally concatenated with mean‑pooling).

### Pretraining the GNN
**Before RL training**, we will leverage our 1049 datasets to pretrain the GNN in a supervised manner:
- **Task 1**: Predict the scalar penalty (or CG offset) from the current layout. Train with MSE loss.
- **Task 2**: (If actions are available in the dataset) predict the optimal shift for a given component via regression or imitation learning.

This pretraining yields a meaningful latent representation, speeds up RL convergence, and reduces the number of episodes needed.

### Policy Head
- **Component selector**: linear layer on node embeddings → logits over N components (categorical distribution). Mask inactive components (those not violating constraints).
- **Shift predictor**: takes the embedding of the *selected* component (concatenated with the global graph embedding) → outputs (μₓ, logσₓ, μᵧ, logσᵧ) via an MLP with 2 hidden layers (256 units, ReLU). The shift is sampled from a Normal distribution and squashed with `tanh` then scaled by δ_max.

### Value Head (Critic)
- Takes the global graph embedding → scalar state value V(s) via an MLP with 2 hidden layers (128 units, ReLU).  
- **Pretraining**: We can initialise the critic by predicting the discounted return (or the final penalty) from the dataset, giving it a sensible starting point.

---

## 4. Training with PPO

We use standard **Proximal Policy Optimization** with a hybrid action space.

### PPO Objective
- **Policy loss**:  
  - For component selection: categorical log‑probability ratio, clipped (ε = 0.2).  
  - For shift: Gaussian log‑probability ratio, clipped.  
  - The combined probability is the product of the two (we sum log probabilities).  
- **Value loss**: MSE between predicted V(s) and discounted returns.  
- **Entropy bonus** (coefficient = 0.01–0.02) applied to both action parts to encourage exploration.

### Advantage Estimation
- Generalized Advantage Estimation (GAE) with γ = 0.99, λ = 0.95.

### Training Loop
1. Collect **K steps** (e.g., 2048 total steps across parallel environments) using the current policy.  
2. Compute advantages and returns.  
3. Perform **multiple epochs** (e.g., 10) of gradient updates with mini‑batch size 64.  
4. **Gradient clipping** to 0.5 to prevent explosion.  
5. **Learning rate schedule**: linear decay from 3e‑4 after a warm‑up of 1000 steps.

### Parallelisation & MPS Acceleration
- Use **16 parallel environments** running on CPU (each with its own initial layout, sampled from the dataset or randomly generated).  
- Neural network forward/backward passes are performed on the **MPS** device.  
- Data from environments is batched and moved to MPS for training.  
- PyTorch’s `DataLoader` with `pin_memory` can be used to overlap data transfer.

### Using the 1049 Datasets in Training
- **Initial states**: Sample layouts from the dataset (with optional random jittering) to provide realistic starting points, reducing the need for random rejection sampling.
- **Offline pre‑training**: As described, use the dataset to pretrain the encoder and critic.
- **Validation**: Hold out a portion (e.g., 10%) of the dataset for validation to monitor generalisation.

---

## 5. Implementation Plan (Phases)

### Phase 1 – Core Components
- **Penalty wrapper**: wrap the existing Numba‑compiled `compute_fitness_core` function to compute penalty from component positions.  
- **Data loader**: parse `newobj.json`, duplicate components, compute all static attributes (dimensions, offsets, clearances, bounds).  
- **Dataset integration**: write a loader that reads the 1049 `.npz` files, converts them into PyG `Data` objects with static and dynamic features, and extracts penalties/returns for pretraining.

### Phase 2 – Environment (`PCBEnv`)
- Implement a Gym‑like environment with `reset()` and `step(action)`.
  - `reset()`: generates an initial layout by either random placement (within bounds) or sampling from the dataset (with optional jitter). Builds the initial graph.  
  - `step()`: applies the move, computes new penalty, constructs next graph, returns reward, done flag, and info dict.  
- **State construction**: build a PyG `Data` object with `x` (node features), `edge_index`, `edge_attr`, and `global_features` (stored separately).  
- Ensure the environment is efficient (penalty calls are the bottleneck; they are fast enough).

### Phase 3 – GNN Model (`GNNPolicy`)
- Define `GNNPolicy` class inheriting `torch.nn.Module`.  
- Implement forward pass that takes a `Data` object and optionally a mask for components.  
- Returns: action distributions (categorical for component, normal for shift) and state value.

### Phase 4 – Supervised Pretraining
- **Encoder pretraining**: Train the GNN to predict the penalty (or CG offset) from the layout using the dataset. Use MSE loss, Adam, and early stopping.
- **Critic pretraining**: Train a small MLP head on the global embedding to predict the discounted return (or final penalty) from the dataset.
- **Policy warm‑start** (optional): If the dataset contains successful move sequences (e.g., from CMA‑ES), perform behaviour cloning to initialise the policy.

### Phase 5 – PPO Trainer
- Implement a `PPO` class with methods:  
  - `collect_rollouts(num_steps)`: runs parallel environments for `num_steps` steps, stores transitions in a buffer.  
  - `update()`: computes advantages, then performs mini‑batch updates for several epochs.  
- Use `torch.distributions.Categorical` and `Normal` for action sampling.

### Phase 6 – Training & Hyperparameter Tuning
- Train on a diverse set of boards (from the dataset and randomly generated) to learn generalisable policies.  
- Evaluate on the specific `newobj.json` instance.  
- Compare final penalty and CG offset with CMA‑ES results.  
- Optionally, fine‑tune the policy on the target board for a few more episodes.

### Phase 7 – Visualisation & Logging
- Log reward, penalty, CG offset, KL divergence, and value loss during training.  
- Periodically save the best model.  
- At evaluation, generate layout images (front/back views and distance maps) using the existing plotting code.

---

## 6. Timeline (Estimated)

| Phase | Duration |
|-------|----------|
| 1. Penalty wrapper & data loader (including dataset integration) | 1.5 days |
| 2. Environment implementation | 2 days |
| 3. GNN model | 2 days |
| 4. Supervised pretraining | 1 day |
| 5. PPO trainer (parallel) | 3 days |
| 6. Training & hyperparameter tuning | 4 days |
| 7. Evaluation & visualisation | 2 days |
| **Total** | **~15.5 days** |

Add a buffer of 3‑5 days for unforeseen issues (MPS/PyTorch bugs, environment quirks, tuning).

---

## 7. Potential Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| **Continuous action space** – may be sensitive to scaling. | Normalise coordinates to [-1,1] within the network; use δ_max = 20 mm (roughly 10% of board width). |
| **Sparse rewards** – penalty may not change often. | Use dense shaping (reward = penalty_old - penalty_new) plus bonuses and penalties as described. |
| **Long training times** – need many steps. | Parallel environments (16), MPS acceleration; pretraining reduces required RL episodes significantly. |
| **Graph construction overhead** – rebuilding every step. | For N ≤ 20, it’s negligible. Use efficient PyG batching and cache static features. |
| **Instability in PPO** – especially with hybrid actions. | Use adaptive KL penalty or early stopping if KL exceeds threshold; clip ratio 0.2. |
| **Overfitting to initial layouts** | Randomise initial layouts by jittering dataset samples or generating new random layouts each reset. |
| **Penalty function is CPU‑bound** | Already JIT‑compiled; can be called in parallel using `multiprocessing` if needed. |
| **Dataset loading and preprocessing** | Precompute all graph features offline and store in a compressed format; use a memory‑mapped array for fast access. |

---

## 8. Evaluation & Success Criteria

- **Primary metric**: Final penalty (should be < 1e-3 for a valid layout) and CG offset (< 1.0 mm) on the target board.
- **Secondary metrics**: 
  - Success rate (percentage of runs where a valid layout is found within the episode).
  - Average number of steps to reach a valid layout.
  - Comparison against CMA‑ES on the same initial layouts (our agent should achieve similar or better quality faster).
- **Generalisation**: Test on a held‑out set of boards from our dataset to measure zero‑shot performance.

---

## 9. Offline RL Alternative (Optional)

Given the sizeable dataset, we could also train an **offline RL** agent (e.g., CQL or IQL) to learn a policy directly from the data without online interaction. This would avoid the risk of damaging the board during exploration and could serve as a strong baseline. We can combine offline pretraining with a short online fine‑tuning phase.

---

## 10. Expected Outcomes

- A trained PPO agent that can refine any feasible layout for the given board and component set.  
- Final penalty close to zero (all constraints satisfied) after a small number of refinement steps.  
- Comparable or better solution quality (CG offset, no overlaps) compared to CMA‑ES, but with faster inference and potential generalisation to new components/boards.  
- Visualisation of the improvement process (penalty vs. steps) and final layout images.

---

## 11. References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.  
- Kipf & Welling, *Semi‑Supervised Classification with Graph Convolutional Networks*, 2017.  
- Original CMA‑ES code (provided).  
- PyTorch Geometric documentation.

---

## 12. Next Steps

1. Set up the environment: Python 3.10+, PyTorch with MPS support, PyTorch Geometric, Numba.  
2. Inspect the `.npz` files to understand their structure; build a data loader and conversion script.  
3. Implement the penalty wrapper and test it with sample positions.  
4. Build the environment and ensure it works with a dummy agent.  
5. Prototype the GNN model and verify forward pass.  
6. Run supervised pretraining on the dataset and evaluate representation quality.  
7. Begin training on simplified boards, then scale up to the full dataset.  
8. Evaluate on the target board and produce final plots.

---

*This document now incorporates the use of the 1049 `.npz` datasets, detailed pretraining strategies, enhanced reward shaping, action masking, and a revised timeline. The plan is robust and ready for implementation on Apple M3 Pro hardware.*