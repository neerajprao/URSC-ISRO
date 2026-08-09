# PCB Component Placement — PPO Reinforcement Learning Rewrite

## 1. Background / Why This Project Exists

The original solution (`new.py`) solved automatic placement of PCB-mounted
components (front + back side) using **CMA-ES**, an evolutionary optimizer.
It:
- Loads component geometry, mass, insert-hole layout, and board dimensions
  from `newobj.json`.
- Scores any candidate layout with a single scalar penalty combining:
  border violations, front/back overlap (with directional clearance,
  including custom `CF`/`CFLen` face overrides), insert-to-insert minimum
  distance (rule-based on insert diameter: 4mm↔4mm = 24mm min, 6mm↔6mm =
  36mm min, default = 30mm), and Center of Gravity (CG) offset from board
  center.
- Runs CMA-ES from multiple random restarts to produce up to 5 distinct
  valid layouts, then renders each as a matplotlib figure (front/back
  projection, proximity/pin-distance map, and a data table) — the two
  reference images (`layout_5_main.png`, `layout_5_distance_map.png`) are
  outputs of this old pipeline.

**Goal of this rewrite:** replace CMA-ES with a **PPO (Proximal Policy
Optimization)** reinforcement-learning agent as the placement engine,
built in a Jupyter notebook (`.ipynb`), cell by cell, so it can be
reviewed and extended interactively. Hybrid techniques were allowed if
useful, but per the current decision, PPO is being used standalone (no
CMA-ES hybrid).

## 2. Constraint Priorities (as specified by user)

| Priority | Constraint | Notes |
|---|---|---|
| **1 (hard)** | Board border containment | Component footprint + clearance + insert pins must stay inside board minus `Border (mm)` margin |
| **1 (hard)** | Overlap + clearance | Same-side components must not overlap once directional clearance (`Clearance (mm)`, or `CF`/`CFLen` face overrides) is added |
| **1 (hard)** | Insert hole (pin) minimum distance | Distance between any two mounting pins across different components must respect a rule based on insert diameter (4mm/6mm/mixed) |
| **2 (soft/secondary)** | Center of Gravity (CG) offset | Combined mass-weighted CG should stay within `CG_TOLERANCE` (1.0 mm) of board center; back-side components are X-mirrored before computing CG |

This priority ordering is directly encoded in the RL reward function's
weighting (see §5).

## 3. Design Decisions Made (via user Q&A)

| Decision point | Options considered | **Chosen** |
|---|---|---|
| Episode / action structure | (a) one-shot full-layout output, (b) sequential one-component-at-a-time placement, (c) iterative refinement (nudge all components over many steps from a random start) | **(c) Iterative refinement** |
| Hybridization | (a) pure PPO, (b) PPO + CMA-ES polish per episode, (c) PPO global search + CMA-ES fallback stage | **(a) Pure PPO only** |
| Reward shaping | (a) sparse (one reward at episode end), (b) dense (per-step reward as penalty decreases) | **(b) Dense**, since iterative refinement was chosen |

**Resulting environment concept:** each episode starts every component at
a random valid-bounds position. At every step, the policy outputs a
small `(dx, dy)` nudge for **all components simultaneously**. Reward is
the reduction in a weighted, log-compressed penalty score from the
previous step, plus a terminal bonus if the layout becomes fully valid
(border/overlap/insert all zero). This is functionally like teaching a
learned "gradient descent" policy over the same penalty landscape
CMA-ES searched, but trained once and reusable for inference.

## 4. Tooling / Stack

- **Environment**: `gymnasium` (not legacy `gym`) custom `Env` subclass
- **RL algorithm**: `stable-baselines3` PPO (`MlpPolicy`)
- **Parallelism**: `SubprocVecEnv` (8 parallel envs) + `VecNormalize`
  (observation & reward normalization)
- **Fast constraint evaluation**: same `numba`-JIT-compiled geometry math
  as the original CMA-ES script, refactored to return a **penalty
  breakdown** (border, overlap, insert, cg separately) instead of one
  lumped score, so the RL reward can weight priority-1 vs priority-2
  terms differently
- **Hardware target**: MacBook, 18GB RAM, Apple Silicon — PPO's neural
  net runs on **MPS** (`torch.backends.mps`); the environment/reward
  math stays on CPU via numba (cheap, doesn't need GPU)
- **Logging**: TensorBoard (via SB3's built-in `tensorboard_log`)

## 5. Reward Function Design

```
composite_score = 5.0·log1p(border) + 5.0·log1p(overlap)
                 + 5.0·log1p(insert) + 0.5·log1p(cg)

reward_t = (composite_score_{t-1} - composite_score_t) - STEP_COST
         + SUCCESS_BONUS (only on the step the layout becomes fully valid)
```

- `log1p` compression is necessary because raw overlap/insert penalties
  can reach ~1e9–1e10 (squared-area / squared-distance terms), which
  would destabilize PPO's value function if used raw.
- Border/overlap/insert (priority 1) are weighted 10× higher than CG
  (priority 2), matching the stated priority ordering.
- Per-component coordinate bounds (`lower_bounds`/`upper_bounds`) are
  precomputed so that **border violations are structurally impossible**
  — the agent's action is always clipped inside a box that guarantees
  full footprint + clearance + insert pins stay on the board. This means
  the agent only ever has to *actually learn* to resolve overlap, insert
  spacing, and CG — border is a safety net, not something it must learn.
- Episode ends early (terminated=True) the moment a layout is fully
  valid (border/overlap/insert == 0), with a `SUCCESS_BONUS`. Otherwise
  it truncates at `MAX_STEPS` (250).

## 6. What's Been Done So Far (Notebook Cells 1–6)

| Cell | Purpose | Status |
|---|---|---|
| **1** | Imports (`gymnasium`, `stable-baselines3`, `numba`, `torch`, matplotlib), MPS device detection, output dir, seeding | ✅ Done, confirmed `mps` device detected |
| **2** | Parse `newobj.json` → per-component arrays (name, shape, mass, dims, insert offsets, `CF`/`CFLen` clearance overrides, back/front side flag) — identical logic to `new.py` | ✅ Done, confirmed 18 components parsed correctly (matches JSON: PDS, CCRR0/1, LD0–5, RS, BM1, PB0–3, RPU, MES, MT20) |
| **3** | Per-component coordinate bounds (`lower_bounds`/`upper_bounds`) guaranteeing board containment; numba-JIT `compute_penalty_breakdown()` returning (border, overlap, insert, cg) separately; `evaluate_layout()` convenience wrapper | ✅ Done, bounds validated (no inversions), numba engine compiled and smoke-tested |
| **4** | `PCBPlacementEnv(gym.Env)` — iterative refinement env: reset (random valid start), step (nudge + clip + reward), obs = normalized positions + log-penalties + step fraction, action = 36-dim `[-1,1]` nudges. **Includes `get_state()` method** (added after fixing a `SubprocVecEnv` bug — see §8) | ✅ Done, passed `check_env`, random-rollout sanity test passed |
| **5** | `SubprocVecEnv` (8 parallel envs) + `VecNormalize` + PPO model (`MlpPolicy`, 256×256 net, tuned hyperparameters, `device="mps"`) | ✅ Done, model created successfully, policy architecture confirmed (41→256→256→36) |
| **6** | `BestLayoutCallback` (tracks best fully-valid layout by CG, and best invalid layout by composite score, across all parallel envs via IPC-safe `env_method("get_state")`) + training loop (`model.learn(...)`, 500,000 timesteps starting budget) + model/`VecNormalize` saving | ✅ **Code finalized and fixed**, but **training run has not been executed/completed yet** — this is the current step |

## 7. What's Left To Do

1. **Run Cell 6 training to completion** (re-run Cell 4 → Cell 5 fresh
   first, since the `SubprocVecEnv` subprocesses must be respawned with
   the corrected `PCBPlacementEnv` class that includes `get_state()`).
   Need to observe: training time, `n_valid_found`, `best_valid_cg`,
   `best_invalid_score` trend, and TensorBoard reward/episode-length
   curves.
2. **Evaluate whether 500,000 timesteps is enough.** If no fully-valid
   layout is found, likely next steps (to be decided together once we
   see numbers):
   - Extend training timesteps
   - Adjust `MAX_STEP_MM` (nudge size) — too large causes oscillation,
     too small causes slow convergence
   - Adjust reward weights or `log1p` compression
   - Add curriculum: e.g. start episodes from a partially-valid
     layout instead of fully random, or shrink `MAX_STEPS` gradually
   - Consider reducing PENALTY_WEIGHT-equivalent scaling if overlap
     term dominates too much early on
3. **Cell 7 (not yet written): Extract & render best layout(s).**
   - Pull `best_layout_cb.best_valid_layout` (or best invalid if no
     valid found) raw `(cx, cy)` coordinates.
   - Re-run through `evaluate_layout()` to confirm/report final penalty
     breakdown.
   - Render in the **same visual style as the original CMA-ES output**:
     front/back projection view, proximity/pin-distance map with
     labeled close-pin pairs (<40mm), data table (component name, side,
     mass, dimensions, CoG X/Y) — matching `layout_5_main.png` and
     `layout_5_distance_map.png` structure.
4. **Cell 8+ (not yet written): Generate multiple distinct layouts.**
   The original CMA-ES script produced up to 5 distinct valid layouts
   from different restarts. Equivalent for PPO: run several independent
   `model.predict()` rollouts from different random resets (post-training,
   deterministic or stochastic policy) and collect distinct valid
   solutions — this still needs to be designed and built.
5. **Inference-only path.** Once trained, need a clean "load trained
   model + VecNormalize stats → run N rollouts → output best layout(s)"
   cell, separate from the training cell, so the notebook can be re-run
   for new layouts without retraining.
6. **Comparison / validation against CMA-ES baseline.** Not yet done —
   worth comparing PPO's best found layout's penalty breakdown against
   the original `new.py` CMA-ES results on the same `newobj.json`, to
   sanity-check the RL approach is actually competitive.
7. **Hyperparameter tuning pass** (learning rate, `n_steps`, `ent_coef`,
   `MAX_STEP_MM`, `MAX_STEPS`, `SCORE_WEIGHTS`) — deferred until we see
   first training run results, since tuning blind is wasted effort.
8. **Output/export**: saving final chosen layout(s) back into a
   JSON/plot format consistent with existing project outputs (not yet
   built for the RL path — Cell 7/8 will handle this).

## 8. Issues Encountered & Fixes

- **`SubprocVecEnv` has no `.envs` attribute.** Direct attribute access
  (`env.cx`) only works for in-process vec envs (`DummyVecEnv`); with
  `SubprocVecEnv`, each env lives in a separate OS process, so nothing
  is directly reachable by attribute. **Fix:** added a `get_state()`
  method to `PCBPlacementEnv` and switched the callback to use SB3's
  IPC-safe `VecEnv.env_method("get_state", indices=[idx])`. Required
  re-running Cell 4 (env class definition) and Cell 5 (respawn
  subprocesses with the corrected class) before Cell 6 would work.
- **Legacy `gym` deprecation warning** seen once during setup — harmless
  noise from an SB3 internal compatibility check; project uses
  `gymnasium` throughout, so no action was needed.

## 9. Key Files

| File | Role |
|---|---|
| `newobj.json` | Board + component specification (input, unchanged from original project) |
| `new.py` | Original CMA-ES reference implementation (not modified, kept for comparison) |
| `layout_5_main.png`, `layout_5_distance_map.png` | Example outputs from the old CMA-ES pipeline (front/back view + proximity map) — the RL pipeline's Cell 7 should aim to reproduce this same visual/report style |
| `<notebook>.ipynb` | New PPO-based pipeline, built cell by cell (Cells 1–6 done per §6) |
| `optimized_layouts_rl/` | Output dir created by the notebook: will hold `tb_logs/` (TensorBoard), saved model (`ppo_pcb_placement.zip`), `vecnormalize.pkl`, and eventually rendered layout images |

## 10. How to Resume

1. Open the notebook, run Cells 1–3 (setup, config parsing, numba engine)
   fresh if starting a new kernel session.
2. Run Cell 4 (env class, with `get_state()` included).
3. Run Cell 5 (spawns `SubprocVecEnv`, builds PPO model) — must be run
   *after* Cell 4 in the same kernel session so subprocesses inherit the
   current class definition.
4. Run Cell 6 to start/continue training. Watch console + TensorBoard
   (`tensorboard --logdir optimized_layouts_rl/tb_logs`) for progress.
5. Report back training time and final `n_valid_found` /
   `best_valid_cg` / `best_invalid_score` numbers so we can decide next
   steps (§7) before writing Cell 7.
