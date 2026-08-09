# PCB Component Placement — PPO Reinforcement Learning Rewrite

_Last updated: after Cell 7 produced the first independently-verified fully valid layout._

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
built in a Jupyter notebook (`.ipynb`), cell by cell. Hybrid techniques
were allowed if useful; per the current decision, PPO is used standalone
(no CMA-ES hybrid).

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

**Environment concept:** each episode starts every component at a random
valid-bounds position. At every step, the policy outputs a small `(dx,
dy)` nudge for **all components simultaneously**, with nudge magnitude
**annealed within the episode** (large early for fast untangling, small
late for fine insert-distance/CG precision — see §7 for why this was
added). Reward is the reduction in a weighted, log-compressed penalty
score from the previous step, plus a terminal bonus if the layout becomes
fully valid (border/overlap/insert all zero).

## 4. Tooling / Stack

- **Environment**: `gymnasium` custom `Env` subclass
- **RL algorithm**: `stable-baselines3` PPO (`MlpPolicy`, 256×256 net)
- **Parallelism**: `SubprocVecEnv` (8 parallel envs) + `VecNormalize`
  (observation & reward normalization)
- **Fast constraint evaluation**: `numba`-JIT-compiled geometry math
  (same rules as the original CMA-ES script), returning a **penalty
  breakdown** (border, overlap, insert, cg separately) so the reward can
  weight priority-1 vs priority-2 terms differently
- **Hardware**: MacBook, 18GB RAM, Apple Silicon — PPO's neural net runs
  on **MPS**; environment/reward math stays on CPU via numba
- **Logging**: TensorBoard (`tb_logs/`)

## 5. Reward Function Design (final)

```
composite_score = 3.0·log1p(border) + 3.0·log1p(overlap)
                 + 10.0·log1p(insert) + 0.5·log1p(cg)

reward_t = (composite_score_{t-1} - composite_score_t) - STEP_COST
         + SUCCESS_BONUS (only on the step the layout becomes fully valid)
```

- `log1p` compression: raw overlap/insert penalties can reach ~1e9–1e10,
  which would destabilize PPO's value function if used raw.
- **Weights were retuned mid-project**: originally border/overlap/insert
  were all weighted 5.0. Once training showed border and overlap were
  reliably solved (0.0) while insert lagged, insert's weight was raised
  to 10.0 (highest) and border/overlap lowered to 3.0 each, since
  gradient signal was better spent on the still-unsolved constraint. CG
  stays lowest (0.5) as priority 2.
- Border violations are **structurally near-impossible** by construction:
  per-component coordinate bounds (`lower_bounds`/`upper_bounds`,
  computed once from board size + component footprint + clearance +
  pin offsets) guarantee any clipped action keeps the full footprint on
  the board.
- **Step size is annealed within each episode**: `MAX_STEP_MM = 12.0` at
  the start (fast coarse untangling of overlap) linearly decaying to
  `MIN_STEP_MM = 0.25` by the end of the episode (fine control needed to
  satisfy exact insert-distance thresholds, which are precision-sensitive
  in a way overlap-resolution isn't).
- Episode ends early (terminated=True, `SUCCESS_BONUS` awarded) the
  moment a layout is fully valid; otherwise truncates at `MAX_STEPS=250`.

## 6. Current Notebook State (Cells 1–7)

| Cell | Purpose | Status |
|---|---|---|
| **1** | Imports, MPS device detection, output dir, seeding | ✅ Done |
| **2** | Parse `newobj.json` → per-component arrays | ✅ Done — 18 components parsed correctly |
| **3** | Per-component coordinate bounds; numba `compute_penalty_breakdown()`; `evaluate_layout()` wrapper | ✅ Done, bounds validated, numba engine tested |
| **4** | `PCBPlacementEnv(gym.Env)` — **final version**: annealed step size, `cx`/`cy` embedded directly in `step()`'s returned `info` dict | ✅ Done (see §7 for why this specific detail matters) |
| **5** | Builds `SubprocVecEnv` (8 envs) + `VecNormalize`, auto-loading existing normalization stats from disk if present | ✅ Done — restructured to NOT create the model (model moved to Cell 6) so re-running this cell can't accidentally discard a trained model |
| **6** | Model: loads existing checkpoint from `optimized_layouts_rl/ppo_pcb_placement.zip` if present, else creates fresh PPO. Defines the corrected `BestLayoutCallback` (reads `cx`/`cy` straight from `info`, not via any separate accessor). Trains for `TRAIN_TIMESTEPS`, saves model + `VecNormalize` stats after. **Re-runnable** — safe to bump `TRAIN_TIMESTEPS` and re-run to keep extending training from the latest checkpoint | ✅ Done, consolidated (previously split across 3 separate cells during debugging — those are now deleted, see §7) |
| **7** | Extracts `best_layout_cb.best_valid_layout`, **independently re-verifies** it via `evaluate_layout()` before trusting it, computes CG, renders front/back projection view + proximity/pin-distance map (style matching original CMA-ES output), exports CSV data table | ✅ Done — **first successful independently-verified valid layout produced** (see §8 for results) |

## 7. Issues Encountered & Fixes (chronological)

1. **`SubprocVecEnv` has no `.envs` attribute.**
   Direct attribute access (`env.cx`) only works for in-process vec envs
   (`DummyVecEnv`); `SubprocVecEnv` runs each env in a separate OS
   process. **Fix (superseded by #3 below):** initially added a
   `get_state()` method + `VecEnv.env_method("get_state", ...)`.

2. **Insert-distance penalty not converging (511 → worse after naive
   continued training).**
   Root cause: a single fixed nudge size (`MAX_STEP_MM=12.0`) is fine for
   coarse overlap-untangling but too coarse for sub-mm insert-distance
   precision. **Fix:** annealed step size within each episode
   (`MAX_STEP_MM` → `MIN_STEP_MM` linearly over the episode), plus raised
   `insert`'s reward weight relative to `border`/`overlap` (already
   solved) and lowered `ent_coef` during the fine-tuning continuation run.

3. **Critical bug: `get_state()` snapshots were silently wrong
   (race condition).**
   `SubprocVecEnv` **auto-resets** a sub-environment internally the
   instant an episode terminates/truncates, *before* control returns to
   the training loop. The callback's post-hoc `env_method("get_state")`
   call therefore ran *after* the sub-env had already reset — meaning
   every "best valid layout" snapshot captured during training (across
   multiple training runs) was actually capturing the **next random
   episode's start state**, not the valid layout that had just been
   found. This was only caught when Cell 7's independent
   re-verification step (`evaluate_layout()` on the "best" snapshot)
   failed an assertion, revealing massive overlap/insert penalties that
   contradicted the training-time numbers.
   **Fix:** stopped using any post-hoc state accessor entirely. `cx`/`cy`
   are now copied directly into the `info` dict *inside* `step()`,
   before any auto-reset can occur — `info` travels back through the
   normal synchronous step-return channel, so there is no race window.
   This required editing Cell 4 (`step()`'s final `info = {...}` line)
   and Cell 6's callback (`_snapshot` logic simplified to just read
   `info["cx"]`/`info["cy"]`).
4. **Verification run coming back with 0 valid layouts.**
   Not a bug — valid layouts were appearing at a rate of roughly 1 per
   ~107k timesteps in that training phase, so a 50k-step verification
   run was simply too short a sample. Increasing the timestep budget
   resolved it.
5. **Notebook cell sprawl during debugging (6, 6b, 6c).**
   The training loop got split into three separate patch cells while
   iterating on fixes #2–#4. These were **consolidated back into a
   single Cell 6** once the design stabilized — old Cell 6 / 6b / 6c
   should be deleted from the notebook; only the current Cell 6
   (load-or-create model + fixed callback + train + save, all in one
   cell) should remain.
6. **Legacy `gym` deprecation warning** (harmless) — seen once during
   initial setup; project uses `gymnasium` throughout, no action needed.

## 8. Current Results

First independently-verified fully valid layout (produced by Cell 7,
re-checked via a fresh call to `evaluate_layout()` separate from the
training-time numbers):

| Metric | Value |
|---|---|
| Border penalty | **0.0000** ✅ |
| Overlap penalty | **0.0000** ✅ |
| Insert-distance penalty | **0.0000** ✅ |
| CG penalty | 16.4566 |
| Combined Assembly CG offset | (-3.94, -3.17) mm |

All three priority-1 hard constraints are satisfied exactly. CG
(priority 2) is off-center by ~5mm combined offset — a reasonable first
valid result, with room to improve via further training if desired.

Outputs saved (on the user's machine, under `optimized_layouts_rl/`):
- `ppo_layout_front_back.png` — front/back projection view (style
  matching `layout_5_main.png`)
- `ppo_layout_proximity_map.png` — pin-distance proximity map (style
  matching `layout_5_distance_map.png`)
- `ppo_layout_table.csv` — component name, side, mass, dimensions, CoG
  X/Y for all 18 components
- `ppo_pcb_placement.zip` — trained PPO model checkpoint
- `vecnormalize.pkl` — matching observation/reward normalization stats

## 9. What's Left To Do

1. **Visual sanity-check** of the two saved PNGs by the user — confirm
   clearance boxes don't look suspiciously tight despite reading 0
   violation, and that proximity-map `<40mm` pin-distance labels look
   consistent with the `req_d_matrix` rule.
2. **Cell 8 (not yet written): generate multiple distinct valid
   layouts.** The original CMA-ES script produced up to 5 distinct valid
   layouts from different restarts. Equivalent for PPO: run several
   independent rollouts (`model.predict()`, deterministic or stochastic)
   from different random resets post-training, and collect distinct
   valid solutions, filtering out near-duplicates.
3. **Optional: further training to lower CG penalty** (currently 16.46
   / ~5mm offset). Just re-running Cell 6 with a larger
   `TRAIN_TIMESTEPS` continues from the saved checkpoint — no code
   changes needed.
4. **Baseline comparison against CMA-ES.** Not yet done — worth
   comparing the PPO layout's penalty breakdown / CG offset against the
   original `new.py` CMA-ES output on the same `newobj.json`.
5. **Inference-only path**, decoupled from training: a cell that loads
   the saved model + `VecNormalize` stats and runs N rollouts to produce
   layouts, without needing to touch `model.learn()` at all — useful for
   regenerating layouts later without re-triggering training.
6. **Hyperparameter tuning pass** (learning rate, `n_steps`, `ent_coef`,
   `MAX_STEP_MM`/`MIN_STEP_MM`, `SCORE_WEIGHTS`) if further CG
   improvement or faster convergence is wanted.

## 10. Key Files

| File | Role |
|---|---|
| `newobj.json` | Board + component specification (input, unchanged from original project) |
| `new.py` | Original CMA-ES reference implementation (not modified, kept for comparison) |
| `layout_5_main.png`, `layout_5_distance_map.png` | Example outputs from the old CMA-ES pipeline — visual style target for Cell 7's renders |
| `<notebook>.ipynb` | New PPO-based pipeline, Cells 1–7 finalized per §6 |
| `optimized_layouts_rl/` | `tb_logs/` (TensorBoard), `ppo_pcb_placement.zip` (model), `vecnormalize.pkl`, `ppo_layout_front_back.png`, `ppo_layout_proximity_map.png`, `ppo_layout_table.csv` |

## 11. How to Resume

1. Run Cells 1–3 fresh if starting a new kernel session (setup, config
   parsing, numba engine).
2. Run Cell 4 (env class, final version with annealed step + `info`
   fix).
3. Run Cell 5 (spawns `SubprocVecEnv`, auto-loads existing
   `VecNormalize` stats if present).
4. Run Cell 6 — auto-loads the existing trained checkpoint if
   `ppo_pcb_placement.zip` exists on disk; trains `TRAIN_TIMESTEPS` more
   steps (bump this value to train longer); saves model + normalization
   stats. Safe to re-run repeatedly to keep extending training.
5. Run Cell 7 to extract, independently re-verify, and render the best
   valid layout found so far.
6. Next: build Cell 8 for multi-layout generation (see §9).
