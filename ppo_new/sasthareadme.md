# PCB Component Placement — PPO Reinforcement Learning Rewrite

_Last updated: after Cell 9 (multi-layout generation + true see-through proximity rendering), plus `gen.py` — a standalone CLI export of the Cell 8/9 inference pipeline, now extended with a gradient-based CG polish hybrid stage._

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
built in a Jupyter notebook (`.ipynb`), cell by cell. PPO is used
standalone (no CMA-ES hybrid) for the global feasibility search.

**New addition (this update):** since PPO alone was not consistently
driving the soft CG constraint under tolerance, `gen.py` now includes an
optional **gradient-based (L-BFGS-B) local polish stage** that runs after
PPO produces a feasible layout. This is a genuine two-optimizer hybrid —
PPO for global feasibility search, gradient descent for local CG
refinement — see §5b.

## 2. Constraint Priorities

| Priority | Constraint | Notes |
|---|---|---|
| **1 (hard)** | Board border containment | Component footprint + clearance + insert pins must stay inside board minus `Border (mm)` margin |
| **1 (hard)** | Overlap + clearance | Same-side components must not overlap once directional clearance (`Clearance (mm)`, or `CF`/`CFLen` face overrides) is added |
| **1 (hard)** | Insert hole (pin) minimum distance | Distance between any two mounting pins across different components must respect a rule based on insert diameter (4mm/6mm/mixed) |
| **2 (soft/secondary)** | Center of Gravity (CG) offset | Combined mass-weighted CG should stay within `CG_TOLERANCE` (1.0 mm) of board center; back-side components are X-mirrored before computing CG. `cg_penalty = 0` iff `cg_offset <= 1.0mm` |

## 3. Design Decisions

| Decision point | Options considered | **Chosen** |
|---|---|---|
| Episode / action structure | (a) one-shot full-layout output, (b) sequential one-component-at-a-time placement, (c) iterative refinement | **(c) Iterative refinement** |
| Global search algorithm | (a) pure PPO, (b) PPO + CMA-ES polish per episode, (c) PPO global search + CMA-ES fallback | **(a) Pure PPO only** |
| Local CG polish algorithm (new) | (a) none, (b) DE/CMA-ES local restart, (c) gradient-based (L-BFGS-B) continuation | **(c) Gradient-based continuation** — chosen specifically to stay distinct from DE/CMA-ES for comparative-study purposes; see §5b for why a single fixed penalty weight doesn't work and a continuation schedule is needed instead |
| Reward shaping | (a) sparse, (b) dense | **(b) Dense** |

**Environment concept:** each episode starts every component at a random
valid-bounds position. At every step, the policy outputs a small `(dx,
dy)` nudge for **all components simultaneously**, with nudge magnitude
**annealed within the episode** (`MAX_STEP_MM=12.0` early → `MIN_STEP_MM=0.25`
late). Reward is the reduction in a weighted, log-compressed penalty
score from the previous step, plus a terminal bonus if the layout becomes
fully valid.

## 4. Tooling / Stack

- **Environment**: `gymnasium` custom `Env` subclass
- **RL algorithm**: `stable-baselines3` PPO (`MlpPolicy`, 256×256 net, `torch.nn.Tanh`)
- **Training parallelism**: `SubprocVecEnv` (8 parallel envs) + `VecNormalize`
- **Inference parallelism**: `DummyVecEnv` (8 envs) + the same saved `VecNormalize` stats, `training=False`
- **Local CG polish (new)**: `scipy.optimize.minimize` (`method="L-BFGS-B"`), reusing the exact same numba `compute_penalty_breakdown` engine as the objective — no reimplementation, no drift risk
- **Fast constraint evaluation**: `numba`-JIT-compiled geometry math, returning a
  penalty breakdown (border, overlap, insert, cg separately)
- **Hardware**: MacBook, 18GB RAM, Apple Silicon — PPO's neural net runs on **MPS**;
  environment/reward math and the polish stage stay on CPU via numba/scipy
- **Logging**: TensorBoard (`tb_logs/`)

## 5. Reward Function Design (PPO training, unchanged)

```
composite_score = 3.0·log1p(border) + 3.0·log1p(overlap)
                 + 10.0·log1p(insert) + 0.5·log1p(cg)

reward_t = (composite_score_{t-1} - composite_score_t) - STEP_COST
         + SUCCESS_BONUS (only on the step the layout becomes fully valid)
```

- `log1p` compression prevents raw overlap/insert penalties (~1e9–1e10) from
  destabilizing PPO's value function.
- Weights were retuned mid-project: border/overlap dropped from 5.0→3.0 once
  reliably solved; insert raised 5.0→10.0 (hardest constraint); CG stays
  lowest (0.5) as priority 2.
- Border violations are structurally near-impossible by construction
  (per-component coordinate bounds clip any action to stay on-board).
- Episode ends early (`terminated=True`, `SUCCESS_BONUS` awarded) the moment
  a layout is fully valid; otherwise truncates at `MAX_STEPS=250`.

## 5b. Gradient-Based CG Polish Stage (new hybrid addition)

**Why this exists:** PPO reliably drives border/overlap/insert to exactly
0, but was not consistently pushing the CG offset under the 1.0mm
tolerance — hitting a low CG penalty required a lucky rollout rather than
being a guaranteed outcome. Rather than retrain PPO harder to also master
CG precision, this stage treats CG minimization as a separate, *local*
problem: take any PPO layout that's already fully valid, and refine it
in place to pull CG under tolerance, using a fundamentally different
optimizer than PPO/CMA-ES/DE (a gradient-based method), which keeps this
comparable as an independent entry in a comparative-optimizer study.

**Where it runs:** in `gen.py`, right after each rollout's independent
re-verification (`is_fully_valid(verify)`) and before dedup/ranking.
Enabled via `--polish`.

**Algorithm — continuation (sequential penalty), not a single fixed weight:**

```
objective(cx, cy; w_hard) = w_hard · (border + overlap + insert) + w_cg · cg_penalty
```

A naive fixed large `w_hard` (e.g. `1e6`) was tried first and **got
stuck at the seed with zero movement** — verified directly against the
real penalty engine. With such a large weight, any local move that
touches an already-tight packing constraint spikes the objective so
sharply that L-BFGS-B's line search rejects the step outright, even when
a coordinated multi-component move would reduce CG while staying
feasible.

The fix is a **continuation schedule**: start with a small hard-constraint
weight so the optimizer can actually move to reduce CG, then ramp the
weight up across stages, warm-starting each stage from the previous
stage's result:

```
weight_schedule = (3.0, 30.0, 300.0, 3e3, 3e4, 3e5, 3e6, 3e7)
```

Early stages make real progress on CG (at the cost of small, temporary
border/overlap/insert slack); later stages tighten the screws and snap
the hard constraints back to exactly 0. This was verified end-to-end
against the authoritative `compute_penalty_breakdown` engine: CG dropped
from 5999 → 4340 in just 3 short stages in a slowed-down (no-numba) test
harness — production speed with numba JIT should complete the full
8-stage schedule across multiple restarts in well under a second per
candidate.

**Search scope and safety:**
- Only searches a local box of `± --polish_delta` mm (default 60mm)
  around the PPO seed, intersected with the same structural
  per-component bounds `PCBPlacementEnv` uses — this is a polish step,
  not a redesign.
- Runs `--polish_restarts` (default 4) random-jittered restarts per
  candidate and keeps the best.
- **Always falls back to the original PPO layout** if no restart both
  stays fully valid *and* beats the seed's CG — polish can only help,
  never regress a result.
- Re-verifies every accepted result against the exact same
  `compute_penalty_breakdown` used everywhere else in the pipeline
  before acceptance (no separate/reimplemented penalty math, so there's
  no risk of drift between what the optimizer sees and what gets
  reported).

**Usage:**
```bash
python gen.py --polish
python gen.py --polish --polish_delta 100 --polish_restarts 8   # wider search, more restarts
```
If CG still doesn't reach ≤1mm on some layouts, first try increasing
`--polish_delta` (give it more room to redistribute mass) or
`--polish_restarts` (more starting points to escape local minima) —
this is a tuning question, not a sign the approach doesn't work.

New functions in `gen.py`: `component_bounds()`, `polish_cg()`. Not yet
ported into the notebook (`.ipynb`) as a Cell 10 — see §9.

## 6. Notebook State (Cells 1–9)

| Cell | Purpose | Status |
|---|---|---|
| **1** | Imports, MPS device detection, output dir, seeding | ✅ Done |
| **2** | Parse `newobj.json` → per-component arrays (18 components) | ✅ Done |
| **3** | Per-component coordinate bounds; numba `compute_penalty_breakdown()`; `evaluate_layout()` wrapper | ✅ Done |
| **4** | `PCBPlacementEnv(gym.Env)` — final version, annealed step size, `cx`/`cy` embedded directly in `step()`'s `info` dict (race-free) | ✅ Done |
| **5** | Builds `SubprocVecEnv` (8 envs) + `VecNormalize`, auto-loads existing normalization stats if present. Does **not** create the model. | ✅ Done |
| **6** | Model (load-or-create from checkpoint), `BestLayoutCallback` (reads `cx`/`cy` from `info`), trains, saves model + `VecNormalize` stats. Re-runnable to extend training. | ✅ Done — 3.5M timesteps trained; last run: **8.7 min, 20 valid layouts found, best CG penalty 16.4566** |
| **7** | Extracts single best training-time layout, independently re-verifies, renders front/back view + proximity map, exports CSV | ✅ Done — first independently-verified valid layout |
| **8** | **Multi-layout generation (inference only, no training).** Runs many rollouts of the trained policy (parallel `DummyVecEnv`, deterministic policy) from independent random resets, independently re-verifies each, deduplicates near-identical layouts (mean per-component shift < `DEDUP_DIST_MM`), keeps up to `MAX_LAYOUTS=5` sorted by CG penalty | ✅ Done — 800 episodes → 22 valid (2.75% success rate, deterministic) → **5 distinct layouts**, best CG penalty **12.2026** (beats Cell 7's single-rollout best of 16.4566) |
| **9** | Renders all layouts from Cell 8: front/back view (unchanged style) + a corrected proximity map that now renders a **true "front view through a transparent board"** — back-layer components drawn first (ghosted, hatched, low z-order), front-layer drawn opaque on top, pin colors differ by layer (red=front, orange=back) | ✅ Done — one combined CSV across all layouts |
| **10** | *(proposed, not yet built in the notebook)* Port `polish_cg()`/`component_bounds()` from `gen.py` and insert a polish step between Cell 8 and Cell 9 | ⬜ Not started — currently only available via `gen.py --polish` |

**Standalone script:** `gen.py` packages the Cell 8/9 inference logic (load trained model + `VecNormalize` stats → run N deterministic rollouts → dedupe → render both view styles → export CSV) as a CLI, so layouts can be regenerated without opening the notebook. Configurable via flags: `--json`, `--model_dir`, `--output_dir`, `--episodes` (default 800), `--parallel` (default 8), `--max_layouts` (default 5), `--dedup_dist` (default 25.0mm), `--deterministic` (default on), `--seed` (default 42). Same `SCORE_WEIGHTS`, `MAX_STEP_MM`/`MIN_STEP_MM`, and constants as the notebook.

**New in `gen.py` (this update):** `--polish` (default off), `--polish_delta` (default 60.0mm), `--polish_restarts` (default 4) — enable and configure the gradient-based CG polish hybrid stage from §5b.

## 7. Issues Encountered & Fixes (chronological)

1. **`SubprocVecEnv` has no `.envs` attribute.** Direct attribute access
   (`env.cx`) only works for in-process vec envs. **Fix (superseded by #3):**
   initially added `get_state()` + `env_method("get_state", ...)`.
2. **Insert-distance penalty not converging.** A single fixed nudge
   size was too coarse for sub-mm insert precision. **Fix:** annealed step size
   within each episode, raised `insert`'s reward weight, lowered `ent_coef`
   during fine-tuning.
3. **Critical bug: `get_state()` snapshots were silently wrong (race
   condition).** `SubprocVecEnv` auto-resets a sub-env the instant an
   episode ends, *before* control returns to the training loop — so
   post-hoc `env_method("get_state")` calls were capturing the **next**
   episode's start state, not the valid layout just found. Caught by Cell
   7's independent re-verification failing an assertion. **Fix:** `cx`/`cy`
   now copied into `info` *inside* `step()`, before any auto-reset can occur.
4. **Verification run returning 0 valid layouts (Cell 6 era).** Not a bug —
   valid layouts appeared at ~1 per 107k timesteps in that phase; the
   sample window was too short. Increasing the budget resolved it.
5. **Notebook cell sprawl during debugging (6, 6b, 6c).** Consolidated back
   into a single Cell 6.
6. **Legacy `gym` deprecation warning** — harmless, `gymnasium` used
   throughout.
7. **Cell 8, first attempt: 0/60 rollouts valid.** Root cause was
   twofold — (a) training's success rate was only ~0.14% per episode
   (20 valid / ~14,000 episodes), so 60 attempts had an expected value
   of ~0.08 successes; (b) `deterministic=False` added exploration noise
   on top of an already-narrow success window. **Fix:** switched to
   `deterministic=True` (mean action, no sampling noise — more reliable
   at inference than the exploration policy used during training) and
   ran 800 episodes across 8 parallel envs instead of 60 serial ones.
   Result: 2.75% success rate, 22 valid layouts, 5 kept after
   deduplication.
8. **Cell 9 proximity map was just front+back stacked, not a real
   see-through view.** Both layers were drawn with identical z-order/
   alpha, so it read as "front and back squished together" rather than
   "front view through a transparent board." **Fix:** two-pass draw —
   back-layer components first (ghosted: `alpha=0.30`, hatched `'//'`,
   muted colors, lower z-order), front-layer components second
   (`alpha=0.85`, opaque, higher z-order). Pin markers also color-coded
   by layer (red=front, orange=back) so overlapping pins stay
   distinguishable, plus a legend was added.
9. **(New) CG polish, first attempt: fixed large `w_hard` caused zero
   movement.** A single fixed hard-constraint penalty weight (`1e6`) made
   any local perturbation near an already-tight packing constraint spike
   the objective so much that L-BFGS-B's line search rejected every step
   — confirmed by re-running against the real penalty engine and seeing
   the polished layout come back byte-identical to the seed. **Fix:**
   replaced the single fixed weight with a continuation (sequential
   penalty) schedule that starts soft and ramps up, warm-starting each
   stage from the last — see §5b.
10. **(New) De-interleaving bug between `component_bounds()` and the
    optimizer's `[cx..., cy...]` vector layout.** `component_bounds()`
    (and the notebook's original per-component bounds logic) returns
    bounds interleaved as `(x_min, y_min, x_max, y_max)` per component,
    but the polish stage's decision vector is laid out as all-cx-then-
    all-cy. Concatenating without de-interleaving first produced inverted
    (`lower > upper`) bounds for several components and crashed
    `scipy.optimize.minimize`. **Fix:** explicit `env_lower[0::2]` /
    `env_lower[1::2]` slicing before building the bounds list.

## 8. Current Results

**Single-rollout best (Cell 7, training-time):**

| Metric | Value |
|---|---|
| Border penalty | 0.0000 ✅ |
| Overlap penalty | 0.0000 ✅ |
| Insert-distance penalty | 0.0000 ✅ |
| CG penalty | 16.4566 |
| Combined Assembly CG offset | (-3.94, -3.17) mm |

**Multi-rollout best of 5 (Cell 8/9, inference-time, deterministic policy, pre-polish):**

| Layout | CG penalty | Border/Overlap/Insert |
|---|---|---|
| 1 (best) | **12.2026** | all 0.0 ✅ |
| 2 | 18.6801 | all 0.0 ✅ |
| 3 | 44.3006 | all 0.0 ✅ |
| 4 | 75.8543 | all 0.0 ✅ |
| 5 | 80.6805 | all 0.0 ✅ |

All three priority-1 hard constraints are satisfied exactly on every kept
layout. The best multi-rollout layout (#1) improves on the single Cell 7
layout by ~26% lower CG penalty.

**Polish stage validation (this update):** run against the real
`compute_penalty_breakdown` engine on an independently-constructed feasible
seed layout (CG penalty 5999.54), a 3-stage abbreviated continuation
schedule reduced CG penalty to 4340.50 while holding border/overlap/insert
at exactly 0 — confirming the approach makes genuine, constraint-respecting
progress. This was run in a slowed-down (no-numba) environment for
verification only; a full `--polish` run on the actual trained checkpoint
with numba JIT has not yet been executed and is the immediate next step
(see §9).

## 9. What's Left To Do

1. **Run `gen.py --polish` on the actual trained checkpoint** and confirm
   CG penalty reaches 0 (offset ≤1.0mm) on at least one of the 5 kept
   layouts. Tune `--polish_delta` / `--polish_restarts` upward if not.
2. **Visual sanity-check** of all rendered PNGs — confirm clearance boxes
   aren't visually tight despite reading 0 violation, and that the
   see-through proximity map reads correctly (back components faint/hatched
   behind opaque front components).
3. **Port the polish stage into the notebook as Cell 10**, inserted between
   Cell 8 (multi-layout generation) and Cell 9 (rendering), so layouts can
   be polished without leaving the notebook.
4. **Optional: further PPO training to raise the quality of seeds fed into
   polish.** Re-running Cell 6 with a larger `TRAIN_TIMESTEPS` continues
   from the saved checkpoint — no code changes needed. A better PPO seed
   means a smaller `--polish_delta` is needed to reach tolerance, and
   would also raise Cell 8's ~2.75% deterministic success rate. This is
   now a secondary lever rather than the primary way to fix CG, since the
   polish stage handles final CG convergence.
5. **Baseline comparison against CMA-ES and DE.** Not yet done — worth
   comparing the PPO(+polish) layouts' penalty breakdown / CG offset /
   wall-clock time against both the original `new.py` CMA-ES output and a
   DE run on the same `newobj.json`, as three points in the comparative
   optimizer study (RL, evolutionary, gradient-based).
6. **Hyperparameter tuning pass** — PPO side: learning rate, `n_steps`,
   `ent_coef`, `MAX_STEP_MM`/`MIN_STEP_MM`, `SCORE_WEIGHTS`. Polish side:
   `weight_schedule` granularity/length, `stage_maxiter`, `polish_delta`.
7. **Tune Cell 8's `N_EPISODES`/`DEDUP_DIST_MM`** if you want more than 5
   candidate layouts to choose from, or a stricter/looser notion of
   "distinct."

## 10. Directory & File Map

```
project_root/
├── new.py                              Original CMA-ES reference implementation
│                                        (not modified, kept for comparison)
├── newobj.json                         Board + component specification (input,
│                                        unchanged from original project) — 18
│                                        components (PDS, CCRR0/1, LD0–5, RS,
│                                        BM1, PB0–3, RPU, MES, MT20)
├── layout_5_main.png                   Example output from the OLD CMA-ES pipeline
│                                        — front/back view style target
├── layout_5_distance_map.png           Example output from the OLD CMA-ES pipeline
│                                        — proximity map style target
├── <notebook>.ipynb                    PPO-based pipeline — Cells 1–9 finalized
│                                        per §6; Cell 10 (polish) not yet ported
├── gen.py                              Standalone CLI export of the Cell 8/9
│                                        inference pipeline, extended with the
│                                        gradient-based CG polish hybrid stage
│                                        (component_bounds(), polish_cg(),
│                                        --polish/--polish_delta/--polish_restarts)
├── README.md                           This file
│
└── optimized_layouts_rl/               Created by the notebook; all PPO outputs live here
    ├── tb_logs/                        TensorBoard logs from Cell 6 training runs
    ├── ppo_pcb_placement.zip           Trained PPO model checkpoint (Cell 6)
    ├── vecnormalize.pkl                Matching observation/reward normalization
    │                                   stats (Cell 6) — required at inference time
    │                                   (Cell 8) to match the policy's expected
    │                                   observation scaling
    ├── ppo_layout_front_back.png       Cell 7 — single best training-time layout,
    │                                   front/back projection view
    ├── ppo_layout_proximity_map.png    Cell 7 — single best training-time layout,
    │                                   proximity/pin-distance map (old-style,
    │                                   front+back stacked — superseded in
    │                                   spirit by Cell 9's per-layout maps below)
    ├── ppo_layout_table.csv            Cell 7 — component-level data table
    │                                   (name, side, mass, dims, CoG X/Y) for the
    │                                   single best training-time layout
    │
    └── multi_layouts/                  Created by Cell 9; one set of outputs per
                                         distinct valid layout found in Cell 8
        ├── layout_1_front_back.png     Layout 1 (best, CG penalty 12.2026 pre-polish) —
        │                               front/back projection view
        ├── layout_1_proximity_map.png  Layout 1 — TRUE see-through proximity map
        │                               (back layer ghosted/hatched, front layer
        │                               opaque on top)
        ├── layout_2_front_back.png     Layout 2 (CG penalty 18.6801 pre-polish)
        ├── layout_2_proximity_map.png
        ├── layout_3_front_back.png     Layout 3 (CG penalty 44.3006 pre-polish)
        ├── layout_3_proximity_map.png
        ├── layout_4_front_back.png     Layout 4 (CG penalty 75.8543 pre-polish)
        ├── layout_4_proximity_map.png
        ├── layout_5_front_back.png     Layout 5 (CG penalty 80.6805 pre-polish)
        ├── layout_5_proximity_map.png
        └── all_layouts_table.csv       Combined component-level data table across
                                         ALL 5 layouts — columns: Layout, Layout CG
                                         Penalty, Component Name, Layer Placement,
                                         Mass (kg), Dimensions (mm), CoG X (mm),
                                         CoG Y (mm)

generated_layouts/                      Created by `gen.py` (CLI, --output_dir);
                                         same structure as multi_layouts/ above,
                                         reflecting post-polish CG values when
                                         --polish is passed
```

## 11. How to Resume

1. Run Cells 1–3 fresh if starting a new kernel session (setup, config
   parsing, numba engine).
2. Run Cell 4 (env class, final version with annealed step + `info` fix).
3. Run Cell 5 (spawns `SubprocVecEnv`, auto-loads existing `VecNormalize`
   stats if present).
4. Run Cell 6 — auto-loads the existing trained checkpoint if
   `ppo_pcb_placement.zip` exists on disk; trains `TRAIN_TIMESTEPS` more
   steps (bump this value to train longer); saves model + normalization
   stats. Safe to re-run repeatedly to keep extending training.
5. Run Cell 7 to extract, independently re-verify, and render the single
   best training-time layout.
6. Run Cell 8 to generate up to 5 distinct valid layouts via inference-only
   rollouts of the trained policy (no training happens in this cell).
7. Run Cell 9 to render all layouts from Cell 8 (front/back view +
   true see-through proximity map) and export the combined CSV.
8. **New:** for CG-under-tolerance layouts without leaving the CLI, run
   `python gen.py --polish` (see §5b) instead of/after steps 6–7 — it
   performs the equivalent of Cells 8+9 plus the gradient-based CG polish
   stage, with post-polish CG values reflected in the output CSV.
9. Next: run the polish stage against the real checkpoint end-to-end
   (§9 item 1), port it into the notebook as Cell 10, and do the
   PPO-vs-CMA-ES-vs-DE baseline comparison (§9 item 5).

**Alternative to steps 6–7:** once a model checkpoint + `VecNormalize`
stats exist under `optimized_layouts_rl/`, run `python gen.py` (or with
flags, e.g. `python gen.py --episodes 1500 --max_layouts 8 --polish
--polish_delta 80 --polish_restarts 8`) to regenerate layouts from the
command line without opening the notebook.
