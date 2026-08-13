# PCB Component Placement — PPO Rewrite, Adapted to New Board (30 components)

_This file picks up from the original project's `README.md` (18-component
board, `layout_5_main.png` reference images). Paste this whole file into a
new chat to resume exactly where this session left off._

## 1. What This Is

Same PPO-based placement pipeline as the original project — **no code
architecture changed.** The only difference is a new board specification
(`newobj.json`) with more components on a bigger board. The notebook
(`new.ipynb`, Cells 1–9) is fully data-driven off the parsed component count
`n`, so almost nothing needed to change to point it at the new board.

**Do NOT go looking for a CMA-ES-generated training dataset (e.g. "1049
variations").** Confirmed from the original README: the CMA-ES script
(`new.py`) was the *original standalone solution being replaced*, kept only
as a comparison baseline — it was never used to generate training data for
PPO. PPO trains directly via RL rollouts against the penalty function
computed live from the JSON config. No such dataset exists or is needed for
this new board either.

## 2. New Board Spec (`newobj.json`)

- Board: 3285.0 × 1458.0 mm, Border 20.0mm, Clearance 20.0mm
- All components are under `"FRONT"` only — there is no `"BACK"` key in this
  JSON (unlike the old board, which had both). This is handled correctly by
  existing code (`data['COMPONENTS'].get('BACK', [])` defaults to `[]`) — no
  fix needed, just means every component in this board is front-side.
- After `Object qty` expansion, this parses to **n = 30 components**
  (confirmed via env obs shape = 65 = `2n + 5` → n = 30).
- One component, `ICP`, has no `"CF"` key (unlike most others which do).
  Confirmed **not a bug**: `component.get('CF', [])` already defaults to an
  empty list, so `ICP` correctly falls back to the board's default
  `Clearance (mm)` = 20mm on all four sides.

## 3. What Changed vs. the Old 18-Component Version

Only **one deliberate code change**, in Cell 1:

```python
OUTPUT_DIR = "optimized_layouts_rl_29comp"   # was "optimized_layouts_rl"
```

Reason: the old board's saved `ppo_pcb_placement.zip` / `vecnormalize.pkl`
have observation/action dimensions sized for 18 components (different
n → different `2n+5` obs dim, `2n` action dim). Pointing the new run at the
same folder would make Cells 5/6 try to load incompatible checkpoints and
crash or silently corrupt training. Using a fresh output folder keeps the
two runs fully separate.

**Every other cell (1–9) was confirmed to need zero code changes** —
config parsing, bounds computation, the numba penalty engine, the Gym env,
vec-env setup, the PPO model/training loop, and both rendering cells (7, 9)
all index purely off `n`, `data`, and `OUTPUT_DIR`, none of which are
hardcoded to 18 anywhere in the notebook.

### Constants intentionally left untouched (for now)

These were tuned for the old 18-component/smaller-board case and may or may
not still be optimal here — **decision made to NOT preemptively retune
them**, and only touch them if training signal (below) says so:

- `MAX_STEP_MM = 12.0` / `MIN_STEP_MM = 0.25` (Cell 4) — nudge size annealing.
  Board is ~6x bigger in area now; 12mm max nudge may be proportionally
  smaller relative to the search space than it was on the old board.
- `SCORE_WEIGHTS = {"border": 3.0, "overlap": 3.0, "insert": 10.0, "cg": 0.5}`
  (Cell 4) — relative difficulty of constraints may shift with 30 components
  vs. 18.
- `learning_rate=3e-4`, `batch_size=256` etc. (Cell 6 PPO hyperparameters) —
  flagged as a *candidate* thing to touch (see §5) because `approx_kl` has
  been consistently high (~0.15, target is usually 0.01–0.03) with
  `clip_fraction` ~0.61, likely because the action space grew from 36-dim
  (18 components) to 60-dim (30 components) at the same learning rate.

## 4. Notebook Cell Status (this board)

| Cell | Purpose | Status |
|---|---|---|
| 1 | Imports, device, seeding, output dir | ✅ Done — `OUTPUT_DIR` changed to `optimized_layouts_rl_29comp` |
| 2 | Parse `newobj.json` → component arrays | ✅ Done, unchanged — n=30 parsed correctly |
| 3 | Coordinate bounds + numba penalty engine | ✅ Done, unchanged — bounds check passed, engine compiles/tests fine |
| 4 | Gym environment | ✅ Done, unchanged — `check_env` passed, obs shape (65,) confirmed |
| 5 | Vectorized envs (SubprocVecEnv + VecNormalize) | ✅ Done, unchanged — fresh VecNormalize created (no prior stats in new folder) |
| 6 | Model (load-or-create) + training | 🔄 **In progress** — see training log below |
| 7 | Extract best layout, verify, render | ⏳ **Blocked** — waiting for `n_valid_found > 0` (hard-asserts otherwise) |
| 8 | Multi-layout generation (inference) | ⏳ Not started — needs Cell 7 done first / a saved model |
| 9 | Render all layouts | ⏳ Not started |

## 5. Training Log (Cell 6, cumulative)

| Run | TRAIN_TIMESTEPS this call | Cumulative timesteps | Wall time | Valid layouts found | Best invalid composite score | border | overlap | insert | cg |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,000,000 | 1,000,000 | 7.6 min | 0 | 71.3562 | 0.0 | 7,359,557,166.57 | 0.0 | 598.95 |
| 2 | 2,000,000 | 3,000,000 | 16.6 min | 0 | 69.9753 | 0.0 | 4,150,403,443.22 | 0.0 | 1177.31 |

**Reading on progress:** `border` and `insert` are already fully resolved
(0.0) in the best-invalid snapshot from run 1 onward. `overlap` dropped
~44% from run 1 → run 2 (7.36e9 → 4.15e9) — real, trending progress, not a
stall. The `cg` value bouncing up between runs is not a regression signal:
`composite_score` is currently dominated by `overlap` (weight 3.0, and it's
still enormous pre-log1p), so the callback's "best invalid layout" snapshot
is selected almost entirely on overlap improvement; `cg` in that specific
snapshot is just whatever it happened to be, not a trend.

No valid layout is expected yet at this stage — the old 18-component board
didn't find its first valid layout until well into a 3.5M-step budget
either, and this board is a harder search (30 components, larger board).

**PPO training diagnostics to watch, not yet acted on:** `approx_kl` has
been ~0.146–0.154 and `clip_fraction` ~0.61–0.62 across both runs — both
higher than the usual PPO target range. Not treated as a problem yet since
overlap is still trending down run-over-run. **If overlap plateaus or
worsens over 2 consecutive future runs, this diagnostic — not
`SCORE_WEIGHTS` — is the first thing to address** (e.g. lower
`learning_rate`, raise `batch_size`/`n_steps`).

## 6. Immediate Next Step (where we left off)

Re-run **Cell 6 unchanged** except bump the timestep count further:

```python
TRAIN_TIMESTEPS = 3_000_000
```

This resumes from the saved checkpoint (`reset_num_timesteps=False`), so
it's cumulative, not a restart — will bring the total to 6,000,000 steps
once complete.

**After that run, check the printed breakdown:**
- If `n_valid_found > 0` → stop training, move to **Cell 7** (extract,
  independently re-verify, render the best valid layout).
- If `n_valid_found` still 0 but `overlap` continues trending down →
  keep training, bump `TRAIN_TIMESTEPS` again, same pattern.
- If `overlap` plateaus/worsens across two runs in a row → this is the
  trigger to revisit PPO hyperparameters (see §5) rather than keep blindly
  training longer.

## 7. How to Resume in a Fresh Chat

1. Paste this README first.
2. Re-run Cells 1–5 fresh in a new kernel (fast, no training involved) —
   they'll pick back up using `optimized_layouts_rl_29comp/` and
   auto-load the saved `vecnormalize.pkl` from Cell 5 if present.
3. Run Cell 6 with `TRAIN_TIMESTEPS` per §6 above — auto-loads the saved
   `ppo_pcb_placement.zip` checkpoint and continues training rather than
   starting over.
4. Once a run reports `n_valid_found > 0`, proceed to Cells 7 → 8 → 9
   exactly as in the original project (all unchanged, generic code).
5. Baseline comparison against `new.py` (CMA-ES) on this new board is still
   an open item from the original project's "What's Left To Do," never done
   for either board.
