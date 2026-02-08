# Continuation: Repo Restructuring — Documentation & Presentation

**Original plan:** `~/.claude/plans/zazzy-launching-donut.md`
**Branch:** `project/refactor`
**Date:** 2026-02-06
**Status:** ~40% complete — structure built, first algorithm doc done, remainder is content writing

---

## Parallel Workstreams

This repo has **two active continuation docs**. They are mostly independent but share a few touch points.

| Workstream | File | Focus | Status |
|---|---|---|---|
| **This doc** — Repo Restructuring | `CONTINUATION.md` | Documentation, presentation, algorithm docs | ~40% |
| **Forge Module** | `continue-forge-module.md` | Infrastructure code (`modules/forge/`) | Phase 1 not started |

**Touch points (handle when you get there):**
- **README.md repo map** (Step 6) — if `modules/forge/` has code by then, include it in the tree
- **Architecture index** (`docs/architecture/README.md`) — may need a forge entry once it ships
- **Weekly reports** (Step 7, W06+) — forge work should be covered if it happened that week
- **Forge plan spec** — `forge-module-plan.md` at project root (655 lines, full API spec)

These are not blockers — the two workstreams can proceed independently.

---

## What This Plan Does

Restructures koe-tts documentation into three visibility layers:
1. **Project Overview** (README — anyone) — not yet written
2. **Technical Architecture** (engineer-readable) — moved, indexed
3. **Deep Technical Reports** (specialist-readable) — already existed, untouched

Also creates: algorithm docs with LaTeX + pseudocode + images, labeler docs, weekly progress reports, milestone reports, and a showcase directory.

---

## Completed Steps

### Step 1: Create directories — DONE
All directories exist:
- `docs/architecture/` `docs/algorithms/` `docs/labeler/` `docs/internal/`
- `reports/` `reports/milestones/` `showcase/`
- `images/trim_optimizer/` `image-generation/commons/` `image-generation/trim_optimizer/`

### Step 2: Move 5 files — DONE
All originals removed from root, confirmed at destinations:
| From (deleted) | To (exists) |
|---|---|
| `datalake-architecture.md` | `docs/architecture/data-platform.md` |
| `labeler-architecture.md` | `docs/architecture/labeler.md` |
| `staged-trim-pause-optimizer-rfc.md` | `docs/architecture/staged-optimizer-rfc.md` |
| `HANDOFF.md` | `docs/internal/HANDOFF.md` |
| `NEXT_STEPS.md` | `docs/internal/NEXT_STEPS.md` |

### Step 3: Architecture index — DONE
- `docs/architecture/README.md` — index linking data-platform, labeler, staged-optimizer-rfc, plus cross-links to GAN_CONTROLLER.md and MONITOR_DESIGN.md

### Step 4a: trim-optimizer.md — DONE (user reviewed, iterated)
- `docs/algorithms/trim-optimizer.md` — complete with:
  - Audio Energy Primer section (dBFS reference table)
  - LaTeX formulas with `where:` clauses explaining every variable and **why defaults were chosen**
  - Embedded matplotlib images below where clauses (3 PNGs)
  - Pseudocode with vectorization note
  - Design note: confidence k/b are fixed heuristics, not optimized (speaker-dependent limitation noted)
  - Observation on learned params: asymmetry shifted from threshold domain to padding domain
  - Results tables, parameter space, output format, source file references

### Image generation infrastructure — DONE
- `image-generation/commons/style.py` — shared `apply_style()` and `get_output_dir()`
- `image-generation/trim_optimizer/asymmetric_thresholds.py` → `images/trim_optimizer/asymmetric_thresholds.png`
- `image-generation/trim_optimizer/logistic_confidence.py` → `images/trim_optimizer/logistic_confidence.png`
- `image-generation/trim_optimizer/capped_l1_loss.py` → `images/trim_optimizer/capped_l1_loss.png`

Each script is standalone: `python image-generation/trim_optimizer/<name>.py`

---

## Remaining Steps

### Step 4b: pause-optimizer.md — NOT STARTED
**File:** `docs/algorithms/pause-optimizer.md`
**Source material:**
- `modules/labeler/heuristic.py` — `optimize_heuristic()`, `_evaluate_config()`, `process_utterance()`
- `modules/data_engineering/common/audio.py` — `find_silent_runs()`
- `docs/architecture/staged-optimizer-rfc.md` — design rationale for cascaded stages
- Optimizer run results: loss 3.300→1.148 (65.2%), recall 0%→78.3%, MAE 60.2ms

**Content to cover:**
- Problem: detect internal silence regions for segmentation
- RMS energy analysis, adaptive thresholding, merge gap
- Hungarian matching for GT-to-candidate alignment (with slack band 200ms)
- Key design: trains on predicted trims (Stage 1 output), not user GT
- 5-param DE with k-fold CV
- Results table

**IMPORTANT — follow the established pattern:**
1. LaTeX `$$...$$` with `where:` clauses defining all variables + intuitive justification for defaults
2. Pseudocode in ```python blocks with vectorization notes where applicable
3. matplotlib images: create scripts in `image-generation/pause_optimizer/`, output to `images/pause_optimizer/`
4. Embed images below where clauses
5. Add Audio Energy Primer if doc uses dB concepts (or reference trim-optimizer.md's primer)
6. **Ask user to review before moving to next file**

### Step 4c: stability-controller.md — NOT STARTED
**File:** `docs/algorithms/stability-controller.md`
**Source material:**
- `docs/training/GAN_CONTROLLER.md` — full 980-line reference (DO NOT duplicate, summarize + link)
- `modules/training/common/gan_controller.py` — implementation
- `docs/postmortems/gan_stability_log.md` — incident history

**Content to cover:**
- State machine: HEALTHY → UNSTABLE (L0-L3) → EMERGENCY
- Key triggers, escalation ladder, de-escalation gates
- Healthy baselines table (g_clip_coef, d_clip_coef, grad norms)
- Danger bands (10k-12k, 27k-30k)
- Link to full reference: `docs/training/GAN_CONTROLLER.md`

### Step 5: Labeler docs — NOT STARTED
**Files:** `docs/labeler/DESIGN.md`, `docs/labeler/WORKFLOW.md`
**Source material:**
- `docs/architecture/labeler.md` — system architecture (already moved)
- `modules/labeler/app/backend.py` — FastAPI server
- `modules/labeler/app/data.py` — data layer
- `modules/labeler/app/frontend/src/` — React SPA

### Step 6: README.md rewrite — NOT STARTED
**File:** `README.md` (overwrite)
**IMPORTANT:** Write this AFTER steps 4-5 so all doc links resolve.
**Structure:** See plan section 1 — neutral tone, metrics table, architecture summary, root cause paragraph, repo map, quick start, doc links.
**Key line:** "This shifted the project from reactive instability mitigation to proactive data quality control."

### Step 7: Reports — NOT STARTED
**Files:**
- `reports/TEMPLATE.md` — weekly report template (see plan for structure)
- `reports/2026-W03.md` through `reports/2026-W06.md` — backfill from git log + postmortems
- `reports/milestones/001-medallion-lakehouse-v1.md`
- `reports/milestones/002-root-cause-data-misalignment.md`
- `reports/milestones/003-staged-optimizer-v1.md`

**Source material for backfill:**
- W03-04: git log, architecture doc creation dates, lakehouse CONTRACT.md
- W05: 14 postmortems in `docs/postmortems/`, `gan_stability_log.md`
- W06: labeler app commits, heuristic.py, this conversation's optimizer results

### Step 8: Showcase manifest — NOT STARTED
**File:** `showcase/README.md` — visual asset manifest (see plan section 6)

### Step 9: Verify links + clean up — NOT STARTED
- Grep for `](./` and `](../` to verify internal links resolve
- Delete empty dirs: `docs/experiments/` and `docs/phoneme_mapping/` (confirmed empty)
- Verify README renders correctly

### Step 10: Security sweep — NOT STARTED
- No IPs, no personal paths, no credentials in any new/moved file

---

## User Workflow Preferences

**CRITICAL — the user has set an explicit review workflow:**
> "When I approve a .md file don't move immediately, ask me to review it and I'll come back with questions if I have any or additional edits."

- Write ONE doc at a time
- Present it to user for review
- Wait for feedback / approval
- Only then proceed to the next file

**Documentation standard (from user feedback + memory):**
1. LaTeX formulas with `where:` clauses — define every variable, explain why defaults were chosen
2. Pseudocode in code blocks — note vectorization opportunities
3. matplotlib images — one script per image, embedded below where clauses
4. Audio Energy Primer for docs using dB concepts
5. Design notes for limitations and arbitrary constants
6. No corner-cutting — show the math, show the code, show the picture

---

## Identified Future Work (not part of this plan)

**LaTeX conversion of existing docs** — identified but not yet scheduled:
- HIGH: `docs/training/GAN_CONTROLLER.md` (~10 formulas), `docs/architecture/labeler.md`, `docs/architecture/staged-optimizer-rfc.md`
- MEDIUM: `docs/architecture/data-platform.md`, select postmortems
- LOWER: `docs/dashboard/MONITOR_DESIGN.md`, `docs/lakehouse/CONTRACT.md`

**Vectorization of audio.py** — for loops identified that should use torch ops:
- `compute_rms_db()`: frame-by-frame loop → `torch.unfold()`
- Forward/backward scans in `detect_trim_region()`: scalar `.item()` loops → `torch.nonzero()`
- `find_silent_runs()`: `.tolist()` iteration → `torch.diff()` + `torch.nonzero()`

---

## Key File References

| File | Role |
|------|------|
| `docs/algorithms/trim-optimizer.md` | Completed algorithm doc (reference for style) |
| `image-generation/commons/style.py` | Shared plot styling (import for new scripts) |
| `modules/labeler/heuristic.py` | Source for pause-optimizer.md content |
| `modules/data_engineering/common/audio.py` | Source for trim detection + audio DSP |
| `modules/training/common/gan_controller.py` | Source for stability-controller.md |
| `docs/training/GAN_CONTROLLER.md` | Full GAN controller reference (don't duplicate) |
| `docs/postmortems/` | 14 incident reports (source for weekly report backfills) |
| `docs/postmortems/gan_stability_log.md` | Rolling trend log (source for W05 report) |
| `.claude/plans/zazzy-launching-donut.md` | Original approved plan with full spec |
| `.claude/projects/-home-john-Repos-koe-tts/memory/MEMORY.md` | Project memory with doc pattern, security checklist |
