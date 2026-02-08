# Labeler Architecture: Data Storage & Optimization

## 1. Storage Locations

```
runs/labeling/
  {session_id}/
    session.json              # Session metadata (dataset, stratum, utterance_ids, heuristic ref)
    labels.jsonl              # Append-only user annotations (last-write-wins per utterance_id)

  published/{dataset}/
    labels.jsonl              # Canonical ground truth (deduped, with session_id provenance)
    manifest.json             # Published session registry + total_labels counter

  heuristics/
    {dataset}_{params_hash}.jsonl       # Per-utterance breakpoints cache (pause)
    {dataset}_{params_hash}.meta.json   # Config snapshot + run stats (pause)
    {dataset}_trim_{params_hash}.jsonl  # Per-utterance trim predictions cache
    {dataset}_trim_{params_hash}.meta.json  # Trim config snapshot + run stats

lake/silver/{dataset}/
  segment_breaks/             # Delta table (fallback if no heuristic JSONL)

lake/gold/{dataset}/
  manifests/train.jsonl       # Source utterances (text, phonemes, audio_relpath, duration_sec)

Browser localStorage:
  koe-labeler:{session_id}    # Transient UI recovery (survives browser crash, not canonical)
                              # Stores: currentIdx, pauBreaksMap, trimMap, savedIndices
```

> **Known gap:** `noise_zone_ms` is accepted by the `PauBreakSave` API model but
> dropped in `backend.py:315-323` when constructing the dict for persistence.
> Noise zone state currently only survives in localStorage, not in `labels.jsonl`.

## 2. End-to-End Data Flow

```mermaid
flowchart TD
    subgraph Sources["Source Data"]
        GOLD["lake/gold/{ds}/manifests/train.jsonl<br/><i>utterance_id, text, phonemes,<br/>audio_relpath, duration_sec</i>"]
        AUDIO["data/{audio_relpath}<br/><i>WAV files</i>"]
    end

    subgraph Heuristic["Heuristic Runner (heuristic.py)"]
        PDC["PauseDetectionConfig<br/><i>min_pause_ms, margin_db, floor_db,<br/>merge_gap_ms, pad_ms</i>"]
        RMS["detect_silence_regions()<br/><i>RMS dB sliding window<br/>→ adaptive threshold<br/>→ contiguous silent runs<br/>→ merge + pad</i>"]
        BP["regions_to_breakpoints()<br/><i>silence midpoints,<br/>filtered by lead/tail margins</i>"]
        CACHE["runs/labeling/heuristics/<br/>{ds}_{hash}.jsonl<br/><i>utterance_id → breakpoints_ms[]</i>"]
    end

    subgraph SessionMgmt["Session Management (data.py)"]
        CREATE["create_session()<br/><i>stratify by pau count (0,1,2,3+)<br/>random sample, exclude assigned+published</i>"]
        SJSON["runs/labeling/{sid}/session.json<br/><i>utterance_ids, heuristic ref,<br/>stratum, batch_size</i>"]
        GETBATCH["get_batch()<br/><i>enumerate pau tokens<br/>map pau → breakpoint (greedy nearest)<br/>merge saved labels<br/>trim: saved > auto-trim > None</i>"]
        AUTOTRIM["load_auto_trim()<br/><i>trim predictions from<br/>Stage 1 cache</i>"]
    end

    subgraph UI["React Frontend (LabelView)"]
        WAVE["Wavesurfer.js waveform<br/>+ draggable markers"]
        DECIDE["User decisions per pau:<br/>- accept/reject (use_break)<br/>- drag to adjust (ms)<br/>- set noise zone (b marker)<br/>- set trim boundaries"]
    end

    subgraph Persist["Label Persistence"]
        SAVE["POST /api/sessions/{sid}/item/{idx}/labels"]
        LJSONL["runs/labeling/{sid}/labels.jsonl<br/><i>append-only, schema v1:<br/>breaks[], trim_start_ms, trim_end_ms,<br/>status (labeled|skipped)</i>"]
    end

    subgraph Publish["Publication"]
        PUB["POST /api/sessions/{sid}/publish"]
        PUBFILE["runs/labeling/published/{ds}/labels.jsonl<br/><i>deduped (last wins per uid),<br/>+ session_id, published_at</i>"]
        MANIFEST["runs/labeling/published/{ds}/manifest.json<br/><i>published_sessions[], total_labels</i>"]
    end

    subgraph Optimizer["Differential Evolution Optimizer (heuristic.py)"]
        LOAD_PUB["_load_published_labels()<br/><i>status=labeled only</i>"]
        PRELOAD["Preload audio waveforms<br/><i>avoid re-reading per DE eval</i>"]
        DE["differential_evolution()<br/><i>5 params, k-fold CV,<br/>popsize=10, maxiter=50</i>"]
        LOSS["_compute_loss_utterance()<br/><i>Hungarian matching + composite loss</i>"]
        WINNER["Save optimized config<br/>→ run_heuristic() with new params"]
    end

    %% Source → Heuristic
    GOLD --> CREATE
    GOLD --> PDC
    AUDIO --> RMS
    PDC --> RMS --> BP --> CACHE

    %% Session flow
    CACHE --> CREATE
    CREATE --> SJSON
    SJSON --> GETBATCH
    CACHE --> GETBATCH
    CACHE --> AUTOTRIM
    AUTOTRIM --> GETBATCH
    GOLD --> GETBATCH

    %% UI flow
    GETBATCH --> WAVE
    WAVE --> DECIDE --> SAVE --> LJSONL

    %% Reload loop
    LJSONL -.->|"reload on navigate"| GETBATCH

    %% Publication
    LJSONL --> PUB --> PUBFILE
    PUB --> MANIFEST

    %% Optimization loop
    PUBFILE --> LOAD_PUB --> PRELOAD
    AUDIO --> PRELOAD
    PRELOAD --> DE
    DE -->|"each candidate config"| LOSS
    LOSS -->|"mean CV loss"| DE
    DE --> WINNER --> CACHE

    %% Feedback arrow
    CACHE -.->|"next session uses<br/>optimized params"| CREATE
```

## 3. Label Schema v1 (per utterance)

```json
{
    "utterance_id": "JSUT_0042",
    "breaks": [
        {
            "pau_idx": 1,
            "token_position": 8,
            "ms_proposed": 630,
            "ms": 680,
            "delta_ms": 50,
            "use_break": true,
            "noise_zone_ms": 750
        },
        {
            "pau_idx": 2,
            "token_position": 14,
            "ms_proposed": 1200,
            "ms": 1200,
            "delta_ms": 0,
            "use_break": false,
            "noise_zone_ms": null
        }
    ],
    "trim_start_ms": 100,
    "trim_end_ms": 4500,
    "label_schema_version": 1,
    "heuristic_version": "pau_v1_adaptive",
    "heuristic_params_hash": "sha1:abc123",
    "sample_rate": 22050,
    "labeled_at": "2026-02-05T10:30:00+00:00",
    "status": "labeled"
}
```

Every pau is recorded (both `use_break=true` and `false`) so negative labels feed the optimizer.

## 4. Staged Optimization Pipeline

> **Implementation:** `detect_trim_region()` in `audio.py`; `optimize_trim()`, `run_trim_heuristic()`, `load_trim_cache()` in `heuristic.py`; trim-aware `_evaluate_config()` and `optimize_heuristic()` in `heuristic.py`; `load_auto_trim()` and auto-trim seeding in `data.py`. Design rationale in `staged-trim-pause-optimizer-rfc.md`.

The optimization is split into two cascaded stages to eliminate the train/inference distribution mismatch:

```mermaid
flowchart TD
    subgraph Stage1[\"Stage 1: Trim Detection\"]
        T1[\"Full audio waveform\"]
        T2[\"detect_trim_region()<br/><i>RMS onset/offset detection<br/>asymmetric thresholds</i>\"]
        T3[\"Predicted trim: ŝ, ê<br/><i>Always 2 points</i>\"]
        T4[\"Clamp + validity check<br/><i>ê-ŝ ≥ min_content_ms<br/>else fallback to full audio</i>\"]
        T1 --> T2 --> T3 --> T4
    end

    subgraph Optimize1[\"optimize_trim()\"]
        O1[\"Published labels with<br/>trim_start_ms, trim_end_ms\"]
        O2[\"DE search: onset_margin_db,<br/>offset_margin_db, floor_db,<br/>percentile, min_content_ms,<br/>pad_start_ms, pad_end_ms\"]
        O3[\"Loss: capped position error<br/>+ duration error<br/>+ fallback penalty\"]
        O4[\"Freeze best T*\"]
        O1 --> O2 --> O3 --> O4
    end

    subgraph Cache[\"Trim Predictions Cache\"]
        C1[\"run_trim_heuristic()<br/>→ {uid → (ŝ, ê)}.jsonl\"]
    end

    subgraph Stage2[\"Stage 2: Pause Detection\"]
        P1[\"Candidates within [ŝ-m, ê+m]<br/><i>m = slack band (200ms)</i>\"]
        P2[\"GT filtered to same window\"]
        P3[\"Hungarian matching + loss\"]
        P4[\"If trim invalid:<br/>full audio, no GT filter\"]
    end

    subgraph Optimize2[\"optimize_heuristic()\"]
        H1[\"Published labels with<br/>breaks[].ms, breaks[].use_break\"]
        H2[\"Pre-load trim predictions<br/>(frozen, not recomputed)\"]
        H3[\"DE search: min_pause_ms,<br/>margin_db, floor_db,<br/>merge_gap_ms, pad_ms\"]
        H4[\"Loss: positional + miss<br/>+ orphan + count + neg\"]
    end

    T4 --> C1
    O4 --> C1
    C1 --> H2
    H2 --> P1 & P2
    P1 --> P3
    P2 --> P3
    P4 -.-> P3
    H1 --> H3 --> H4
    P3 --> H4
```

### Training Order (No Circular Dependency)

1. `optimize_trim(dataset)` → freeze best `TrimDetectionConfig` T*
2. `run_trim_heuristic(dataset, **T*)` → cache predicted trims as JSONL artifact
3. `optimize_heuristic(dataset, trim_predictions=cached, trim_margin_ms=200)` → pause optimizer trains on Stage 1 output

### Inference (No Slack by Default)

1. `detect_trim_region(wav, T*)` → [ŝ, ê] (clamped)
2. `detect_silence_regions(wav, P*)` → breakpoints, filtered to [ŝ, ê]

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Separate stages** | Trim (2-point boundary) and pause (N-point interior) have different signal regimes |
| **Train on predicted trims** | Matches inference distribution; avoids user-GT-as-window circular dependency |
| **Slack band (200ms)** | Absorbs small Stage 1 boundary drift during training |
| **Fallback penalty** | Discourages configs that trigger frequent clamping |
| **Decoupled safety constant** | `MIN_TRIM_WINDOW_MS_FOR_PAUSE=200` is Stage 2's own guardrail |

## 5. Original Single-Stage Optimization (Legacy)

> **Note:** This section documents the original single-stage approach, preserved for reference.
> The staged pipeline (Section 4) supersedes this for new optimization runs.

## 6. Single-Stage Optimization Detail (Legacy)

```mermaid
flowchart TD
    subgraph Input["Inputs"]
        PUB_LABELS["Published labels<br/><i>G+ = accepted pau positions (ms)<br/>G- = rejected pau positions (ms)</i>"]
        AUDIO_WAV["Preloaded waveforms<br/><i>uid → (tensor, sr, duration_ms)</i>"]
        BOUNDS["DE search bounds:<br/>min_pause_ms: [20, 250]<br/>margin_db: [2.0, 20.0]<br/>floor_db: [-80.0, -35.0]<br/>merge_gap_ms: [0, 200]<br/>pad_ms: [0, 80]"]
    end

    subgraph DE_Loop["Differential Evolution Loop"]
        PROPOSE["DE proposes candidate<br/>(min_pause, margin, floor, merge, pad)"]
        BUILD_CFG["Build PauseDetectionConfig"]
        KFOLD["K-Fold split (k=3)"]

        subgraph EvalFold["Per-Fold Evaluation"]
            DETECT["detect_silence_regions(wav, cfg)<br/>→ regions"]
            BKPT["regions_to_breakpoints(regions)<br/>→ candidates_ms[]"]
            TRIM["Filter candidates to<br/>[trim_start, trim_end]"]
            HUNGARIAN["Hungarian assignment<br/>(scipy linear_sum_assignment)<br/><i>cost[i,k] = |g - c| if ≤ τ<br/>else τ+1 (infeasible)</i>"]
        end

        subgraph LossCalc["Composite Loss"]
            POS["Positional (α=1.0)<br/>Huber(|g-c|/τ) over matched pairs"]
            MISS["Miss rate (β=3.0)<br/>unmatched G+ / |G+|"]
            ORPHAN["Orphan rate (γ=auto)<br/>unmatched candidates / |G+|<br/><i>γ=1.0 if no G-, else 0.25</i>"]
            COUNT["Count reg (λ=0.3)<br/>||C| - |G+|| / |G+|"]
            NEG["Negative proximity (δ=0.5)<br/>Σ exp(-d²/2σ²) for G- near C"]
            SUM["L = α·pos + β·miss + γ·orphan<br/>+ λ·count + δ·neg"]
        end

        MEAN_CV["Mean loss across folds"]
        CONVERGE{"Converged?<br/>tol=1e-4 or<br/>maxiter=50"}
    end

    subgraph Output["Output"]
        BEST["Best config params"]
        SAVE_RUN["run_heuristic(dataset, **best)<br/>→ new .jsonl + .meta.json"]
        REPORT["Baseline vs Optimized<br/>loss, recall, MAE"]
    end

    PUB_LABELS --> KFOLD
    AUDIO_WAV --> DETECT
    BOUNDS --> PROPOSE

    PROPOSE --> BUILD_CFG --> KFOLD
    KFOLD --> DETECT --> BKPT --> TRIM --> HUNGARIAN

    HUNGARIAN --> POS
    HUNGARIAN --> MISS
    HUNGARIAN --> ORPHAN
    HUNGARIAN --> COUNT
    HUNGARIAN --> NEG
    POS --> SUM
    MISS --> SUM
    ORPHAN --> SUM
    COUNT --> SUM
    NEG --> SUM

    SUM --> MEAN_CV --> CONVERGE
    CONVERGE -->|No| PROPOSE
    CONVERGE -->|Yes| BEST --> SAVE_RUN
    BEST --> REPORT
```

### Loss Function Breakdown

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| **Positional** | α=1.0 | `(1/\|M\|) Σ Huber(\|g-c\|/τ)` | Penalize misaligned matched pairs. Huber is quadratic near 0, linear for outliers (robust). |
| **Miss rate** | β=3.0 | `\|unmatched G+\| / \|G+\|` | Heavy penalty for failing to detect user-accepted pauses. |
| **Orphan rate** | γ=auto | `\|unmatched C\| / \|G+\|` | Penalize proposing breaks where user didn't accept. Auto-schedules: γ=1.0 when no G- data, γ=0.25 otherwise. |
| **Count reg** | λ=0.3 | `\|\|C\| - \|G+\|\| / \|G+\|` | Regularize candidate count toward ground truth count. |
| **Neg proximity** | δ=0.5 | `(1/\|G-\|) Σ exp(-d²/2σ²)` | Gaussian penalty when candidates land near rejected pau positions. σ=τ/2. |

**τ (tau)** = 120ms matching radius. Pairs beyond this distance are infeasible in the cost matrix.

## 7. Pau-to-Breakpoint Mapping

```mermaid
flowchart LR
    subgraph PhonemeString["Phoneme String"]
        P["k o N n i ch i w a <b>pau</b> o g e N k i d e s u <b>pau</b> k a"]
    end

    subgraph Enumerate["_enumerate_pau()"]
        E1["pau_idx=1, token_pos=9"]
        E2["pau_idx=2, token_pos=19"]
    end

    subgraph Expected["Uniform Prior"]
        X1["expected_ms = duration × 9/21"]
        X2["expected_ms = duration × 19/21"]
    end

    subgraph AutoBP["Heuristic Breakpoints"]
        B1["breakpoint 640ms"]
        B2["breakpoint 1850ms"]
        B3["breakpoint 3100ms"]
    end

    subgraph Match["Greedy Nearest Match"]
        M["Sort all (distance, bp, pau) pairs<br/>Assign closest unmatched pairs"]
    end

    subgraph Result["PauBreak Objects"]
        R1["pau_idx=1: ms=640, ms_proposed=640"]
        R2["pau_idx=2: ms=1850, ms_proposed=1850"]
        R3["breakpoint 3100ms → unmatched (orphan)"]
    end

    P --> E1 & E2
    E1 --> X1
    E2 --> X2
    X1 & X2 --> Match
    B1 & B2 & B3 --> Match
    Match --> R1 & R2 & R3
```

## 8. Feedback Loop (Staged)

```mermaid
flowchart LR
    H["Heuristic<br/>(default params)"]
    S["Labeling<br/>Session"]
    L["Published<br/>Labels"]

    subgraph Staged["Staged Optimization"]
        OT["optimize_trim()<br/><i>Stage 1: trim params</i>"]
        TC["Trim Cache<br/><i>frozen predictions</i>"]
        OP["optimize_heuristic()<br/><i>Stage 2: pause params<br/>on predicted trims</i>"]
    end

    H2["Heuristic<br/>(optimized params)<br/>+ auto-trim proposals"]
    S2["Next Session"]

    H -->|"proposes breakpoints"| S
    S -->|"user refines + publishes"| L
    L -->|"trim GT"| OT
    OT -->|"run_trim_heuristic()"| TC
    L -->|"pause GT"| OP
    TC -->|"trim_predictions"| OP
    OP -->|"best pause config"| H2
    H2 -->|"better proposals"| S2
    S2 -->|"more labels"| L
```

Each cycle: more labels → better trim fit → better pause fit → better proposals → less user effort per utterance.
Training order: Stage 1 (trim) freezes first, Stage 2 (pause) trains on Stage 1 predictions with slack band.
`optimize_heuristic()` supports legacy mode (`trim_predictions=None`) for backward compatibility.

**Auto-trim seeding:** `get_batch()` calls `load_auto_trim()` to seed `trim_start_ms`/`trim_end_ms` from the Stage 1 cache for utterances without saved trims. Precedence: saved trims > auto-trim > None. Trims are clamped and validated before reaching the UI.

## 9. What's NOT Built Yet

The path from **published labels → training pipeline** is not yet implemented. The published labels sit at `runs/labeling/published/{ds}/labels.jsonl` but nothing currently:
- Merges them into silver/gold Delta tables
- Creates "Tier 2" labeled segments for training
- Updates the training dataset with segment-level phoneme alignment

This is the bridge between the labeling app and the training pipeline that would close the loop for supervised segment training.
