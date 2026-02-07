# KOE Monitor Dashboard Design

## Goal

Provide a local, low-friction training monitor that:

- streams training metrics live (no Spark boot tax, no coupling)
- exposes GPU telemetry
- links to existing eval artifacts (HTML, audio, manifests)
- optionally sends simple control commands to training via a file-backed contract (control.json)

This is intended for single-user local dev, running alongside `koe train ...`.

## Non-Goals (for now)

- Multi-node orchestration (K8s), auth, remote access
- A full experiment tracker (MLflow parity)
- Tight in-process callbacks into the training loop
- Real-time tensorboard replacement

---

## Run Directory Contract

For a given run `runs/<run_id>/`:

```
runs/<run_id>/
├── config.yaml
├── RUN.md
├── train/
│   ├── metrics.jsonl          # append-only; monitor tails this (Phase A)
│   └── metrics.latest.json    # optional: last event snapshot (nice-to-have)
├── events.jsonl               # append-only; checkpoint/eval events (Phase A)
├── control.json               # monitor writes; training reads (Phase B)
├── eval/                      # existing eval artifacts
│   ├── eval_*/index.html
│   └── multispeaker_*/index.html
└── checkpoints/
    ├── best.pt
    └── final.pt
```

## Principles

- Append-only JSONL for streaming (`metrics.jsonl`, `events.jsonl`)
- Single small JSON for control (`control.json`) so training can poll safely
- Training always works without monitor present
- Monitor never mutates training artifacts besides `control.json`

---

## CLI UX Contract

### Primary entry point

```bash
koe monitor <run_id>
```

Starts monitor backend + UI for a run.

### Useful variants

```bash
koe monitor <run_id> --port 8080
koe monitor <run_id> --no-ui           # backend only
koe monitor --list                      # recent runs sorted by activity
koe monitor --latest [--dataset multi]  # optional convenience: pick newest run matching prefix
```

### Process model

- **Option A (simple):** `koe monitor` spawns backend + frontend dev server
- **Option B (cleaner):** backend serves bundled static frontend

**Recommendation:**

- Dev: backend + Vite dev server (fast iteration)
- Prod: backend serves `frontend/dist/` static bundle

---

## Phase A: Read-only Monitor

### Panels (minimum viable)

- Loss curves (mel / kl / dur / adv / fm where applicable)
- Step / epoch / throughput
- GPU temp / mem / util (polled)
- Controller status (alarm, throttle, adv ramp)
- Eval artifacts list (links to `eval/**/index.html`)
- Recent events (checkpoint saved, eval finished, stage change)

### Data sources

- `train/metrics.jsonl` (tail every ~2s or via SSE stream)
- `events.jsonl` (tail every ~2s or merged into SSE)
- GPU via `nvidia-smi` poll every ~5s

---

## Phase B: Control Plane (File-backed)

### control.json semantics

Monitor writes `runs/<run_id>/control.json`. Training polls it (e.g. every `log_interval` steps).

### File format

```json
{
  "schema_version": 1,
  "updated_at": "2026-01-25T18:03:12Z",
  "nonce": "d9b3d1b4-7bb0-4a31-9c19-5b0a0a2f1b3e",
  "commands": [
    { "type": "stop", "reason": "GPU temp too high" },
    { "type": "save_checkpoint", "tag": "manual" },
    { "type": "set_lr_mult", "value": 0.5 },
    { "type": "set_mel_weight_mult", "value": 1.1 }
  ]
}
```

### Rules

- `nonce` changes on each update (training can ignore repeats)
- training should acknowledge commands by emitting an event (see below)
- training should treat unknown command types as no-ops (log warning)
- monitor can keep the last command list; training should be idempotent

### Emergency stop (auto-trigger)

Monitor can issue a `stop` command if:

- `gpu_temp_c >= threshold` sustained for N polls (e.g. 3 samples)
- OR GPU memory near OOM and rising quickly (optional)
- OR controller raises critical alarm (optional)

Stop behavior should be graceful:

1. save checkpoint (tag: `emergency_stop`)
2. write event
3. exit cleanly

---

## JSONL Schemas

### metrics.jsonl (append-only)

One JSON object per training step (or log interval). Keep it flat and sparse.

#### Required fields

| Field   | Type              | Description                                      |
|---------|-------------------|--------------------------------------------------|
| `ts`    | string (ISO8601)  | UTC timestamp                                    |
| `step`  | integer           | Training step                                    |
| `epoch` | integer or float  | Epoch (optional)                                 |
| `stage` | string            | `"core"` \| `"gan"` \| `"baseline"` \| `"duration"` \| etc. |

#### Recommended fields

| Field          | Type   | Description                        |
|----------------|--------|------------------------------------|
| `lr`           | float  | Learning rate                      |
| `loss_total`   | float  | Total loss                         |
| `loss_mel`     | float  | Mel reconstruction loss            |
| `loss_kl`      | float  | KL divergence loss                 |
| `loss_dur`     | float  | Duration prediction loss           |
| `loss_adv`     | float  | Adversarial loss (GAN)             |
| `loss_fm`      | float  | Feature matching loss (GAN)        |
| `loss_disc`    | float  | Discriminator loss (GAN)           |
| `grad_norm_g`  | float  | Generator gradient norm (nullable) |
| `grad_norm_d`  | float  | Discriminator gradient norm        |
| `throughput_sps` | float | Steps per second                  |
| `num_speakers` | int    | Number of speakers                 |
| `batch_size`   | int    | Batch size                         |
| `segment_seconds` | float | Segment length in seconds        |

#### Controller fields (if present)

| Field                   | Type   | Description                                |
|-------------------------|--------|--------------------------------------------|
| `ctrl_alarm`            | string | `"healthy"` \| `"d_overpowering"` \| `"g_collapse"` \| `"instability"` |
| `ctrl_adv_weight_scale` | float  | Current adversarial weight multiplier      |
| `ctrl_d_throttle_active`| bool   | Whether discriminator is being throttled   |

#### Example

```json
{
  "ts": "2026-01-25T18:04:01Z",
  "step": 6200,
  "epoch": 1.84,
  "stage": "core",
  "lr": 0.0002,
  "loss_total": 2.311,
  "loss_mel": 0.441,
  "loss_kl": 0.032,
  "loss_dur": 0.301,
  "grad_norm_g": 4.92,
  "throughput_sps": 47.3,
  "ctrl_alarm": "healthy",
  "ctrl_adv_weight_scale": 0.0,
  "ctrl_d_throttle_active": false
}
```

### events.jsonl (append-only)

Discrete events that should be easy to display and query.

#### Event fields

| Field     | Type             | Description                |
|-----------|------------------|----------------------------|
| `ts`      | string (ISO8601) | UTC timestamp              |
| `type`    | string           | Event type (see below)     |
| `payload` | object           | Event-specific data        |

#### Event types

- `checkpoint_saved`
- `eval_started`
- `eval_finished`
- `control_ack`
- `stage_changed`

#### Examples

```json
{"ts":"2026-01-25T18:10:00Z","type":"checkpoint_saved","payload":{"path":"checkpoints/step_006000.pt","step":6000,"tag":"auto"}}
{"ts":"2026-01-25T18:20:12Z","type":"eval_finished","payload":{"eval_dir":"eval/multispeaker_42","mean_inter_speaker_distance":1.15}}
{"ts":"2026-01-25T18:21:01Z","type":"control_ack","payload":{"nonce":"d9b3d1b4-...","command":"stop","status":"ok"}}
```

---

## FastAPI Backend Contract

**Base URL:** `http://localhost:<port>`

### Endpoints (Phase A)

#### `GET /api/runs`

List runs from `runs/` sorted by last modified time.

**Response:**
```json
[
  {"run_id":"multi_vits_core_...","updated_at":"...","stage":"core","step":6200},
  ...
]
```

#### `GET /api/runs/{run_id}/config`

Reads `config.yaml` and returns parsed JSON.

#### `GET /api/runs/{run_id}/metrics?after_step=0&limit=2000`

Paginated read for initial page load.

#### `GET /api/runs/{run_id}/events?limit=200`

Recent events.

#### `GET /api/runs/{run_id}/artifacts`

Return eval artifact list (paths + titles):

```json
{
  "eval": [
    {"name":"multispeaker_42","path":"runs/<run>/eval/multispeaker_42/index.html","updated_at":"..."},
    ...
  ]
}
```

#### `GET /api/gpu`

Polled GPU snapshot.

**Response:**
```json
{
  "ts":"...",
  "gpus":[
    {"index":0,"name":"RTX 3090","temp_c":71,"util_pct":92,"mem_used_mb":12680,"mem_total_mb":24576,"power_w":315}
  ]
}
```

### Live streaming

#### `GET /api/runs/{run_id}/stream` (SSE)

Server-sent events streaming new lines from:

- `train/metrics.jsonl`
- optionally `events.jsonl`

**Event types:**

- `metrics`
- `event`
- `gpu` (optional: can be separate polling endpoint)

**Example SSE frames:**

```
event: metrics
data: {"ts":"...","step":6201,"loss_mel":0.439,...}

event: event
data: {"ts":"...","type":"checkpoint_saved","payload":{...}}
```

### Control endpoint (Phase B)

#### `POST /api/runs/{run_id}/control`

Writes `control.json` (atomic write recommended: write temp then rename).

**Request:**
```json
{
  "commands": [{"type":"stop","reason":"manual"}]
}
```

**Response:**
```json
{"ok": true, "nonce": "...", "path": "runs/<run_id>/control.json"}
```

---

## Frontend UX (React + Vite)

### Core components

| Component          | Description                                    |
|--------------------|------------------------------------------------|
| `LossChart`        | Plots selectable series (mel/kl/dur/adv/fm)    |
| `GpuPanel`         | Simple gauges or numbers + warning colors      |
| `ControllerBadges` | Alarm + throttle + ramp status                 |
| `ArtifactsPanel`   | Links to eval pages                            |
| `EventsFeed`       | Recent events list                             |
| `ControlPanel`     | Stop/pause/checkpoint buttons (Phase B)        |

### State model

1. **Initial load:** fetch metrics batch + config
2. **Live updates:** subscribe SSE stream; append to in-memory ring buffer
3. Keep last N points (e.g. 10k) to avoid browser slowdown

---

## File Watching / Tail Strategy

Use a robust tailer that:

- handles file rotation/truncation safely
- reopens file if inode changes
- resumes from last byte offset per client session

**Avoid blocking training:**

- training writes JSONL with `flush()` at log interval
- monitor polls/tails independently

---

## Integration with Existing Training Code

### Training loop additions (minimal)

1. Ensure `metrics.jsonl` is written consistently (already true)
2. Add `events.jsonl` writes on:
   - checkpoint save
   - eval start/finish
   - stage transition
3. Add optional control poll:
   - read `control.json`, apply if nonce changed
   - emit `control_ack` event
   - clear commands? (optional; better to rely on nonce)

---

## Implementation Notes

### Portability

Monitor should work even when `nvidia-smi` is missing:

- return `"available": false` in `/api/gpu`
- Support CPU training runs without crashing UI

### Security

- Bind to `127.0.0.1` by default
- No auth assumed (local-only)

---

## Milestones

### Phase A (Read-only)

- [ ] backend endpoints + SSE stream
- [ ] frontend charts + GPU panel + artifacts links

### Phase B (Control)

- [ ] control endpoint + `control.json` contract
- [ ] training loop polls + ack event
- [ ] emergency stop (GPU temp threshold)

### Phase C (Quality-of-life)

- [ ] run filters (dataset, stage)
- [ ] artifact preview embed (iframe for eval HTML)
- [ ] lightweight regression gates view (consume compare outputs)

---

## Open Questions / Defaults

| Setting                  | Default                          | Notes                        |
|--------------------------|----------------------------------|------------------------------|
| SSE polling cadence      | 2s metrics, 5s GPU               |                              |
| Emergency stop threshold | 83°C sustained 15s               | Configurable                 |
| Client metrics buffer    | Last 10k points                  |                              |
| Server paging            | Read last 50k lines              | Configurable                 |
