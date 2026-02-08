# Architecture Documentation

System design documents for the koe-tts platform.

## Documents

### [Data Platform](./data-platform.md)
Medallion lakehouse architecture (Bronze → Silver → Gold) built on PySpark + Delta Lake. Covers ingest pipeline, schema enforcement, QC, phoneme generation, training manifest creation, and the two-tier storage model (archive + local SSD).

### [Labeler System](./labeler.md)
Segmentation labeling application for pause boundary refinement and trim boundary annotation. Covers the FastAPI backend, React/wavesurfer.js frontend, session management, label persistence, and the human-in-the-loop feedback cycle.

### [Staged Optimizer RFC](./staged-optimizer-rfc.md)
Design rationale for the two-stage cascaded optimization pipeline (trim detection → pause detection). Explains why pause must train on predicted trims rather than user GT, the slack band mechanism, and the invariant that Stage 2 never filters GT on invalid trims.

### Related Documents

- [GAN Stability Controller](../training/GAN_CONTROLLER.md) — State machine, escalation ladder, detection triggers, and baseline metrics for VITS GAN training stability.
- [Monitoring Dashboard](../dashboard/MONITOR_DESIGN.md) — FastAPI + SSE streaming design for live training metrics and control plane.
- [Data Contracts](../lakehouse/CONTRACT.md) — Schema contracts, validation rules, and partition strategy for Delta tables.
- [Algorithm Details](../algorithms/) — Standalone documentation for trim optimizer, pause optimizer, and stability controller algorithms.
