"""
Training module: Model training for koe-tts.

Structure:
    dataloading/  - Cache snapshots, dataset readers
    pipelines/    - Train, eval, synthesize entrypoints
    common/       - Logging, checkpoints, metrics utilities

Workflow:
    1. Cache a snapshot of gold data locally
       python -m modules.training.pipelines.cache --gold-version jsut_v1 --mode symlink

    2. Train from cached snapshot
       python -m modules.training.pipelines.train --cache-snapshot jsut_v1 --config configs/training/vits.yaml

    3. Evaluate checkpoint
       python -m modules.training.pipelines.eval --checkpoint runs/jsut-v1/checkpoints/best.ckpt

    4. Synthesize audio
       python -m modules.training.pipelines.synthesize --checkpoint runs/jsut-v1/checkpoints/best.ckpt --text "こんにちは"
"""

# TODO: Implement training module
