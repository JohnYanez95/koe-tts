"""
Labeler module: Human-in-the-loop data labeling for koe-tts.

Structure:
    app/        - FastAPI + React labeling UI
    pipelines/  - Pull batches, write labels back to lake
    common/     - Label schemas, validators, audio utils

Workflow:
    1. Launch the labeling server
       koe label serve jsut

    2. Or pull a batch headlessly
       koe label pull --dataset jsut --batch-size 100

    3. Write labels back to silver/gold
       koe label commit <batch_id>
"""
