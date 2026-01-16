# TODO - Roadmap to v0

## Phase 1: Data Preprocessing

- [ ] Extract and validate JSUT corpus (~7,700 utterances)
- [ ] Create `scripts/preprocess_jsut.py`:
  - Build metadata CSV: `audio_path | transcript | phonemes | duration`
  - Resample audio to 22kHz, normalize, trim silence
  - Filter out samples > 12 seconds
- [ ] Split into train/val/test (80/10/10)
- [ ] Verify phoneme coverage and quality

## Phase 2: Training Pipeline

- [ ] Choose base model for fine-tuning:
  - **XTTS-v2** (recommended) - best quality, voice cloning, heavier
  - MeloTTS - lightweight, fast inference
  - VITS - good baseline, needs more data

- [ ] Implement `src/training/dataset.py`:
  - PyTorch Dataset class for preprocessed JSUT
  - Collate function for batching variable-length audio
  - Data augmentation (optional)

- [ ] Implement `src/training/train.py`:
  - Load pretrained checkpoint
  - Fine-tuning loop with mixed precision
  - Validation and checkpoint saving
  - TensorBoard logging

## Phase 3: Inference

- [ ] Implement `src/inference/synthesize.py`:
  - Load fine-tuned checkpoint
  - Text → phonemes → audio pipeline
  - Save output as WAV

- [ ] Create `scripts/demo.py` for quick testing

## Phase 4: Polish

- [ ] Add evaluation metrics (MOS estimation, mel-cepstral distortion)
- [ ] Optimize inference speed
- [ ] Export model for serving (ONNX or TorchScript)
- [ ] API endpoint via FastAPI (`[serve]` dependencies)

## Notes

- Hardware: RTX 3090 (24GB VRAM) - config tuned for this
- Target: Single-speaker Japanese TTS fine-tuned on JSUT
- Stretch: Multi-speaker with JVS corpus
