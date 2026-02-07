# Reference Papers

Papers referenced in the koe-tts implementation.

## Directory Structure

```
papers/
├── pdf/      # Original paper PDFs
└── notes/    # Implementation notes and annotations
```

## Papers

### VITS (arXiv:2106.06103)

**Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech**

Kim et al., 2021

- PDF: `pdf/2106.06103_VITS.pdf`
- [arXiv](https://arxiv.org/abs/2106.06103)

Key contributions used in koe-tts:
- Variational inference with normalizing flows for speech synthesis
- Stochastic duration predictor for realistic rhythm
- Adversarial training with multi-period and multi-scale discriminators
- End-to-end text-to-waveform without separate vocoder

Implementation notes:
- `modules/training/models/vits/` - Core VITS architecture
- `modules/training/models/vits/discriminators.py` - MPD/MSD implementations
- `modules/training/pipelines/train_vits.py` - Two-stage training (core + GAN)
