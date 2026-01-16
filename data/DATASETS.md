# Japanese TTS Datasets

## Priority Order (Start Here)

### 1. JSUT (Japanese Speech Corpus of Saruwatari Lab, U-Tokyo)
**Recommended first dataset** - Single speaker, very clean, studio quality.

- **Size**: ~10 hours, 7,696 utterances
- **Speaker**: Single female speaker
- **License**: CC-BY-SA 4.0
- **Quality**: Studio recorded, 48kHz
- **Download**: https://sites.google.com/site/shinaborumorimoto/jsut

```bash
# Download script
cd data/raw
wget http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
unzip jsut_ver1.1.zip
```

Subsets:
- `basic5000/` - Basic sentences (main TTS training set)
- `travel1000/` - Travel domain
- `countersuffix26/` - Counter expressions
- `loanword128/` - Loanwords
- `voiceactress100/` - Emotional/acted speech

---

### 2. JVS (Japanese Versatile Speech) Corpus
**Multi-speaker diversity** - 100 speakers, parallel sentences.

- **Size**: ~30 hours total (100 speakers × ~100 utterances each)
- **Speakers**: 100 (50M/50F)
- **License**: CC-BY-SA 4.0
- **Quality**: Studio recorded, 24kHz
- **Download**: https://sites.google.com/site/shinaborumorimoto/jvs-corpus

```bash
cd data/raw
# Download from the site (requires form submission)
# or via Hugging Face:
# https://huggingface.co/datasets/japanese-asr/jvs
```

Subsets:
- `parallel100/` - Same 100 sentences across all speakers
- `nonpara30/` - Speaker-specific sentences
- `whisper10/` - Whispered speech
- `falset10/` - Falsetto

---

### 3. Common Voice Japanese (Mozilla)
**Large scale, varied speakers** - Crowdsourced, filter for quality.

- **Size**: 100+ hours (growing)
- **Speakers**: Thousands (crowdsourced)
- **License**: CC-0 (public domain)
- **Quality**: Variable (needs filtering)
- **Download**: https://commonvoice.mozilla.org/ja/datasets

```bash
# Via Hugging Face datasets
python -c "
from datasets import load_dataset
ds = load_dataset('mozilla-foundation/common_voice_16_1', 'ja', split='train')
print(f'Loaded {len(ds)} samples')
"
```

**Important**: Filter by `up_votes` and `down_votes` for quality.

---

## Secondary Sources

### 4. LJ Speech Japanese Equivalent - LibriTTS-style
Check OpenSLR for Japanese entries:
- https://www.openslr.org/resources.php (search for Japanese)

### 5. JTubeSpeech
YouTube-derived Japanese speech corpus.
- **Size**: 1,300+ hours
- **Quality**: Variable (in-the-wild)
- **Link**: https://github.com/sarulab-speech/jtubespeech

### 6. JNAS (Japanese Newspaper Article Sentences)
- Read newspaper text
- Contact required for access

---

## Data Preparation Checklist

After downloading, run through:

1. [ ] Verify audio format (target: 22kHz or 44kHz mono WAV)
2. [ ] Check transcript alignment (text matches audio)
3. [ ] Normalize audio levels
4. [ ] Remove silence/noise
5. [ ] Split into train/val/test (80/10/10)
6. [ ] Generate phoneme alignments (pyopenjtalk)

---

## Storage Estimates

| Dataset | Raw Size | Processed |
|---------|----------|-----------|
| JSUT | ~3 GB | ~5 GB |
| JVS | ~8 GB | ~15 GB |
| Common Voice (50h subset) | ~10 GB | ~20 GB |
| **Total starter set** | **~21 GB** | **~40 GB** |

Your 2TB is plenty.

---

## Quick Start Script

```bash
#!/bin/bash
# scripts/download_jsut.sh

DATA_DIR="data/raw/jsut"
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "Downloading JSUT corpus..."
wget -q https://ss-takashi.github.io/jsut-lab/data/jsut_ver1.1.zip
unzip -q jsut_ver1.1.zip
rm jsut_ver1.1.zip

echo "Done. JSUT downloaded to $DATA_DIR"
ls -la
```
