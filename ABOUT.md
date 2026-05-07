# About Music Style Transfer

## Overview

**Music Style Transfer** is an intelligent audio analysis and generative music system that analyzes the stylistic characteristics of reference audio and generates original compositions that preserve the identified style while creating entirely new content.

Think of it as a **musical translator** - it understands what makes a song sound the way it does, and creates new music with the same "voice."

## The Problem We Solve

Musicians, composers, and producers often need to:
- Generate multiple variations of a piece maintaining consistent style
- Create content in a specific musical style without having the expertise
- Explore creative variations while preserving brand identity
- Accelerate composition and ideation workflows

Traditional approaches require:
- Manual composition skills
- Hours of creative work
- Domain expertise in music theory and production

## The Solution

Music Style Transfer automates this by:
1. **Analyzing** reference audio to extract 12+ style metrics
2. **Classifying** the musical genre and characteristics
3. **Generating** new audio that matches the identified style
4. **Preserving** the essential qualities while creating novel content

## How It Works

### Three-Phase Architecture

```
Reference Audio (Input)
    ↓
[PHASE 1: ANALYSIS]
  Extract: Tempo, Loudness, Spectral Profile, Rhythm, etc.
    ↓
[PHASE 2: CLASSIFICATION]
  Detect: Genre (Jazz/Pop/Rock/etc), Beat Regularity, Syncopation
    ↓
[PHASE 3: GENERATION]
  Create: New audio matching the style profile
    ↓
Generated Audio (Output)
```

### Technical Highlights

- **Audio Analysis**: librosa (12+ acoustic features)
- **Genre Classification**: music21 + custom algorithms (9 genres)
- **Music Generation**: Transformers-based MusicGen (Facebook/Meta)
- **Fallback System**: Always produces output via synthetic generation
- **Zero Configuration**: Works out-of-the-box with sensible defaults

## Use Cases

### 1. **Creative Variation**
```
Input: Artist's demo track
Output: 10 variations in the same style for brainstorming
```

### 2. **Style Transfer**
```
Input A: Classical music
Input B: Electronic music
Output: Classical piece with electronic style
```

### 3. **Content Generation at Scale**
```
Input: Brand signature music
Output: 100+ variations for ads, videos, notifications
```

### 4. **Production Acceleration**
```
Input: Reference track
Output: Instant starting point for further composition
```

## Key Features

✅ **Automatic Style Detection**
- 9-genre classification system
- Confidence scoring
- Detailed rhythm analysis (beat regularity, syncopation, breaks)

✅ **Multi-Method Generation**
1. Local MusicGen (Transformers) - Highest quality
2. Hugging Face API - Cloud-based
3. Synthetic Fallback - Always works

✅ **Full Audio Feature Extraction**
- Tempo (BPM)
- Loudness (RMS energy)
- Spectral Centroid (brightness/darkness)
- Zero Crossing Rate (harshness)
- MFCC (13 coefficients - "fingerprint" of sound)
- And 6+ more metrics

✅ **Complete CLI Interface**
```bash
# Analyze any audio file
python -m src analyze reference.wav

# Generate music matching the style
python -m src generate reference.wav --duration 30
```

✅ **Production-Ready**
- 11/11 tests passing
- Comprehensive error handling
- Full documentation
- Fallback chains (never fails)

## Technology Stack

### Core Libraries
- **librosa** (0.10.0) - Audio analysis, feature extraction
- **music21** (8.3.0) - Music theory, analysis
- **transformers** - MusicGen model (Facebook/Meta)
- **torch** - Neural network inference
- **numpy/scipy** - Numerical computing
- **soundfile** - Audio I/O

### Infrastructure
- Python 3.11
- Virtual environment (venv)
- pytest (11 unit tests)
- Git + GitHub

## Getting Started

### Quick Start (5 minutes)

```bash
# 1. Clone and setup
git clone https://github.com/Jakub-Syrek/MusicStyleTransfer.git
cd MusicStyleTransfer
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# 2. Generate test audio
python create_test_audio.py

# 3. Analyze
python -m src analyze reference.wav

# 4. Generate
python -m src generate reference.wav --duration 30

# 5. Check output
# → output/generated_music.wav
```

### With Hugging Face API (Optional)

For cloud-based generation:
```bash
# 1. Get API key: https://huggingface.co/settings/tokens
# 2. Create .env file
echo "HUGGINGFACE_API_KEY=hf_your_key_here" > .env

# 3. Generate via cloud
python -m src generate reference.wav --provider huggingface
```

## Project Structure

```
MusicStyleTransfer/
├── src/
│   ├── __main__.py              # CLI interface
│   ├── style_analyzer.py        # Audio feature extraction
│   ├── genre_detector.py        # Style classification
│   ├── music_generator.py       # Generation orchestrator
│   └── api_client.py            # AI model integration
├── tests/
│   ├── test_style_analyzer.py   # 5 unit tests
│   └── test_api_client.py       # 6 unit tests
├── CLAUDE.md                    # Development directives
├── TECHNICAL_OVERVIEW.md        # Deep technical docs
├── EXAMPLES.md                  # Usage examples
├── requirements.txt             # Dependencies
└── .env.example                 # Configuration template
```

## Example Output

### Input Analysis
```
[ANALYSIS] Musical Style Analysis
==================================================
Tempo: 125.0 BPM
Loudness: 0.369
Spectral Centroid: 2868.3 Hz
Zero Crossing Rate: 0.0717
MFCC (13 coefficients): ['-23.43', '19.51', '7.39', ...]

[GENRE] Classification
==================================================
Genre: Pop music
Confidence: 68.1%

[RHYTHM] Beat & Rhythm Analysis
==================================================
Beat Regularity: 93.7%
Onset Density: 0.132
Strong Rhythm: Yes
Detected Breaks: 0
```

### Generation Process
```
[GENERATING] Music based on: reference.wav
   Duration: 30s, Provider: huggingface
[INFO] Loading facebook/musicgen-small...
[OK] Generated music saved to: output/generated_music.wav
```

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│            INPUT: reference.wav                      │
│         (e.g., Jazz, 85 BPM, Smooth)               │
└────────────────────┬─────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  STYLE ANALYZER       │
         │  (librosa + music21)  │
         │                       │
         │ Extract: 12+ metrics  │
         │ • Tempo               │
         │ • Loudness            │
         │ • Spectral Profile    │
         │ • MFCC Fingerprint    │
         │ • Rhythm Pattern      │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  GENRE CLASSIFIER     │
         │  (9 genres)           │
         │                       │
         │ Detect: Jazz, 94%     │
         │ Beat Regularity: 93%  │
         │ Syncopation: 0.08     │
         └───────────┬───────────┘
                     │
         ┌───────────▼────────────────────────┐
         │  PROMPT GENERATOR                  │
         │                                    │
         │ "Create smooth jazz music,        │
         │  85 BPM, warm and relaxing"       │
         └───────────┬────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     ▼               ▼               ▼
┌─────────┐  ┌─────────┐  ┌──────────────┐
│ Local   │  │Hugging  │  │   Synthetic  │
│MusicGen │  │Face API │  │  Fallback    │
│(GPU)    │  │(Cloud)  │  │(Instant)     │
└────┬────┘  └────┬────┘  └──────┬───────┘
     │            │              │
     └────────────┼──────────────┘
                  │
         ┌────────▼────────┐
         │ AUDIO VOCODER   │
         │ Tokens → WAV    │
         └────────┬────────┘
                  │
         ┌────────▼────────────────────┐
         │ OUTPUT: generated_music.wav │
         │ (New composition, same style) │
         └─────────────────────────────┘
```

## Performance Metrics

| Metric | Value | Note |
|--------|-------|------|
| Analysis Time | ~2-3s | Depends on audio length |
| Generation Time | ~30-120s | First run: model download |
| Generated Quality | ⭐⭐⭐⭐⭐ | MusicGen; ⭐⭐⭐ Fallback |
| Test Coverage | 11/11 passing | 100% of critical paths |
| Supported Formats | WAV, MP3, FLAC | Via librosa |
| Output Format | WAV (16kHz, 16-bit) | Standard, universal |

## Limitations & Future Work

### Current Limitations
- Monophonic audio generation (single track)
- ~30 second maximum length per generation
- No instrument-specific control
- Accuracy depends on quality of training data

### Planned Improvements
- [ ] Multi-track orchestration
- [ ] Real-time parameter control
- [ ] MIDI export capability
- [ ] Fine-tuning on custom styles
- [ ] Web UI interface
- [ ] Batch processing
- [ ] Audio visualization
- [ ] Genre-specific models

## Contributing

This project follows professional development standards:
- ✅ JSDoc documentation for all functions
- ✅ SOLID principles throughout
- ✅ Comprehensive error handling
- ✅ Unit tests (11/11 passing)
- ✅ English-only code and commits

### Development Setup
```bash
# Create worktree
git worktree add feature/new-feature

# Run tests
pytest tests/ -v

# Follow CLAUDE.md directives
# Commit with: git commit -m "Feature: description"
```

## License

MIT License - Free for personal and commercial use

## Author

Jakub Syrek

## Acknowledgments

- **Facebook/Meta** - MusicGen model and transformers library
- **librosa team** - Audio analysis library
- **music21 project** - Music theory toolkit
- **Hugging Face** - Model hosting and inference API

## Contact & Support

- 🐛 **Issues**: Report bugs on GitHub Issues
- 💬 **Discussions**: Ask questions in Discussions
- ✉️ **Email**: See GitHub profile

## Resources

- 📚 [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) - Deep dive into technology
- 📖 [EXAMPLES.md](EXAMPLES.md) - Usage examples and workflows
- 🎵 [README.md](README.md) - Quick start guide
- 🛠️ [CLAUDE.md](CLAUDE.md) - Development guidelines

---

**Music Style Transfer** - Transform audio into infinite creative possibilities. 🎵✨

**Repository**: https://github.com/Jakub-Syrek/MusicStyleTransfer
