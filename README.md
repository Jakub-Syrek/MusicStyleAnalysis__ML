# MusicStyleAnalysis__ML

Analyse musical style from audio and classify it into 45+ electronic genres with an adaptive Random Forest model.

![CI](https://github.com/Jakub-Syrek/MusicStyleAnalysis__ML/actions/workflows/tests.yml/badge.svg)
![Release](https://img.shields.io/github/v/release/Jakub-Syrek/MusicStyleAnalysis__ML)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/github/license/Jakub-Syrek/MusicStyleAnalysis__ML)
![Last commit](https://img.shields.io/github/last-commit/Jakub-Syrek/MusicStyleAnalysis__ML)

## Overview

MusicStyleAnalysis__ML extracts musical features from any audio source
(local file, HTTP URL or YouTube video), classifies the track against 45+
electronic-music genre profiles, and learns from user corrections through an
adaptive scikit-learn Random Forest classifier. It is a small, fast,
dependency-light tool intended for DJs, producers, music librarians and MIR
researchers.

## Features

- 4-D audio feature extraction: tempo, loudness, spectral centroid,
  zero-crossing rate, plus 13-coefficient MFCC.
- 45+ genre profiles organised into families (house, techno, trance,
  breakbeat, ambient, hardcore, rave, electronic, hip-hop, reggae,
  industrial, dance, jazz).
- Top-5 ranked genre classification with confidence scores and genre family
  display.
- Rhythm analysis: beat regularity, onset density, structural breaks,
  strong-rhythm flag.
- Adaptive ML model (`RandomForestClassifier`, 50 trees) that retrains from
  user corrections saved to `training_data.csv`.
- Multi-source audio loading: local files (WAV, MP3, FLAC, ...), direct
  HTTP/HTTPS URLs and YouTube videos.
- FFmpeg is bundled automatically via `static-ffmpeg` -- no system install
  required.

## Tech stack

- **Audio**: [librosa](https://librosa.org/) 0.10 for feature extraction and
  beat tracking, [soundfile](https://pysoundfile.readthedocs.io/) for I/O.
- **ML**: [scikit-learn](https://scikit-learn.org/) (`RandomForestClassifier`,
  `LabelEncoder`), [numpy](https://numpy.org/),
  [scipy](https://scipy.org/), [joblib](https://joblib.readthedocs.io/).
- **Loaders**: [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube,
  [static-ffmpeg](https://pypi.org/project/static-ffmpeg/) for a bundled
  FFmpeg binary, [requests](https://requests.readthedocs.io/) for HTTP.
- **Misc**: [music21](https://web.mit.edu/music21/) (reserved for future
  music-theory features), [python-dotenv](https://pypi.org/project/python-dotenv/)
  for optional API keys.

## Installation

```bash
# Clone repository
git clone https://github.com/Jakub-Syrek/MusicStyleAnalysis__ML.git
cd MusicStyleAnalysis__ML

# Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

For development (tests + coverage):

```bash
pip install -e ".[dev]"
```

## Usage examples

### Analyse audio

```bash
# Local audio file
python -m src analyze sample.wav

# Audio from HTTP URL
python -m src analyze https://example.com/music/song.mp3

# YouTube video (FFmpeg auto-downloaded on first use)
python -m src analyze https://youtube.com/watch?v=dQw4w9WgXcQ
```

Sample output:

```
[ANALYSIS] Musical Style Analysis
==================================================
Tempo: 170.5 BPM
Loudness: 0.272
Spectral Centroid: 2077.6 Hz
Zero Crossing Rate: 0.0954
MFCC (13 coefficients): ['-29.71', '59.95', '25.88', ...]
==================================================

[GENRE] Top 5 Genre Classification
==================================================
1. Drum & Bass                     50.2%  (Breakbeat)
2. Jungle                          49.3%  (Breakbeat)
3. Breakcore/Jungle                48.5%  (Breakbeat)
4. Ambient Breakbeat               46.7%  (Breakbeat)
5. Breakbeat                       46.3%  (Breakbeat)
==================================================

[RHYTHM] Beat & Rhythm Analysis
==================================================
Beat Regularity: 97.9% (how steady)
Onset Density: 0.134 (syncopation)
Strong Rhythm: Yes
Detected Breaks: 253 (silent sections)
==================================================
```

### Correct a genre and train

```bash
# Save a correction for future training
python -m src analyze sample.wav --correct breakcore

# Retrain the model once you have 3+ samples
python -m src train

# Show all genres with strength bars
python -m src analyze sample.wav --verbose
```

### Generate music (experimental)

```bash
python -m src generate reference.wav --duration 30 --output generated.wav
```

The `generate` command tries local MusicGen, then the Hugging Face API
(requires `HUGGINGFACE_API_KEY` in `.env`), then a synthetic-audio fallback.

## Project structure

```
MusicStyleAnalysis__ML/
├── src/
│   ├── __init__.py            # Package metadata (__version__)
│   ├── __main__.py            # CLI entry: analyze / train / generate
│   ├── audio_loader.py        # Local files, HTTP URLs, YouTube
│   ├── style_analyzer.py      # librosa feature extraction
│   ├── genre_database.py      # 45+ genre profiles + families
│   ├── genre_detector.py      # GenreDetector + RhythmAnalyzer
│   ├── ml_trainer.py          # Random Forest training + persistence
│   ├── api_client.py          # MusicGen / HF API / synthetic fallback
│   └── music_generator.py     # Generation orchestrator
├── tests/
│   ├── test_style_analyzer.py
│   ├── test_genre_detector.py
│   ├── test_genre_database.py
│   ├── test_ml_trainer.py
│   ├── test_audio_loader.py
│   └── test_api_client.py
├── .github/
│   ├── workflows/             # CI and auto-version
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
├── SECURITY.md
├── LICENSE
└── README.md
```

## Testing

```bash
# Run the full suite
pytest -q

# With coverage on the src package
pytest -q --cov=src --cov-report=term-missing
```

The suite mocks heavy I/O (`librosa.load`, network calls, file system writes)
so it runs in seconds and does not require real audio files or network
access.

## Versioning

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Commits use the [Conventional Commits](https://www.conventionalcommits.org/)
prefixes (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`, `refactor:`).
The `.github/workflows/version.yml` workflow inspects commits since the last
tag and bumps the version in `pyproject.toml` accordingly:

- `BREAKING CHANGE:` -> major
- `feat:` -> minor
- anything else -> patch

A matching Git tag and GitHub Release are created automatically.

See [CHANGELOG.md](CHANGELOG.md) for the human-readable history.

## License

Released under the [MIT License](LICENSE) (c) 2026 Jakub Syrek.

## Contact

- Author: **Jakub Syrek**
- Email: `jakubvonsyrek@gmail.com`
- Issues: <https://github.com/Jakub-Syrek/MusicStyleAnalysis__ML/issues>
- Security: see [SECURITY.md](SECURITY.md)
