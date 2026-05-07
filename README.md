# Music Analysis & Genre Classification

Analyze musical style from audio files and classify music genre based on comprehensive audio features.

## Features

- Extract musical features (tempo, loudness, spectral centroid, zero crossing rate, MFCC)
- Classify music into **40+ electronic and mainstream genres** (House, Techno, Trance, Drum & Bass, Breakcore, Ambient, Industrial, etc.)
- Return **top 5 genre matches** with confidence scores and genre families
- Analyze rhythm and beat patterns (beat regularity, syncopation, structural breaks)
- **Adaptive ML model** that learns from user feedback and improves predictions
- Support for multiple audio sources:
  - Local audio files (WAV, MP3, FLAC, etc.)
  - Direct HTTP/HTTPS URLs
  - YouTube videos (automatic extraction)

## Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/Jakub-Syrek/MusicStyleAnalysisML.git
cd MusicStyleAnalysisML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Analyze Musical Style

Extract features and classify genre from audio sources:

```bash
# Local audio file
python -m src analyze sample.wav

# Audio from HTTP URL
python -m src analyze https://example.com/music/song.mp3

# YouTube video (FFmpeg bundled via static-ffmpeg)
python -m src analyze https://youtube.com/watch?v=dQw4w9WgXcQ
```

FFmpeg is bundled automatically via `static-ffmpeg` (included in `requirements.txt`). No system install required—the binary is downloaded on first YouTube use and cached in the virtualenv.

Output:
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

#### Correct Genre & Train ML Model

Provide feedback to improve predictions:

```bash
# Analyze and correct genre
python -m src analyze sample.wav --correct breakcore

# Train model on collected feedback (requires 3+ samples)
python -m src train

# Show detailed analysis of all genres
python -m src analyze sample.wav --verbose
```

The ML model uses **Random Forest Classification** trained on collected corrections. Each correction is saved to `training_data.csv` and integrated into predictions after retraining.

#### Run Tests
```bash
pytest tests/
pytest tests/test_style_analyzer.py -v
pytest tests/test_genre_detector.py -v
```

## Project Status

- [x] Audio feature extraction (tempo, loudness, spectral features, MFCC)
- [x] Rhythm analysis (beat tracking, break detection, syncopation)
- [x] Comprehensive genre classification (40+ genres with top 5 ranking)
- [x] GenreFamily architecture for organized subgenres
- [x] Multi-dimensional feature matching (tempo, loudness, spectral, ZCR)
- [x] Confidence scoring and probability ranking
- [x] CLI interface (analyze command with top 5 output)
- [x] Adaptive ML model with feedback training (Random Forest, sklearn)
- [x] User feedback system and model retraining
- [x] Verbose mode with visual strength indicators
- [x] Multi-source audio loading (local files, HTTP URLs, YouTube)
- [ ] Spectral features (chroma, rolloff, flatness)
- [ ] Web UI for visualization
- [ ] Real-time analysis streaming

## Architecture

### Style Analyzer
Extracts key musical features:
- Tempo and beat information (via librosa.beat.tempo)
- Loudness (RMS)
- Spectral characteristics (centroid, rolloff)
- Zero crossing rate (noisiness indicator)
- Mel-frequency cepstral coefficients (MFCC)

### Genre Detector
Classifies music into genres using multi-dimensional feature matching:
- Tempo ranges and confidence scoring
- Loudness characteristics for each genre
- Spectral profile matching
- Zero crossing rate patterns
- Weighted scoring system (tempo 30%, spectral 30%, loudness 20%, ZCR 20%)

### Rhythm Analyzer
Analyzes beat and rhythm characteristics:
- Beat tracking and regularity
- Onset detection (syncopation)
- Break/silence detection
- Rhythm strength estimation

## Supported Genres (40+)

**Electronic Base Genres:**
- Acid (130 BPM)
- Ambient (40-90 BPM)
- Breakbeat (140-180 BPM)
- Breakcore/Jungle (160-200 BPM)
- Club (124-130 BPM)
- Dance/EDM (115-150 BPM)
- Dub (80-110 BPM)
- Dubstep (135-150 BPM)
- Electro (110-140 BPM)
- Electronic (100-160 BPM)
- House (115-135 BPM)
- Minimal Techno (120-140 BPM)
- Techno (120-150 BPM)
- Trance (130-150 BPM)
- Trip Hop (85-120 BPM)

**House Subgenres:**
- Acid House, Ambient House, Deep House, Future House, Tech House, Progressive House

**Trance Subgenres:**
- Goa Trance, Dark Trance, Hard Trance, Psytrance, Progressive Trance, Minimal Trance

**Breakbeat Subgenres:**
- Drum & Bass (160-180 BPM)
- Jungle (155-180 BPM)
- Hardcore, Happy Hardcore, Gabber (160-200 BPM)

**Other Styles:**
- Industrial, IDM, Downtempo, Isolationism, Detroit, Dub, Trip Hop, Techstep

## Development

Run tests before committing:
```bash
pytest tests/
```

## License

MIT
