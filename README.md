# Music Analysis & Genre Classification

Analyze musical style from audio files and classify music genre based on comprehensive audio features.

## Features

- Extract musical features (tempo, loudness, spectral centroid, zero crossing rate, MFCC)
- Classify music into **40+ electronic and mainstream genres** (House, Techno, Trance, Drum & Bass, Breakcore, Ambient, Industrial, etc.)
- Return **top 5 genre matches** with confidence scores (probability rankings)
- Analyze rhythm and beat patterns (beat regularity, syncopation, structural breaks)
- Support for multiple audio formats (WAV, MP3, FLAC)

## Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/Jakub-Syrek/MusicStyleTransfer.git
cd MusicStyleTransfer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Analyze Musical Style

Extract features and classify genre from a reference audio file:

```bash
python -m src analyze sample.wav
```

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
1. Drum & Bass                     50.2%
2. Ambient Breakbeat               49.3%
3. Downbeat/Downtempo              46.7%
4. Jungle                          44.5%
5. IDM                             44.3%
==================================================

[RHYTHM] Beat & Rhythm Analysis
==================================================
Beat Regularity: 97.9% (how steady)
Onset Density: 0.134 (syncopation)
Strong Rhythm: Yes
Detected Breaks: 253 (silent sections)
==================================================
```

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
- [x] Comprehensive test coverage
- [ ] Spectral features (chroma, rolloff, flatness)
- [ ] Machine learning classifier with training capability
- [ ] Web UI for visualization

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

Follow code standards in CLAUDE.md.

## License

MIT
