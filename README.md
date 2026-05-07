# Music Analysis & Genre Classification

Analyze musical style from audio files and classify music genre based on comprehensive audio features.

## Features

- Extract musical features (tempo, loudness, spectral centroid, zero crossing rate, MFCC)
- Classify music into 10 genres: Classical, Ambient, Jazz, Blues, Rock, Pop, Hip-Hop, Electronic, Dance, Breakcore
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

[GENRE] Classification
==================================================
Genre: Electronic/Synth
Confidence: 78.5%
==================================================

[RHYTHM] Beat & Rhythm Analysis
==================================================
Beat Regularity: 97.9% (how steady)
Onset Density: 0.134 (syncopation)
Strong Rhythm: Yes
Detected Breaks: 5 (silent sections)
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
- [x] Basic rhythm analysis (beat tracking, break detection)
- [x] Improved genre classification (10 genres with multi-feature matching)
- [x] CLI interface (analyze command)
- [x] Comprehensive test coverage
- [ ] Spectral features (chroma, rolloff, flatness)
- [ ] Support for more genres (Techno, House, Drum & Bass, etc.)
- [ ] Machine learning genre classifier (optional future enhancement)

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

## Supported Genres

1. **Classical/Orchestral** - Tempo: 60-120 BPM, orchestral textures
2. **Ambient/Atmospheric** - Tempo: 40-90 BPM, subtle, peaceful
3. **Jazz** - Tempo: 70-130 BPM, syncopated rhythms
4. **Blues** - Tempo: 60-110 BPM, soulful character
5. **Rock** - Tempo: 90-150 BPM, high energy
6. **Pop** - Tempo: 85-135 BPM, radio-friendly
7. **Hip-Hop/Rap** - Tempo: 80-120 BPM, rhythmic focus
8. **Electronic/Synth** - Tempo: 100-160 BPM, synthetic sounds
9. **Dance/EDM** - Tempo: 115-150 BPM, energetic, regular beats
10. **Breakcore/Jungle** - Tempo: 160-200+ BPM, chaotic syncopation, fast breaks

## Development

Run tests before committing:
```bash
pytest tests/
```

Follow code standards in CLAUDE.md.

## License

MIT
