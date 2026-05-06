# Music Style Transfer

Analyze musical style from reference audio and generate new music with similar characteristics.

## Features

- Extract musical features (tempo, key, instrumentation, mood) from reference audio
- Generate new music based on analyzed style using state-of-the-art AI models
- Support for multiple audio formats (WAV, MP3, FLAC)
- Modular design for easy API/model swapping

## Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/MusicStyleTransfer.git
cd MusicStyleTransfer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API credentials
```

### Usage

#### Analyze Musical Style
Extract features from a reference audio file:

```bash
python -m src analyze reference_track.mp3
```

Output:
```
📊 Musical Style Analysis
==================================================
Tempo: 128.5 BPM
Loudness: 0.456
Spectral Centroid: 2048.3 Hz
Zero Crossing Rate: 0.0145
MFCC (13 coefficients): [12.34, 5.67, ...]
==================================================
```

#### Generate Music
Create new music based on a reference style:

```bash
# Basic usage
python -m src generate reference_track.mp3

# Custom duration and output
python -m src generate reference_track.mp3 --duration 60 --output my_music.wav

# Specify API provider
python -m src generate reference_track.mp3 --provider huggingface
```

#### Run Tests
```bash
pytest tests/
pytest tests/test_style_analyzer.py -v
pytest tests/test_api_client.py -v
```

## Project Status

- [x] Project scaffold with CLAUDE.md directives
- [x] Style analyzer implementation (tempo, loudness, spectral features, MFCC)
- [x] Music generator orchestrator
- [x] API client for Hugging Face integration
- [x] CLI interface (analyze, generate commands)
- [x] Unit tests with mocked dependencies
- [ ] Real audio file testing
- [ ] Web UI/API wrapper
- [ ] Documentation and examples

## Architecture

### Style Analyzer
Extracts key musical features:
- Tempo and beat information
- Key and harmonic content
- Instrumentation
- Loudness and dynamics
- Genre/mood indicators

### Music Generator
Generates audio using:
- Hugging Face MusicGen model (default)
- Fallback options: Jukebox, other open-source models
- API-based generation for production use

## API Integrations

Currently configured for:
- **Hugging Face Inference API** - MusicGen, other open-source models
- **Optional**: Google MusicLM, OpenAI (if credentials provided)

## Development

Run tests before committing:
```bash
pytest tests/
```

Follow code standards in CLAUDE.md.

## License

MIT
