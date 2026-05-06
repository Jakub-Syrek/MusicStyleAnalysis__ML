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

```bash
# Analyze a reference audio file
python -m src.style_analyzer --input reference_track.mp3

# Generate music based on reference style
python -m src.music_generator --reference reference_track.mp3 --duration 30

# Full pipeline
python -m src.main --reference reference_track.mp3 --output generated_music.wav
```

## Project Status

- [x] Project scaffold
- [ ] Style analyzer implementation
- [ ] Music generator integration
- [ ] CLI interface
- [ ] Testing suite

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
