# Music Style Transfer - Examples & Development Guide

## Quick Start Examples

### 1. Analyzing Audio Style

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Analyze a reference track
python -m src analyze sample.wav
```

**Expected output:**
```
📊 Musical Style Analysis
==================================================
Tempo: 95.3 BPM
Loudness: 0.523
Spectral Centroid: 3125.7 Hz
Zero Crossing Rate: 0.0234
MFCC (13 coefficients): ...
==================================================
```

### 2. Generating Music

#### With Hugging Face API
```bash
# Set API key in .env
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx

# Generate 30 seconds of music matching reference style
python -m src generate sample.wav --duration 30 --output generated.wav
```

#### With Different Providers
```bash
# Google MusicLM (if configured)
python -m src generate sample.wav --provider google

# Custom output location
python -m src generate sample.wav --output ./results/my_music.wav
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_style_analyzer.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src
```

### Adding New Features

1. **New analyzer feature:**
   - Add method to `StyleAnalyzer` class
   - Write unit test in `test_style_analyzer.py`
   - Update CLI if needed

2. **New API provider:**
   - Add method to `MusicGenerationClient` (e.g., `_generate_google()`)
   - Handle in `generate()` method
   - Add tests in `test_api_client.py`
   - Update `--provider` choices in `__main__.py`

3. **Example:**
   ```python
   class StyleAnalyzer:
       def extract_key(self, y: np.ndarray, sr: int) -> str:
           """
           Detect musical key.

           @param {array} y - Audio time series
           @param {number} sr - Sample rate
           @returns {string} Detected key (C, D, E, etc)
           """
           # Implementation
           return "C"
   ```

### Code Standards

Professional development standards:
- ✓ English only (code, comments, commits)
- ✓ JSDoc for all public functions
- ✓ SOLID principles (max 30 lines per function)
- ✓ Dependency Injection
- ✓ Comprehensive error handling
- ✓ ES6+ / Python 3.10+ async/await

### Git Workflow

```bash
# Before committing
pytest tests/          # Run all tests
python -m src analyze samples/test.mp3  # Manual testing

# Commit with descriptive message
git add -A
git commit -m "Add key detection feature to StyleAnalyzer

- Implement Krumhansl-Schmuckler algorithm
- Add unit tests with mocked audio
- Update CLI help text"

# Push
git push
```

## Troubleshooting

### "API key not found"
```bash
# Make sure .env file exists in project root
cp .env.example .env
# Edit .env with actual API key
```

### "librosa could not load audio"
```bash
# Ensure audio file format is supported (WAV, MP3, FLAC)
# On Windows, you may need ffmpeg for MP3 support
pip install ffmpeg-python
```

### Tests failing with import errors
```bash
# Reinstall in development mode
pip install -e .
pytest tests/
```

## API Integration Notes

### Hugging Face MusicGen
- Free tier: Limited requests per month
- Model: `facebook/musicgen-medium` (default)
- Supports: Prompt-based generation
- Max duration: ~30 seconds per request

### Google MusicLM (if implemented)
- Requires Google Cloud API key
- Better quality but slower
- Different prompt format

### Adding New Providers

1. Update `.env.example` with new credentials
2. Add method in `MusicGenerationClient`
3. Handle in `generate()` method's switch logic
4. Update CLI `--provider` choices
5. Add tests

## File Structure Reference

```
src/
├── __init__.py           # Package initialization
├── __main__.py           # CLI entry point
├── style_analyzer.py     # Audio feature extraction
├── music_generator.py    # Orchestrator
└── api_client.py         # API integration

tests/
├── __init__.py
├── test_style_analyzer.py
└── test_api_client.py
```

## Next Steps

- [ ] Implement key/scale detection
- [ ] Add genre classification
- [ ] Support for MIDI output
- [ ] Web UI with Flask/FastAPI
- [ ] Batch processing for multiple files
- [ ] Audio visualization
