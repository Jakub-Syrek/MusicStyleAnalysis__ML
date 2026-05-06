# Music Style Transfer - Project Directives

## Overview
MVP application for musical style analysis and automatic music generation based on reference styles. Uses existing AI/ML APIs and libraries to avoid custom model training.

## Technology Stack
- **Language**: Python 3.10+
- **Audio Processing**:
  - `librosa` - audio feature extraction and analysis
  - `music21` - music theory and analysis
  - `essentia` - advanced audio descriptors
- **API Clients**:
  - `requests` - HTTP client for external APIs
  - `python-dotenv` - environment configuration
- **Optional**: FastAPI for backend if expanding beyond CLI

## Project Structure
```
MusicStyleTransfer/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── style_analyzer.py      # Extract musical features from reference audio
│   ├── music_generator.py     # Generate music based on analyzed style
│   └── api_client.py          # Handle calls to music generation APIs
├── examples/                  # Sample input files (small)
└── tests/
```

## Development Approach
**Phase 1 (MVP)**: CLI tool that takes reference audio → extracts style features → generates new audio via API

**Key Decisions**:
- Use existing music generation models (MusicGen, etc) - no custom ML training
- Focus on feature extraction and API integration
- Simple CLI interface first, Web UI later if needed
- Keep models/APIs swappable - modular design

## Git Standards
- Commits: Descriptive, imperative form ("Add style analyzer", "Fix MIDI parsing")
- Author: Jakub Syrek <jakubvonsyrek@gmail.com>
- Test before every commit
- Push immediately after merge to main
- No long-running branches

## Code Standards
- English only (code, comments, commits)
- JSDoc for all public functions (@param, @returns)
- SOLID principles, max 30 lines per function
- Dependency Injection pattern
- Comprehensive error handling (try/catch)
- ES6+ async/await, no callbacks
- No var, use const/let
- No inline scripts (CSP compliant)
- Meaningful variable names

## Quality Checklist Before Commit
✓ English code and comments
✓ JSDoc documentation
✓ SOLID principles applied
✓ Error handling present
✓ No console errors
✓ Tested locally
✓ Descriptive commit message
✓ Author: Jakub Syrek only
