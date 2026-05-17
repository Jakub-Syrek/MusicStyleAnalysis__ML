# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-05-17

### Added

- `StyleAnalyzer` extracting tempo, loudness, spectral centroid, zero-crossing
  rate and 13-coefficient MFCC features via `librosa`.
- `GenreDetector` with 45+ electronic-music genre profiles organised into
  families (house, techno, trance, breakbeat, ambient, hardcore, rave, etc.).
- Top-5 ranked classification with weighted 4-D scoring
  (tempo 30%, spectral 30%, loudness 20%, ZCR 20%).
- `RhythmAnalyzer` reporting beat regularity, onset density, break detection
  and strong-rhythm flag.
- Adaptive `MLTrainer` (scikit-learn `RandomForestClassifier`, 50 trees,
  Gini split) that learns from user corrections stored in
  `training_data.csv` and persists models as `genre_model.pkl`.
- `AudioLoader` supporting local files, direct HTTP/HTTPS URLs and YouTube
  videos (via `yt-dlp` with `static-ffmpeg`-bundled FFmpeg).
- `MusicGenerator` orchestrator and `MusicGenerationClient` with local
  MusicGen, Hugging Face inference API and synthetic-audio fallback.
- CLI entry point `python -m src` exposing `analyze`, `train` and `generate`
  subcommands.
- Pytest test suite with mocked audio I/O for fast, deterministic runs.

### Changed

- Project renamed to **MusicStyleAnalysis__ML**.

### Documentation

- README with shields, installation, usage, project structure and testing.
- ABOUT, EXAMPLES and TECHNICAL_OVERVIEW documents.
- SECURITY policy for private vulnerability reports.
- MIT License (2026, Jakub Syrek).

[Unreleased]: https://github.com/Jakub-Syrek/MusicStyleAnalysis__ML/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Jakub-Syrek/MusicStyleAnalysis__ML/releases/tag/v1.0.0
