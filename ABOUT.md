# About MusicStyleAnalysis__ML

MusicStyleAnalysis__ML is a lightweight Python tool that extracts musical
features from any audio source (local file, HTTP URL or YouTube video) and
classifies the track against 45+ electronic-music genre profiles. It pairs a
deterministic rule-based scorer (4-D weighted matching on tempo, spectral
centroid, loudness and zero-crossing rate) with an adaptive scikit-learn
Random Forest classifier that retrains on user corrections.

Under the hood it relies on `librosa` for tempo, beat tracking and MFCC
extraction, `scikit-learn` for the ensemble model, and `yt-dlp` plus a
bundled `static-ffmpeg` binary for YouTube ingestion -- so no system-wide
FFmpeg install is required. A `RhythmAnalyzer` complements the genre
prediction with beat regularity, onset density and structural-break
detection, useful for DJs and music librarians.

The project ships a small CLI (`python -m src analyze | train | generate`),
a fully mocked pytest suite that runs in seconds, semantic-versioned
releases driven by Conventional Commits, and CI on every push and pull
request. See [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) for the
architectural deep dive and [EXAMPLES.md](EXAMPLES.md) for end-to-end
walkthroughs.
