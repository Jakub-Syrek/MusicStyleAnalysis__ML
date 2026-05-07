"""
Detect music genre and rhythm characteristics from audio features.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import librosa
from src.genre_database import GENRE_DATABASE


class GenreFamily:
  """Base class for music genre families."""

  def __init__(self, name: str, genres: Dict[str, Dict]):
    """Initialize genre family with genres."""
    self.name = name
    self.genres = genres

  def match_score(
    self,
    tempo: float,
    loudness: float,
    spectral: float,
    zcr: float
  ) -> Dict[str, float]:
    """Calculate match score for each genre in family."""
    scores = {}
    for genre_name, criteria in self.genres.items():
      score = self._calculate_score(
        tempo, loudness, spectral, zcr, criteria
      )
      if score > 0:
        scores[genre_name] = score
    return scores

  def _calculate_score(
    self,
    tempo: float,
    loudness: float,
    spectral: float,
    zcr: float,
    criteria: Dict
  ) -> float:
    """Calculate genre match score."""
    score = 0.0
    weights = {
      "tempo_range": 0.3,
      "loudness_range": 0.2,
      "spectral_centroid_range": 0.3,
      "zcr_range": 0.2
    }

    tempo_min, tempo_max = criteria["tempo_range"]
    if tempo_min <= tempo <= tempo_max:
      center = (tempo_min + tempo_max) / 2
      range_size = (tempo_max - tempo_min) / 2 + 1e-6
      tempo_score = 1.0 - abs(tempo - center) / range_size
      score += weights["tempo_range"] * max(0, tempo_score)

    loud_min, loud_max = criteria["loudness_range"]
    if loud_min <= loudness <= loud_max:
      center = (loud_min + loud_max) / 2
      range_size = (loud_max - loud_min) / 2 + 1e-6
      loud_score = 1.0 - abs(loudness - center) / range_size
      score += weights["loudness_range"] * max(0, loud_score)

    spec_min, spec_max = criteria["spectral_centroid_range"]
    if spec_min <= spectral <= spec_max:
      center = (spec_min + spec_max) / 2
      range_size = (spec_max - spec_min) / 2 + 1e-6
      spec_score = 1.0 - abs(spectral - center) / range_size
      score += weights["spectral_centroid_range"] * max(0, spec_score)

    zcr_min, zcr_max = criteria["zcr_range"]
    if zcr_min <= zcr <= zcr_max:
      center = (zcr_min + zcr_max) / 2
      range_size = (zcr_max - zcr_min) / 2 + 1e-6
      zcr_score = 1.0 - abs(zcr - center) / range_size
      score += weights["zcr_range"] * max(0, zcr_score)

    return score


class GenreDetector:
  """Classify music genre and return top 5 matches."""

  def __init__(self):
    """Initialize genre detector with families."""
    self._build_families()

  def _build_families(self):
    """Organize genres into families."""
    families_dict = {}
    for genre_name, criteria in GENRE_DATABASE.items():
      family = criteria["family"]
      if family not in families_dict:
        families_dict[family] = {}
      families_dict[family][genre_name] = criteria

    self.families = {
      name: GenreFamily(name, genres)
      for name, genres in families_dict.items()
    }

  def classify_top5(self, features: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Classify genre and return top 5 matches.

    Args:
      features: Audio features from StyleAnalyzer

    Returns:
      List of (genre_name, confidence_score) sorted by confidence
    """
    tempo = features.get("tempo", 100)
    loudness = features.get("loudness", 0.3)
    spectral = features.get("spectral_centroid", 2000)
    zcr = features.get("zero_crossing_rate", 0.05)

    all_scores = {}
    for family in self.families.values():
      scores = family.match_score(tempo, loudness, spectral, zcr)
      all_scores.update(scores)

    if not all_scores:
      return [("unknown", 0.0)]

    sorted_genres = sorted(
      all_scores.items(),
      key=lambda x: x[1],
      reverse=True
    )

    top5 = sorted_genres[:5]
    return [
      (name, min(score / 1.0, 1.0))
      for name, score in top5
    ]

  def classify_all(self, features: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Classify all genres with scores.

    Args:
      features: Audio features from StyleAnalyzer

    Returns:
      List of all (genre_name, confidence_score) sorted by confidence
    """
    tempo = features.get("tempo", 100)
    loudness = features.get("loudness", 0.3)
    spectral = features.get("spectral_centroid", 2000)
    zcr = features.get("zero_crossing_rate", 0.05)

    all_scores = {}
    for family in self.families.values():
      scores = family.match_score(tempo, loudness, spectral, zcr)
      all_scores.update(scores)

    sorted_genres = sorted(
      all_scores.items(),
      key=lambda x: x[1],
      reverse=True
    )

    return [
      (name, min(score / 1.0, 1.0))
      for name, score in sorted_genres
    ]

  def get_genre_description(self, genre: str) -> str:
    """Get human-readable genre description."""
    if genre in GENRE_DATABASE:
      return GENRE_DATABASE[genre]["description"]
    return "Unknown"


class RhythmAnalyzer:
  """Analyze rhythm, beat patterns, and structural breaks in audio."""

  def __init__(self, sample_rate: int = 16000):
    """Initialize rhythm analyzer."""
    self.sample_rate = sample_rate

  def analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze rhythm and beat characteristics.

    Args:
      y: Audio time series
      sr: Sample rate

    Returns:
      Dictionary with rhythm analysis results
    """
    try:
      onset_env = librosa.onset.onset_strength(y=y, sr=sr)
      tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr
      )

      if len(beats) > 1:
        beat_intervals = np.diff(beats)
        beat_regularity = 1.0 - (np.std(beat_intervals) / (
          np.mean(beat_intervals) + 1e-6
        ))
      else:
        beat_regularity = 0.0

      breaks = self._detect_breaks(onset_env)

      onset_density = len(librosa.onset.onset_detect(
        onset_envelope=onset_env,
        units='frames'
      )) / max(len(onset_env), 1)

      return {
        "tempo": float(tempo),
        "beat_regularity": max(0, min(1.0, beat_regularity)),
        "onset_density": float(onset_density),
        "num_breaks": len(breaks),
        "has_strong_rhythm": beat_regularity > 0.7,
      }
    except Exception as error:
      raise ValueError(f"Rhythm analysis failed: {str(error)}")

  def _detect_breaks(
    self,
    onset_env: np.ndarray,
    threshold: float = 0.1
  ) -> List[int]:
    """Detect breaks (silent or low-energy sections)."""
    onset_env_norm = onset_env / (np.max(onset_env) + 1e-6)
    breaks = np.where(onset_env_norm < threshold)[0]

    if len(breaks) == 0:
      return []

    break_groups = []
    current_group = [breaks[0]]

    for b in breaks[1:]:
      if b - current_group[-1] <= 1:
        current_group.append(b)
      else:
        if len(current_group) > 5:
          break_groups.append(len(current_group))
        current_group = [b]

    return break_groups
