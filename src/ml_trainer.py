"""
Machine learning model trainer for adaptive genre classification.

Trains on user feedback data and improves predictions over time.
"""

import os
import csv
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


class MLTrainer:
  """Train and manage ML models for genre classification."""

  def __init__(self, data_file: str = "training_data.csv",
               model_file: str = "genre_model.pkl"):
    """Initialize trainer with file paths."""
    self.data_file = Path(data_file)
    self.model_file = Path(model_file)
    self.model = None
    self.label_encoder = None
    self._load_model()

  def save_correction(self, features: Dict[str, Any],
                     correct_genre: str) -> None:
    """Save user correction to training data.

    Args:
      features: Audio features (tempo, loudness, spectral, zcr)
      correct_genre: Ground truth genre label
    """
    feature_row = [
      features.get("tempo", 0),
      features.get("loudness", 0),
      features.get("spectral_centroid", 0),
      features.get("zero_crossing_rate", 0),
      correct_genre
    ]

    file_exists = self.data_file.exists()
    with open(self.data_file, 'a', newline='') as f:
      writer = csv.writer(f)
      if not file_exists:
        writer.writerow([
          "tempo", "loudness", "spectral_centroid", "zcr", "genre"
        ])
      writer.writerow(feature_row)

  def train(self) -> bool:
    """Train model on collected feedback data.

    Returns:
      True if training succeeded, False otherwise
    """
    if not self.data_file.exists():
      return False

    try:
      data = []
      labels = []
      with open(self.data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
          data.append([
            float(row["tempo"]),
            float(row["loudness"]),
            float(row["spectral_centroid"]),
            float(row["zcr"])
          ])
          labels.append(row["genre"])

      if len(data) < 3:
        return False

      X = np.array(data)
      self.label_encoder = LabelEncoder()
      y = self.label_encoder.fit_transform(labels)

      self.model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
      )
      self.model.fit(X, y)
      self._save_model()
      return True

    except Exception as e:
      print(f"[ERROR] Training failed: {str(e)}")
      return False

  def predict(self, features: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Predict genre using trained model.

    Args:
      features: Audio features

    Returns:
      (genre, confidence) or None if model unavailable
    """
    if self.model is None or self.label_encoder is None:
      return None

    try:
      X = np.array([[
        features.get("tempo", 0),
        features.get("loudness", 0),
        features.get("spectral_centroid", 0),
        features.get("zero_crossing_rate", 0)
      ]])

      genre_idx = self.model.predict(X)[0]
      genre = self.label_encoder.inverse_transform([genre_idx])[0]
      confidence = float(np.max(self.model.predict_proba(X)))
      return (genre, confidence)

    except Exception:
      return None

  def _save_model(self) -> None:
    """Save trained model and encoder to disk."""
    with open(self.model_file, 'wb') as f:
      pickle.dump({
        'model': self.model,
        'encoder': self.label_encoder
      }, f)

  def _load_model(self) -> None:
    """Load trained model from disk if exists."""
    if not self.model_file.exists():
      return

    try:
      with open(self.model_file, 'rb') as f:
        data = pickle.load(f)
        self.model = data['model']
        self.label_encoder = data['encoder']
    except Exception:
      pass

  def get_training_size(self) -> int:
    """Get number of training samples collected."""
    if not self.data_file.exists():
      return 0

    with open(self.data_file, 'r') as f:
      return sum(1 for _ in f) - 1
