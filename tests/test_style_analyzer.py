"""
Unit tests for StyleAnalyzer module.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from src.style_analyzer import StyleAnalyzer


class TestStyleAnalyzer(unittest.TestCase):
  """Test cases for StyleAnalyzer class."""

  def setUp(self) -> None:
    """
    Set up test fixtures.
    """
    self.analyzer = StyleAnalyzer(sample_rate=16000)

  def test_initialization(self) -> None:
    """
    Test StyleAnalyzer initialization.
    """
    self.assertEqual(self.analyzer.sample_rate, 16000)

  @patch("src.style_analyzer.librosa.feature.mfcc")
  @patch("src.style_analyzer.librosa.feature.zero_crossing_rate")
  @patch("src.style_analyzer.librosa.feature.spectral_centroid")
  @patch("src.style_analyzer.librosa.beat.tempo")
  @patch("src.style_analyzer.librosa.onset.onset_strength")
  @patch("src.style_analyzer.librosa.feature.rms")
  @patch("src.style_analyzer.librosa.load")
  def test_analyze_success(
    self,
    mock_load: MagicMock,
    mock_rms: MagicMock,
    mock_onset: MagicMock,
    mock_tempo: MagicMock,
    mock_spec: MagicMock,
    mock_zcr: MagicMock,
    mock_mfcc: MagicMock
  ) -> None:
    """
    Test successful audio analysis.

    @param {MagicMock} mock_load - Mocked librosa.load function
    @param {MagicMock} mock_rms - Mocked librosa.feature.rms
    @param {MagicMock} mock_onset - Mocked librosa.onset.onset_strength
    @param {MagicMock} mock_tempo - Mocked librosa.beat.tempo
    @param {MagicMock} mock_spec - Mocked librosa.feature.spectral_centroid
    @param {MagicMock} mock_zcr - Mocked librosa.feature.zero_crossing_rate
    @param {MagicMock} mock_mfcc - Mocked librosa.feature.mfcc
    """
    mock_y = np.random.randn(16000)
    mock_sr = 16000
    mock_load.return_value = (mock_y, mock_sr)
    mock_rms.return_value = np.array([[0.5]])
    mock_onset.return_value = np.random.randn(100)
    mock_tempo.return_value = 120.0
    mock_spec.return_value = np.array([[2000.0]])
    mock_zcr.return_value = np.array([[0.02]])
    mock_mfcc.return_value = np.random.randn(13, 100)

    features = self.analyzer.analyze("test_audio.wav")

    self.assertIn("tempo", features)
    self.assertIn("loudness", features)
    self.assertIn("spectral_centroid", features)
    self.assertIn("zero_crossing_rate", features)
    self.assertIn("mfcc", features)

  @patch("src.style_analyzer.librosa.load")
  def test_analyze_missing_file(self, mock_load: MagicMock) -> None:
    """
    Test analysis with missing file.

    @param {MagicMock} mock_load - Mocked librosa.load function
    """
    mock_load.side_effect = FileNotFoundError("File not found")

    with self.assertRaises(ValueError):
      self.analyzer.analyze("nonexistent.wav")

  @patch("src.style_analyzer.librosa.feature.rms")
  @patch("src.style_analyzer.librosa.load")
  def test_extract_loudness(
    self,
    mock_load: MagicMock,
    mock_rms: MagicMock
  ) -> None:
    """
    Test loudness extraction.

    @param {MagicMock} mock_load - Mocked librosa.load
    @param {MagicMock} mock_rms - Mocked librosa.feature.rms
    """
    mock_y = np.random.randn(16000)
    mock_rms.return_value = np.array([[0.5, 0.6, 0.7]])

    loudness = self.analyzer._extract_loudness(mock_y)

    self.assertAlmostEqual(loudness, 0.6)

  @patch("src.style_analyzer.librosa.beat.tempo")
  @patch("src.style_analyzer.librosa.onset.onset_strength")
  @patch("src.style_analyzer.librosa.load")
  def test_extract_tempo(
    self,
    mock_load: MagicMock,
    mock_onset: MagicMock,
    mock_tempo: MagicMock
  ) -> None:
    """
    Test tempo extraction.

    @param {MagicMock} mock_load - Mocked librosa.load
    @param {MagicMock} mock_onset - Mocked librosa.onset.onset_strength
    @param {MagicMock} mock_tempo - Mocked librosa.beat.tempo
    """
    mock_y = np.random.randn(16000)
    mock_onset.return_value = np.random.randn(100)
    mock_tempo.return_value = 120.0

    tempo = self.analyzer._extract_tempo(mock_y, 16000)

    self.assertEqual(tempo, 120.0)


if __name__ == "__main__":
  unittest.main()
