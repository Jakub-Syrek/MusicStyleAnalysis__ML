"""Unit tests for :mod:`src.genre_detector`."""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from src.genre_detector import GenreDetector, GenreFamily, RhythmAnalyzer


class TestGenreFamily(unittest.TestCase):
    """Exercise :class:`GenreFamily` scoring rules."""

    def setUp(self) -> None:
        """Build a minimal in-memory family fixture."""
        self.family = GenreFamily(
            name="test",
            genres={
                "house_like": {
                    "tempo_range": (115, 135),
                    "loudness_range": (0.3, 0.6),
                    "spectral_centroid_range": (2000, 4000),
                    "zcr_range": (0.06, 0.13),
                },
                "ambient_like": {
                    "tempo_range": (40, 90),
                    "loudness_range": (0.05, 0.3),
                    "spectral_centroid_range": (1000, 2500),
                    "zcr_range": (0.01, 0.05),
                },
            },
        )

    def test_match_score_returns_positive_for_central_values(self) -> None:
        """A track exactly at the centre of the house ranges scores > 0."""
        scores = self.family.match_score(
            tempo=125.0,
            loudness=0.45,
            spectral=3000.0,
            zcr=0.095,
        )
        self.assertIn("house_like", scores)
        self.assertGreater(scores["house_like"], 0.0)

    def test_match_score_drops_out_of_range_features(self) -> None:
        """Features outside every range yield no matches."""
        scores = self.family.match_score(
            tempo=300.0,
            loudness=2.0,
            spectral=20_000.0,
            zcr=1.0,
        )
        self.assertEqual(scores, {})

    def test_score_is_bounded_by_total_weights(self) -> None:
        """The weighted score must not exceed 1.0 for any input."""
        scores = self.family.match_score(
            tempo=125.0,
            loudness=0.45,
            spectral=3000.0,
            zcr=0.095,
        )
        for value in scores.values():
            self.assertLessEqual(value, 1.0)
            self.assertGreaterEqual(value, 0.0)


class TestGenreDetector(unittest.TestCase):
    """Behaviour tests for :class:`GenreDetector`."""

    def setUp(self) -> None:
        """Instantiate the detector once per test."""
        self.detector = GenreDetector()

    def test_classify_top5_returns_at_most_five_entries(self) -> None:
        """``classify_top5`` truncates the ranking to five rows."""
        features = {
            "tempo": 125.0,
            "loudness": 0.45,
            "spectral_centroid": 3000.0,
            "zero_crossing_rate": 0.09,
        }
        ranking = self.detector.classify_top5(features)
        self.assertLessEqual(len(ranking), 5)
        self.assertGreater(len(ranking), 0)

    def test_classify_top5_is_sorted_descending(self) -> None:
        """The ranking is sorted by confidence in descending order."""
        features = {
            "tempo": 125.0,
            "loudness": 0.45,
            "spectral_centroid": 3000.0,
            "zero_crossing_rate": 0.09,
        }
        ranking = self.detector.classify_top5(features)
        confidences = [conf for _, conf in ranking]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_classify_top5_falls_back_to_unknown(self) -> None:
        """When nothing matches the detector reports ``unknown``."""
        features = {
            "tempo": 9_999.0,
            "loudness": 9_999.0,
            "spectral_centroid": 9_999.0,
            "zero_crossing_rate": 9_999.0,
        }
        ranking = self.detector.classify_top5(features)
        self.assertEqual(ranking, [("unknown", 0.0)])

    def test_get_genre_description_unknown_returns_unknown(self) -> None:
        """Unknown labels must return the literal ``"Unknown"``."""
        self.assertEqual(
            self.detector.get_genre_description("definitely_not_a_genre"),
            "Unknown",
        )

    def test_get_genre_description_known_returns_string(self) -> None:
        """Known labels return a non-empty description."""
        description = self.detector.get_genre_description("house")
        self.assertIsInstance(description, str)
        self.assertTrue(description.strip())


class TestRhythmAnalyzer(unittest.TestCase):
    """Tests for :class:`RhythmAnalyzer` with mocked ``librosa`` calls."""

    @patch("src.genre_detector.librosa.onset.onset_detect")
    @patch("src.genre_detector.librosa.beat.beat_track")
    @patch("src.genre_detector.librosa.onset.onset_strength")
    def test_analyze_rhythm_returns_expected_keys(
        self,
        mock_onset_strength: MagicMock,
        mock_beat_track: MagicMock,
        mock_onset_detect: MagicMock,
    ) -> None:
        """``analyze_rhythm`` returns the documented schema."""
        mock_onset_strength.return_value = np.ones(200)
        mock_beat_track.return_value = (
            128.0,
            np.array([10, 20, 30, 40, 50, 60]),
        )
        mock_onset_detect.return_value = np.array([10, 30, 50])

        analyzer = RhythmAnalyzer(sample_rate=16_000)
        result = analyzer.analyze_rhythm(np.zeros(16_000), 16_000)

        for key in (
            "tempo",
            "beat_regularity",
            "onset_density",
            "num_breaks",
            "has_strong_rhythm",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["tempo"], 128.0)
        self.assertGreaterEqual(result["beat_regularity"], 0.0)
        self.assertLessEqual(result["beat_regularity"], 1.0)


if __name__ == "__main__":
    unittest.main()
