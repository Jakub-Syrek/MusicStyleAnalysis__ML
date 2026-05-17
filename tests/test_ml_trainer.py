"""Unit tests for :mod:`src.ml_trainer`."""

import os
import tempfile
import unittest

from src.ml_trainer import MLTrainer


SAMPLE_FEATURES = {
    "tempo": 128.0,
    "loudness": 0.45,
    "spectral_centroid": 2500.0,
    "zero_crossing_rate": 0.09,
}


class TestMLTrainer(unittest.TestCase):
    """Verify training-data persistence and predict/train round-trips."""

    def setUp(self) -> None:
        """Create an isolated temp directory for every test run."""
        self.tmp = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.tmp.name, "training_data.csv")
        self.model_path = os.path.join(self.tmp.name, "genre_model.pkl")
        self.trainer = MLTrainer(
            data_file=self.data_path,
            model_file=self.model_path,
        )

    def tearDown(self) -> None:
        """Remove the temp directory after each test."""
        self.tmp.cleanup()

    def test_get_training_size_empty(self) -> None:
        """A fresh trainer reports zero collected samples."""
        self.assertEqual(self.trainer.get_training_size(), 0)

    def test_save_correction_writes_header_and_row(self) -> None:
        """``save_correction`` creates the CSV with header and one data row."""
        self.trainer.save_correction(SAMPLE_FEATURES, "house")

        self.assertTrue(os.path.exists(self.data_path))
        with open(self.data_path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh.readlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            lines[0],
            "tempo,loudness,spectral_centroid,zcr,genre",
        )
        self.assertTrue(lines[1].endswith(",house"))

    def test_save_correction_appends_without_duplicating_header(self) -> None:
        """Subsequent corrections append rows without rewriting the header."""
        self.trainer.save_correction(SAMPLE_FEATURES, "house")
        self.trainer.save_correction(SAMPLE_FEATURES, "techno")

        with open(self.data_path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh.readlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(self.trainer.get_training_size(), 2)

    def test_train_rejects_too_few_samples(self) -> None:
        """The trainer refuses to fit with fewer than three samples."""
        self.trainer.save_correction(SAMPLE_FEATURES, "house")
        self.trainer.save_correction(SAMPLE_FEATURES, "techno")
        self.assertFalse(self.trainer.train())

    def test_train_and_predict_round_trip(self) -> None:
        """With three labelled samples the model trains and predicts."""
        self.trainer.save_correction(
            {**SAMPLE_FEATURES, "tempo": 125.0}, "house"
        )
        self.trainer.save_correction(
            {**SAMPLE_FEATURES, "tempo": 128.0}, "house"
        )
        self.trainer.save_correction(
            {**SAMPLE_FEATURES, "tempo": 170.0}, "drum_bass"
        )

        self.assertTrue(self.trainer.train())
        self.assertTrue(os.path.exists(self.model_path))

        prediction = self.trainer.predict(
            {**SAMPLE_FEATURES, "tempo": 126.0}
        )
        self.assertIsNotNone(prediction)
        genre, confidence = prediction
        self.assertIn(genre, {"house", "drum_bass"})
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_predict_without_model_returns_none(self) -> None:
        """An untrained trainer cannot predict."""
        self.assertIsNone(self.trainer.predict(SAMPLE_FEATURES))


if __name__ == "__main__":
    unittest.main()
