"""Unit tests for :mod:`src.audio_loader`."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from src.audio_loader import AudioLoader


class TestAudioLoader(unittest.TestCase):
    """Verify dispatch and cleanup behaviour without touching real I/O."""

    def setUp(self) -> None:
        """Create a fresh loader rooted in a temp directory."""
        self.tmp = tempfile.TemporaryDirectory()
        self.loader = AudioLoader(temp_dir=self.tmp.name)

    def tearDown(self) -> None:
        """Drop temporary files."""
        self.tmp.cleanup()

    def test_load_rejects_invalid_source(self) -> None:
        """A non-existent, non-HTTP source raises ``ValueError``."""
        with self.assertRaises(ValueError):
            self.loader.load("not-a-real-path-or-url")

    @patch("src.audio_loader.librosa.load")
    def test_load_dispatches_local_file(self, mock_load: MagicMock) -> None:
        """Existing local files go through ``_load_file``."""
        mock_load.return_value = (np.zeros(16_000), 16_000)

        path = os.path.join(self.tmp.name, "fake.wav")
        with open(path, "wb") as fh:
            fh.write(b"\x00")

        y, sr = self.loader.load(path, sr=16_000)

        mock_load.assert_called_once()
        self.assertEqual(sr, 16_000)
        self.assertEqual(y.shape[0], 16_000)

    @patch("src.audio_loader.requests.get")
    @patch("src.audio_loader.librosa.load")
    def test_load_dispatches_direct_http_url(
        self,
        mock_librosa_load: MagicMock,
        mock_get: MagicMock,
    ) -> None:
        """Direct HTTP URLs are downloaded then loaded with librosa."""
        mock_response = MagicMock()
        mock_response.content = b"RIFF....fakewav"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        mock_librosa_load.return_value = (np.zeros(8_000), 16_000)

        y, sr = self.loader.load(
            "https://example.com/track.wav",
            sr=16_000,
        )

        mock_get.assert_called_once()
        mock_librosa_load.assert_called_once()
        self.assertEqual(sr, 16_000)
        self.assertEqual(y.shape[0], 8_000)
        self.assertGreaterEqual(len(self.loader.cleanup_paths), 1)

    def test_cleanup_removes_tracked_files(self) -> None:
        """``cleanup`` deletes every tracked temporary path."""
        path = os.path.join(self.tmp.name, "scratch.wav")
        with open(path, "wb") as fh:
            fh.write(b"data")
        self.loader.cleanup_paths.append(path)

        self.loader.cleanup()

        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.loader.cleanup_paths, [])

    def test_cleanup_is_idempotent(self) -> None:
        """Calling ``cleanup`` twice does not raise even when files are gone."""
        path = os.path.join(self.tmp.name, "ghost.wav")
        self.loader.cleanup_paths.append(path)

        self.loader.cleanup()
        self.loader.cleanup()  # second call must be a no-op


if __name__ == "__main__":
    unittest.main()
