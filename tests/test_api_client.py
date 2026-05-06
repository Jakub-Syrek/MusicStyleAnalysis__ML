"""
Unit tests for API client module.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.api_client import MusicGenerationClient


class TestMusicGenerationClient(unittest.TestCase):
  """Test cases for MusicGenerationClient class."""

  @patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "test_key_123"})
  def test_initialization(self) -> None:
    """
    Test client initialization with valid API key.
    """
    client = MusicGenerationClient(provider="huggingface")
    self.assertEqual(client.provider, "huggingface")
    self.assertEqual(client.api_key, "test_key_123")

  def test_initialization_missing_key(self) -> None:
    """
    Test client initialization without API key.
    """
    with self.assertRaises(ValueError):
      MusicGenerationClient(provider="nonexistent")

  @patch("src.api_client.requests.post")
  @patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "test_key_123"})
  def test_generate_huggingface(
    self,
    mock_post: MagicMock
  ) -> None:
    """
    Test music generation via Hugging Face API.

    @param {MagicMock} mock_post - Mocked requests.post
    """
    mock_response = MagicMock()
    mock_response.content = b"audio_data"
    mock_post.return_value = mock_response

    client = MusicGenerationClient(provider="huggingface")
    result = client._generate_huggingface(
      "Create upbeat electronic music",
      duration=30
    )

    self.assertEqual(result, b"audio_data")
    mock_post.assert_called_once()

  @patch("src.api_client.requests.post")
  @patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "test_key_123"})
  def test_generate_api_error(
    self,
    mock_post: MagicMock
  ) -> None:
    """
    Test error handling on API failure.

    @param {MagicMock} mock_post - Mocked requests.post
    """
    import requests
    mock_post.side_effect = requests.exceptions.RequestException("API Error")

    client = MusicGenerationClient(provider="huggingface")

    with self.assertRaises(RuntimeError):
      client._generate_huggingface("prompt", duration=30)

  @patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "test_key_123"})
  def test_format_style_prompt(self) -> None:
    """
    Test style features to prompt conversion.
    """
    client = MusicGenerationClient(provider="huggingface")
    style_features = {
      "tempo": 140.0,
      "loudness": 0.7,
    }

    prompt = client.format_style_prompt(style_features)

    self.assertIn("140", prompt)
    self.assertIn("energetic", prompt)

  @patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "test_key_123"})
  def test_format_style_prompt_quiet(self) -> None:
    """
    Test prompt generation for quiet music.
    """
    client = MusicGenerationClient(provider="huggingface")
    style_features = {
      "tempo": 80.0,
      "loudness": 0.2,
    }

    prompt = client.format_style_prompt(style_features)

    self.assertIn("80", prompt)
    self.assertIn("soft", prompt)


if __name__ == "__main__":
  unittest.main()
