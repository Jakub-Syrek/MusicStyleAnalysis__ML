"""
Generate test audio file for testing Music Style Transfer.
"""

import numpy as np
import soundfile as sf


def create_test_audio(
    filename: str = "reference.wav",
    duration: int = 10,
    sample_rate: int = 16000,
    frequency: float = 440.0
) -> None:
  """
  Create a test sine wave audio file.

  @param {string} filename - Output filename
  @param {number} duration - Duration in seconds
  @param {number} sample_rate - Sample rate in Hz
  @param {number} frequency - Base frequency in Hz
  """
  # Generate time array
  t = np.linspace(0, duration, int(sample_rate * duration), False)

  # Create multi-frequency signal (more realistic)
  audio = (
    0.3 * np.sin(2 * np.pi * frequency * t) +
    0.2 * np.sin(2 * np.pi * frequency * 1.5 * t) +
    0.1 * np.sin(2 * np.pi * frequency * 2 * t) +
    0.05 * np.random.randn(len(t))  # Add slight noise
  )

  # Normalize
  audio = audio / np.max(np.abs(audio))

  # Save to file
  sf.write(filename, audio, sample_rate)
  print(f"[OK] Created {filename} ({duration}s @ {sample_rate}Hz)")


if __name__ == "__main__":
  create_test_audio("sample.wav", duration=15)
  print("\nTest audio ready! Run: python -m src analyze sample.wav")
