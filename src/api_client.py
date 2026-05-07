"""
Handle API calls to music generation services.
"""

import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv


load_dotenv()


class MusicGenerationClient:
    """
    Interface for music generation APIs.

    Supports Hugging Face and other music generation models.
    """

    def __init__(self, provider: str = "huggingface"):
        """
        Initialize music generation client.

        @param {string} provider - API provider ("huggingface", "google", etc)
        """
        self.provider = provider
        self.api_key = os.getenv(f"{provider.upper()}_API_KEY")

        if not self.api_key:
            raise ValueError(f"API key not found for provider: {provider}")

    def generate(
        self,
        prompt: str,
        duration: int = 30,
        **options: Any
    ) -> bytes:
        """
        Generate music based on prompt.

        @param {string} prompt - Text description of desired music
        @param {number} duration - Duration in seconds
        @param {object} options - Additional generation parameters
        @returns {bytes} Generated audio data
        """
        if self.provider == "huggingface":
            return self._generate_huggingface(prompt, duration, **options)

        raise NotImplementedError(f"Provider {self.provider} not implemented")

    def _generate_huggingface(
        self,
        prompt: str,
        duration: int,
        model: str = "facebook/musicgen-small",
        **options: Any
    ) -> bytes:
        """
        Generate music using Hugging Face API or local generation.

        For production: Use local transformers or Gradio endpoints.
        This demo generates synthetic audio based on style features.

        @param {string} prompt - Generation prompt
        @param {number} duration - Duration in seconds
        @param {string} model - Model identifier
        @param {object} options - Additional parameters
        @returns {bytes} Audio data (WAV format)
        """
        try:
            import numpy as np
            import soundfile as sf
            from io import BytesIO

            # Parse tempo from prompt
            tempo = self._extract_tempo_from_prompt(prompt)
            frequency = self._tempo_to_frequency(tempo)

            # Generate synthetic audio based on style
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration), False)

            # Create musically coherent signal
            audio = (
                0.3 * np.sin(2 * np.pi * frequency * t) +
                0.2 * np.sin(2 * np.pi * frequency * 1.5 * t) +
                0.1 * np.sin(2 * np.pi * frequency * 2 * t) +
                0.02 * np.random.randn(len(t))
            )

            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-6)

            # Save to bytes
            output = BytesIO()
            sf.write(output, audio, sample_rate, format='WAV')
            return output.getvalue()

        except Exception as error:
            raise RuntimeError(f"Audio generation failed: {str(error)}")

    def _extract_tempo_from_prompt(self, prompt: str) -> float:
        """
        Extract tempo value from generation prompt.

        @param {string} prompt - Generation prompt text
        @returns {number} Tempo in BPM
        """
        import re
        match = re.search(r"(\d+)\s*BPM", prompt)
        return float(match.group(1)) if match else 120.0

    def _tempo_to_frequency(self, tempo: float) -> float:
        """
        Convert tempo (BPM) to base frequency (Hz).

        @param {number} tempo - Tempo in BPM
        @returns {number} Base frequency in Hz
        """
        # A4 = 440 Hz, adjust based on tempo
        base_freq = 440.0
        tempo_ratio = tempo / 120.0
        return base_freq * tempo_ratio

    def format_style_prompt(self, style_features: Dict[str, Any]) -> str:
        """
        Convert style analysis results to generation prompt.

        @param {object} style_features - Analyzed style characteristics
        @returns {string} Natural language prompt for music generation
        """
        tempo = style_features.get("tempo", 120)
        loudness = style_features.get("loudness", 0.5)

        prompt_parts = [
            f"Create music with a tempo around {int(tempo)} BPM",
        ]

        if loudness > 0.6:
            prompt_parts.append("energetic and loud")
        elif loudness < 0.3:
            prompt_parts.append("soft and quiet")
        else:
            prompt_parts.append("moderate volume")

        return ", ".join(prompt_parts)
