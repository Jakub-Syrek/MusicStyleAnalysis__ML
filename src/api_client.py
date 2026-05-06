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
        model: str = "facebook/musicgen-medium",
        **options: Any
    ) -> bytes:
        """
        Generate music using Hugging Face API.

        @param {string} prompt - Generation prompt
        @param {number} duration - Duration in seconds
        @param {string} model - Model identifier
        @param {object} options - Additional parameters
        @returns {bytes} Audio data
        """
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model}"

            headers = {"Authorization": f"Bearer {self.api_key}"}

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": duration,
                    **options,
                },
            }

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120
            )

            response.raise_for_status()
            return response.content

        except requests.exceptions.RequestException as error:
            raise RuntimeError(f"API request failed: {str(error)}")

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
