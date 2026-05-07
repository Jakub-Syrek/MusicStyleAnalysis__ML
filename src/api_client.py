"""
Handle API calls to music generation services.

Supports:
- Local transformers (MusicGen via transformers library)
- Hugging Face HTTP API (when available)
- Fallback synthetic generation
"""

import os
import warnings
from typing import Optional, Dict, Any
from dotenv import load_dotenv

try:
  import requests
except ImportError:
  requests = None

load_dotenv()
warnings.filterwarnings("ignore")


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

        Tries: Local transformers → HTTP API → Fallback synthesis

        @param {string} prompt - Text description of desired music
        @param {number} duration - Duration in seconds
        @param {object} options - Additional generation parameters
        @returns {bytes} Generated audio data
        """
        if self.provider == "huggingface":
            # Try local model first
            try:
                return self._generate_local_musicgen(prompt, duration, **options)
            except Exception as e:
                print(f"[INFO] Local model unavailable: {str(e)[:50]}")
                # Fallback to HTTP API
                try:
                    return self._generate_huggingface(prompt, duration, **options)
                except Exception:
                    # Final fallback: synthetic generation
                    return self._generate_fallback(prompt, duration)

        raise NotImplementedError(f"Provider {self.provider} not implemented")

    def _generate_local_musicgen(
        self,
        prompt: str,
        duration: int,
        model_name: str = "facebook/musicgen-small",
        **options: Any
    ) -> bytes:
        """
        Generate music using local MusicGen model (transformers).

        Downloads and caches model locally on first use.

        @param {string} prompt - Generation prompt
        @param {number} duration - Duration in seconds
        @param {string} model_name - HF model identifier
        @param {object} options - Additional parameters
        @returns {bytes} Audio data (WAV format)
        """
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            import soundfile as sf
            from io import BytesIO

            print(f"[INFO] Loading {model_name}...")

            processor = AutoProcessor.from_pretrained(model_name)
            model = MusicgenForConditionalGeneration.from_pretrained(model_name)

            # Process input
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")

            # Generate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = model.generate(**inputs, max_length=duration * 50)

            # Convert to audio
            audio_values = outputs[0, 0].cpu().numpy()

            # Save to bytes
            output = BytesIO()
            sf.write(output, audio_values, model.config.sample_rate, format='WAV')
            return output.getvalue()

        except Exception as error:
            raise RuntimeError(f"Local MusicGen failed: {str(error)}")

    def _generate_huggingface(
        self,
        prompt: str,
        duration: int,
        model: str = "facebook/musicgen-small",
        **options: Any
    ) -> bytes:
        """
        Generate music using Hugging Face HTTP API.

        @param {string} prompt - Generation prompt
        @param {number} duration - Duration in seconds
        @param {string} model - Model identifier
        @param {object} options - Additional parameters
        @returns {bytes} Audio data (WAV format)
        """
        if not requests:
            raise RuntimeError("requests library required for HTTP API")

        try:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"inputs": prompt}

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=180
            )

            response.raise_for_status()
            return response.content

        except Exception as error:
            raise RuntimeError(f"API request failed: {str(error)}")

    def _generate_fallback(
        self,
        prompt: str,
        duration: int
    ) -> bytes:
        """
        Fallback: Generate synthetic audio based on style analysis.

        @param {string} prompt - Generation prompt
        @param {number} duration - Duration in seconds
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
            raise RuntimeError(f"Fallback generation failed: {str(error)}")

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
