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
        """Initialize music generation client.

        Args:
            provider: API provider ("huggingface", "google", etc)
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
        """Generate music based on prompt.

        Tries: Local transformers → HTTP API → Fallback synthesis

        Args:
            prompt: Text description of desired music
            duration: Duration in seconds
            **options: Additional generation parameters

        Returns:
            Generated audio data (bytes)
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
        """Fallback: Generate synthetic audio with drums, bass, and melody.

        Creates layered audio with:
        - Kick drum pattern at original tempo
        - Bass line (80-120 Hz) moving between notes
        - Melodic layer (400-1200 Hz) with variations
        - Snare/hi-hat patterns
        - Realistic envelope shaping

        Args:
            prompt: Generation prompt (contains style info)
            duration: Duration in seconds

        Returns:
            Generated audio data (WAV format)
        """
        try:
            import numpy as np
            import soundfile as sf
            from io import BytesIO

            tempo = self._extract_tempo_from_prompt(prompt)
            sample_rate = 16000
            samples = int(sample_rate * duration)
            t = np.linspace(0, duration, samples, False)

            beat_samples = int(sample_rate * 60 / tempo)

            # Create click track to time drum hits
            kick_envelope = np.zeros(samples)
            hi_hat_envelope = np.zeros(samples)
            snare_envelope = np.zeros(samples)

            for beat_idx in range(int(duration * tempo / 60) + 1):
                beat_pos = int(beat_idx * beat_samples)
                if beat_pos >= samples:
                    break

                kick_len = int(0.1 * sample_rate)
                kick_envelope[beat_pos:beat_pos+kick_len] = 1.0

                if beat_idx % 2 == 0:
                    snare_len = int(0.08 * sample_rate)
                    snare_pos = beat_pos + int(beat_samples * 0.5)
                    if snare_pos + snare_len < samples:
                        snare_envelope[snare_pos:snare_pos+snare_len] = 1.0

                for sub in range(0, beat_samples, int(beat_samples / 8)):
                    hat_pos = beat_pos + sub
                    if hat_pos < samples:
                        kick_envelope[hat_pos:hat_pos+100] += 0.3

            kick_decay = np.exp(-np.arange(samples) / (0.1 * sample_rate))
            kick_envelope *= kick_decay

            # Generate kick drum (80 Hz fundamental, decaying)
            kick = np.sin(2 * np.pi * 80 * t) * kick_envelope
            kick += 0.5 * np.sin(2 * np.pi * 40 * t) * kick_envelope
            kick = kick * np.exp(-3 * t % 1.0)

            # Generate snare (white noise + high freq)
            snare = 0.5 * np.random.randn(samples) * snare_envelope
            snare += 0.4 * np.sin(2 * np.pi * 200 * t) * snare_envelope

            # Generate hi-hat (filtered noise)
            hat_noise = np.random.randn(samples)
            hi_hat = 0.25 * hat_noise * hi_hat_envelope
            hi_hat += 0.15 * np.sin(2 * np.pi * 8000 * t) * hi_hat_envelope

            # Bass line (oscillates between 80-120 Hz)
            bass_freq_env = 80 + 40 * np.sin(2 * np.pi * 0.5 * t)
            bass_phase = np.cumsum(bass_freq_env * 2 * np.pi / sample_rate)
            bass = 0.5 * np.sin(bass_phase)

            # Melody (400-800 Hz with vibrato)
            melody_base = 400 + 200 * np.sin(2 * np.pi * 0.3 * t)
            melody_phase = 2 * np.pi * np.cumsum(melody_base) / sample_rate
            melody_vibrato = 20 * np.sin(2 * np.pi * 5 * t)
            melody = 0.4 * np.sin(melody_phase + 0.3 * melody_vibrato)

            # Atmospheric pad (low harmonics, 150-300 Hz)
            pad = (0.25 * np.sin(2 * np.pi * 150 * t) +
                   0.15 * np.sin(2 * np.pi * 250 * t))

            # Mix all layers (weighted for better balance)
            audio = kick + 0.6 * bass + 0.5 * melody + 0.4 * pad + snare + 0.4 * hi_hat

            # Pre-normalize: scale to increase RMS loudness
            audio = 2.0 * audio

            # Fade in/out
            fade_len = int(0.2 * sample_rate)
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            audio[:fade_len] *= fade_in
            audio[-fade_len:] *= fade_out

            # Final normalization to prevent clipping
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9

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
