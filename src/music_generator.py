"""
Generate music based on analyzed style features.
"""

import os
from pathlib import Path
from typing import Optional

from src.style_analyzer import StyleAnalyzer
from src.api_client import MusicGenerationClient


class MusicGenerator:
    """
    Main orchestrator for music generation workflow.

    Combines style analysis and API calls to generate styled music.
    """

    def __init__(self, output_dir: str = "./output"):
        """
        Initialize music generator.

        @param {string} output_dir - Directory for generated audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = StyleAnalyzer()

    def generate_from_reference(
        self,
        reference_path: str,
        duration: int = 30,
        output_filename: Optional[str] = None,
        provider: str = "huggingface"
    ) -> str:
        """
        Generate music based on a reference audio file.

        @param {string} reference_path - Path to reference audio file
        @param {number} duration - Duration of generated music in seconds
        @param {string} output_filename - Output file name (auto-generated if not provided)
        @param {string} provider - API provider to use
        @returns {string} Path to generated audio file
        """
        try:
            # Analyze reference style
            style_features = self.analyzer.analyze(reference_path)

            # Initialize API client
            client = MusicGenerationClient(provider=provider)

            # Create generation prompt
            prompt = client.format_style_prompt(style_features)

            # Generate audio
            audio_data = client.generate(prompt, duration=duration)

            # Save output
            if not output_filename:
                output_filename = "generated_music.wav"

            output_path = self.output_dir / output_filename
            with open(output_path, "wb") as f:
                f.write(audio_data)

            return str(output_path)

        except Exception as error:
            raise RuntimeError(f"Music generation failed: {str(error)}")

    def get_style_features(self, audio_path: str) -> dict:
        """
        Get analyzed style features for an audio file.

        @param {string} audio_path - Path to audio file
        @returns {object} Style features dictionary
        """
        return self.analyzer.analyze(audio_path)
