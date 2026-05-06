"""
Command-line interface for Music Style Transfer.

Usage:
  python -m src analyze <audio_file>
  python -m src generate <reference_audio> [--duration SECONDS] [--output FILE]
"""

import argparse
import sys
from pathlib import Path

from src.style_analyzer import StyleAnalyzer
from src.music_generator import MusicGenerator


def create_parser() -> argparse.ArgumentParser:
  """
  Create and configure argument parser.

  @returns {ArgumentParser} Configured parser
  """
  parser = argparse.ArgumentParser(
    prog="MusicStyleTransfer",
    description="Analyze musical style and generate new music"
  )

  subparsers = parser.add_subparsers(dest="command", help="Command to execute")

  # Analyze command
  analyze_parser = subparsers.add_parser(
    "analyze",
    help="Analyze musical style from audio file"
  )
  analyze_parser.add_argument(
    "audio",
    help="Path to audio file"
  )

  # Generate command
  generate_parser = subparsers.add_parser(
    "generate",
    help="Generate music based on reference style"
  )
  generate_parser.add_argument(
    "reference",
    help="Path to reference audio file"
  )
  generate_parser.add_argument(
    "--duration",
    type=int,
    default=30,
    help="Duration of generated music in seconds (default: 30)"
  )
  generate_parser.add_argument(
    "--output",
    default="generated_music.wav",
    help="Output file path (default: generated_music.wav)"
  )
  generate_parser.add_argument(
    "--provider",
    default="huggingface",
    help="API provider (default: huggingface)"
  )

  return parser


def analyze_command(audio_path: str) -> None:
  """
  Execute analyze command.

  @param {string} audio_path - Path to audio file
  """
  try:
    analyzer = StyleAnalyzer()
    features = analyzer.analyze(audio_path)

    print("\n📊 Musical Style Analysis")
    print("=" * 50)
    print(f"Tempo: {features['tempo']:.1f} BPM")
    print(f"Loudness: {features['loudness']:.3f}")
    print(f"Spectral Centroid: {features['spectral_centroid']:.1f} Hz")
    print(f"Zero Crossing Rate: {features['zero_crossing_rate']:.4f}")
    print(f"MFCC (13 coefficients): {[f'{m:.2f}' for m in features['mfcc']]}")
    print("=" * 50)

  except Exception as error:
    print(f"❌ Error: {str(error)}", file=sys.stderr)
    sys.exit(1)


def generate_command(
  reference: str,
  duration: int,
  output: str,
  provider: str
) -> None:
  """
  Execute generate command.

  @param {string} reference - Path to reference audio file
  @param {number} duration - Duration in seconds
  @param {string} output - Output file path
  @param {string} provider - API provider name
  """
  try:
    print(f"\n🎵 Generating music based on: {reference}")
    print(f"   Duration: {duration}s, Provider: {provider}")

    generator = MusicGenerator()
    result_path = generator.generate_from_reference(
      reference,
      duration=duration,
      output_filename=output,
      provider=provider
    )

    print(f"✓ Generated music saved to: {result_path}")

  except Exception as error:
    print(f"❌ Error: {str(error)}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
  """
  Main entry point for CLI.
  """
  parser = create_parser()
  args = parser.parse_args()

  if not args.command:
    parser.print_help()
    sys.exit(0)

  if args.command == "analyze":
    analyze_command(args.audio)
  elif args.command == "generate":
    generate_command(
      args.reference,
      args.duration,
      args.output,
      args.provider
    )


if __name__ == "__main__":
  main()
