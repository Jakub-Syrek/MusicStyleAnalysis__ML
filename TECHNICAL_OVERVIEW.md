# Music Style Transfer - Technical Overview

## What Does This Application Do?

**Music Style Transfer** analyzes an original musical track and generates new music while preserving its stylistic characteristics.

### Example:
```
INPUT: reference.wav (Pop song, 125 BPM, energetic)
         ↓
   [STYLE ANALYSIS]
         ↓
OUTPUT: generated_music.wav (new melody, same style)
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  MUSIC STYLE TRANSFER                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT AUDIO                                               │
│  └─→ reference.wav (e.g., 15 seconds, Pop, 125 BPM)       │
│                                                             │
│  ╔═══════════════════════════════════════════════════════╗ │
│  ║ PHASE 1: STYLE ANALYSIS (StyleAnalyzer)              ║ │
│  ╠═══════════════════════════════════════════════════════╣ │
│  ║ Extract 12+ audio features:                          ║ │
│  ║ • Tempo (125.0 BPM)                                  ║ │
│  ║ • Loudness (0.369)                                   ║ │
│  ║ • Spectral Centroid (2868 Hz)                        ║ │
│  ║ • Zero Crossing Rate (0.0717)                        ║ │
│  ║ • MFCC 13 coefficients                               ║ │
│  ║ • And more...                                        ║ │
│  ╚═══════════════════════════════════════════════════════╝ │
│                                                             │
│  ╔═══════════════════════════════════════════════════════╗ │
│  ║ PHASE 2: CLASSIFICATION (GenreDetector)              ║ │
│  ╠═══════════════════════════════════════════════════════╣ │
│  ║ Determine genre based on features:                   ║ │
│  ║ • Genre: Pop (68% confidence)                        ║ │
│  ║ • Beat Regularity: 93.7%                             ║ │
│  ║ • Syncopation: 0.132                                 ║ │
│  ║ • Breaks detected: 0                                 ║ │
│  ╚═══════════════════════════════════════════════════════╝ │
│                                                             │
│  ╔═══════════════════════════════════════════════════════╗ │
│  ║ PHASE 3: GENERATION (MusicGenerator)                 ║ │
│  ╠═══════════════════════════════════════════════════════╣ │
│  ║ Select generation method (fallback chain):           ║ │
│  ║ 1. Local MusicGen (transformers)                     ║ │
│  ║    └─→ Convert features → text prompt                ║ │
│  ║        "Create upbeat Pop music, 125 BPM..."         ║ │
│  ║        └─→ Neural network generates audio            ║ │
│  ║                                                       ║ │
│  ║ 2. Hugging Face API                                  ║ │
│  ║    └─→ Cloud-based inference                        ║ │
│  ║                                                       ║ │
│  ║ 3. Synthetic Fallback                                ║ │
│  ║    └─→ Algorithmic audio generation                 ║ │
│  ╚═══════════════════════════════════════════════════════╝ │
│                                                             │
│  OUTPUT AUDIO                                              │
│  └─→ generated_music.wav (30 seconds, Pop, 125 BPM)      │
│      [New melody, preserved style]                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Technologies in Detail

### 1. AUDIO ANALYSIS (Analysis Phase)

#### Library: **librosa**
- Industry standard for music information retrieval
- Convolution-based spectral analysis

**Extracted Features:**

| Feature | Meaning | Technology |
|---------|---------|-----------|
| **Tempo** | Beats per minute (BPM) | Onset detection + beat tracking |
| **Loudness** | Amplitude RMS | Root Mean Square energy |
| **Spectral Centroid** | "Brightness" of sound (Hz) | Weighted average of frequencies |
| **Zero Crossing Rate** | Harshness/noise level | Rate of sign changes in waveform |
| **MFCC (13)** | "Fingerprint" of sound | Mel-Frequency Cepstral Coefficients |

**Code:**
```python
# Tempo extraction
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

# MFCC (how humans hear)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```

**Why MFCC?**
- Humans hear logarithmically, not linearly
- MFCC captures human auditory perception
- Industry standard in speech recognition and music analysis

---

### 2. GENRE CLASSIFICATION (Classification Phase)

#### Library: **music21**
- Music analysis at the music theory level
- Classification based on features

**Algorithm:**
```
For each genre:
  IF tempo in range(90-130) AND loudness > 0.3:
    Confidence = (tempo_match + loudness_match) / 2
```

**9 Supported Genres:**
- Jazz, Blues, Classical, Ambient (slow)
- Pop, Hip-Hop (medium)
- Rock, Electronic, Dance (fast)

**Rhythm Analysis:**
```python
# Beat regularity - how steady is the beat
beat_intervals = np.diff(beat_positions)
regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)

# Onset density - how syncopated
syncopation = len(onsets) / len(onset_envelope)
```

---

### 3. MUSIC GENERATION (Generation Phase)

#### Method 1: Local MusicGen (Recommended)

**Architecture:**
```
Input: "Create upbeat Pop music, 125 BPM, energetic"
  ↓
[Tokenizer - breaks text into tokens]
  ↓
[Transformer Encoder - creates text embeddings]
  ↓
[Autoregressive Decoder - generates audio token by token]
  ↓
[Vocoder - converts tokens into waveform]
  ↓
Output: audio waveform (16kHz, mono)
```

**Model: facebook/musicgen-small**
- Parameters: 300M
- Training data: AudioSet (10M+ audio clips)
- Capability: Text-to-Audio generation

**Generation Code:**
```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(text=[prompt], return_tensors="pt")
outputs = model.generate(**inputs, max_length=duration * 50)
```

**How Text Prompt Influences Output?**
- "Upbeat electronic dance music" → Fast, synthy, energetic
- "Soft ambient music" → Slow, atmospheric, reverberant
- "Rock guitar solo" → Hard attack, higher frequencies

---

#### Method 2: Hugging Face Inference API

**Protocol:**
```
POST https://api-inference.huggingface.co/models/{model}
Headers: Authorization: Bearer {API_KEY}
Body: {"inputs": "Create upbeat Pop music"}
Response: <audio bytes in WAV format>
```

**Advantages:**
- Cloud-based (no local GPU needed)
- Automatic scaling
- Always latest model version

**Disadvantages:**
- Requires API key
- Slower (network latency)
- API rate limits

---

#### Method 3: Fallback - Synthetic Generation

**Algorithm:**
```python
# Convert Tempo → Frequency
base_freq = 440 Hz (A4)
tempo_ratio = tempo / 120 BPM (reference)
frequency = 440 * tempo_ratio

# Generate multi-harmonic wave
t = [0, duration]
wave = sin(2π * f * t)         +  # Fundamental
       0.5 * sin(2π * 1.5f * t) +  # Harmonic
       0.25 * sin(2π * 2f * t)     # Overtone
```

**Why Does This Work?**
- Tempo → Frequency (preserves feeling of speed)
- Harmonics → Tonal richness
- Noise → Natural sound quality

---

## Generation Methods Comparison

| Aspect | Local MusicGen | HF API | Synthetic |
|--------|---|---|---|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Speed** | Medium | Fast | ⭐⭐⭐⭐⭐ |
| **Requires GPU** | Yes | No | No |
| **Requires API Key** | No | Yes | No |
| **Cost** | Free (one-time) | $/query | Free |
| **Surprises** | AI-generated | AI-generated | Predictable |

---

## Workflow: From Analysis to Generation

### Example: Original = "Smooth Jazz"

```
STEP 1: ANALYZE original.wav (30s, Jazz, 85 BPM)
├─ Tempo: 85 BPM
├─ Loudness: 0.25 (quiet)
├─ Spectral: 1800 Hz (warm, dark)
├─ ZCR: 0.02 (smooth, no harshness)
├─ MFCC: [low energy, rich harmonics]
└─ Genre: Jazz (94% confidence)

STEP 2: EXTRACT STYLE PROMPT
└─ "Create smooth jazz music, 85 BPM, warm and relaxing"

STEP 3: GENERATE (select method)
└─ MusicGen reads prompt
   └─ Transformer encoder: understands "smooth", "jazz", "85 BPM"
   └─ Decoder: generates audio tokens for 30 seconds
   └─ Vocoder: converts tokens → waveform
   └─ Result: new_jazz.wav (never heard before, but same STYLE)

STEP 4: OUTPUT
└─ generated_music.wav
   ├─ Duration: 30s ✓
   ├─ Tempo: ~85 BPM ✓
   ├─ Genre: Jazz ✓
   ├─ Mood: Smooth, warm ✓
   └─ Content: Completely new composition ✓
```

---

## Why This Works

### Thesis: Style = Features + Patterns

```
ORIGINAL SONG STYLE = {
  "tempo": 125 BPM,
  "loudness": 0.4,
  "spectral_centroid": 2500 Hz,
  "zero_crossing": 0.08,
  "mfcc": [23.4, 19.5, 7.4, ...],
  "beat_regularity": 0.94,
  "genre": "Pop",
  ...
}

TEXT PROMPT (created from style):
"Create upbeat Pop music, 125 BPM, energetic, bright sound, 
 strong beat, 0.4 loudness, 2500 Hz spectral profile"

MUSICGEN → Interprets prompt as constraints
          → Generates audio that satisfies them
          → Result: different melody, SAME STYLE
```

---

## Use Cases

### 1. Creative Variation
```
Original: "Artist's Demo"
Generated: 10 variations in the same style
Use: Brainstorming, inspiration
```

### 2. Style Transfer
```
Original A: "Classical Violin"
Original B: "Electronic Dance"
Result: "Classical EDM hybrid"
```

### 3. Soundtrack Generation
```
Original: "Film Score - Dramatic"
Generated: 30 different dramatic pieces, same mood
Use: Choose best fit for scene
```

### 4. Audio Branding
```
Original: "Company Logo Music"
Generated: 100 variations (same brand identity)
Use: Ads, videos, notifications
```

---

## Current Limitations & Future

### Current Limitations:
- MusicGen generates monophonic audio (single track)
- No control over instruments
- Maximum ~30 seconds per generation
- Quality limited by training data quality

### Future Improvements:
- [ ] Multi-track generation (orchestration)
- [ ] Instrument-specific control
- [ ] Longer sequences (minutes)
- [ ] MIDI export (editable)
- [ ] Real-time parameter adjustment
- [ ] Fine-tuning on custom styles

---

## Technology Stack (Summary)

```
┌─────────────────────────────────────┐
│      AUDIO ANALYSIS LAYER           │
├─────────────────────────────────────┤
│ librosa (spectral analysis)         │
│ numpy (numerical computing)         │
│ scipy (signal processing)           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│    FEATURE EXTRACTION LAYER         │
├─────────────────────────────────────┤
│ music21 (music theory)              │
│ Custom genre classifier             │
│ Rhythm analyzer                     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   MUSIC GENERATION LAYER            │
├─────────────────────────────────────┤
│ transformers (MusicGen)             │
│ torch (neural computation)          │
│ soundfile (audio I/O)               │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│       OUTPUT LAYER                  │
├─────────────────────────────────────┤
│ WAV format (16kHz, 16-bit)          │
│ Ready for playback/editing          │
└─────────────────────────────────────┘
```

---

## Summary

**Music Style Transfer** is a system that:
1. **Analyzes** original audio on 12+ dimensions
2. **Classifies** genre and characteristics
3. **Converts** features to text ("Create upbeat Pop...")
4. **Generates** new music using AI
5. **Preserves** style while changing content (melody)

**The Magic** is that AI understands style at an abstract level and can reproduce it in completely new compositions.

**Analogy:**
```
Just like a language translator translates CONTENT while preserving STYLE,
MusicGen translates MELODY while preserving MUSICAL STYLE.
```

---

**Repository**: https://github.com/Jakub-Syrek/MusicStyleTransfer
