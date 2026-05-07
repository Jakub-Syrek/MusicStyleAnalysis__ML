# Music Style Transfer - Technical Overview

## Co robi aplikacja?

**Music Style Transfer** analizuje oryginalny utwór muzyczny i generuje nową muzykę zachowującą jego charakterystykę stylową.

### Przykład:
```
INPUT: reference.wav (utwór Pop, 125 BPM, energetyczny)
         ↓
   [ANALIZA STYLU]
         ↓
OUTPUT: generated_music.wav (nowa melodia, ale taki sam styl)
```

---

## Architektura Systemu

```
┌─────────────────────────────────────────────────────────────┐
│                  MUSIC STYLE TRANSFER                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT AUDIO                                               │
│  └─→ reference.wav (np. 15 sekund, Pop, 125 BPM)          │
│                                                             │
│  ╔═══════════════════════════════════════════════════════╗ │
│  ║ FAZA 1: ANALIZA STYLU (StyleAnalyzer)                ║ │
│  ╠═══════════════════════════════════════════════════════╣ │
│  ║ Ekstrakcja 12+ cech audio:                           ║ │
│  ║ • Tempo (125.0 BPM)                                  ║ │
│  ║ • Loudness (0.369)                                   ║ │
│  ║ • Spectral Centroid (2868 Hz)                        ║ │
│  ║ • Zero Crossing Rate (0.0717)                        ║ │
│  ║ • MFCC 13 coefficients                               ║ │
│  ║ • i inne...                                          ║ │
│  ╚═══════════════════════════════════════════════════════╝ │
│                                                             │
│  ╔═══════════════════════════════════════════════════════╗ │
│  ║ FAZA 2: KLASYFIKACJA (GenreDetector)                 ║ │
│  ╠═══════════════════════════════════════════════════════╣ │
│  ║ Określenie gatunku na podstawie cech:                ║ │
│  ║ • Genre: Pop (68% confidence)                        ║ │
│  ║ • Beat Regularity: 93.7%                             ║ │
│  ║ • Syncopation: 0.132                                 ║ │
│  ║ • Breaks detected: 0                                 ║ │
│  ╚═══════════════════════════════════════════════════════╝ │
│                                                             │
│  ╔═══════════════════════════════════════════════════════╗ │
│  ║ FAZA 3: GENERACJA (MusicGenerator)                   ║ │
│  ╠═══════════════════════════════════════════════════════╣ │
│  ║ Wybór metody generacji (fallback chain):            ║ │
│  ║ 1. Local MusicGen (transformers)                     ║ │
│  ║    └─→ Konwersja cech → tekst prompt                ║ │
│  ║        "Create upbeat Pop music, 125 BPM..."         ║ │
│  ║        └─→ Neural network generuje audio            ║ │
│  ║                                                       ║ │
│  ║ 2. Hugging Face API                                  ║ │
│  ║    └─→ Cloud-based inference                        ║ │
│  ║                                                       ║ │
│  ║ 3. Synthetic Fallback                                ║ │
│  ║    └─→ Algorithmiczne tworzenie fali audio          ║ │
│  ╚═══════════════════════════════════════════════════════╝ │
│                                                             │
│  OUTPUT AUDIO                                              │
│  └─→ generated_music.wav (30 sekund, Pop, 125 BPM)       │
│      [Nowa melodia, zachowany styl]                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Technologie Szczegółowo

### 1. AUDIO ANALYSIS (Faza Analiza)

#### Biblioteka: **librosa**
- Industry standard dla music information retrieval
- Convolution-based spectral analysis

**Ekstrahowane cechy:**

| Cecha | Znaczenie | Technologia |
|-------|-----------|-------------|
| **Tempo** | BPM (beats per minute) | Onset detection + beat tracking |
| **Loudness** | Amplituda RMS | Root Mean Square energy |
| **Spectral Centroid** | "Jasność" dźwięku (Hz) | Weighted average of frequencies |
| **Zero Crossing Rate** | Szrapnięcia/szumy | Rate of sign changes |
| **MFCC (13)** | "Odcisk palca" dźwięku | Mel-Frequency Cepstral Coefficients |

**Kod:**
```python
# Tempo extraction
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

# MFCC (jak ludzie słyszą)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```

**Dlaczego MFCC?**
- Ludzie słyszą logarytmicznie, nie liniowo
- MFCC oddaje ludzką percepcję dźwięku
- Standard w speech recognition i music analysis

---

### 2. GENRE CLASSIFICATION (Faza Klasyfikacja)

#### Biblioteka: **music21**
- Analiza muzyki na poziomie teorii muzyki
- Klasyfikacja oparta na cechach

**Algorytm:**
```
Dla każdego gatunku:
  IF tempo w range(90-130) AND loudness > 0.3:
    Confidence = (tempo_match + loudness_match) / 2
```

**9 Gatunków obsługiwanych:**
- Jazz, Blues, Classical, Ambient (wolne)
- Pop, Hip-Hop (średnie)
- Rock, Electronic, Dance (szybkie)

**Rhythm Analysis:**
```python
# Beat regularity - jak steady jest beat
beat_intervals = np.diff(beat_positions)
regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)

# Onset density - jak synkopowany
syncopation = len(onsets) / len(onset_envelope)
```

---

### 3. MUSIC GENERATION (Faza Generacja)

#### Metoda 1: Local MusicGen (Recommended)

**Architektura:**
```
Input: "Create upbeat Pop music, 125 BPM, energetic"
  ↓
[Tokenizer - rozbija tekst na tokens]
  ↓
[Transformer Encoder - robi embedding tekstu]
  ↓
[Autoregressive Decoder - generuje audio token po tokenie]
  ↓
[Vocoder - konwertuje tokens na waveform]
  ↓
Output: audio waveform (16kHz, mono)
```

**Model: facebook/musicgen-small**
- Parametry: 300M parameters
- Training data: AudioSet (10M+ clips)
- Tekst → Audio (text-conditioned generation)

**Kod generacji:**
```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(text=[prompt], return_tensors="pt")
outputs = model.generate(**inputs, max_length=duration * 50)
```

**Jak tekstowy prompt wpływa na output?**
- "Upbeat electronic dance music" → Fast, synthy, energetic
- "Soft ambient music" → Slow, atmospheric, reverby
- "Rock guitar solo" → Hard attack, higher frequencies

---

#### Metoda 2: Hugging Face Inference API

**Protokół:**
```
POST https://api-inference.huggingface.co/models/{model}
Headers: Authorization: Bearer {API_KEY}
Body: {"inputs": "Create upbeat Pop music"}
Response: <audio bytes in WAV format>
```

**Zalety:**
- Cloud-based (nie trzeba GPU lokalnie)
- Skaluje się automatycznie
- Zawsze najnowszy model

**Wady:**
- Wymaga API key
- Wolniejsze (network latency)
- API limits

---

#### Metoda 3: Fallback - Synthetic Generation

**Algorytm:**
```python
# Konwersja Tempo → Frequency
base_freq = 440 Hz (A4)
tempo_ratio = tempo / 120 BPM (referencja)
frequency = 440 * tempo_ratio

# Generacja multi-harmoniczna
t = [0, duration]
wave = sin(2π * f * t)         +  # Fundamental
       0.5 * sin(2π * 1.5f * t) +  # Harmonic
       0.25 * sin(2π * 2f * t)     # Overtone
```

**Dlaczego to działa?**
- Tempo → Frequency (zachowuje szybkość poczucia)
- Harmoniki → Bogatość tonalna
- Noise → Naturalne brzmienie

---

## Porównanie Metod Generacji

| Aspekt | Local MusicGen | HF API | Synthetic |
|--------|---|---|---|
| **Jakość** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Szybkość** | Medium | Fast | ⭐⭐⭐⭐⭐ |
| **Wymaga GPU** | Tak | Nie | Nie |
| **Wymaga API key** | Nie | Tak | Nie |
| **Koszt** | 0 (jeden raz) | $/query | 0 |
| **Niespodzianki** | AI-generated | AI-generated | Predictable |

---

## Workflow: Od Analizy do Generacji

### Przykład: Original = "Smooth Jazz"

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

## Dlaczego to Działa?

### Teza: Style = Features + Patterns

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

MUSICGEN → Interpretuje prompt jako constraints
          → Generuje audio które je spełnia
          → Wynik: różna melodia, TAKI SAM STYL
```

---

## Use Cases

### 1. Creative Variation
```
Original: "Artist's Demo"
Generated: 10 variations w tym samym stylu
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

## Limitations & Future

### Obecne ograniczenia:
- MusicGen generuje monophonic audio (1 track)
- Brak kontroli nad instrumentami
- Długość max ~30 sekund
- Quality = Training data quality

### Rozszerzenia:
- [ ] Multi-track generation (orchestration)
- [ ] Instrument-specific control
- [ ] Longer sequences (minutes)
- [ ] MIDI export (edytowalne)
- [ ] Real-time parameter adjustment
- [ ] Fine-tuning na custom styles

---

## Stack Techniczny (Summary)

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

## Podsumowanie

**Music Style Transfer** to system który:
1. **Analizuje** oryginalny utwór na 12+ wymiarach
2. **Klasyfikuje** gatunek i charakterystykę
3. **Tłumaczy** cechy na tekst ("Create upbeat Pop...")
4. **Generuje** nową muzykę używając AI
5. **Zachowuje** styl, ale zmienia treść (melodię)

**Magią** jest to, że AI rozumie styl na poziomie abstrakcyjnym i potrafi go reprodukować w zupełnie nowych kompozycjach.

**Analogia:**
```
Jak tłumacz językowy tłumaczy TREŚĆ zachowując STYL,
tak MusicGen tłumaczy MELODIĘ zachowując STYL MUZYCZNY.
```

---

**Repo:** https://github.com/Jakub-Syrek/MusicStyleTransfer
