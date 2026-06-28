# Plan: Vocal-to-Music-Notes Conversion

Convert vocal-only audio into a table of detected musical notes and their frequencies, building on Kaban's existing CREPE-based pitch detection pipeline.

---

## Problem

Given a monophonic vocal recording (no instruments), produce a structured table where each row is one sung note with its pitch name, frequency in Hz, cent deviation, timing, and confidence.

## Architecture

```
vocal audio file
      │
      ▼
┌──────────────┐
│  Load & Prep │  librosa: resample to 16 kHz mono, normalize amplitude
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Preprocess  │  1. High-pass filter (remove sub-80 Hz rumble/breath noise)
│              │  2. Skip HPSS (no drone in solo vocals)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Onset Det.  │  librosa onset detection — catches repeated syllables
└──────┬───────┘       on the same pitch that CREPE alone would merge
       │
       ▼
┌──────────────────┐
│  Pitch Detection │  CREPE (torchcrepe) in overlapping chunks
│                  │  → per-frame (f0, confidence) at 10 ms steps
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Post-process    │  1. Zero out low-confidence frames (< 0.3)
│                  │  2. Median-filter isolated dropouts
│                  │  3. Optional: vocal range clamp (80–1100 Hz)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Segmentation    │  Group frames into note events using:
│                  │  - cent-tolerance boundary (35 cents)
│                  │  - onset boundaries (syllable attacks)
│                  │  - rest detection via hysteresis
│                  │  - merge short rests & same-pitch neighbors
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Output          │  Table or JSON with columns:
│                  │  Start, End, Duration, Note, Cents, Hz, Confidence
└──────────────────┘
```

## Implementation Steps

### Step 1 — Vocal-specific preprocessing

Add a high-pass filter stage before pitch detection to remove breath noise and microphone rumble that is common in isolated vocal recordings. A 2nd-order Butterworth at ~80 Hz is sufficient.

```python
import scipy.signal

def highpass_filter(audio, sr, cutoff=80, order=2):
    sos = scipy.signal.butter(order, cutoff, btype="high", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, audio)
```

**New dependency:** `scipy` (add to `requirements.txt`).

### Step 2 — Vocal range clamping (post-detection)

After CREPE returns per-frame frequencies, clamp values outside the human vocal range to zero. This eliminates spurious octave jumps and harmonic confusion that CREPE sometimes produces on vocal vibrato.

```
VOCAL_FLOOR = 80   # Hz — low male chest voice
VOCAL_CEIL  = 1100 # Hz — high soprano / head voice
frequency = np.where((frequency >= VOCAL_FLOOR) & (frequency <= VOCAL_CEIL), frequency, 0.0)
```

### Step 3 — Use existing segmentation with tuned defaults

The current `segment_notes()` already handles cent-tolerance boundaries, onset-based splitting, rest hysteresis, and merge logic. For vocals, use slightly adjusted defaults:

| Parameter        | Instrument default | Vocal default | Reason                                    |
|------------------|--------------------|---------------|-------------------------------------------|
| `cent_tolerance` | 35                 | 40            | Vocals have wider natural vibrato          |
| `min_duration`   | 0.03 s             | 0.05 s        | Sung notes are rarely shorter than 50 ms   |
| `min_rest`       | 0.15 s             | 0.10 s        | Syllable gaps are shorter than pluck decay |

### Step 4 — CLI entry point

Add a `--vocal` flag (or a separate command) that enables the vocal pipeline:

```
python src/kaban.py vocals.wav --vocal
```

When `--vocal` is set:
1. Skip HPSS drone separation (no instrumental drone).
2. Apply the high-pass filter from Step 1.
3. Apply vocal range clamping from Step 2.
4. Use vocal-tuned segmentation defaults from Step 3.

### Step 5 — Output the note table

Reuse the existing `print_notes()` function. Output already includes all required columns:

```
   Start       End     Dur  Note          Hz   Conf
-----------------------------------------------------
   0.100     0.450   0.350  A4+12c       443.07  0.912
   0.450     0.800   0.350  C5-8c        521.40  0.887
   0.800     0.850   0.050  rest              —  0.150
   0.850     1.200   0.350  D5+5c        588.20  0.903
```

JSON output is also available via `--format json`.

## File Changes Summary

| File               | Change                                                  |
|--------------------|---------------------------------------------------------|
| `requirements.txt` | Add `scipy`                                             |
| `src/kaban.py`     | Add `highpass_filter()`, vocal range clamping, `--vocal` flag, vocal-tuned defaults |

## Non-Goals (out of scope)

- **Polyphonic vocal detection** — Kaban targets monophonic audio only.
- **Lyrics/phoneme alignment** — pitch detection only, no speech recognition.
- **Real-time streaming** — batch processing of files only.
- **Custom tuning systems** — output is Western note name + cent offset (already captures microtones).
