#!/usr/bin/env python3
"""
Kaban — Microtonal pitch detection CLI for monophonic Arabic & Somali audio.

Uses CREPE neural pitch detection with HPSS drone separation to output
note sequences at cent-level precision.
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe


# ---------------------------------------------------------------------------
# Pitch ↔ note helpers
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def hz_to_note_and_cents(frequency: float) -> tuple[str, int, float]:
    """Convert a frequency (Hz) to the nearest Western note name + cent offset.

    Returns (note_name_with_octave, cent_offset, frequency).
    cent_offset is in the range (-50, +50].
    """
    if frequency <= 0:
        return ("rest", 0, 0.0)

    # MIDI note number (continuous)
    midi = 12.0 * np.log2(frequency / 440.0) + 69.0
    midi_rounded = int(round(midi))
    cent_offset = round((midi - midi_rounded) * 100)

    note_name = NOTE_NAMES[midi_rounded % 12]
    octave = (midi_rounded // 12) - 1

    return (f"{note_name}{octave}", cent_offset, frequency)


def format_note(note: str, cents: int) -> str:
    """Format a note with its cent offset, e.g. 'A4+30c' or 'G#3-15c'."""
    if note == "rest":
        return "rest"
    sign = "+" if cents >= 0 else ""
    return f"{note}{sign}{cents}c"


# ---------------------------------------------------------------------------
# Audio loading & preprocessing
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to *sr* Hz mono."""
    audio, orig_sr = librosa.load(path, sr=sr, mono=True)
    return audio, sr


def separate_drone(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove a sustained drone using Harmonic-Percussive Source Separation.

    The harmonic component retains the melody while suppressing the
    steady-state drone that sits in the percussive/residual part.
    """
    harmonic, _ = librosa.effects.hpss(audio, margin=3.0)
    return harmonic


def detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    """Detect note-onset times (seconds) using librosa.

    Uses a higher delta threshold to only detect clear attacks,
    not minor fluctuations within sustained notes.
    """
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, onset_envelope=onset_env,
        backtrack=True, units="frames",
        delta=0.07, wait=10,  # require stronger/spaced-out onsets
    )
    return librosa.frames_to_time(onset_frames, sr=sr)


# ---------------------------------------------------------------------------
# Pitch detection
# ---------------------------------------------------------------------------

def detect_pitch(
    audio: np.ndarray,
    sr: int,
    model_capacity: str = "full",
    model_path: str | None = None,
    viterbi: bool = True,
    frame_confidence_floor: float = 0.3,
    step_size: int = 10,  # ms
    chunk_seconds: float = 8.0,
    overlap_seconds: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run CREPE pitch detection via torchcrepe (PyTorch) in overlapping chunks.

    torchcrepe needs contextual lead-in to produce reliable periodicity.
    We process the audio in *chunk_seconds* windows with *overlap_seconds*
    overlap, keeping only the non-overlapping middle portion of each chunk
    (except the first and last).

    Returns (time, frequency, confidence) arrays.
    Frequencies below *frame_confidence_floor* are zeroed out (→ rest).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optionally load a fine-tuned model
    if model_path is not None:
        custom_model = torchcrepe.Crepe(model_capacity)
        custom_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        custom_model = custom_model.to(device)
        custom_model.eval()
        # Inject into torchcrepe so predict() uses it
        torchcrepe.infer.model = custom_model
        torchcrepe.infer.capacity = model_capacity

    hop_samples = int(sr * step_size / 1000)
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    stride_samples = chunk_samples - overlap_samples
    total_samples = len(audio)

    all_freq: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []

    pos = 0
    chunk_idx = 0
    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk = audio[pos:end]

        audio_tensor = torch.tensor(
            chunk, dtype=torch.float32).unsqueeze(0).to(device)
        freq_chunk, conf_chunk = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length=hop_samples,
            model=model_capacity,
            decoder=torchcrepe.decode.viterbi if viterbi else torchcrepe.decode.argmax,
            device=device,
            return_periodicity=True,
            batch_size=512,
        )
        freq_chunk = freq_chunk.squeeze(0).cpu().numpy()
        conf_chunk = conf_chunk.squeeze(0).cpu().numpy()

        # Determine which frames to keep (discard overlap lead-in except for first chunk)
        overlap_frames = int(overlap_seconds / (step_size / 1000))
        if chunk_idx == 0:
            # First chunk: keep everything
            keep_start = 0
        else:
            # Later chunks: skip the overlap lead-in
            keep_start = overlap_frames

        all_freq.append(freq_chunk[keep_start:])
        all_conf.append(conf_chunk[keep_start:])

        pos += stride_samples
        chunk_idx += 1

    frequency = np.concatenate(all_freq)
    confidence = np.concatenate(all_conf)

    # Build time array
    time = np.arange(len(frequency)) * hop_samples / sr

    # Zero out low-confidence frames (fixed internal floor for segmentation)
    frequency = np.where(confidence >= frame_confidence_floor, frequency, 0.0)

    # Median-filter isolated dropouts (confidence dips in sustained notes)
    for i in range(1, len(frequency) - 1):
        if frequency[i] <= 0 and frequency[i - 1] > 0 and frequency[i + 1] > 0:
            frequency[i] = (frequency[i - 1] + frequency[i + 1]) / 2.0
            confidence[i] = (confidence[i - 1] + confidence[i + 1]) / 2.0

    return time, frequency, confidence


# ---------------------------------------------------------------------------
# Note segmentation
# ---------------------------------------------------------------------------

def segment_notes(
    time: np.ndarray,
    frequency: np.ndarray,
    confidence: np.ndarray,
    onset_times: np.ndarray | None = None,
    cent_tolerance: int = 35,
    min_duration: float = 0.03,
    min_rest: float = 0.15,
    rest_hysteresis: int = 10,
    median_window: int = 15,
    merge_tolerance: int = 35,
) -> list[dict]:
    """Group consecutive frames into note events.

    A new note begins when the cent distance from a sliding-window median
    exceeds *cent_tolerance*, or a rest/voiced transition occurs.

    *rest_hysteresis* consecutive unvoiced frames are required before a
    voiced→rest transition is recognised (prevents single dropout frames
    from splitting a note).

    *median_window* controls how many recent voiced frames form the
    comparison reference.

    *onset_times* — if provided (from librosa onset detection), forces a
    note boundary at each onset even when pitch is unchanged (repeated
    strikes of the same note).

    *merge_tolerance* — after initial segmentation, adjacent voiced notes
    whose median frequencies are within this many cents **and** that have
    no onset boundary between them are merged.

    Notes shorter than *min_duration* seconds are dropped.
    """
    notes: list[dict] = []
    if len(frequency) == 0:
        return notes

    def _cents_distance(f1: float, f2: float) -> float:
        if f1 <= 0 or f2 <= 0:
            return float("inf")
        return abs(1200.0 * np.log2(f1 / f2))

    # Build a set of frame indices that correspond to detected onsets.
    onset_frames: set[int] = set()
    if onset_times is not None and len(onset_times) > 0 and len(time) > 0:
        step = time[1] - time[0] if len(time) > 1 else 0.01
        for ot in onset_times:
            idx = int(round(ot / step))
            if 0 < idx < len(time):          # skip idx 0 (start)
                onset_frames.add(idx)

    # --- Step 1: smooth out isolated unvoiced frames ----------------------
    freq = frequency.copy()
    conf = confidence.copy()
    gap_start = None
    for i in range(len(freq)):
        if freq[i] <= 0:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_len = i - gap_start
                if gap_len < rest_hysteresis:
                    fill_val = freq[gap_start -
                                    1] if gap_start > 0 else freq[i]
                    freq[gap_start:i] = fill_val
                    # Also fill confidence so gap-filled frames don't
                    # drag down the note's average confidence.
                    fill_conf = conf[gap_start -
                                     1] if gap_start > 0 else conf[i]
                    conf[gap_start:i] = fill_conf
                gap_start = None

    # --- Step 2: segment into notes ---------------------------------------
    current_freqs: list[float] = []
    current_confs: list[float] = []
    start_idx = 0

    def _flush(end_idx: int) -> None:
        if not current_freqs:
            return
        dur = time[end_idx] - time[start_idx]
        if dur < min_duration:
            return
        voiced = [f for f in current_freqs if f > 0]
        if not voiced:
            notes.append({
                "start": float(time[start_idx]),
                "end": float(time[end_idx]),
                "duration": float(dur),
                "note": "rest",
                "cents": 0,
                "frequency_hz": 0.0,
                "confidence": float(np.mean(current_confs)),
            })
        else:
            median_freq = float(np.median(voiced))
            note, cents, _ = hz_to_note_and_cents(median_freq)
            notes.append({
                "start": float(time[start_idx]),
                "end": float(time[end_idx]),
                "duration": float(dur),
                "note": note,
                "cents": int(cents),
                "frequency_hz": round(median_freq, 2),
                "confidence": round(float(np.mean(current_confs)), 3),
            })

    for i in range(len(freq)):
        f = freq[i]
        is_rest = f <= 0

        if not current_freqs:
            current_freqs.append(f)
            current_confs.append(conf[i])
            start_idx = i
            continue

        prev_voiced = any(ff > 0 for ff in current_freqs)

        # Transition: rest ↔ voiced
        if is_rest and prev_voiced:
            _flush(i - 1)
            current_freqs = [f]
            current_confs = [conf[i]]
            start_idx = i
            continue
        if not is_rest and not prev_voiced:
            _flush(i - 1)
            current_freqs = [f]
            current_confs = [conf[i]]
            start_idx = i
            continue

        # Force boundary at detected onsets (repeated strikes) — only for voiced frames
        if i in onset_frames and len(current_freqs) > 2 and not is_rest and any(ff > 0 for ff in current_freqs):
            _flush(i - 1)
            current_freqs = [f]
            current_confs = [conf[i]]
            start_idx = i
            continue

        # Voiced: compare against sliding-window median
        if not is_rest:
            recent_voiced = [
                ff for ff in current_freqs[-median_window:] if ff > 0]
            if recent_voiced:
                median = float(np.median(recent_voiced))
                if _cents_distance(f, median) > cent_tolerance:
                    _flush(i - 1)
                    current_freqs = [f]
                    current_confs = [conf[i]]
                    start_idx = i
                    continue

        current_freqs.append(f)
        current_confs.append(conf[i])

    # Flush remaining
    if current_freqs:
        _flush(len(freq) - 1)

    # --- Step 3: merge short rests into surrounding notes -----------------
    merged: list[dict] = []
    for n in notes:
        if (
            n["note"] == "rest"
            and n["duration"] < min_rest
            and merged
            and merged[-1]["note"] != "rest"
        ):
            merged[-1]["end"] = n["end"]
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            merged.append(n)

    # --- Step 4: merge adjacent same-pitch notes --------------------------
    # Only merge when there is NO onset boundary between the two notes.
    # This preserves repeated strikes of the same pitch.
    final: list[dict] = []
    for n in merged:
        has_onset_between = False
        if final and onset_times is not None and len(onset_times) > 0:
            has_onset_between = any(
                final[-1]["start"] < ot <= n["start"] for ot in onset_times
            )
        if (
            final
            and n["note"] != "rest"
            and final[-1]["note"] != "rest"
            and not has_onset_between
            and _cents_distance(n["frequency_hz"], final[-1]["frequency_hz"]) <= merge_tolerance
        ):
            # Weighted-average frequency by duration
            d1 = final[-1]["duration"]
            d2 = n["duration"]
            avg_freq = (final[-1]["frequency_hz"] * d1 +
                        n["frequency_hz"] * d2) / (d1 + d2)
            avg_conf = (final[-1]["confidence"] * d1 +
                        n["confidence"] * d2) / (d1 + d2)
            note_name, cents, _ = hz_to_note_and_cents(avg_freq)
            final[-1]["end"] = n["end"]
            final[-1]["duration"] = final[-1]["end"] - final[-1]["start"]
            final[-1]["frequency_hz"] = round(avg_freq, 2)
            final[-1]["note"] = note_name
            final[-1]["cents"] = int(cents)
            final[-1]["confidence"] = round(avg_conf, 3)
        else:
            final.append(n)

    return final


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_notes(
    notes: list[dict],
    output_format: str = "table",
    confidence_threshold: float = 0.0,
) -> None:
    """Print detected notes to stdout.

    If *confidence_threshold* > 0, voiced notes whose average confidence
    falls below the threshold are excluded from output.
    """
    if confidence_threshold > 0:
        notes = [
            n for n in notes
            if n["note"] == "rest" or n["confidence"] >= confidence_threshold
        ]

    if output_format == "json":
        import json
        print(json.dumps(notes, indent=2))
        return

    # Table format
    header = f"{'Start':>8s}  {'End':>8s}  {'Dur':>6s}  {'Note':<12s}  {'Hz':>9s}  {'Conf':>5s}"
    print(header)
    print("-" * len(header))

    for n in notes:
        label = format_note(n["note"], n["cents"])
        freq_str = f"{n['frequency_hz']:.2f}" if n["frequency_hz"] > 0 else "—"
        print(
            f"{n['start']:8.3f}  {n['end']:8.3f}  {n['duration']:6.3f}  {label:<12s}  {freq_str:>9s}  {n['confidence']:5.3f}"
        )

    print(f"\nTotal notes: {len(notes)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kaban",
        description="Microtonal pitch detection for monophonic Arabic & Somali audio.",
    )
    p.add_argument(
        "audio", help="Path to the input audio file (WAV, MP3, FLAC, etc.).")
    p.add_argument(
        "--no-drone-separation",
        action="store_true",
        help="Skip HPSS drone-separation preprocessing.",
    )
    p.add_argument(
        "--model-capacity",
        choices=["tiny", "full"],
        default="full",
        help="CREPE model capacity (default: full).",
    )
    p.add_argument(
        '--model-path',
        default=None,
        help="Path to a fine-tuned CREPE .pth file (from finetune.py). "
             "If omitted, uses the stock torchcrepe weights.",
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum CREPE confidence to accept a pitch frame. "
             "Accepts 0–1 (e.g. 0.3) or percent (e.g. 30). Default: 0.3."
    )
    p.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="CREPE step size in ms (default: 10).",
    )
    p.add_argument(
        "--cent-tolerance",
        type=int,
        default=35,
        help="Cent distance to start a new note segment (default: 35)."
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=0.03,
        help="Minimum note duration in seconds (default: 0.03).",
    )
    p.add_argument('--min-rest',
                   type=float,
                   default=0.15,
                   help="Rests shorter than this (seconds) are absorbed into the melody (default: 0.15).",
                   )
    p.add_argument("--format",
                   dest="output_format",
                   choices=["table", "json"],
                   default="table",
                   help="Output format (default: table).",
                   )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Resample audio to this rate in Hz (default: 16000).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Normalize confidence threshold: accept both 0.8 and 80 as 80%
    if args.confidence_threshold > 1.0:
        args.confidence_threshold = args.confidence_threshold / 100.0

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load audio
    print(f"Loading {audio_path} …", file=sys.stderr)
    audio, sr = load_audio(str(audio_path), sr=args.sample_rate)
    print(f"  {len(audio)/sr:.2f}s @ {sr} Hz", file=sys.stderr)

    # 2. Drone separation
    if not args.no_drone_separation:
        print("Applying HPSS drone separation …", file=sys.stderr)
        audio = separate_drone(audio, sr)

    # 3. Onset detection
    print("Detecting onsets …", file=sys.stderr)
    onset_times = detect_onsets(audio, sr)
    print(f"  {len(onset_times)} onsets found", file=sys.stderr)

    # 4. Pitch detection
    model_label = args.model_path or args.model_capacity
    print(f"Running CREPE ({model_label}) …", file=sys.stderr)
    time, frequency, confidence = detect_pitch(
        audio,
        sr,
        model_capacity=args.model_capacity,
        model_path=args.model_path,
        viterbi=True,
        step_size=args.step_size,
    )
    print(f"  {len(time)} frames detected", file=sys.stderr)

    # 5. Segment into notes
    notes = segment_notes(
        time,
        frequency,
        confidence,
        onset_times=onset_times,
        cent_tolerance=args.cent_tolerance,
        min_duration=args.min_duration,
        min_rest=args.min_rest,
    )

    # 6. Output
    print_notes(notes, output_format=args.output_format, confidence_threshold=args.confidence_threshold)


if __name__ == "__main__":
    main()
