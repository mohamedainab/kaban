#!/usr/bin/env python3
"""
Kaban — Microtonal pitch detection CLI for monophonic Arabic & Somali audio.

Uses CREPE neural pitch detection with HPSS drone separation to output
note sequences at cent-level precision.
"""

import argparse
import gc
import sys
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import torch
import torchcrepe


# ---------------------------------------------------------------------------
# Pitch ↔ note helpers
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

SHEET_FORMAT_CHOICES = ["musicxml", "midi", "both"]
INSTRUMENT_CHOICES = ["guitar", "piano"]


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


def highpass_filter(
    audio: np.ndarray, sr: int, cutoff: int = 80, order: int = 2
) -> np.ndarray:
    """Remove sub-*cutoff* Hz rumble and breath noise with a Butterworth HPF."""
    sos = scipy.signal.butter(order, cutoff, btype="high", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, audio)


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
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

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
            batch_size=128,
        )
        freq_chunk = freq_chunk.squeeze(0).cpu().numpy()
        conf_chunk = conf_chunk.squeeze(0).cpu().numpy()

        # Free GPU / MPS memory between chunks
        del audio_tensor
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        gc.collect()

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

    # Median-filter isolated single-frame dropouts (vectorised)
    if len(frequency) > 2:
        drop = (
            (frequency[1:-1] <= 0)
            & (frequency[:-2] > 0)
            & (frequency[2:] > 0)
        )
        idx = np.flatnonzero(drop) + 1  # offset back to original indices
        frequency[idx] = (frequency[idx - 1] + frequency[idx + 1]) / 2.0
        confidence[idx] = (confidence[idx - 1] + confidence[idx + 1]) / 2.0

    return time, frequency, confidence


def clamp_vocal_range(
    frequency: np.ndarray,
    floor_hz: float = 80.0,
    ceil_hz: float = 1100.0,
) -> np.ndarray:
    """Zero out frequencies outside the human vocal range."""
    return np.where(
        (frequency >= floor_hz) & (frequency <= ceil_hz), frequency, 0.0
    )


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


def cents_distance(f1: float, f2: float) -> float:
    """Return absolute cent distance between two frequencies."""
    if f1 <= 0 or f2 <= 0:
        return float("inf")
    return abs(1200.0 * np.log2(f1 / f2))


def prepare_sheet_notes_for_export(
    raw_notes: list[dict],
    simplify_sheet: bool = False,
    ornament_max_duration: float = 0.10,
    ornament_cents_tolerance: float = 120.0,
    repeat_gap_seconds: float = 0.10,
    repeat_cents_tolerance: float = 30.0,
) -> list[dict]:
    """Build a strictly monophonic note stream for sheet export.

    Always drops rests and removes overlaps so the exported score never emits
    simultaneous notes. When *simplify_sheet* is enabled, short ornaments are
    absorbed into nearby notes and repeated nearby notes are collapsed.
    """
    if not raw_notes:
        return []

    events = [dict(n) for n in raw_notes if n["note"] != "rest"]
    events.sort(key=lambda n: (float(n.get("start", 0.0)), float(n.get("end", 0.0))))

    mono: list[dict] = []
    for ev in events:
        start = float(ev.get("start", 0.0))
        dur = max(0.0, float(ev.get("duration", 0.0)))
        end = float(ev.get("end", start + dur))
        if end <= start:
            continue

        if mono:
            prev = mono[-1]
            prev_start = float(prev.get("start", 0.0))
            prev_end = float(prev.get("end", prev_start + float(prev.get("duration", 0.0))))

            if start < prev_end:
                trimmed_prev_dur = max(0.0, start - prev_start)
                if trimmed_prev_dur <= 0.0:
                    prev_conf = float(prev.get("confidence", 0.0))
                    curr_conf = float(ev.get("confidence", 0.0))
                    if curr_conf > prev_conf:
                        mono[-1] = ev
                    continue

                prev["duration"] = trimmed_prev_dur
                prev["end"] = prev_start + trimmed_prev_dur

        mono.append(ev)

    mono = [n for n in mono if float(n.get("duration", 0.0)) > 0.0]
    if not simplify_sheet:
        return mono

    # Absorb only short sandwiched ornaments, not standalone melodic steps.
    simplified = [dict(n) for n in mono]
    changed = True
    while changed and len(simplified) > 1:
        changed = False
        next_notes: list[dict] = []
        i = 0
        while i < len(simplified):
            current = dict(simplified[i])
            current_dur = float(current.get("duration", 0.0))
            if current_dur >= ornament_max_duration:
                next_notes.append(current)
                i += 1
                continue

            prev_note = next_notes[-1] if next_notes else None
            next_note = simplified[i + 1] if i + 1 < len(simplified) else None
            if prev_note is None or next_note is None:
                next_notes.append(current)
                i += 1
                continue

            dist_prev = cents_distance(
                float(prev_note.get("frequency_hz", 0.0)),
                float(current.get("frequency_hz", 0.0)),
            )
            dist_next = cents_distance(
                float(next_note.get("frequency_hz", 0.0)),
                float(current.get("frequency_hz", 0.0)),
            )
            dist_neighbours = cents_distance(
                float(prev_note.get("frequency_hz", 0.0)),
                float(next_note.get("frequency_hz", 0.0)),
            )

            if (
                dist_neighbours > repeat_cents_tolerance
                or min(dist_prev, dist_next) > ornament_cents_tolerance
            ):
                next_notes.append(current)
                i += 1
                continue

            changed = True
            if dist_prev <= dist_next:
                prev_note["duration"] = float(prev_note["duration"]) + current_dur
                prev_note["end"] = max(
                    float(prev_note.get("end", 0.0)),
                    float(current.get("end", 0.0)),
                )
            else:
                next_note["duration"] = float(next_note["duration"]) + current_dur
                next_note["start"] = min(
                    float(next_note.get("start", 0.0)),
                    float(current.get("start", 0.0)),
                )
            i += 1

        simplified = next_notes

    # Collapse repeated nearby notes of nearly the same pitch.
    collapsed: list[dict] = []
    for current in simplified:
        if not collapsed:
            collapsed.append(dict(current))
            continue

        prev = collapsed[-1]
        gap = float(current.get("start", 0.0)) - float(prev.get("end", 0.0))
        dist = cents_distance(
            float(prev.get("frequency_hz", 0.0)),
            float(current.get("frequency_hz", 0.0)),
        )
        if gap <= repeat_gap_seconds and dist <= repeat_cents_tolerance:
            prev["duration"] = float(prev["duration"]) + float(current.get("duration", 0.0))
            prev["end"] = max(float(prev.get("end", 0.0)), float(current.get("end", 0.0)))
            prev["frequency_hz"] = round(
                (float(prev.get("frequency_hz", 0.0)) + float(current.get("frequency_hz", 0.0))) / 2.0,
                2,
            )
            prev["confidence"] = round(
                max(float(prev.get("confidence", 0.0)), float(current.get("confidence", 0.0))),
                3,
            )
            note_name, cents, _ = hz_to_note_and_cents(float(prev["frequency_hz"]))
            prev["note"] = note_name
            prev["cents"] = int(cents)
        else:
            collapsed.append(dict(current))

    return [n for n in collapsed if float(n.get("duration", 0.0)) > 0.0]


def export_music_sheet(
    notes: list[dict],
    output_path: str,
    sheet_format: str,
    tempo_bpm: int = 90,
    instrument_name: str = "guitar",
    simplify_sheet: bool = False,
) -> None:
    """Export detected notes to MusicXML or MIDI.

    Durations are mapped from seconds to quarter lengths via tempo:
    quarter_length = duration_seconds * tempo_bpm / 60.
    """
    try:
        from music21 import clef, instrument, layout, meter, metadata, note as m21note, pitch, stream, tempo
    except ImportError as exc:
        raise RuntimeError(
            "music21 is required for sheet export. Install with: pip install music21"
        ) from exc

    def _to_music21_note(n: dict, ql: float) -> m21note.Note:
        ev = m21note.Note(n["note"], quarterLength=ql)
        cents = int(n.get("cents", 0))
        if cents != 0:
            ev.pitch.microtone = pitch.Microtone(cents)
        return ev

    def _build_guitar_part(sheet_notes: list[dict], ql_step: float) -> stream.Part:
        part = stream.Part(id="guitar")
        oud = instrument.Instrument()
        oud.instrumentName = "Arabic Oud"
        part.partName = "Arabic Oud"
        part.append(oud)
        part.append(tempo.MetronomeMark(number=tempo_bpm))
        # Oud lead notation is monophonic on a single treble staff.
        part.append(clef.TrebleClef())
        part.append(meter.TimeSignature("4/4"))

        for n in sheet_notes:
            ql = _quantize_quarter_length(float(n["duration"]), ql_step)
            part.append(_to_music21_note(n, ql))

        return part

    def _build_piano_parts(sheet_notes: list[dict], ql_step: float) -> tuple[stream.PartStaff, stream.PartStaff]:
        right = stream.PartStaff(id="piano-rh")
        left = stream.PartStaff(id="piano-lh")

        right.append(instrument.Piano())
        right.append(tempo.MetronomeMark(number=tempo_bpm))
        right.append(clef.TrebleClef())
        right.append(meter.TimeSignature("4/4"))

        left.append(clef.BassClef())
        left.append(meter.TimeSignature("4/4"))

        split_midi = 60  # C4 split between LH/RH
        for n in sheet_notes:
            ql = _quantize_quarter_length(float(n["duration"]), ql_step)
            ev = _to_music21_note(n, ql)
            if ev.pitch.midi >= split_midi:
                right.append(ev)
            else:
                left.append(ev)

        return right, left

    score = stream.Score(id="kaban-score")
    score.metadata = metadata.Metadata()
    score.metadata.title = "Kaban Export"

    def _quantize_quarter_length(duration_seconds: float, step: float) -> float:
        # Quantize to the profile-specific rhythmic grid for stable MusicXML.
        raw_ql = float(duration_seconds) * float(tempo_bpm) / 60.0
        return max(step, round(raw_ql / step) * step)

    sheet_notes = prepare_sheet_notes_for_export(notes, simplify_sheet=simplify_sheet)

    if instrument_name == "piano":
        right, left = _build_piano_parts(sheet_notes, ql_step=0.25)
        score.append(right)
        score.append(left)
        score.insert(0, layout.StaffGroup([right, left], symbol="brace", barTogether=True, name="Piano"))
    else:
        ql_step = 0.5 if simplify_sheet else 0.25
        score.append(_build_guitar_part(sheet_notes, ql_step=ql_step))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fmt = "musicxml" if sheet_format == "musicxml" else "midi"
    score.write(fmt=fmt, fp=str(target))


def resolve_sheet_targets(
    audio_path: Path,
    sheet_format: str,
    sheet_output: str | None,
) -> list[tuple[str, Path]]:
    """Resolve concrete export targets based on CLI options."""
    if sheet_format != "both":
        if sheet_output:
            return [(sheet_format, Path(sheet_output))]
        ext = ".musicxml" if sheet_format == "musicxml" else ".mid"
        return [(sheet_format, audio_path.with_suffix(ext))]

    # both: infer two paths from provided output stem or input stem
    if sheet_output:
        base = Path(sheet_output)
        stem = base.with_suffix("")
    else:
        stem = audio_path.with_suffix("")

    return [
        ("musicxml", stem.with_suffix(".musicxml")),
        ("midi", stem.with_suffix(".mid")),
    ]


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
    p.add_argument(
        "--vocal",
        action="store_true",
        help="Vocal mode: high-pass filter, vocal range clamping, "
             "skip drone separation, and vocal-tuned segmentation defaults.",
    )
    p.add_argument(
        "--sheet-format",
        choices=SHEET_FORMAT_CHOICES,
        default=None,
        help="Export a score file in this format (musicxml or midi).",
    )
    p.add_argument(
        "--sheet-output",
        default=None,
        help="Output path for score export. If omitted, derives from input filename.",
    )
    p.add_argument(
        "--sheet-tempo",
        type=int,
        default=90,
        help="Tempo used to map seconds to note lengths for score export (default: 90 BPM).",
    )
    p.add_argument(
        "--instrument",
        choices=INSTRUMENT_CHOICES,
        default="guitar",
        help="Sheet export instrument profile (default: guitar, interpreted as Arabic Oud lead).",
    )
    p.add_argument(
        "--simplify-sheet",
        action="store_true",
        help="Apply an Oud-focused lead-sheet simplification preset before export. Enabled by default for guitar/Oud sheets.",
    )
    p.add_argument(
        "--no-simplify-sheet",
        action="store_true",
        help="Disable Oud/guitar sheet simplification and export a more faithful note stream.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.simplify_sheet and args.no_simplify_sheet:
        print("Error: use only one of --simplify-sheet or --no-simplify-sheet", file=sys.stderr)
        sys.exit(1)

    if args.instrument == "guitar" and not args.no_simplify_sheet and "--simplify-sheet" not in sys.argv:
        args.simplify_sheet = True

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

    # 2. Preprocessing
    if args.vocal:
        print("Vocal mode: applying high-pass filter …", file=sys.stderr)
        audio = highpass_filter(audio, sr)
    elif not args.no_drone_separation:
        print("Applying HPSS drone separation …", file=sys.stderr)
        audio = separate_drone(audio, sr)

    # 3. Onset detection
    print("Detecting onsets …", file=sys.stderr)
    onset_times = detect_onsets(audio, sr)
    print(f"  {len(onset_times)} onsets found", file=sys.stderr)

    # 4. Pitch detection
    # Vocal mode: use faster defaults unless the user overrode them.
    model_cap = args.model_capacity
    step = args.step_size
    if args.vocal:
        if "--model-capacity" not in sys.argv:
            model_cap = "tiny"
        if "--step-size" not in sys.argv:
            step = 20

    model_label = args.model_path or model_cap
    print(f"Running CREPE ({model_label}, step={step}ms) …", file=sys.stderr)
    time, frequency, confidence = detect_pitch(
        audio,
        sr,
        model_capacity=model_cap,
        model_path=args.model_path,
        viterbi=True,
        step_size=step,
    )
    print(f"  {len(time)} frames detected", file=sys.stderr)

    # 4b. Vocal range clamping
    if args.vocal:
        print("Clamping to vocal range (80–1100 Hz) …", file=sys.stderr)
        frequency = clamp_vocal_range(frequency)

    # 5. Segment into notes
    # Vocal mode overrides segmentation defaults unless the user set them explicitly.
    cent_tol = args.cent_tolerance
    min_dur = args.min_duration
    min_rst = args.min_rest
    if args.vocal:
        if "--cent-tolerance" not in sys.argv:
            cent_tol = 40
        if "--min-duration" not in sys.argv:
            min_dur = 0.05
        if "--min-rest" not in sys.argv:
            min_rst = 0.10

    notes = segment_notes(
        time,
        frequency,
        confidence,
        onset_times=onset_times,
        cent_tolerance=cent_tol,
        min_duration=min_dur,
        min_rest=min_rst,
    )

    # 6. Output
    print_notes(notes, output_format=args.output_format, confidence_threshold=args.confidence_threshold)

    # 7. Optional score export
    if args.sheet_format is not None:
        targets = resolve_sheet_targets(
            audio_path=audio_path,
            sheet_format=args.sheet_format,
            sheet_output=args.sheet_output,
        )

        for fmt, sheet_path in targets:
            print(
                f"Exporting {fmt} sheet to {sheet_path} ...",
                file=sys.stderr,
            )
            export_music_sheet(
                notes,
                output_path=str(sheet_path),
                sheet_format=fmt,
                tempo_bpm=args.sheet_tempo,
                instrument_name=args.instrument,
                simplify_sheet=args.simplify_sheet,
            )

        print("  sheet export complete", file=sys.stderr)


if __name__ == "__main__":
    main()
