#!/usr/bin/env python3
"""
Kaban fine-tuning pipeline for torchcrepe.

Workflow:
  1. Generate  — Run finetune.py on audio files, export frame-level predictions
  2. Correct   — Edit the CSV to fix wrong pitches (ground truth)
  3. Train     — Fine-tune torchcrepe on the corrected data
  4. Use       — Pass the fine-tuned model to kaban.py via --model-path

This script does NOT modify kaban.py or the default torchcrepe weights.
The fine-tuned model is saved as a separate .pth file.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchcrepe

# Import note helpers from kaban
from kaban import hz_to_note_and_cents, format_note


# ---------------------------------------------------------------------------
# Step 1: Generate frame-level predictions for manual correction
# ---------------------------------------------------------------------------

def generate_annotations(
    audio_path: str,
    output_csv: str,
    sr: int = 16000,
    step_size: int = 10,
    model_capacity: str = "full",
) -> None:
    """Run CREPE on an audio file and write per-frame predictions to CSV.

    The CSV has columns: time, frequency_hz, note, confidence
    You then manually correct the frequency_hz or note column where CREPE was wrong.
    """
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    hop = int(sr * step_size / 1000)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_tensor = torch.tensor(
        audio, dtype=torch.float32).unsqueeze(0).to(device)
    freq, conf = torchcrepe.predict(
        audio_tensor, sr,
        hop_length=hop,
        model=model_capacity,
        decoder=torchcrepe.decode.viterbi,
        device=device,
        return_periodicity=True,
        batch_size=512,
    )
    freq = freq.squeeze(0).cpu().numpy()
    conf = conf.squeeze(0).cpu().numpy()
    time = np.arange(len(freq)) * hop / sr

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "frequency_hz", "note", "confidence"])
        for t, fr, c in zip(time, freq, conf):
            if fr > 0 and c > 0.1:
                note, cents, _ = hz_to_note_and_cents(float(fr))
                note_label = format_note(note, cents)
            else:
                note_label = "rest"
            writer.writerow([f"{t:.4f}", f"{fr:.2f}", note_label, f"{c:.4f}"])

    print(f"Wrote {len(freq)} frames to {output_csv}")
    print(f"Edit the note/frequency_hz columns to correct wrong pitches, then run 'train'.")


# ---------------------------------------------------------------------------
# Step 2: Dataset — load corrected CSVs + audio for training
# ---------------------------------------------------------------------------

CREPE_FRAME_SIZE = 1024  # torchcrepe expects 1024-sample input frames


def hz_to_bin(frequency: float) -> int:
    """Convert Hz to CREPE pitch bin (0–359).

    CREPE bins span 1997.77 cents total: 20 cents per bin,
    starting at C1 (32.70 Hz), MIDI note 24.
    """
    if frequency <= 0:
        return 0
    cents = 1200.0 * np.log2(frequency / 32.70)
    bin_idx = int(round(cents / 20.0))
    return max(0, min(torchcrepe.PITCH_BINS - 1, bin_idx))


class PitchDataset(Dataset):
    """Dataset of (audio_frame, target_bin) pairs from corrected CSVs."""

    def __init__(self, audio_paths: list[str], csv_paths: list[str], sr: int = 16000, step_size: int = 10):
        self.frames: list[torch.Tensor] = []
        self.targets: list[int] = []

        for audio_path, csv_path in zip(audio_paths, csv_paths):
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)
            hop = int(sr * step_size / 1000)

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            for i, row in enumerate(rows):
                freq = float(row["frequency_hz"])
                if freq <= 0:
                    continue  # skip unvoiced frames

                # Extract the 1024-sample frame centred on this time step
                centre = i * hop
                start = centre - CREPE_FRAME_SIZE // 2
                end = start + CREPE_FRAME_SIZE

                if start < 0 or end > len(audio):
                    continue

                frame = audio[start:end]
                # Normalise like torchcrepe does
                frame = frame - np.mean(frame)
                std = np.std(frame)
                if std > 0:
                    frame = frame / std

                self.frames.append(torch.tensor(frame, dtype=torch.float32))
                self.targets.append(hz_to_bin(freq))

        print(
            f"Loaded {len(self.frames)} voiced frames from {len(audio_paths)} file(s)")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.frames[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Step 3: Fine-tune
# ---------------------------------------------------------------------------

def train(
    audio_paths: list[str],
    csv_paths: list[str],
    output_path: str = "kaban_crepe.pth",
    base_model: str = "full",
    epochs: int = 30,
    lr: float = 1e-4,
    freeze_layers: int = 4,
    batch_size: int = 64,
    sr: int = 16000,
    step_size: int = 10,
) -> None:
    """Fine-tune torchcrepe on corrected annotation data.

    By default, freezes the first 4 conv layers and only trains
    conv5, conv6, and the classifier — preserving low-level acoustic
    features while adapting high-level pitch mapping.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained model
    model = torchcrepe.Crepe(base_model)
    weights_file = os.path.join(
        os.path.dirname(torchcrepe.__file__), "assets", f"{base_model}.pth"
    )
    model.load_state_dict(torch.load(
        weights_file, map_location=device, weights_only=True))
    model = model.to(device)

    # Freeze early layers
    freeze_names = []
    for i in range(1, freeze_layers + 1):
        freeze_names.extend([f"conv{i}", f"conv{i}_BN"])

    for name, param in model.named_parameters():
        layer_name = name.split(".")[0]
        if layer_name in freeze_names:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {total:,} params, {trainable:,} trainable (layers {freeze_layers+1}–6 + classifier)")

    # Dataset
    dataset = PitchDataset(audio_paths, csv_paths, sr=sr, step_size=step_size)
    if len(dataset) == 0:
        print("Error: no voiced frames found. Check your CSVs.", file=sys.stderr)
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=False)

    # Loss and optimiser
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total_frames = 0

        for frames, targets in loader:
            frames = frames.to(device)
            targets = targets.to(device)

            # Build soft target (Gaussian blur around the true bin)
            target_probs = torch.zeros(
                len(targets), torchcrepe.PITCH_BINS, device=device)
            bins = torch.arange(torchcrepe.PITCH_BINS, device=device).float()
            for i, t in enumerate(targets):
                target_probs[i] = torch.exp(-0.5 *
                                            ((bins - t.float()) / 1.5) ** 2)
                target_probs[i] /= target_probs[i].sum()

            # Forward
            logits = model(frames)
            loss = criterion(logits, target_probs)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)
            predicted = logits.argmax(dim=1)
            correct += (predicted == targets).sum().item()
            total_frames += len(targets)

        scheduler.step()
        acc = 100 * correct / total_frames if total_frames > 0 else 0
        avg_loss = running_loss / total_frames if total_frames > 0 else 0
        print(
            f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.1f}%  lr={scheduler.get_last_lr()[0]:.2e}")

    # Save
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved fine-tuned model to {output_path}")
    print(f"Use with: python kaban.py audio.m4a --model-path {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kaban-finetune",
        description="Fine-tune CREPE for oud/kaban pitch detection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---------------------------------------------------------
    gen = sub.add_parser(
        "generate", help="Generate annotation CSV from audio.")
    gen.add_argument("audio", help="Path to audio file.")
    gen.add_argument("-o", "--output",
                     help="Output CSV path (default: <audio>.csv).")
    gen.add_argument("--model-capacity",
                     choices=["tiny", "full"], default="full")
    gen.add_argument("--step-size", type=int, default=10)

    # --- train ------------------------------------------------------------
    tr = sub.add_parser(
        "train", help="Fine-tune CREPE on corrected annotations.")
    tr.add_argument("--audio", nargs="+", required=True, help="Audio file(s).")
    tr.add_argument("--csv", nargs="+", required=True,
                    help="Corrected CSV file(s), same order as --audio.")
    tr.add_argument("-o", "--output", default="kaban_crepe.pth",
                    help="Output model path.")
    tr.add_argument("--base-model", choices=["tiny", "full"], default="full")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--freeze-layers", type=int, default=4,
                    help="Freeze first N conv layers (0–5).")
    tr.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    if args.command == "generate":
        output = args.output or (Path(args.audio).stem + ".csv")
        generate_annotations(
            args.audio, output, model_capacity=args.model_capacity, step_size=args.step_size)

    elif args.command == "train":
        if len(args.audio) != len(args.csv):
            print(
                "Error: --audio and --csv must have the same number of files.", file=sys.stderr)
            sys.exit(1)
        train(
            audio_paths=args.audio,
            csv_paths=args.csv,
            output_path=args.output,
            base_model=args.base_model,
            epochs=args.epochs,
            lr=args.lr,
            freeze_layers=args.freeze_layers,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
