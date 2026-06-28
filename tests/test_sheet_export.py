import tempfile
import unittest
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

try:
    from kaban import export_music_sheet, prepare_sheet_notes_for_export, resolve_sheet_targets
    _MISSING_DEP = None
except ModuleNotFoundError as exc:
    export_music_sheet = None
    prepare_sheet_notes_for_export = None
    resolve_sheet_targets = None
    _MISSING_DEP = exc


@unittest.skipIf(_MISSING_DEP is not None, f"Missing dependency: {_MISSING_DEP}")
class TestSheetExport(unittest.TestCase):
    def test_resolve_sheet_targets_both(self):
        audio = Path("input.wav")
        targets = resolve_sheet_targets(audio, "both", None)
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0][0], "musicxml")
        self.assertEqual(targets[1][0], "midi")
        self.assertEqual(targets[0][1].suffix, ".musicxml")
        self.assertEqual(targets[1][1].suffix, ".mid")

    def test_export_musicxml_and_midi(self):
        notes = [
            {
                "start": 0.0,
                "end": 0.5,
                "duration": 0.5,
                "note": "A4",
                "cents": 20,
                "frequency_hz": 440.0,
                "confidence": 0.9,
            },
            {
                "start": 0.5,
                "end": 1.0,
                "duration": 0.5,
                "note": "rest",
                "cents": 0,
                "frequency_hz": 0.0,
                "confidence": 0.8,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            musicxml_path = tmp_path / "melody.musicxml"
            midi_path = tmp_path / "melody.mid"

            export_music_sheet(notes, str(musicxml_path), "musicxml", tempo_bpm=90)
            export_music_sheet(notes, str(midi_path), "midi", tempo_bpm=90)

            self.assertTrue(musicxml_path.exists())
            self.assertTrue(midi_path.exists())
            self.assertGreater(musicxml_path.stat().st_size, 0)
            self.assertGreater(midi_path.stat().st_size, 0)
            xml = musicxml_path.read_text(encoding="utf-8")
            self.assertNotIn("<chord/>", xml)

    def test_export_musicxml_piano_profile(self):
        notes = [
            {
                "start": 0.0,
                "end": 0.5,
                "duration": 0.5,
                "note": "C3",
                "cents": 0,
                "frequency_hz": 130.81,
                "confidence": 0.95,
            },
            {
                "start": 0.5,
                "end": 1.0,
                "duration": 0.5,
                "note": "E4",
                "cents": 0,
                "frequency_hz": 329.63,
                "confidence": 0.95,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "piano.musicxml"
            export_music_sheet(
                notes,
                str(out),
                "musicxml",
                tempo_bpm=90,
                instrument_name="piano",
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("Piano", xml)
            self.assertIn("<sign>G</sign>", xml)
            self.assertIn("<sign>F</sign>", xml)
            self.assertNotIn("<chord/>", xml)

    def test_prepare_sheet_notes_for_export_simplifies_ornaments_and_repeats(self):
        notes = [
            {
                "start": 0.0,
                "end": 0.40,
                "duration": 0.40,
                "note": "D4",
                "cents": 0,
                "frequency_hz": 293.66,
                "confidence": 0.9,
            },
            {
                "start": 0.40,
                "end": 0.47,
                "duration": 0.07,
                "note": "Db4",
                "cents": 15,
                "frequency_hz": 279.0,
                "confidence": 0.7,
            },
            {
                "start": 0.47,
                "end": 0.90,
                "duration": 0.43,
                "note": "D4",
                "cents": 2,
                "frequency_hz": 294.1,
                "confidence": 0.92,
            },
        ]

        simplified = prepare_sheet_notes_for_export(notes, simplify_sheet=True)
        self.assertEqual(len(simplified), 1)
        self.assertGreater(float(simplified[0]["duration"]), 0.80)

    def test_prepare_sheet_notes_for_export_keeps_short_melodic_step(self):
        notes = [
            {
                "start": 0.0,
                "end": 0.40,
                "duration": 0.40,
                "note": "D4",
                "cents": 0,
                "frequency_hz": 293.66,
                "confidence": 0.9,
            },
            {
                "start": 0.40,
                "end": 0.47,
                "duration": 0.07,
                "note": "Db4",
                "cents": 0,
                "frequency_hz": 277.18,
                "confidence": 0.8,
            },
            {
                "start": 0.47,
                "end": 0.90,
                "duration": 0.43,
                "note": "C4",
                "cents": 0,
                "frequency_hz": 261.63,
                "confidence": 0.92,
            },
        ]

        simplified = prepare_sheet_notes_for_export(notes, simplify_sheet=True)
        self.assertEqual(len(simplified), 3)

    def test_export_musicxml_guitar_simplify_uses_coarser_quantization(self):
        notes = [
            {
                "start": 0.0,
                "end": 0.33,
                "duration": 0.33,
                "note": "A3",
                "cents": 0,
                "frequency_hz": 220.0,
                "confidence": 0.95,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "oud.musicxml"
            export_music_sheet(
                notes,
                str(out),
                "musicxml",
                tempo_bpm=90,
                instrument_name="guitar",
                simplify_sheet=True,
            )

            xml = out.read_text(encoding="utf-8")
            self.assertIn("Arabic Oud", xml)
            self.assertIn("<type>eighth</type>", xml)


if __name__ == "__main__":
    unittest.main()
