"""
RAVDESS metadata: parse filenames and (optionally) merge the human validation data.

Filename convention (Livingstone & Russo, 2018, Table 1):
    Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor.wav
    e.g. 03-01-06-01-02-01-12.wav = audio-only, speech, fearful, normal,
         "Dogs are sitting by the door", 1st repetition, actor 12 (female)

Human agreement comes from S1 Table of the RAVDESS validation paper
(per-stimulus "proportion correct" from 10 listener ratings).
NOTE: in that paper, "neutral" and "calm" responses were accepted as correct
for both neutral and calm stimuli.
"""
from pathlib import Path
import re
import pandas as pd

EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
INTENSITY = {1: "normal", 2: "strong"}
STATEMENT = {1: "kids", 2: "dogs"}


def parse_filename(path):
    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Not a RAVDESS filename: {stem}")
    mod, chan, emo, inten, stmt, rep, actor = map(int, parts)
    return {
        "file": str(path),
        "stimulus_id": stem,
        "modality": mod,
        "channel": chan,
        "emotion_id": emo,
        "emotion": EMOTIONS[emo],
        "intensity": INTENSITY[inten],
        "statement": STATEMENT[stmt],
        "repetition": rep,
        "actor": actor,
        "gender": "male" if actor % 2 == 1 else "female",
    }


def build_metadata(audio_root):
    """Scan audio_root for audio-only speech files (03-01-*.wav)."""
    files = sorted(Path(audio_root).rglob("03-01-*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No RAVDESS speech .wav files found under {audio_root}. "
            "Download Audio_Speech_Actors_01-24.zip from Zenodo and unzip it there.")
    return pd.DataFrame([parse_filename(f) for f in files])


def _find_col(columns, patterns):
    for p in patterns:
        for c in columns:
            if re.search(p, str(c), flags=re.I):
                return c
    return None


def load_human_validation(path):
    """
    Load the per-stimulus validation table (S1 Table). Column names are matched
    loosely so minor formatting differences in the spreadsheet don't break it.
    Returns columns: stimulus_id, human_agreement, human_intensity, human_genuineness
    """
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)

    file_col = _find_col(df.columns, [r"file", r"stimul", r"name"])
    acc_col = _find_col(df.columns, [r"proportion.*correct", r"accuracy", r"^acc"])
    int_col = _find_col(df.columns, [r"^intensity", r"intensity.*mean", r"intensity"])
    gen_col = _find_col(df.columns, [r"genuin"])
    if file_col is None or acc_col is None:
        raise ValueError(f"Could not find filename / proportion-correct columns in {list(df.columns)}")

    out = pd.DataFrame({
        "stimulus_id": df[file_col].astype(str).str.extract(r"(\d\d-\d\d-\d\d-\d\d-\d\d-\d\d-\d\d)")[0],
        "human_agreement": pd.to_numeric(df[acc_col], errors="coerce"),
    })
    if int_col is not None and int_col != acc_col:
        out["human_intensity"] = pd.to_numeric(df[int_col], errors="coerce")
    if gen_col is not None:
        out["human_genuineness"] = pd.to_numeric(df[gen_col], errors="coerce")
    return out.dropna(subset=["stimulus_id"])


def merge_human(meta, human):
    """Validation ratings use the same stimulus code as the audio-only files."""
    merged = meta.merge(human, on="stimulus_id", how="left")
    n_missing = merged["human_agreement"].isna().sum()
    print(f"[meta] matched human ratings for {len(merged) - n_missing}/{len(merged)} files")
    return merged
