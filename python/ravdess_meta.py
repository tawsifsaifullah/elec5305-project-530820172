"""
This file is purely data reading and preparation for Ravdess no analasys or feauture extraction
Build the RAVDESS speech metadata table and merge human validation ratings.

Output: results/metadata.csv (one row per audio file, 1440 rows)
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths (relative to the repo root)
# ---------------------------------------------------------------------------
AUDIO_DIR = Path("data/ravdess_speech")
RATINGS_FILE = Path("data/validation/pone.0196391.s004.xlsx")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "metadata.csv"


# ---------------------------------------------------------------------------
# Code -> label dictionaries (from the RAVDESS filename convention)
# ---------------------------------------------------------------------------
EMOTION_NAMES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_NAMES = {
    "01": "normal",
    "02": "strong",
}

STATEMENT_NAMES = {
    "01": "kids",
    "02": "dogs",
}


# ---------------------------------------------------------------------------
# Step 1: find all speech wav files (expect 1440)
# ---------------------------------------------------------------------------
wav_files = sorted(AUDIO_DIR.glob("Actor_*/*.wav"))
print(f"Found {len(wav_files)} wav files in {AUDIO_DIR}")


# ---------------------------------------------------------------------------
# Step 2: parse each filename into its seven fields
# Filename format: modality-channel-emotion-intensity-statement-repetition-actor.wav
# e.g. 03-01-05-02-01-02-12.wav
# ---------------------------------------------------------------------------
rows = []
for wav_path in wav_files:
    filename = wav_path.name
    parts = wav_path.stem.split("-")

    modality = parts[0]
    channel = parts[1]
    emotion_code = parts[2]
    intensity_code = parts[3]
    statement_code = parts[4]
    repetition = int(parts[5])
    actor = int(parts[6])

    # Odd-numbered actors are male, even-numbered actors are female
    if actor % 2 == 1:
        gender = "male"
    else:
        gender = "female"

    row = {
        "filename": filename,
        "path": wav_path.as_posix(),
        "emotion": EMOTION_NAMES[emotion_code],
        "emotion_id": int(emotion_code),
        "intensity": INTENSITY_NAMES[intensity_code],
        "statement": STATEMENT_NAMES[statement_code],
        "repetition": repetition,
        "actor": actor,
        "gender": gender,
    }
    rows.append(row)

meta = pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: load the human validation ratings (Livingstone & Russo, 2018, S4)
# Keep only the audio-only (AO) rows, since we use audio-only speech
# ---------------------------------------------------------------------------
ratings = pd.read_excel(RATINGS_FILE, sheet_name="Speech", header=1)
ratings = ratings[ratings["Modality"] == "AO"]

ratings = ratings[["Filename", "PropCorr", "UHR", "Intensity(M)", "Genuineness(M)"]]
ratings = ratings.rename(columns={
    "Filename": "filename",
    "PropCorr": "human_agreement",
    "UHR": "uhr",
    "Intensity(M)": "rated_intensity",
    "Genuineness(M)": "rated_genuineness",
})
print(f"Loaded {len(ratings)} audio-only rating rows from {RATINGS_FILE.name}")


# ---------------------------------------------------------------------------
# Step 4: attach the ratings to each audio file (left merge on filename)
# ---------------------------------------------------------------------------
meta = meta.merge(ratings, on="filename", how="left")


# ---------------------------------------------------------------------------
# Step 5: sort by actor, then filename
# ---------------------------------------------------------------------------
meta = meta.sort_values(["actor", "filename"])
meta = meta.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 6: sanity checks
# ---------------------------------------------------------------------------
print()
print("Sanity checks")
print("-------------")

# 6a. total number of files
n_rows = len(meta)
if n_rows == 1440:
    print(f"OK    total rows = {n_rows}")
else:
    print(f"FAIL  total rows = {n_rows} (expected 1440)")

# 6b. every actor should have 60 files
actor_counts = meta["actor"].value_counts().sort_index()
bad_actors = []
for actor, count in actor_counts.items():
    if count != 60:
        bad_actors.append(actor)
if len(actor_counts) == 24 and len(bad_actors) == 0:
    print("OK    24 actors, 60 files each")
else:
    print(f"FAIL  actors with wrong file count: {bad_actors} "
          f"(found {len(actor_counts)} actors)")

# 6c. neutral has 96 files (no strong intensity), every other emotion has 192
emotion_counts = meta["emotion"].value_counts()
for emotion in EMOTION_NAMES.values():
    count = emotion_counts.get(emotion, 0)
    if emotion == "neutral":
        expected = 96
    else:
        expected = 192
    if count == expected:
        print(f"OK    {emotion:<10} {count} files")
    else:
        print(f"FAIL  {emotion:<10} {count} files (expected {expected})")

# 6d. every file should have a human agreement score
n_missing = meta["human_agreement"].isna().sum()
if n_missing == 0:
    print("OK    no missing human_agreement values")
else:
    print(f"FAIL  {n_missing} files have no human_agreement value")

# 6e. mean human agreement per emotion (for information)
print()
print("Mean human agreement per emotion")
print("--------------------------------")
mean_agreement = meta.groupby("emotion")["human_agreement"].mean()
mean_agreement = mean_agreement.sort_values(ascending=False)
for emotion, value in mean_agreement.items():
    print(f"{emotion:<10} {value:.3f}")


# ---------------------------------------------------------------------------
# Step 7: save the table
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
meta.to_csv(OUTPUT_FILE, index=False)
print()
print(f"Saved {len(meta)} rows to {OUTPUT_FILE}")
