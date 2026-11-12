"""
Extract frozen self-supervised (SSL) speech representations for every clip.

Run from the repo root:
    python python/extract_ssl.py

For each model, every clip is passed through the frozen network and all 13
hidden states (CNN/embedding output + 12 transformer layers) are mean-pooled
over time. Output per model: an array of shape (1440, 13, 768), float32.

    results/wavlm_layers.npy
    results/unispeechsat_layers.npy

Row i of each array matches row i of results/metadata.csv.
"""
# Python SSL models: modern, learned features. The network discovered them from huge amounts of speech. They’re high-performing, but hard to interpret.
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, UniSpeechSatModel, WavLMModel


# ---------------------------------------------------------------------------
# Settings
# TEST_N = 5    -> quick test on the first 5 clips, saved as *_test.npy
# TEST_N = None -> full run on all 1440 clips, saved under the final names
# ---------------------------------------------------------------------------
TEST_N = None

METADATA_FILE = Path("results/metadata.csv")
RESULTS_DIR = Path("results")
SAMPLE_RATE = 16000

# (Hugging Face model name, model class, output file stem)
MODELS = [
    ("microsoft/wavlm-base-plus", WavLMModel, "wavlm_layers"),
    ("microsoft/unispeech-sat-base-plus", UniSpeechSatModel, "unispeechsat_layers"),
]


# ---------------------------------------------------------------------------
# Step 1: load the metadata table (row order defines the array row order)
# ---------------------------------------------------------------------------
meta = pd.read_csv(METADATA_FILE)
if TEST_N is not None:
    meta = meta.head(TEST_N)
print(f"Clips to process: {len(meta)}")

# Use the GPU if there is one, otherwise the CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


# ---------------------------------------------------------------------------
# Step 2: run each frozen model, one after the other
# ---------------------------------------------------------------------------
saved_files = []

for model_name, model_class, file_stem in MODELS:

    # Test runs get a "_test" suffix so they never overwrite the final files
    if TEST_N is None:
        out_file = RESULTS_DIR / f"{file_stem}.npy"
    else:
        out_file = RESULTS_DIR / f"{file_stem}_test.npy"
    saved_files.append(out_file)

    # Skip a model whose final output already exists (safe to rerun)
    if TEST_N is None and out_file.exists():
        print()
        print(f"{out_file} already exists -> skipping {model_name}")
        continue

    print()
    print(f"Loading {model_name}")

    # Load the feature extractor (input normalisation) and the pretrained model.
    # eval() switches off dropout; the weights are never updated (frozen).
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_clips = []
    start_time = time.time()

    for i, row in meta.iterrows():

        # Step 3: load the wav as 16 kHz mono
        audio, sr = librosa.load(row["path"], sr=SAMPLE_RATE, mono=True)

        # Convert the waveform into model input (one clip at a time, no batching)
        inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs["input_values"].to(device)

        # Forward pass with no gradients, asking for every hidden state
        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)

        # Step 4: mean-pool each of the 13 hidden states over time
        # each hidden state has shape (1, n_frames, 768) -> (768,)
        layer_vectors = []
        for hidden_state in outputs.hidden_states:
            pooled = hidden_state[0].mean(dim=0)
            layer_vectors.append(pooled.cpu().numpy())

        clip_features = np.stack(layer_vectors)  # shape (13, 768)
        all_clips.append(clip_features)

        # Step 6: progress message every 100 clips
        n_done = len(all_clips)
        if n_done % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {n_done}/{len(meta)} clips done ({elapsed:.1f} s)")

    total_time = time.time() - start_time

    # Step 5: stack all clips -> (n_clips, 13, 768) and save as float32
    features = np.stack(all_clips).astype(np.float32)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_file, features)

    print(f"  Finished {model_name}: {len(all_clips)} clips in {total_time:.1f} s "
          f"({total_time / len(all_clips):.3f} s per clip)")
    print(f"  Saved {features.shape} to {out_file}")

    # Free GPU memory before loading the next model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 7: check the saved array shapes
# ---------------------------------------------------------------------------
print()
print("Shape check")
print("-----------")
expected_shape = (len(meta), 13, 768)
for out_file in saved_files:
    features = np.load(out_file)
    if features.shape == expected_shape:
        print(f"OK    {out_file}: {features.shape}")
    else:
        print(f"FAIL  {out_file}: {features.shape} (expected {expected_shape})")
