"""
Milestone 3/4 (in progress): frozen WavLM / UniSpeech-SAT layer-wise embeddings.

For every utterance, run the frozen pretrained model once, mean-pool every
hidden layer over time, and save to disk so downstream experiments are fast.

Output: features/<model>_layers.npz with
    X:   (n_utterances, n_layers+1, hidden_dim)   layer 0 = CNN/feature-projection output
    ids: stimulus ids, in the same order as metadata.csv

Usage:
    python src/extract_ssl.py --model wavlm
    python src/extract_ssl.py --model unispeech-sat
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModel

MODELS = {
    "wavlm": "microsoft/wavlm-base-plus",
    "unispeech-sat": "microsoft/unispeech-sat-base-plus",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="wavlm")
    ap.add_argument("--meta", default="results/metadata.csv")
    ap.add_argument("--out", default="features")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = MODELS[args.model]
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_hidden_states=True).to(device).eval()
    for p in model.parameters():          # frozen: no fine-tuning
        p.requires_grad = False

    meta = pd.read_csv(args.meta)
    feats = []
    with torch.no_grad():
        for i, f in enumerate(meta["file"]):
            y, _ = librosa.load(f, sr=16000, mono=True)       # models expect 16 kHz
            y, _ = librosa.effects.trim(y, top_db=30)
            inp = fe(y, sampling_rate=16000, return_tensors="pt").to(device)
            hs = model(**inp).hidden_states                   # tuple of (1, T, D)
            feats.append(torch.stack([h.mean(dim=1)[0] for h in hs]).cpu().numpy())
            if i % 100 == 0:
                print(f"[{args.model}] {i}/{len(meta)}")

    Path(args.out).mkdir(exist_ok=True)
    np.savez_compressed(Path(args.out) / f"{args.model}_layers.npz",
                        X=np.stack(feats).astype(np.float32),
                        ids=meta["stimulus_id"].values)
    print("saved", np.stack(feats).shape)


if __name__ == "__main__":
    main()
