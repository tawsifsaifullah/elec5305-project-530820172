"""
Preliminary results for Project Feedback Two.

  1. Build RAVDESS speech metadata (+ human validation ratings if available)
  2. Extract interpretable acoustic features (cached)
  3. Leave-one-actor-out classification: prosody-only vs full acoustic baseline
  4. Compare machine confidence with human agreement (if ratings are available)
  5. Write figures to results/figures and a summary to results/summary.md

Usage:
    python run_prelim.py --audio data/ravdess_speech --human data/ravdess_validation.xlsx
    (--human is optional)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ravdess_meta import build_metadata, load_human_validation, merge_human, EMOTIONS
from acoustic_features import extract, load_audio, SR, HOP
from baseline import run_loao, summarise

RES = Path("results")
FIG = RES / "figures"
ORDER = list(EMOTIONS.values())
PROSODY = ["f0_mean_hz", "f0_std_st", "f0_range_st", "voiced_frac",
           "int_mean_db", "int_std_db", "int_range_db", "duration_s"]


def get_features(meta, cache=RES / "acoustic_features.csv"):
    if cache.exists():
        df = pd.read_csv(cache)
        if len(df) == len(meta):
            print("[feat] loaded cache")
            return df
    rows = []
    for i, f in enumerate(meta["file"]):
        rows.append(extract(f))
        if i % 100 == 0:
            print(f"[feat] {i}/{len(meta)}")
    df = pd.DataFrame(rows)
    df.insert(0, "stimulus_id", meta["stimulus_id"].values)
    df.to_csv(cache, index=False)
    return df


def plot_confusion(df, title, path, normalise=True):
    labels = [e for e in ORDER if e in set(df["emotion"])]
    cm = confusion_matrix(df["emotion"], df["pred"], labels=labels)
    if normalise:
        cm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Intended expression")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_example(meta):
    """Waveform, spectrogram and pitch contour: neutral vs angry, same actor & sentence."""
    import librosa, librosa.display
    sel = meta[(meta.actor == 1) & (meta.statement == "kids") & (meta.repetition == 1)]
    picks = [sel[sel.emotion == "neutral"].iloc[0],
             sel[(sel.emotion == "angry") & (sel.intensity == "strong")].iloc[0]]
    fig, axes = plt.subplots(3, 2, figsize=(10, 7))
    for col, row in enumerate(picks):
        y = load_audio(row.file)
        t = np.arange(len(y)) / SR
        axes[0, col].plot(t, y, lw=0.5); axes[0, col].set_title(f"{row.emotion} ({row.intensity}) - actor 1")
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=HOP)), ref=np.max)
        librosa.display.specshow(S, sr=SR, hop_length=HOP, x_axis="time", y_axis="hz", ax=axes[1, col])
        axes[1, col].set_ylim(0, 5000)
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=HOP)
        axes[2, col].plot(librosa.times_like(f0, sr=SR, hop_length=HOP), f0, ".", ms=2)
        axes[2, col].set_ylim(50, 500); axes[2, col].set_ylabel("F0 (Hz)"); axes[2, col].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude"); axes[1, 0].set_ylabel("Frequency (Hz)")
    fig.tight_layout(); fig.savefig(FIG / "example_neutral_vs_angry.png", dpi=150); plt.close(fig)


def plot_distributions(meta, feats):
    d = meta[["emotion", "intensity"]].join(feats[["f0_mean_hz", "f0_range_st", "int_mean_db", "duration_s"]])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (col, lab) in zip(axes, [("f0_mean_hz", "Mean F0 (Hz)"), ("f0_range_st", "F0 range (semitones)"),
                                     ("int_mean_db", "Mean intensity (dB)"), ("duration_s", "Duration (s)")]):
        for k, inten in enumerate(["normal", "strong"]):
            vals = [d[(d.emotion == e) & (d.intensity == inten)][col].dropna() for e in ORDER]
            pos = np.arange(len(ORDER)) + (k - 0.5) * 0.35
            bp = ax.boxplot([v if len(v) else [np.nan] for v in vals], positions=pos, widths=0.3,
                            patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor("#9ecae1" if inten == "normal" else "#3182bd")
        ax.set_xticks(range(len(ORDER)), ORDER, rotation=45, ha="right"); ax.set_title(lab, fontsize=10)
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc="#9ecae1"), plt.Rectangle((0, 0), 1, 1, fc="#3182bd")],
                   ["normal", "strong"], fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "acoustic_distributions.png", dpi=150); plt.close(fig)


def human_machine(pred, name):
    """Relationship between human agreement and machine probability of the intended label."""
    d = pred.dropna(subset=["human_agreement"])
    if len(d) == 0:
        return None
    rho, p = spearmanr(d["human_agreement"], d["p_true"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for e in ORDER:
        s = d[d.emotion == e]
        ax.scatter(s["human_agreement"] + np.random.uniform(-0.02, 0.02, len(s)), s["p_true"], s=8, alpha=0.6, label=e)
    ax.set_xlabel("Human agreement (proportion of listeners choosing intended expression)")
    ax.set_ylabel(f"{name}: P(intended expression)")
    ax.set_title(f"Human agreement vs machine probability  (Spearman rho = {rho:.2f})", fontsize=10)
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(FIG / "human_vs_machine.png", dpi=150); plt.close(fig)

    # 2x2 quadrant counts: human easy/hard x machine correct/wrong
    hard = d["human_agreement"] < d["human_agreement"].median()
    quad = pd.crosstab(pd.Series(np.where(hard, "human difficult", "human easy"), name="listeners"),
                       pd.Series(np.where(d["correct"] == 1, "machine correct", "machine wrong"), name="classifier"))
    by_int = d.groupby("intensity")[["human_agreement", "correct", "p_true"]].mean()
    return {"rho": rho, "p": p, "n": len(d), "quadrants": quad, "by_intensity": by_int}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/ravdess_speech")
    ap.add_argument("--human", default=None)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    meta = build_metadata(args.audio)
    print(f"[meta] {len(meta)} speech files, {meta.actor.nunique()} actors")
    if args.human:
        meta = merge_human(meta, load_human_validation(args.human))
    meta.to_csv(RES / "metadata.csv", index=False)

    feats = get_features(meta)
    plot_example(meta)
    plot_distributions(meta, feats)

    results, preds = {}, {}
    for name, cols in [("Prosody only (F0, intensity, duration)", PROSODY),
                       ("Full acoustic (prosody + spectral + MFCC stats)",
                        [c for c in feats.columns if c != "stimulus_id"])]:
        pred = run_loao(meta, feats[cols].values.astype(float))
        preds[name], results[name] = pred, summarise(pred)
        print(name, results[name])

    best = "Full acoustic (prosody + spectral + MFCC stats)"
    preds[best].to_csv(RES / "predictions_acoustic.csv", index=False)
    plot_confusion(preds[best], "Acoustic baseline - leave-one-actor-out", FIG / "confusion_acoustic.png")

    per_class = f1_score(preds[best]["emotion"], preds[best]["pred"], labels=ORDER, average=None)
    per_int = preds[best].groupby("intensity")["correct"].mean()
    hm = human_machine(preds[best], "Acoustic baseline")

    # ---------- summary.md ----------
    L = ["# Preliminary results (auto-generated by run_prelim.py)\n",
         f"Data: {len(meta)} RAVDESS audio-only speech files, {meta.actor.nunique()} actors, "
         f"8 intended expressions. Evaluation: leave-one-actor-out ({meta.actor.nunique()} folds), logistic regression.\n",
         "| Feature set | Accuracy | Macro-F1 | Macro-F1 SD across actors | Acc. (neutral+calm merged) | Chance |",
         "|---|---|---|---|---|---|"]
    for k, r in results.items():
        L.append(f"| {k} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['macro_f1_actor_sd']:.3f} | "
                 f"{r['accuracy_neutral_calm_merged']:.3f} | {r['chance']:.3f} |")
    L += ["\n**Per-emotion F1 (full acoustic):** " +
          ", ".join(f"{e} {f:.2f}" for e, f in zip(ORDER, per_class)),
          "\n**Accuracy by intensity (full acoustic):** " +
          ", ".join(f"{k} {v:.3f}" for k, v in per_int.items())]
    if hm:
        L += [f"\n**Human agreement vs machine P(intended):** Spearman rho = {hm['rho']:.3f} "
              f"(p = {hm['p']:.2g}, n = {hm['n']})\n",
              "Human-difficulty (median split) x machine outcome:\n",
              hm["quadrants"].to_markdown(), "\nBy intensity:\n", hm["by_intensity"].round(3).to_markdown()]
    else:
        L.append("\n_Human validation ratings not yet merged (run with --human)._")
    (RES / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
