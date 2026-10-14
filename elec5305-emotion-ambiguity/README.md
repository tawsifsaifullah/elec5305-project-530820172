# Acoustic and Self-Supervised Representations of Ambiguous Emotional Speech

**ELEC5305 Project, Semester 2 2026, Saifullah Tawsif (University of Sydney)**

> Status: **Project Feedback Two, preliminary implementation.** The data pipeline, speaker-independent evaluation and interpretable acoustic baseline are implemented. Frozen WavLM / UniSpeech-SAT extraction is in progress.

---

## 1. Revised research question

Following the Project Feedback One comments, the project has moved away from asking *"which acoustic feature set gives the highest speech-emotion accuracy?"*, which has already been studied extensively (including in prior USYD work on SSL features for SER). The new question is:

> **When human listeners find an emotional vocal expression ambiguous, do modern self-supervised speech representations (frozen WavLM and UniSpeech-SAT) find it ambiguous too?**

Secondary question:

> Do self-supervised representations explain difficult emotional expressions better than conventional, interpretable acoustic features (F0, intensity, duration, spectral and MFCC statistics)?

The comparison focuses on *what each representation tells us about emotional ambiguity*, rather than on which model scores highest.

### Terminology: intended vs perceived expression
A recording does not give direct access to a speaker's internal emotional state. RAVDESS actors were instructed to produce particular expressions, so this project uses:
- **intended emotional expression**: the label the actor was asked to portray (from the filename)
- **listener-perceived emotional expression**: what human raters judged it to be (from the RAVDESS validation study)

The project does **not** claim to detect what someone is actually feeling, and does not claim to measure engagement or similar real-world states. Its scope is the analysis and classification of *vocal emotional expression*.

---

## 2. Data

**RAVDESS** (Livingstone & Russo, 2018): 1440 audio-only speech files, 24 professional actors (12 F / 12 M), 2 sentences × 2 repetitions, 8 intended expressions (neutral, calm, happy, sad, angry, fearful, disgust, surprised), normal and strong intensity (neutral only at normal).

Its key advantage here is the **human perceptual validation**. Every stimulus was rated 10 times (247 raters) for emotion category, intensity and genuineness. From this we define, per utterance:

```
human agreement = proportion of listeners selecting the intended expression
```

Note: the validation study scored *neutral* and *calm* responses as interchangeable, so results are also reported with neutral and calm merged for a like-for-like comparison.

### Download
1. Audio: `Audio_Speech_Actors_01-24.zip` from Zenodo, https://zenodo.org/records/1188976. Unzip to `data/ravdess_speech/` (gives `Actor_01/ … Actor_24/`).
2. Human ratings: **S1 Table** (per-stimulus proportion correct, intensity, genuineness) from the RAVDESS paper's Supporting Information, https://doi.org/10.1371/journal.pone.0196391. Save as `data/ravdess_validation.xlsx`. The loader matches column names loosely (`src/ravdess_meta.py`).

Data files are not committed to this repository (licence: CC BY-NC-SA 4.0).

---

## 3. Method

```
                 ┌─> interpretable acoustic features (F0, intensity, duration, spectral, MFCC stats)  ─┐
speech waveform ─┤                                                                                     ├─> same simple classifier ─> predictions + confidence ─> compare with human agreement & confusion
                 └─> frozen WavLM / UniSpeech-SAT ─> hidden layer L ─> mean pooling over time ────────┘
```

- **Speaker-independent evaluation:** leave-one-actor-out (24 folds). Every recording from an actor stays on one side of the split, which is checked with an assertion in `src/splits.py`.
- **Same downstream classifier for every representation:** standardised features, then L2-regularised logistic regression. Any difference therefore comes from the representation, not the classifier.
- **Keep confidence, not just correct/incorrect:** for each utterance we store the probability of the intended label, the max probability and the top-1/top-2 margin.
- **Frozen SSL models (no fine-tuning):** `microsoft/wavlm-base-plus` and `microsoft/unispeech-sat-base-plus` (16 kHz). All hidden layers are mean-pooled and cached, so the layer-wise analysis is cheap.
- **Signal processing in MATLAB, SSL in Python:** MATLAB handles waveform, STFT, F0, intensity and MFCC inspection; Python/Hugging Face handles SSL extraction and classifiers.

---

## 4. Progress to date

| Milestone (from Feedback One) | Status |
|---|---|
| 1. RAVDESS metadata parser (actor, intended expression, intensity, statement, repetition) | ✅ `src/ravdess_meta.py` |
| 1. Human validation ratings merged per utterance (human agreement) | ✅ loader written; 🔄 verifying every file matches |
| Actor-grouped (leave-one-actor-out) splits with leakage check | ✅ `src/splits.py` |
| 2. Interpretable acoustic baseline: F0, intensity, duration, spectral, MFCC stats | ✅ `src/acoustic_features.py` |
| 2. MATLAB acoustic analysis: waveform, spectrogram, pitch, intensity, MFCC | ✅ `matlab/acoustic_analysis.m` |
| 2. Speaker-independent baseline classifier + confusion matrix | ✅ `src/baseline.py`, `run_prelim.py` |
| First look at human agreement vs machine confidence | ✅ (acoustic baseline only) |
| 3. Frozen WavLM layer-wise embeddings | 🔄 `src/extract_ssl.py` written, running next |
| 4. Frozen UniSpeech-SAT embeddings (same pipeline) | ⏳ |
| 5. Layer-wise analysis: macro-F1 and human alignment per layer | ⏳ |
| 6. Human/machine disagreement case study | ⏳ |
| Optional extension (one of: intensity generalisation / neutral-normalised acoustics / weighted layers) | ⏳ |

---

## 5. Preliminary results

All results are speaker-independent (leave-one-actor-out). Full auto-generated output is in [`results/summary.md`](results/summary.md).

<!-- Paste the table from results/summary.md here after running run_prelim.py -->

| Feature set | Accuracy | Macro-F1 | Acc. (neutral+calm merged) | Chance |
|---|---|---|---|---|
| Prosody only (F0, intensity, duration) | _see summary.md_ | | | 0.125 |
| Full acoustic (prosody + spectral + MFCC stats) | _see summary.md_ | | | 0.125 |

**Figures** (in `results/figures/`):
- `example_neutral_vs_angry.png`: waveform, spectrogram and F0 contour for neutral vs strong angry (same actor and sentence)
- `acoustic_distributions.png`: F0, F0 range, intensity and duration by expression and intensity
- `confusion_acoustic.png`: confusion matrix of the acoustic baseline
- `human_vs_machine.png`: human agreement vs machine probability of the intended expression

**Initial observations** *(update after running)*:
- …which expressions the acoustic baseline confuses most (e.g. neutral/calm, and which others)
- …whether strong-intensity utterances are easier for both listeners and the classifier
- …how strongly machine confidence tracks human agreement for the acoustic baseline; this is the reference point the SSL layers will be compared against

---

## 6. Next steps
1. Extract WavLM Base Plus hidden states (all layers, mean-pooled) and rerun the same leave-one-actor-out classifier per layer.
2. Repeat with UniSpeech-SAT Base Plus using identical audio, folds, pooling and classifier.
3. Plot layer vs macro-F1, and layer vs correlation(machine confidence, human agreement).
4. Identify human-easy/machine-hard and human-ambiguous/machine-confident utterances and inspect them acoustically.

---

## 7. How to run

```bash
pip install -r requirements.txt

# Preliminary acoustic baseline (CPU, roughly 10-20 min the first time; features are cached)
python run_prelim.py --audio data/ravdess_speech --human data/ravdess_validation.xlsx

# SSL embeddings (GPU recommended; CPU works but is slower)
python src/extract_ssl.py --model wavlm
python src/extract_ssl.py --model unispeech-sat
```
MATLAB: open `matlab/acoustic_analysis.m`, set `dataRoot`, and run.

### Repository structure
```
run_prelim.py              # Feedback Two pipeline: features -> LOAO classifier -> figures -> summary.md
src/ravdess_meta.py        # filename parsing + human validation loader
src/splits.py              # leave-one-actor-out folds + leakage assertion
src/acoustic_features.py   # interpretable acoustic features
src/baseline.py            # shared downstream classifier + metrics + confidence
src/extract_ssl.py         # frozen WavLM / UniSpeech-SAT layer-wise embeddings
matlab/acoustic_analysis.m # waveform / STFT / F0 / intensity / MFCC analysis
results/                   # generated outputs
```

---

## 8. References
- S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)," *PLoS ONE*, 13(5), e0196391, 2018. Data: https://zenodo.org/records/1188976
- Major & Chatterjee, "Acoustic analyses of the RAVDESS corpus," https://pmc.ncbi.nlm.nih.gov/articles/PMC12888863/
- S. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," *IEEE JSTSP*, 2022. Model: https://huggingface.co/microsoft/wavlm-base-plus
- S. Chen et al., "UniSpeech-SAT: Universal Speech Representation Learning with Speaker Aware Pre-Training," *ICASSP*, 2022. Model: https://huggingface.co/microsoft/unispeech-sat-base-plus
- H. Wu et al., "EMO-SUPERB: An In-depth Look at Speech Emotion Recognition," arXiv:2402.13018. https://arxiv.org/abs/2402.13018
- F. Eyben et al., "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing," *IEEE Trans. Affective Computing*, 2016. openSMILE: https://audeering.github.io/opensmile/
- M. Pratihast, *Exploring Self-Supervised Learning for Speech Emotion Recognition: Feature Analysis, Dimensional Enhancement…*, PhD thesis, University of Sydney, 2025.

---

*Acknowledgement: generative AI (Claude) was used to help draft parts of the code scaffolding and this README. All experiments, results and interpretation are my own.*
