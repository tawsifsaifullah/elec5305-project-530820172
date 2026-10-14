# Acoustic and Self-Supervised Representations of Ambiguous Emotional Speech

**ELEC5305 Project, Semester 2 2026, Saifullah Tawsif (SID 530820172), University of Sydney**

> Status: **Project Feedback Two, early work in progress.** The project has been redesigned around the Feedback One comments. The first speaker-independent acoustic baseline results are in. The SSL (WavLM / UniSpeech-SAT) work has not started yet.
> All code is in [`elec5305-emotion-ambiguity/`](elec5305-emotion-ambiguity/). The original proposal is [`ELEC5305_Project_Proposal_one.pdf`](ELEC5305_Project_Proposal_one.pdf).

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
1. Audio: `Audio_Speech_Actors_01-24.zip` from Zenodo, https://zenodo.org/records/1188976. Unzip into `elec5305-emotion-ambiguity/data/ravdess_speech/`.
2. Human ratings: **S1 Table** (per-stimulus proportion correct, intensity, genuineness) from the RAVDESS paper's Supporting Information, https://doi.org/10.1371/journal.pone.0196391. Save as `elec5305-emotion-ambiguity/data/ravdess_validation.xlsx`.

Data files are not committed to this repository (licence: CC BY-NC-SA 4.0).

---

## 3. Method

```
                 ┌─> interpretable acoustic features (F0, intensity, duration, spectral, MFCC stats) ─┐
speech waveform ─┤                                                                                    ├─> same simple classifier ─> predictions + confidence ─> compare with human agreement & confusion
                 └─> frozen WavLM / UniSpeech-SAT ─> hidden layer L ─> mean pooling over time ───────┘
```

- **Speaker-independent evaluation:** leave-one-actor-out (24 folds). Every recording from an actor stays on one side of the split, which is checked with an assertion in `src/splits.py`.
- **Same downstream classifier for every representation:** standardised features, then L2-regularised logistic regression. Any difference therefore comes from the representation, not the classifier.
- **Keep confidence, not just correct/incorrect:** for each utterance we store the probability of the intended label, the max probability and the top-1/top-2 margin.
- **Frozen SSL models (no fine-tuning):** `microsoft/wavlm-base-plus` and `microsoft/unispeech-sat-base-plus` (16 kHz). All hidden layers are mean-pooled and cached.
- **Signal processing in MATLAB, SSL in Python:** MATLAB handles waveform, STFT, F0, intensity and MFCC inspection; Python/Hugging Face handles SSL extraction and classifiers.

---

## 4. Progress to date

| Component | Status |
|---|---|
| RAVDESS metadata parsing + leave-one-actor-out splits | Initial code |
| Human validation ratings loader | Initial code (ratings not merged yet) |
| Interpretable acoustic features | Initial code |
| MATLAB acoustic analysis | Draft |
| Baseline classifier on acoustic features | Early results |
| Human agreement vs machine confidence | Planned |
| Frozen WavLM / UniSpeech-SAT extraction | Script drafted |
| Layer-wise analysis and human/machine disagreement study | Planned |

---

## 5. Preliminary results

First run of the interpretable acoustic baseline on all 1440 RAVDESS speech files. Evaluation is leave-one-actor-out (24 folds), so every test actor is unseen, and the classifier is logistic regression. Full output: [`results/summary.md`](elec5305-emotion-ambiguity/results/summary.md).

| Feature set | Accuracy | Macro-F1 | Macro-F1 SD across actors | Acc. (neutral+calm merged) | Chance |
|---|---|---|---|---|---|
| Prosody only (F0, intensity, duration) | 0.440 | 0.419 | 0.119 | 0.468 | 0.125 |
| Full acoustic (prosody + spectral + MFCC stats) | 0.522 | 0.517 | 0.120 | 0.540 | 0.125 |

**Per-emotion F1 (full acoustic):** neutral 0.45, calm 0.65, happy 0.44, sad 0.38, angry 0.59, fearful 0.46, disgust 0.51, surprised 0.65

**Accuracy by intensity (full acoustic):** normal 0.467, strong 0.585

### Figures
![Neutral vs angry](elec5305-emotion-ambiguity/results/figures/example_neutral_vs_angry.png)
*Waveform, spectrogram and F0 contour: neutral vs strong angry, same actor and sentence.*

![Acoustic distributions](elec5305-emotion-ambiguity/results/figures/acoustic_distributions.png)
*F0, F0 range, intensity and duration by intended expression and intensity.*

![Confusion matrix](elec5305-emotion-ambiguity/results/figures/confusion_acoustic.png)
*Confusion matrix of the acoustic baseline (leave-one-actor-out).*

### Initial observations
- With only 8 prosodic features (pitch, loudness, duration), the classifier reaches 44% accuracy on unseen actors, about 3.5× chance. Adding spectral and MFCC statistics raises this to 52%.
- **Strong-intensity expressions are clearly easier (58.5% vs 46.7%).** This matches the direction of the RAVDESS human validation study, where listeners were also more accurate on strong expressions. It motivates the planned human-agreement comparison.
- Calm, surprised and angry are recognised best (F1 0.59–0.65), and sad, happy and neutral worst (0.38–0.45).
- Merging neutral and calm (as the human study did) improves accuracy by only about 2 points, so most errors are not neutral/calm mix-ups.
- There is substantial variation between actors (macro-F1 SD ≈ 0.12), which supports using speaker-independent evaluation.

These are first-pass results. The human ratings have not been merged yet, so the human agreement vs machine confidence analysis is still to come.

---

## 6. Next steps
1. Merge the human validation ratings and compare human agreement with the classifier's confidence.
2. Extract WavLM Base Plus hidden states (all layers, mean-pooled) and rerun the same leave-one-actor-out classifier per layer.
3. Repeat with UniSpeech-SAT Base Plus using identical audio, folds, pooling and classifier.
4. Plot layer vs macro-F1, and layer vs correlation(machine confidence, human agreement).
5. Identify human-easy/machine-hard and human-ambiguous/machine-confident utterances and inspect them acoustically.

---

## 7. How to run

```bash
cd elec5305-emotion-ambiguity
pip install -r requirements.txt

# Acoustic baseline (CPU, roughly 10-20 min the first time; features are cached)
python run_prelim.py --audio data/ravdess_speech
# add --human data/ravdess_validation.xlsx once the ratings are downloaded

# SSL embeddings (GPU recommended)
python src/extract_ssl.py --model wavlm
python src/extract_ssl.py --model unispeech-sat
```
MATLAB: open `matlab/acoustic_analysis.m`, set `dataRoot`, and run.

### Repository structure
```
elec5305-emotion-ambiguity/
├── run_prelim.py              # features -> LOAO classifier -> figures -> summary.md
├── requirements.txt
├── src/
│   ├── ravdess_meta.py        # filename parsing + human validation loader
│   ├── splits.py              # leave-one-actor-out folds + leakage assertion
│   ├── acoustic_features.py   # interpretable acoustic features
│   ├── baseline.py            # shared classifier + metrics + confidence
│   └── extract_ssl.py         # frozen WavLM / UniSpeech-SAT layer-wise embeddings
├── matlab/
│   └── acoustic_analysis.m    # waveform / STFT / F0 / intensity / MFCC analysis
└── results/                   # generated outputs and figures
ELEC5305_Project_Proposal_one.pdf   # original proposal (Feedback One)
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

*Acknowledgement: generative AI (Claude) was used to help draft parts of the code scaffolding and this README.*
