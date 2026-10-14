# Acoustic and Self-Supervised Representations of Ambiguous Emotional Speech

**ELEC5305 Acoustics, Speech and Signal Processing, Semester 2 2026, Saifullah Tawsif (SID 530820172)**

> **Status: Project Feedback Two, early work in progress.** The project has been redesigned around the Feedback One comments, and initial pipeline code is in place. Results are pending.
> All current code and documentation live in [`elec5305-emotion-ambiguity/`](elec5305-emotion-ambiguity/).

---

## Research question

Following the Feedback One comments, the project has moved away from asking *"which acoustic feature set gives the highest speech-emotion accuracy?"*, which has already been studied extensively. The new question is:

> **When human listeners find an emotional vocal expression ambiguous, do modern self-supervised speech representations (frozen WavLM and UniSpeech-SAT) find it ambiguous too?**

Secondary question: do self-supervised representations explain difficult emotional expressions better than interpretable acoustic features (F0, intensity, duration, spectral and MFCC statistics)?

**Scope.** The project studies *intended* (actor-portrayed) and *listener-perceived* emotional expression. It does not claim to detect what a speaker is actually feeling.

## Data

[RAVDESS](https://zenodo.org/records/1188976): 1440 audio-only speech files, 24 actors, 8 intended expressions, normal and strong intensity. Every recording was rated by human listeners, which gives each utterance a **human agreement** score (the proportion of listeners who chose the intended expression).

## Method

```
speech ─┬─> interpretable acoustic features ─────────────────────────────┐
        └─> frozen WavLM / UniSpeech-SAT ─> hidden layer ─> mean pooling ─┴─> same simple classifier
                                                                               ─> confidence vs human agreement
```

- Speaker-independent evaluation: leave-one-actor-out (no actor appears in both training and testing)
- The same lightweight classifier (logistic regression) is used for every representation
- Per-utterance confidence is kept and compared with human agreement and human confusion patterns
- SSL models are used frozen (no fine-tuning), with a layer-by-layer analysis

## Progress

| Component | Status |
|---|---|
| RAVDESS metadata parsing + leave-one-actor-out splits | Initial code |
| Human validation ratings loader | Initial code |
| Interpretable acoustic features | Initial code |
| MATLAB acoustic analysis | Draft |
| Baseline classifier on acoustic features | Code written, first run pending |
| Human agreement vs machine confidence | Planned |
| Frozen WavLM / UniSpeech-SAT extraction | Script drafted |
| Layer-wise analysis and human/machine disagreement study | Planned |

## Repository structure

```
elec5305-emotion-ambiguity/
├── README.md                  # full project documentation
├── run_prelim.py              # features -> leave-one-actor-out classifier -> figures -> summary
├── requirements.txt
├── src/
│   ├── ravdess_meta.py        # filename parsing + human validation loader
│   ├── splits.py              # leave-one-actor-out folds + leakage check
│   ├── acoustic_features.py   # F0, intensity, duration, spectral, MFCC features
│   ├── baseline.py            # shared classifier, metrics, confidence
│   └── extract_ssl.py         # frozen WavLM / UniSpeech-SAT layer embeddings
├── matlab/
│   └── acoustic_analysis.m    # waveform / spectrogram / F0 / intensity / MFCC
└── results/                   # generated figures and summaries
ELEC5305_Project_Proposal_one.pdf   # original proposal (Feedback One)
```

## Quick start

```bash
cd elec5305-emotion-ambiguity
pip install -r requirements.txt
# put RAVDESS speech audio in data/ravdess_speech/
python run_prelim.py --audio data/ravdess_speech
```

See [`elec5305-emotion-ambiguity/README.md`](elec5305-emotion-ambiguity/README.md) for full details, data download instructions and references.

## Project history

- **Feedback One (Aug 2026) proposal:** original proposal on acoustic-feature speech emotion recognition ([PDF](ELEC5305_Project_Proposal_one.pdf))
- **Feedback Two (Oct 2026) prposal:** redesigned around human perceptual ambiguity and self-supervised representations

---

*Generative AI (Claude) was used to help draft code scaffolding and documentation.*
