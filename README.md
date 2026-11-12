# Acoustic and Self-Supervised Representations of Ambiguous Emotional Speech

ELEC5305 Acoustics, Speech and Signal Processing, University of Sydney, Semester 2 2026
Saifullah Tawsif (SID 530820172)

- Report: [report/ELEC5305_Report_530820172.pdf](report/ELEC5305_Report_530820172.pdf)
- Video: [Project video (Google Drive)](https://drive.google.com/file/d/1kFRbgu61ARlcvqPdfcNoPMfjc3iEq7Ua/view?usp=sharing)
- Project page: https://tawsifsaifullah.github.io/elec5305-project-530820172

## Research question
Do frozen WavLM and UniSpeech-SAT representations reflect human perceptual agreement and ambiguity in emotional speech, and which network layers hold the strongest emotion-relevant information? How do interpretable hand-crafted acoustic features compare?

## Approach
1. RAVDESS audio-only speech (1,440 clips, 24 actors, 8 emotions) with per-clip listener agreement from the RAVDESS validation data.
2. 36 interpretable acoustic features in MATLAB (F0, intensity, duration, spectral centroid, MFCC 1–13), based on the course labs.
3. Frozen WavLM Base+ and UniSpeech-SAT Base+ (Hugging Face): all 13 hidden layers, mean-pooled over time (768-dim per layer).
4. The same linear probe for all 27 systems (StandardScaler + logistic regression, C = 0.01, balanced classes), leave-one-actor-out cross-validation (24 folds), macro-F1.
5. Human–machine comparison: Spearman ρ between model confidence in the intended emotion and listener agreement, per layer, at the clip and emotion level; confusion and arousal analysis; normal vs strong intensity; acoustic case studies.

## Key results
| System | Macro-F1 | Alignment ρ with listeners |
|---|---|---|
| Acoustic features (36) | 0.447 | 0.324 |
| WavLM (best layer) | 0.755 (layer 5) | 0.428 (layer 6) |
| UniSpeech-SAT (best layer) | 0.745 (layer 6) | 0.419 (layer 7) |

- Emotion information peaks in the middle layers; the middle layer beats the final layer for 22/24 (WavLM) and 21/24 (UniSpeech-SAT) actors.
- Hand-crafted features match listeners on which emotions are hard (ρ = 0.83 across emotions); SSL features better match which individual clips are hard (within-emotion ρ ≈ 0.40 vs 0.22).

## Repository structure
```
data/                          (not included – see Data below)
matlab/
  extract_acoustic_features.m        36 acoustic features -> results/acoustic_features.csv
  extract_acoustic_features_live.m   Live Script version of the same extraction
  plot_case_studies.m                waveform / spectrogram / F0 / intensity case studies
python/
  ravdess_meta.py                    builds results/metadata.csv (clip labels + listener agreement)
  make_folds.py                      builds results/folds.csv (leave-one-actor-out, 24 folds)
  Extract_SSL_features_from_pre_trained_models.ipynb   frozen WavLM / UniSpeech-SAT features, 13 layers, mean-pooled
  Running_classifier_on_featuers.ipynb                 logistic regression probe for all 27 systems, leave-one-actor-out
  analysis.ipynb                     all figures, tables and statistics
results/                       CSV outputs and tables
results/figures/               all report figures
report/                        final report PDF
requirements.txt               Python packages
```

## Requirements
- Python 3.10+: `pip install -r requirements.txt`
- MATLAB R2023 or later with Signal Processing Toolbox and Audio Toolbox

## Data (not included in this repo)
1. RAVDESS speech audio: download `Audio_Speech_Actors_01-24.zip` from Zenodo (https://zenodo.org/records/1188976) and extract to `data/ravdess_speech/Actor_01 … Actor_24/`.
2. Listener ratings: download Supporting Information file S4 (`pone.0196391.s004.xlsx`) from Livingstone & Russo (2018), PLoS ONE, doi:10.1371/journal.pone.0196391, into `data/validation/`.

## How to run (in order, from the repository root unless stated)
1. `python python/ravdess_meta.py` → `results/metadata.csv`
2. `python python/make_folds.py` → `results/folds.csv`
3. Open `python/Extract_SSL_features_from_pre_trained_models.ipynb` and Run All → `results/wavlm_layers.npy`, `results/unispeechsat_layers.npy` (about 10–30 min on CPU; a model is skipped if its `.npy` file already exists). These `.npy` files are not in the repo because of their size.
4. MATLAB (current folder = repository root): `run('matlab/extract_acoustic_features.m')` → `results/acoustic_features.csv`
5. Open `python/Running_classifier_on_featuers.ipynb` and Run All → `results/predictions.csv`, `results/classifier_summary.csv`.
6. Open `python/analysis.ipynb` and Run All. The notebook runs from the `python/` folder (its paths use `../results/`). It writes the figures and tables to `results/figures/` (e.g. `fig1a_layer_f1_ci.png`, `fig2_layer_alignment.png`, `fig3_confidence_by_agreement.png`, `quadrants.png`, `fig4a_confusions.png`, `fig5_intensity.png`, `table1_paired_layers.png`) and the analysis tables to `results/` (e.g. `alignment_by_layer.csv`, `intensity_effect.csv`, `table1_paired_layers.csv`, `cases_easy_wrong.csv`).
7. MATLAB (current folder = repository root): `run('matlab/plot_case_studies.m')` → `results/figures/fig6_case_studies.png`

For the extraction and classifier notebooks, run `import os; os.chdir('..')` in the first cell (or start Jupyter from the repository root). `analysis.ipynb` runs from `python/` as it is.

## AI acknowledgement
Claude (Anthropic) was used to help debug and refine code, explain literature and statistical results, and improve the structure and wording of the report. All experiments were run by the author and all results were checked against the code outputs.
