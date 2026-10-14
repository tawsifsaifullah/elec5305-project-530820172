"""
Interpretable acoustic baseline features (one vector per utterance).

  - F0 (pYIN): mean, std, range (5th-95th percentile), voiced fraction
  - Intensity (RMS, dB): mean, std, range
  - Duration of the speech region (leading/trailing silence trimmed)
  - Spectral centroid, bandwidth, roll-off (mean, std)
  - MFCC 1-13: mean and std over frames
"""
import numpy as np
import librosa

SR = 16000
HOP = 160      # 10 ms
NFFT = 512     # 32 ms window


def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    return y_trim if len(y_trim) > sr * 0.2 else y


def extract(path):
    y = load_audio(path)
    feats = {}

    # --- F0 ---
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=SR,
                                 frame_length=1024, hop_length=HOP)
    f0v = f0[voiced & ~np.isnan(f0)] if f0 is not None else np.array([])
    if len(f0v) > 2:
        st = 12 * np.log2(f0v / 100.0)   # semitones re 100 Hz (more perceptual)
        feats.update(f0_mean_hz=np.mean(f0v), f0_std_st=np.std(st),
                     f0_range_st=np.percentile(st, 95) - np.percentile(st, 5))
    else:
        feats.update(f0_mean_hz=np.nan, f0_std_st=np.nan, f0_range_st=np.nan)
    feats["voiced_frac"] = float(np.mean(voiced)) if voiced is not None else 0.0

    # --- Intensity ---
    rms = librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-8)
    active = db[db > db.max() - 40]
    feats.update(int_mean_db=np.mean(active), int_std_db=np.std(active),
                 int_range_db=np.percentile(active, 95) - np.percentile(active, 5))

    # --- Duration ---
    feats["duration_s"] = len(y) / SR

    # --- Spectral descriptors ---
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    for name, v in [("centroid", librosa.feature.spectral_centroid(S=S, sr=SR)[0]),
                    ("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]),
                    ("rolloff", librosa.feature.spectral_rolloff(S=S, sr=SR)[0])]:
        feats[f"spec_{name}_mean"] = np.mean(v)
        feats[f"spec_{name}_std"] = np.std(v)

    # --- MFCC statistics ---
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=NFFT, hop_length=HOP)
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        feats[f"mfcc{i+1}_std"] = np.std(mfcc[i])

    return feats
