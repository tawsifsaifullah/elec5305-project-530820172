%% Acoustic Feature Extraction for RAVDESS Speech (Interpretable Baseline)

%
% This live script computes 36 interpretable acoustic features for every clip
% listed in results/metadata.csv and saves them to results/acoustic_features.csv
% (one row per clip, same order as metadata.csv). These features are the
% hand-crafted baseline compared against the WavLM and UniSpeech-SAT layers.
%
% Feature groups: pitch (F0), intensity, duration, spectral centroid and MFCCs.
% Run from the repository root.

clear all ; close all ; clc ;

%% Analysis settings
% Short-time analysis uses a 25 ms Hamming window with a 10 ms hop, as in the
% framing used in the Week 4 STFT lab. Pitch uses a longer 40 ms window so that
% at least two periods of the lowest F0 fit in a frame (50 Hz -> 20 ms period).
%
% Pitch search ranges are gender-specific to avoid octave errors on low male
% voices (similar to Praat defaults). The voicing threshold of 0.65 was chosen
% after 0.8 proved too strict for breathy or low male voices.

fs        = 16000 ;            % analysis sample rate (Hz), same as the SSL models
winTime   = 0.025 ;            % 25 ms window (Week 4)
hopTime   = 0.010 ;            % 10 ms hop (Week 4)
pitchWinTime = 0.040 ;         % 40 ms window for pitch
f0RangeMale   = [60 350] ;     % F0 search range, male actors (Hz)
f0RangeFemale = [100 500] ;    % F0 search range, female actors (Hz)
vadThreshDb = 35 ;             % speech frames = within 35 dB of the loudest frame
voicedThresh = 0.65 ;          % harmonic ratio above this = voiced frame
numCoeffs = 13 ;               % MFCC 1-13
nTest     = [] ;               % set to e.g. 5 to test on the first 5 clips

winLen   = round(winTime * fs) ;          % 400 samples
hopSize  = round(hopTime * fs) ;          % 160 samples
pWinLen  = round(pitchWinTime * fs) ;     % 640 samples
winFun   = hamming(winLen, 'periodic') ;  % Hamming window (Week 4)
nfft     = 512 ;
freqAxis = (0:nfft/2).' * fs / nfft ;     % one-sided frequency axis (Hz)

%% Load the clip list
% The metadata table (from ravdess_meta.py) gives the path, label, actor and
% gender of each clip. The feature names are set up here so the output table
% has one clearly named column per feature.

meta = readtable('results/metadata.csv', 'TextType', 'string') ;
numClips = height(meta) ;
if ~isempty(nTest)
    numClips = nTest ;
end

featNames = ["f0_mean","f0_std_st","f0_range_st", ...
             "int_mean_db","int_std_db","int_range_db", ...
             "speech_dur_s","speech_ratio", ...
             "centroid_mean","centroid_std"] ;
for I = 1:numCoeffs
    featNames(end+1) = "mfcc" + I + "_mean" ;
end
for I = 1:numCoeffs
    featNames(end+1) = "mfcc" + I + "_std" ;
end
feats = nan(numClips, numel(featNames)) ;

%% Feature extraction loop
% Each clip goes through the same steps:
%
% # Read the audio, convert to mono and resample to 16 kHz (Week 1B, Week 10B).
% # Frame-by-frame analysis with a Hamming window and FFT (Week 4), giving the
%   frame RMS level (Week 6C) and the spectral centroid.
% # Energy-based speech detection (Week 6C idea) to ignore silence.
% # Intensity and duration statistics over speech frames (Week 6C).
% # F0 from MATLAB's pitch() with the NCF method, a normalised autocorrelation
%   estimator (Week 7 source-filter / autocorrelation idea), kept only on voiced
%   frames and smoothed against octave jumps.
% # MFCCs from mfcc(): mel filterbank, log, DCT (Week 6A filterbank idea).

tic ;
for clipIdx = 1:numClips

    % --- Read audio, mono, resample (Week 1B, Week 10B) ---
    [sig, fsOrig] = audioread(meta.path(clipIdx)) ;
    sig = mean(sig, 2) ;                       % mono
    sig = resample(sig, fs, fsOrig) ;          % 48 kHz -> 16 kHz

    % --- Frame-by-frame analysis loop (Week 4 framing; RMS from Week 6C) ---
    numFrames = floor((length(sig) - winLen) / hopSize) + 1 ;
    frameRms  = zeros(numFrames, 1) ;
    centroid  = zeros(numFrames, 1) ;
    pIn = 0 ;
    for k = 1:numFrames
        grain = sig(pIn+1 : pIn+winLen) .* winFun ;      % windowed frame
        frameRms(k) = sqrt(mean(grain.^2)) ;             % frame RMS
        frameSpec = abs(fft(grain, nfft)) ;
        frameSpec = frameSpec(1:nfft/2+1) ;              % one-sided magnitude
        centroid(k) = sum(freqAxis .* frameSpec) / (sum(frameSpec) + eps) ;  % spectral centroid (Hz)
        pIn = pIn + hopSize ;
    end
    rmsDb = 20*log10(frameRms + 1e-10) ;                 % level in dB

    % --- Speech frames: simple energy threshold (Week 6C idea) ---
    speechFrames = rmsDb > (max(rmsDb) - vadThreshDb) ;

    % --- Intensity and duration (Week 6C) ---
    speechDb = rmsDb(speechFrames) ;
    intMean  = mean(speechDb) ;
    intStd   = std(speechDb) ;
    intRange = Percentile(speechDb, 95) - Percentile(speechDb, 5) ;
    speechDur   = sum(speechFrames) * hopTime ;          % seconds of speech
    speechRatio = sum(speechFrames) / numFrames ;        % speech / total

    % --- Spectral centroid over speech frames (Week 4 spectrum) ---
    centMean = mean(centroid(speechFrames)) ;
    centStd  = std(centroid(speechFrames)) ;

    % --- F0 (Week 7: source-filter model, autocorrelation pitch) ---
    if meta.gender(clipIdx) == "male"
        f0Range = f0RangeMale ;
    else
        f0Range = f0RangeFemale ;
    end
    [f0, loc] = pitch(sig, fs, 'Method', 'NCF', 'Range', f0Range, ...
                      'WindowLength', pWinLen, 'OverlapLength', pWinLen - hopSize) ;
    % pitch() returns a value for every frame, including unvoiced sounds with no
    % real pitch, so a harmonic-ratio check keeps only voiced frames (where the
    % vocal folds - the "source" - are vibrating)
    hr = harmonicRatio(sig, fs, 'Window', hamming(pWinLen, 'periodic'), ...
                       'OverlapLength', pWinLen - hopSize) ;
    nP = min(length(f0), length(hr)) ;
    f0 = f0(1:nP) ; hr = hr(1:nP) ; loc = loc(1:nP) ;
    pitchDb = zeros(nP, 1) ;                             % level of each pitch frame
    for k = 1:nP
        seg = sig(loc(k)-pWinLen+1 : loc(k)) ;
        pitchDb(k) = 20*log10(sqrt(mean(seg.^2)) + 1e-10) ;
    end
    f0 = medfilt1(f0, 5) ;                               % remove one-frame octave jumps
    voiced = (hr > voicedThresh) & (pitchDb > (max(pitchDb) - vadThreshDb)) ;
    f0Voiced = f0(voiced) ;
    if numel(f0Voiced) >= 5
        f0Mean = mean(f0Voiced) ;                        % Hz
        f0St   = 12*log2(f0Voiced / 100) ;               % semitones re 100 Hz
        f0Std  = std(f0St) ;
        f0RangeVal = Percentile(f0St, 95) - Percentile(f0St, 5) ;
    else
        f0Mean = NaN ; f0Std = NaN ; f0RangeVal = NaN ;  % filled later in Python
    end

    % --- MFCCs (Week 6A filterbank idea: mel filterbank -> log -> DCT) ---
    coeffs = mfcc(sig, fs, 'Window', winFun, 'OverlapLength', winLen - hopSize, ...
                  'NumCoeffs', numCoeffs, 'LogEnergy', 'Ignore') ;
    nUse = min(size(coeffs,1), numFrames) ;              % align frame counts
    coeffs = coeffs(1:nUse, 1:numCoeffs) ;
    mfccSpeech = coeffs(speechFrames(1:nUse), :) ;
    mfccMean = mean(mfccSpeech, 1) ;
    mfccStd  = std(mfccSpeech, 0, 1) ;

    % --- Store this clip's feature row ---
    feats(clipIdx, :) = [f0Mean, f0Std, f0RangeVal, ...
                         intMean, intStd, intRange, ...
                         speechDur, speechRatio, ...
                         centMean, centStd, ...
                         mfccMean, mfccStd] ;

    if mod(clipIdx, 100) == 0
        fprintf('%d / %d clips done (%.1f s)\n', clipIdx, numClips, toc) ;
    end
end
fprintf('Feature extraction finished in %.1f s\n', toc) ;

%% Sanity checks
% Any missing F0 values are reported but left in the file; they are filled
% inside each training fold in Python, so nothing leaks from test actors.
% Expected: male mean F0 roughly 110-170 Hz, female roughly 180-270 Hz, and
% actor 9 (the worst earlier octave-error case) around 120-170 Hz.

for c = 1:size(feats, 2)
    nanRows = isnan(feats(:, c)) ;
    if any(nanRows)
        fprintf('Column %s: %d NaN(s) left for fold-wise filling in Python\n', ...
                featNames(c), sum(nanRows)) ;
    end
end
isMale = meta.gender(1:numClips) == "male" ;
fprintf('Mean F0: male %.0f Hz, female %.0f Hz\n', ...
        mean(feats(isMale,1), 'omitnan'), mean(feats(~isMale,1), 'omitnan')) ;
fprintf('F0 std (semitones): male %.2f, female %.2f\n', ...
        mean(feats(isMale,2), 'omitnan'), mean(feats(~isMale,2), 'omitnan')) ;
isA9 = meta.actor(1:numClips) == 9 & ismember(meta.emotion(1:numClips), ["calm","neutral"]) ;
if any(isA9)
    fprintf('Actor 9 calm/neutral mean F0: %.0f Hz\n', mean(feats(isA9,1), 'omitnan')) ;
end

%% Save the feature table
% One row per clip with the filename first, so it lines up with metadata.csv
% and the SSL embeddings.

featTable = array2table(feats, 'VariableNames', cellstr(featNames)) ;
featTable = addvars(featTable, meta.filename(1:numClips), ...
                    'Before', 1, 'NewVariableNames', 'filename') ;
if isempty(nTest)
    outFile = 'results/acoustic_features.csv' ;
else
    outFile = 'results/acoustic_features_test.csv' ;
end
writetable(featTable, outFile) ;
fprintf('Saved %s : %d rows x %d features\n', outFile, height(featTable), numel(featNames)) ;

%% Local functions
% Simple percentile (sort and index), so the Statistics Toolbox is not needed.

function p = Percentile(v, q)
v = sort(v(:)) ;
idx = max(1, min(numel(v), round(q/100 * numel(v)))) ;
p = v(idx) ;
end
