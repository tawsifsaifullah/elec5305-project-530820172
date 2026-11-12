%% extract_acoustic_features.m
% ELEC5305 Project - Interpretable acoustic baseline for RAVDESS speech
% Saifullah Tawsif (530820172)
%
% Computes a compact, physically interpretable feature set for every clip in
% results/metadata.csv and writes results/acoustic_features.csv (one row per
% clip, same order as metadata.csv).
%
% Course labs this script is based on:
%   Week 1B  - reading audio with audioread
%   Week 4   - STFT framing: Hamming window, hop size, fft per frame,
%              20*log10 magnitudes (frame-by-frame analysis loop)
%   Week 6C  - frame RMS level and a simple energy-based voice activity
%              detector (speech frames vs silence frames)
%   Week 7A/B- source-filter model of speech; F0 via correlation of the
%              signal with itself (here MATLAB's pitch() with the NCF
%              method, a normalised autocorrelation estimator)
%   Week 6A  - filterbank analysis of the spectrum; here the mel filterbank
%              + log + DCT inside MATLAB's mfcc() gives cepstral features
%   Week 10B - resample() to change sample rate
%
% F0 uses voiced frames only (harmonic ratio > 0.8), a 5-point median
% filter against octave jumps, and semitone units for F0 variation/range.
%
% Toolboxes: Signal Processing (resample, hamming, medfilt1),
%            Audio (pitch, harmonicRatio, mfcc).
% Run from the repository root:  >> run('matlab/extract_acoustic_features.m')

clear all ; close all ; clc ;

%% ------------------------- Settings -------------------------------------
fs        = 16000 ;            % analysis sample rate (Hz), same as SSL models
winTime   = 0.025 ;            % 25 ms analysis window (Week 4 framing)
hopTime   = 0.010 ;            % 10 ms hop
pitchWinTime = 0.040 ;         % 40 ms window for pitch: must hold >= 2 periods
                               % of the lowest F0 (50 Hz -> 20 ms period)
f0RangeMale   = [60 350] ;     % F0 search range for male actors (Hz)
f0RangeFemale = [100 500] ;    % F0 search range for female actors (Hz)
                               % (similar to Praat defaults 75-300 / 100-500)
vadThreshDb = 35 ;             % speech frames = within 35 dB of loudest frame
voicedThresh = 0.65 ;          % harmonic ratio above this = voiced frame
                               % (0.8 was too strict for low/breathy male voices)
numCoeffs = 13 ;               % MFCC 1-13
nTest     = [] ;               % set to e.g. 5 to test on the first 5 clips

winLen   = round(winTime * fs) ;      % 400 samples
hopSize  = round(hopTime * fs) ;      % 160 samples
pWinLen  = round(pitchWinTime * fs) ; % 640 samples
winFun   = hamming(winLen, 'periodic') ;   % Hamming window (Week 4)
nfft     = 512 ;
freqAxis = (0:nfft/2).' * fs / nfft ;      % one-sided frequency axis (Hz)

%% ------------------------- Load metadata --------------------------------
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

%% ------------------------- Main loop over clips -------------------------
tic ;
for clipIdx = 1:numClips

    % ---- 1. Read audio, mono, resample (Week 1B, Week 10B) ----
    [sig, fsOrig] = audioread(meta.path(clipIdx)) ;
    sig = mean(sig, 2) ;                       % mono
    sig = resample(sig, fs, fsOrig) ;          % 48 kHz -> 16 kHz

    % ---- 2. Frame-by-frame analysis (Week 4 framing loop) ----
    numFrames = floor((length(sig) - winLen) / hopSize) + 1 ;
    frameRms  = zeros(numFrames, 1) ;
    centroid  = zeros(numFrames, 1) ;
    pIn = 0 ;
    for k = 1:numFrames
        grain = sig(pIn+1 : pIn+winLen) .* winFun ;      % windowed frame
        frameRms(k) = sqrt(mean(grain.^2)) ;             % frame RMS (Week 6C)
        frameSpec = abs(fft(grain, nfft)) ;
        frameSpec = frameSpec(1:nfft/2+1) ;              % one-sided magnitude
        % spectral centroid = "centre of mass" of the spectrum (Hz)
        centroid(k) = sum(freqAxis .* frameSpec) / (sum(frameSpec) + eps) ;
        pIn = pIn + hopSize ;
    end
    rmsDb = 20*log10(frameRms + 1e-10) ;                 % level in dB (Week 4/6C)

    % ---- 3. Simple energy VAD (Week 6C idea) ----
    speechFrames = rmsDb > (max(rmsDb) - vadThreshDb) ;

    % ---- 4. Intensity and duration features (Week 6C) ----
    speechDb = rmsDb(speechFrames) ;
    intMean  = mean(speechDb) ;
    intStd   = std(speechDb) ;
    intRange = Percentile(speechDb, 95) - Percentile(speechDb, 5) ;
    speechDur   = sum(speechFrames) * hopTime ;          % seconds of speech
    speechRatio = sum(speechFrames) / numFrames ;        % speech / total

    % ---- 5. Spectral centroid on speech frames (Week 4 spectrum) ----
    centMean = mean(centroid(speechFrames)) ;
    centStd  = std(centroid(speechFrames)) ;

    % ---- 6. F0 with pitch() - normalised correlation (Week 7 idea) ----
    % gender-specific search range stops the tracker locking onto a
    % harmonic (octave error) for low male voices. Gender comes from the
    % actor number in the metadata, not from the emotion label.
    if meta.gender(clipIdx) == "male"
        f0Range = f0RangeMale ;
    else
        f0Range = f0RangeFemale ;
    end
    [f0, loc] = pitch(sig, fs, 'Method', 'NCF', 'Range', f0Range, ...
                      'WindowLength', pWinLen, 'OverlapLength', pWinLen - hopSize) ;
    % Voicing check: pitch() returns a value for EVERY frame, including
    % unvoiced sounds (/s/, /t/, breath) that have no real pitch. The
    % harmonic ratio (same window/hop) is high only when the vocal cords
    % vibrate (the "source" in the Week 7 source-filter model).
    hr = harmonicRatio(sig, fs, 'Window', hamming(pWinLen, 'periodic'), ...
                       'OverlapLength', pWinLen - hopSize) ;
    nP = min(length(f0), length(hr)) ;
    f0 = f0(1:nP) ; hr = hr(1:nP) ; loc = loc(1:nP) ;
    % energy of each pitch frame (same VAD rule as above)
    pitchDb = zeros(nP, 1) ;
    for k = 1:nP
        seg = sig(loc(k)-pWinLen+1 : loc(k)) ;
        pitchDb(k) = 20*log10(sqrt(mean(seg.^2)) + 1e-10) ;
    end
    % median filter removes one-frame octave jumps in the contour
    f0 = medfilt1(f0, 5) ;
    voiced = (hr > voicedThresh) & (pitchDb > (max(pitchDb) - vadThreshDb)) ;
    f0Voiced = f0(voiced) ;
    if numel(f0Voiced) >= 5
        f0Mean = mean(f0Voiced) ;                         % Hz
        f0St   = 12*log2(f0Voiced / 100) ;                % semitones re 100 Hz
        f0Std  = std(f0St) ;                              % variation in semitones
        f0RangeVal = Percentile(f0St, 95) - Percentile(f0St, 5) ;
    else
        f0Mean = NaN ; f0Std = NaN ; f0RangeVal = NaN ;   % filled later in Python
    end

    % ---- 7. MFCCs (Week 6A filterbank -> log -> DCT, via mfcc()) ----
    coeffs = mfcc(sig, fs, 'Window', winFun, 'OverlapLength', winLen - hopSize, ...
                  'NumCoeffs', numCoeffs, 'LogEnergy', 'Ignore') ;
    nUse = min(size(coeffs,1), numFrames) ;              % align frame counts
    coeffs = coeffs(1:nUse, 1:numCoeffs) ;
    mfccSpeech = coeffs(speechFrames(1:nUse), :) ;
    mfccMean = mean(mfccSpeech, 1) ;
    mfccStd  = std(mfccSpeech, 0, 1) ;

    % ---- 8. Store this clip's feature row ----
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

%% ------------------------- Checks and save -------------------------------
% Report NaNs (e.g. too few voiced frames) but leave them in the CSV:
% they are filled inside each training fold in Python, so no information
% leaks from test actors into training.
for c = 1:size(feats, 2)
    nanRows = isnan(feats(:, c)) ;
    if any(nanRows)
        fprintf('Column %s: %d NaN(s) left for fold-wise filling in Python\n', ...
                featNames(c), sum(nanRows)) ;
    end
end
% quick sanity check on pitch by gender (expect male ~110-160 Hz)
isMale = meta.gender(1:numClips) == "male" ;
fprintf('Mean F0: male %.0f Hz, female %.0f Hz\n', ...
        mean(feats(isMale,1), 'omitnan'), mean(feats(~isMale,1), 'omitnan')) ;
fprintf('F0 std (semitones): male %.2f, female %.2f\n', ...
        mean(feats(isMale,2), 'omitnan'), mean(feats(~isMale,2), 'omitnan')) ;
% actor 9 was the worst octave-error case: check its calm/neutral F0
isA9 = meta.actor(1:numClips) == 9 & ismember(meta.emotion(1:numClips), ["calm","neutral"]) ;
if any(isA9)
    fprintf('Actor 9 calm/neutral mean F0: %.0f Hz (expect ~120-170)\n', ...
            mean(feats(isA9,1), 'omitnan')) ;
end

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

%% ------------------------- Local functions -------------------------------
function p = Percentile(v, q)
% Simple percentile without the Statistics Toolbox (sort + index)
v = sort(v(:)) ;
idx = max(1, min(numel(v), round(q/100 * numel(v)))) ;
p = v(idx) ;
end