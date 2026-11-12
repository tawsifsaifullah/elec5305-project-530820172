%% plot_case_studies.m
% ELEC5305 Project - acoustic case studies of human/machine disagreement
% Saifullah Tawsif (530820172)
%
% For each selected clip, plots (one column per clip):
%   row 1 - waveform                      (Week 1B)
%   row 2 - spectrogram                   (Week 4: Hamming window STFT)
%   row 3 - F0 contour, voiced frames     (Week 7 source-filter idea; pitch()
%                                          NCF + harmonicRatio voicing, same
%                                          settings as extract_acoustic_features.m)
%   row 4 - frame intensity (RMS dB)      (Week 6C)
%
% Run from the repository root. Saves results/figures/fig6_case_studies.png

clear all ; close all ; clc ;

%% ------------------------- Clips to compare ------------------------------
% Case 1: actor 13 "calm" - listeners 100% correct, both SSL models -> neutral
% Reference: actor 13 "neutral", same sentence and repetition
% Case 2: duplicate happy clip - listeners 0.2 (twin file 0.7), models correct
files  = ["03-01-02-01-01-02-13.wav", ...
          "03-01-01-01-01-02-13.wav", ...
          "03-01-03-01-02-02-07.wav"] ;
titles = ["Actor 13 CALM (humans 1.0, models: neutral)", ...
          "Actor 13 NEUTRAL (reference)", ...
          "Actor 7 HAPPY (humans 0.2, models correct)"] ;

%% ------------------------- Settings (match feature script) --------------
fs = 16000 ;
winLen  = round(0.025*fs) ; hopSize = round(0.010*fs) ;
pWinLen = round(0.040*fs) ;
voicedThresh = 0.65 ; vadThreshDb = 35 ;

numClips = numel(files) ;
figure('Position', [50 50 1400 900]) ;

for c = 1:numClips
    % ---- read audio (path from filename: last field = actor) ----
    parts = split(files(c), "-") ;
    actor = str2double(extractBefore(parts(7), ".wav")) ;
    path  = sprintf('data/ravdess_speech/Actor_%02d/%s', actor, files(c)) ;
    [sig, fsOrig] = audioread(path) ;
    sig = resample(mean(sig, 2), fs, fsOrig) ;
    t = (0:length(sig)-1) / fs ;

    % ---- gender-specific F0 range (odd actor = male) ----
    if mod(actor, 2) == 1
        f0Range = [60 350] ;
    else
        f0Range = [100 500] ;
    end

    % ---- row 1: waveform ----
    subplot(4, numClips, c) ;
    plot(t, sig) ; xlim([0 t(end)]) ;
    title(titles(c), 'FontSize', 9) ; ylabel('Amplitude') ;

    % ---- row 2: spectrogram (Week 4) ----
    subplot(4, numClips, numClips + c) ;
    spectrogram(sig, hamming(winLen, 'periodic'), winLen - hopSize, 512, fs, 'yaxis') ;
    ylim([0 5]) ; colorbar off ; title('Spectrogram (0-5 kHz)', 'FontSize', 9) ;

    % ---- row 3: F0 contour on voiced frames (Week 7) ----
    [f0, loc] = pitch(sig, fs, 'Method', 'NCF', 'Range', f0Range, ...
                      'WindowLength', pWinLen, 'OverlapLength', pWinLen - hopSize) ;
    hr = harmonicRatio(sig, fs, 'Window', hamming(pWinLen, 'periodic'), ...
                       'OverlapLength', pWinLen - hopSize) ;
    nP = min(length(f0), length(hr)) ;
    f0 = medfilt1(f0(1:nP), 5) ; hr = hr(1:nP) ; loc = loc(1:nP) ;
    pitchDb = zeros(nP, 1) ;
    for k = 1:nP
        seg = sig(loc(k)-pWinLen+1 : loc(k)) ;
        pitchDb(k) = 20*log10(sqrt(mean(seg.^2)) + 1e-10) ;
    end
    voiced = (hr > voicedThresh) & (pitchDb > max(pitchDb) - vadThreshDb) ;
    f0Plot = f0 ; f0Plot(~voiced) = NaN ;          % gaps where unvoiced
    subplot(4, numClips, 2*numClips + c) ;
    plot(loc / fs, f0Plot, '.-') ; xlim([0 t(end)]) ; ylim([50 400]) ;
    ylabel('F0 (Hz)') ;
    title(sprintf('F0: mean %.0f Hz, range %.1f st', mean(f0(voiced)), ...
          12*log2(prctile_simple(f0(voiced),95) / prctile_simple(f0(voiced),5))), 'FontSize', 9) ;

    % ---- row 4: frame intensity (Week 6C) ----
    numFrames = floor((length(sig) - winLen) / hopSize) + 1 ;
    rmsDb = zeros(numFrames, 1) ; win = hamming(winLen, 'periodic') ;
    for k = 1:numFrames
        grain = sig((k-1)*hopSize+1 : (k-1)*hopSize+winLen) .* win ;
        rmsDb(k) = 20*log10(sqrt(mean(grain.^2)) + 1e-10) ;
    end
    speech = rmsDb > max(rmsDb) - vadThreshDb ;
    subplot(4, numClips, 3*numClips + c) ;
    tf = ((0:numFrames-1) * hopSize + winLen/2) / fs ;
    plot(tf, rmsDb) ; xlim([0 t(end)]) ; ylim([-90 -10]) ;
    xlabel('Time (s)') ; ylabel('Level (dB)') ;
    title(sprintf('Intensity: mean %.1f dB (speech frames)', mean(rmsDb(speech))), 'FontSize', 9) ;
end

sgtitle('Figure 6: Acoustic case studies of human-machine disagreement') ;
exportgraphics(gcf, 'results/figures/fig6_case_studies.png', 'Resolution', 200) ;
disp('Saved results/figures/fig6_case_studies.png') ;

%% ------------------------- Local function --------------------------------
function p = prctile_simple(v, q)
v = sort(v(:)) ;
p = v(max(1, min(numel(v), round(q/100 * numel(v))))) ;
end