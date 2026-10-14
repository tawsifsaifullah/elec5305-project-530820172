%% ELEC5305 - Acoustic analysis of RAVDESS emotional speech
% Compares a neutral and an emotional utterance from the SAME actor and
% sentence: waveform, spectrogram (STFT), F0 contour, intensity contour, MFCCs.
% Requires: Signal Processing Toolbox + Audio Toolbox (pitch, mfcc).
%
% Usage: set dataRoot below, then run. Figures are saved to results/figures/.

clear; close all;
dataRoot = fullfile('..', 'data', 'ravdess_speech');
outDir   = fullfile('..', 'results', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

actor = 1;
% Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor
files = { sprintf('03-01-01-01-01-01-%02d.wav', actor), ...   % neutral
          sprintf('03-01-05-02-01-01-%02d.wav', actor) };     % angry, strong
labels = {'Neutral', 'Angry (strong)'};

fsTarget = 16000;
win  = round(0.032 * fsTarget);   % 32 ms analysis window
hop  = round(0.010 * fsTarget);   % 10 ms hop
nfft = 512;

figure('Position', [100 100 1100 850]);
summary = table();
for k = 1:2
    f = fullfile(dataRoot, sprintf('Actor_%02d', actor), files{k});
    [x, fs] = audioread(f);
    x = resample(mean(x, 2), fsTarget, fs);
    x = trimSilence(x, fsTarget);
    t = (0:numel(x)-1) / fsTarget;

    % --- waveform ---
    subplot(4, 2, k);
    plot(t, x); xlabel('Time (s)'); ylabel('Amplitude'); title(labels{k}); grid on;

    % --- spectrogram (STFT) ---
    subplot(4, 2, 2 + k);
    [S, F, T] = spectrogram(x, hamming(win), win - hop, nfft, fsTarget);
    imagesc(T, F, 20*log10(abs(S) + eps)); axis xy; ylim([0 5000]);
    xlabel('Time (s)'); ylabel('Frequency (Hz)'); colorbar; caxis([-80 0] + max(caxis));

    % --- F0 contour ---
    subplot(4, 2, 4 + k);
    [f0, loc] = pitch(x, fsTarget, 'Range', [60 500], ...
                      'WindowLength', 2*win, 'OverlapLength', 2*win - hop);
    e = movmean(x.^2, 2*win);
    voiced = e(loc) > 0.01 * max(e);           % simple energy-based voicing
    f0(~voiced) = NaN;
    plot(loc / fsTarget, f0, '.'); ylim([50 500]);
    xlabel('Time (s)'); ylabel('F0 (Hz)'); grid on;

    % --- intensity contour ---
    subplot(4, 2, 6 + k);
    frames = buffer(x, win, win - hop, 'nodelay');
    rmsDb = 20*log10(sqrt(mean(frames.^2, 1)) + 1e-8);
    plot((0:numel(rmsDb)-1) * hop / fsTarget, rmsDb);
    xlabel('Time (s)'); ylabel('Intensity (dB)'); grid on;

    % --- MFCC statistics ---
    c = mfcc(x, fsTarget, 'Window', hamming(win, 'periodic'), 'OverlapLength', win - hop, ...
             'NumCoeffs', 13);

    f0v = f0(~isnan(f0));
    st  = 12 * log2(f0v / 100);
    row = table(string(labels{k}), mean(f0v), std(st), prctile(st, 95) - prctile(st, 5), ...
                mean(rmsDb(rmsDb > max(rmsDb) - 40)), numel(x) / fsTarget, mean(c(:, 2)), ...
        'VariableNames', {'Condition', 'F0_mean_Hz', 'F0_std_st', 'F0_range_st', ...
                          'Intensity_mean_dB', 'Duration_s', 'MFCC1_mean'});
    summary = [summary; row]; %#ok<AGROW>
end
sgtitle(sprintf('RAVDESS actor %d, "Kids are talking by the door"', actor));
exportgraphics(gcf, fullfile(outDir, 'matlab_neutral_vs_angry.png'), 'Resolution', 150);
disp(summary);
writetable(summary, fullfile('..', 'results', 'matlab_feature_summary.csv'));

function y = trimSilence(x, fs)
    % Remove leading/trailing silence using a 30 dB-below-peak energy threshold
    frame = round(0.02 * fs);
    e = movmean(x.^2, frame);
    idx = find(10*log10(e + eps) > 10*log10(max(e)) - 30);
    y = x(max(idx(1) - frame, 1) : min(idx(end) + frame, numel(x)));
end
