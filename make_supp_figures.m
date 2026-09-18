%% make_supp_figures.m
%  Supplementary figures S1 (group-mean spectra per cohort) and S5 (single-feature
%  participant-level AUC on D3 Mendeley). Reads the Stage 1b epoch stores and the
%  Stage 2b / Stage 8 feature stores. Writes vector PDF and 600-dpi PNG into <ROOT>\Figures.
%  NaN-marked bad channels (dry-electrode datasets) are skipped in the spectra.

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN1 = fullfile(ROOT, 'Stage1b_Epochs'); IN2 = fullfile(ROOT, 'Stage2b_Features'); IN8 = fullfile(ROOT, 'Stage8_Features');
OUT = fullfile(ROOT, 'Figures'); if ~isfolder(OUT), mkdir(OUT); end
set(groot, 'defaultAxesFontName', 'Arial', 'defaultAxesFontSize', 8, 'defaultTextFontName', 'Arial', 'defaultTextFontSize', 8, 'defaultAxesLineWidth', 0.6, 'defaultLineLineWidth', 1);
col = [0.00 0.45 0.70; 0.84 0.37 0.00];

%% ---------------- Figure S1: group-mean PSD per cohort
stores = {'D1_IEEE_epochs.mat', 'D1 IEEE'; 'D2_TDBRAIN_epochs.mat', 'D2 TDBRAIN'; 'D3_MENDELEY_epochs.mat', 'D3 Mendeley'; 'D4_BALLADEER_epochs.mat', 'D4 BALLADEER'; 'D5_OSF_epochs.mat', 'D5 OSF'};
fig = figure('Units', 'centimeters', 'Position', [2 2 17 10], 'Color', 'w');
for d = 1:size(stores, 1)
    fp = fullfile(IN1, stores{d, 1}); if ~isfile(fp), continue; end
    T = load(fp); S = T.S; fs = S.fs; win = hamming(min(256, size(S.X, 2)));
    [subj, ia] = unique(S.subject); labU = S.label(ia); Pa = []; Pc = []; f = [];
    for g = [1 0]
        subs = subj(labU == g); acc = [];
        for s = subs'
            X = S.X(S.subject == s, :, :); psum = 0; cnt = 0;
            for c = 1:size(X, 3)
                for e = 1:size(X, 1)
                    x = double(squeeze(X(e, :, c)))';
                    if any(~isfinite(x)), continue; end
                    [pxx, f] = pwelch(x, win, [], 256, fs); psum = psum + pxx; cnt = cnt + 1;
                end
            end
            if cnt > 0, acc = [acc, psum / cnt]; end %#ok<AGROW>
        end
        m = mean(acc, 2);
        if g == 1, Pa = m; else, Pc = m; end
    end
    subplot(2, 3, d); hold on; grid on
    plot(f, 10 * log10(Pa + eps), 'Color', col(1, :)); plot(f, 10 * log10(Pc + eps), 'Color', col(2, :));
    xlim([0.5 45]); xlabel('Frequency (Hz)'); if mod(d - 1, 3) == 0, ylabel('Power (dB)'); end; title(stores{d, 2});
    if d == 1, legend({'ADHD', 'Control'}, 'Location', 'northeast', 'FontSize', 6.5); end
end
exportgraphics(fig, fullfile(OUT, 'FigS1_spectra.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT, 'FigS1_spectra.png'), 'Resolution', 600);

%% ---------------- Figure S5: single-feature participant-level AUC on D3
families = {'PSR', 'BandPower', 'LogVar', 'timedomain', 'entropy', 'nonlinear', 'wavelet', 'vmd'};
allAUC = []; allName = {};
for fam = 1:numel(families)
    switch families{fam}
        case {'PSR', 'BandPower', 'LogVar'}
            fp = fullfile(IN2, 'D3_MENDELEY_features.mat'); if ~isfile(fp), continue; end
            T = load(fp); F = T.F;
            switch families{fam}, case 'PSR', X = F.psr; nm = F.psrNames; case 'BandPower', X = F.bp; nm = F.bpNames; case 'LogVar', X = F.logvar; nm = F.lvNames; end
        otherwise
            fp = fullfile(IN8, families{fam}, sprintf('D3_MENDELEY_%s.mat', families{fam})); if ~isfile(fp), continue; end
            T = load(fp); F = T.F; X = F.feat; nm = F.featNames;
    end
    [subj, ~, k] = unique(F.subject); lab = accumarray(k, double(F.label), [], @(v) v(1));
    for j = 1:size(X, 2)
        m = accumarray(k, X(:, j), [], @(v) mean(v, 'omitnan')); ok = ~isnan(m); if nnz(ok) < 4 || std(m(ok)) < 1e-9, continue; end
        [~, ~, ~, a] = perfcurve(lab(ok), m(ok), 1); allAUC(end+1) = max(a, 1 - a); allName{end+1} = sprintf('%s:%s', families{fam}, nm{j}); %#ok<AGROW>
    end
end
fig = figure('Units', 'centimeters', 'Position', [2 2 12 7], 'Color', 'w'); hold on; grid on
histogram(allAUC, 0.5:0.02:1.0, 'FaceColor', [0.4 0.5 0.7]);
xline(prctile(allAUC, 95), 'r--', '95th pct', 'FontSize', 7);
xlabel('Participant-level |AUC| of a single feature'); ylabel('Number of features'); xlim([0.5 1.0]);
[~, imax] = max(allAUC);
title(sprintf('D3 Mendeley: %d features, max AUC %.3f (%s)', numel(allAUC), max(allAUC), strrep(allName{imax}, '_', '\_')));
exportgraphics(fig, fullfile(OUT, 'FigS5_mendeley_single_feature.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT, 'FigS5_mendeley_single_feature.png'), 'Resolution', 600);

fprintf('Top 20 single-feature AUC on D3 (paste into Supplement S5):\n');
[~, o] = sort(allAUC, 'descend'); for i = 1:min(20, numel(o)), fprintf('  %.3f  %s\n', allAUC(o(i)), allName{o(i)}); end
fprintf('\nDONE. Figures written to %s\n', OUT);
