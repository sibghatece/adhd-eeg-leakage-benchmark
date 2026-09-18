%% make_figures.m
%  Generates the eight main-text figures from the stage outputs.
%  Reads: Stage9_Master\T_stage9_master.csv, Stage9b_EEGNet\T_stage9b_eegnet.csv,
%         Stage7b_Overlap\T_stage7b_overlap.csv, Stage4_Amplitude\T_channel_auc.csv,
%         Stage5_Metadata\T_stage5_subjects.csv, Stage2b_Features\D3_MENDELEY_features.mat,
%         Stage2b_Features\D4_BALLADEER_features.mat, Stage10_Riemannian\T_stage10.csv,
%         Stage10_Riemannian\T_stage10b.csv, Stage3_Results\T_transfer.csv,
%         Stage4_Amplitude\T_stage4_transfer.csv
%  Writes: Figures\Fig1_design.pdf/png ... Fig8_transfer.pdf/png
%  No readtable/readcell; CSVs are parsed with a local function.

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
OUT = fullfile(ROOT, 'Figures'); if ~isfolder(OUT), mkdir(OUT); end
set(groot, 'defaultAxesFontName', 'Arial', 'defaultAxesFontSize', 8, 'defaultTextFontName', 'Arial', 'defaultTextFontSize', 8, 'defaultAxesLineWidth', 0.6, 'defaultLineLineWidth', 1);
col = [0.00 0.45 0.70; 0.84 0.37 0.00; 0.00 0.62 0.45; 0.80 0.47 0.65; 0.94 0.89 0.26; 0.34 0.71 0.91; 0.90 0.62 0.00; 0.50 0.50 0.50; 0.10 0.10 0.10; 0.60 0.20 0.20];   % Okabe-Ito
fams = {'BandPower', 'LogVar', 'timedomain', 'entropy', 'nonlinear', 'wavelet', 'vmd', 'PSR', 'riemannian'};
famLab = {'Spectral', 'Amplitude', 'Time domain', 'Entropy', 'Fractal', 'Wavelet', 'VMD', 'EMD-EWT PSR', 'Riemannian'};
cohorts = {'D1_IEEE', 'D2m_TDBRAIN', 'D3_MENDELEY', 'D4m_BALLADEER', 'D5_OSF'};
cohLab = {'D1 IEEE', 'D2 TDBRAIN', 'D3 Mendeley', 'D4 BALLADEER', 'D5 OSF'};
exportBoth = @(fig, name) exportPair(fig, fullfile(OUT, name));

%% ---------------- data
M = readCsv(fullfile(ROOT, 'Stage9_Master', 'T_stage9_master.csv'));
E = readCsv(fullfile(ROOT, 'Stage9b_EEGNet', 'T_stage9b_eegnet.csv'));
getM = @(c, f, p, clf, lvl, key) pick(M, {'cohort', c, 'family', f, 'protocol', p, 'classifier', clf, 'level', lvl}, key);
getE = @(c, p, lvl, key) pick(E, {'cohort', c, 'protocol', p, 'level', lvl}, key);
nC = numel(cohorts); nF = numel(fams);
balP3 = nan(nC, nF); sdP3 = balP3; balP1 = balP3; balP3e = balP3; eegP3 = nan(nC, 1); eegP1 = eegP3; eegP3e = eegP3;
for c = 1:nC
    for f = 1:nF
        balP3(c, f) = getM(cohorts{c}, fams{f}, 'P3_subjectCV', 'RF', 'subject', 'BAL'); sdP3(c, f) = getM(cohorts{c}, fams{f}, 'P3_subjectCV', 'RF', 'subject', 'BAL_sd');
        balP3e(c, f) = getM(cohorts{c}, fams{f}, 'P3_subjectCV', 'RF', 'epoch', 'BAL'); balP1(c, f) = getM(cohorts{c}, fams{f}, 'P1_epochCV', 'RF', 'epoch', 'BAL');
    end
    eegP3(c) = getE(cohorts{c}, 'P3_subjectCV', 'subject', 'BAL'); eegP3e(c) = getE(cohorts{c}, 'P3_subjectCV', 'epoch', 'BAL'); eegP1(c) = getE(cohorts{c}, 'P1_epochCV', 'epoch', 'BAL');
end

%% ================= FIGURE 1: study design (drawn)
fig = figure('Units', 'centimeters', 'Position', [2 2 17 9], 'Color', 'w'); ax = axes('Position', [0 0 1 1]); axis off; hold on; xlim([0 17]); ylim([0 9]);
box_ = @(x, y, w, h, txt, c) drawBox(x, y, w, h, txt, c);
arrow = @(x1, y1, x2, y2) annotation('arrow', 'Units', 'centimeters', 'Position', [x1 y1 x2 - x1 y2 - y1], 'HeadLength', 5, 'HeadWidth', 5, 'LineWidth', 0.8);
% column 1: datasets
ds = {sprintf('D1 IEEE DataPort\nchildren, task\n121 / 19 ch'), sprintf('D2 TDBRAIN\nadults, clinic, rest\n168 matched / 26 ch'), sprintf('D3 Mendeley\nadults, task\n79 / 2 ch'), sprintf('D4 BALLADEER\nchildren, one protocol\n86 matched / 29 ch'), sprintf('D5 OSF flanker\nchildren, CSD trials\n144 / 56 ch')};
for i = 1:5, box_(0.3, 8.2 - 1.65 * i, 3.4, 1.4, ds{i}, [0.93 0.95 1.00]); end
text(2.0, 8.6, 'Datasets (617 participants)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% column 2: harmonisation
box_(4.4, 3.0, 3.0, 3.4, sprintf('Harmonised\npreprocessing\n\n128 Hz, 0.5-45 Hz\nnotch, 2 s epochs\nbad-channel and\nepoch rejection\nage / sex matching'), [0.95 0.95 0.95]);
for i = 1:5, arrow(3.7, 8.2 - 1.65 * i + 0.7, 4.4, 4.7); end
% column 3: families
box_(8.1, 3.0, 3.2, 3.4, sprintf('Ten method families\n\nspectral, amplitude,\ntime domain, entropy,\nfractal, wavelet, VMD,\nEMD-EWT phase space,\nRiemannian covariance,\nEEGNet'), [0.95 0.95 0.95]);
arrow(7.4, 4.7, 8.1, 4.7);
% column 4: protocols
box_(12.0, 5.6, 4.7, 1.6, sprintf('P3 leakage-free: participant-grouped nested CV,\nfold-internal selection, uniform priors'), [0.90 0.97 0.92]);
box_(12.0, 3.8, 4.7, 1.6, sprintf('P1 segment-level CV and\nP0 overlapping windows (0-90 %%)'), [1.00 0.94 0.90]);
box_(12.0, 2.0, 4.7, 1.6, sprintf('Confound, specificity and transfer tests\n(TDBRAIN other diagnoses; BALLADEER positive control)'), [0.97 0.93 1.00]);
arrow(11.3, 4.7, 12.0, 6.4); arrow(11.3, 4.7, 12.0, 4.6); arrow(11.3, 4.7, 12.0, 2.8);
text(14.35, 7.6, 'Evaluation', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(0.3, 0.55, 'Outputs: balanced accuracy, AUC, MCC at epoch and participant level; permutation p; inflation = P1 - P3', 'FontSize', 7.5);
exportBoth(fig, 'Fig1_design');

%% ================= FIGURE 2: inflation heat-map
infl = [balP1 - balP3e, eegP1 - eegP3e];
fig = figure('Units', 'centimeters', 'Position', [2 2 12 7], 'Color', 'w');
imagesc(infl); colormap(parula); cb = colorbar; cb.Label.String = 'Inflation (balanced accuracy points, P1 - P3)';
set(gca, 'XTick', 1:nF + 1, 'XTickLabel', [famLab {'EEGNet'}], 'XTickLabelRotation', 40, 'YTick', 1:nC, 'YTickLabel', cohLab);
for c = 1:nC, for f = 1:nF + 1, text(f, c, sprintf('%+.0f', infl(c, f)), 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', ifelse(infl(c, f) > 25, 'k', 'w')); end, end
title('Inflation from segment-level cross-validation');
exportBoth(fig, 'Fig2_inflation');

%% ================= FIGURE 3: overlap curves
O = readCsv(fullfile(ROOT, 'Stage7b_Overlap', 'T_stage7b_overlap.csv'));
ovCoh = {'D1_IEEE_19ch', 'D2m_TDBRAIN_26ch', 'D3_MENDELEY_2ch'}; ovLab = {'D1 IEEE', 'D2 TDBRAIN', 'D3 Mendeley'};
fig = figure('Units', 'centimeters', 'Position', [2 2 17 5.5], 'Color', 'w');
for i = 1:3
    subplot(1, 3, i); hold on; grid on
    for k = 1:2
        fsn = ifelse(k == 1, 'LogVar', 'BandPower'); ls = ifelse(k == 1, '-', '--');
        for pr = 1:2
            prot = ifelse(pr == 1, 'P0_windowCV', 'P3_subjectCV'); mk = ifelse(pr == 1, 'o', 's');
            ov = []; acc = [];
            for r = 1:numel(O.rows)
                if strcmp(O.rows{r}.cohort, ovCoh{i}) && strcmp(O.rows{r}.featureset, fsn) && strcmp(O.rows{r}.protocol, prot), ov(end+1) = str2double(O.rows{r}.overlap); acc(end+1) = str2double(O.rows{r}.ACC); end %#ok<AGROW>
            end
            [ov, o] = sort(ov); plot(100 * ov, acc(o), [ls mk], 'Color', col(pr, :), 'MarkerFaceColor', col(pr, :), 'MarkerSize', 4);
        end
    end
    xlabel('Window overlap (%)'); if i == 1, ylabel('Accuracy (%)'); end; ylim([50 100]); title(ovLab{i});
    if i == 3, legend({'Amplitude, window split', 'Amplitude, participant split', 'Spectral, window split', 'Spectral, participant split'}, 'Location', 'southeast', 'FontSize', 6.5); end
end
exportBoth(fig, 'Fig3_overlap');

%% ================= FIGURE 4: leakage-free benchmark dot plot
fig = figure('Units', 'centimeters', 'Position', [2 2 17 6.5], 'Color', 'w'); hold on; grid on
x0 = 1:nC; w = 0.08;
for f = 1:nF, errorbar(x0 + (f - 5) * w, balP3(:, f), sdP3(:, f), 'o', 'Color', col(f, :), 'MarkerFaceColor', col(f, :), 'MarkerSize', 4, 'CapSize', 2); end
plot(x0 + 5 * w, eegP3, 'd', 'Color', col(10, :), 'MarkerFaceColor', col(10, :), 'MarkerSize', 5);
yline(50, 'k:', 'chance'); set(gca, 'XTick', x0, 'XTickLabel', cohLab); ylabel('Participant-level balanced accuracy (%)'); ylim([40 100]); xlim([0.4 5.6]);
legend([famLab {'EEGNet'}], 'Location', 'northoutside', 'Orientation', 'horizontal', 'NumColumns', 5, 'FontSize', 6.5);
exportBoth(fig, 'Fig4_benchmark');

%% ================= FIGURE 5: (a) topography, (b) specificity scores
A = readCsv(fullfile(ROOT, 'Stage4_Amplitude', 'T_channel_auc.csv'));
chn = {}; auc = []; dd = [];
for r = 1:numel(A.rows), if startsWith(A.rows{r}.cohort, 'D2'), chn{end+1} = A.rows{r}.channel; auc(end+1) = str2double(A.rows{r}.AUC); dd(end+1) = str2double(A.rows{r}.cohen_d); end, end %#ok<AGROW>
pos = scalpPositions(chn);
fig = figure('Units', 'centimeters', 'Position', [2 2 17 7], 'Color', 'w');
subplot(1, 2, 1); hold on; axis equal off
th = linspace(0, 2 * pi, 200); plot(cos(th), sin(th), 'k'); plot([-0.1 0 0.1], [1 1.12 1], 'k');
scatter(pos(:, 1), pos(:, 2), 180, 1 - auc(:), 'filled', 'MarkerEdgeColor', 'k');   % 1-AUC so that ADHD < HC maps to >0.5
for i = 1:numel(chn), text(pos(i, 1), pos(i, 2) - 0.13, chn{i}, 'HorizontalAlignment', 'center', 'FontSize', 6); end
colormap(gca, flipud(hot)); cb = colorbar('southoutside'); cb.Label.String = 'AUC (healthy > ADHD, log power 1-30 Hz)'; caxis([0.5 0.75]);
title('(a) TDBRAIN, per-channel AUC'); xlim([-1.3 1.3]); ylim([-1.3 1.3]);
% (b) specificity scores: recompute ADHD-vs-HC model scores (fast) from the Stage 5 subject table
S = readCsv(fullfile(ROOT, 'Stage5_Metadata', 'T_stage5_subjects.csv'));
grp = cellfun(@(r) r.group, S.rows, 'UniformOutput', false); age = cellfun(@(r) str2double(r.age), S.rows); sex = cellfun(@(r) str2double(r.sex), S.rows);
chCols = find(endsWith(S.header, '_lp')); X = cell2mat(cellfun(@(r) cellfun(@str2double, r.values(chCols)), S.rows, 'UniformOutput', false)');
isHC = strcmp(grp, 'HC'); isA = strcmp(grp, 'ADHD'); oth = {'MDD', 'OCD', 'INSOMNIA'};
kk = matchGroups(find(isA), find(isHC), age, sex, 2, 2026); pos_ = kk{1}; neg_ = kk{2};
[cvScore, cvLab, othScore] = adhdModelScores(X, pos_, neg_, find(ismember(grp, oth)), 5, 200, 2026);
subplot(1, 2, 2); hold on; grid on
data = {cvScore(cvLab == 0), cvScore(cvLab == 1)}; for k = 1:3, data{end+1} = othScore(strcmp(grp(ismember(grp, oth)), oth{k})); end %#ok<AGROW>
labs = {'Healthy (CV)', 'ADHD (CV)', 'Depression', 'OCD', 'Insomnia'};
for k = 1:5
    v = data{k}; q = prctile(v, [25 50 75]);
    rectangle('Position', [k - 0.3, q(1), 0.6, q(3) - q(1)], 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', 'k'); plot([k - 0.3 k + 0.3], [q(2) q(2)], 'k', 'LineWidth', 1.5);
    scatter(k + 0.25 * (rand(numel(v), 1) - 0.5), v, 8, col(ifelse(k <= 2, k, 3), :), 'filled', 'MarkerFaceAlpha', 0.5);
    text(k, 1.04, sprintf('n=%d', numel(v)), 'HorizontalAlignment', 'center', 'FontSize', 6.5);
end
yline(0.5, 'k--'); set(gca, 'XTick', 1:5, 'XTickLabel', labs, 'XTickLabelRotation', 20); ylabel('P(ADHD) from ADHD-vs-healthy model'); ylim([0 1.1]);
title('(b) Specificity test');
exportBoth(fig, 'Fig5_amplitude_specificity');

%% ================= FIGURE 6: global log power by group across cohorts
fig = figure('Units', 'centimeters', 'Position', [2 2 17 6], 'Color', 'w');
% D2 groups from Stage 5 subject table (global_logpower_1_30 column)
gcol = find(strcmp(S.header, 'global_logpower_1_30')); glob = cellfun(@(r) str2double(r.values{gcol}), S.rows);
subplot(1, 3, 1); hold on; grid on; gl = {'HC', 'ADHD', 'MDD', 'OCD', 'INSOMNIA'}; gt = {'Healthy', 'ADHD', 'Depr.', 'OCD', 'Insom.'};
for k = 1:5, v = glob(strcmp(grp, gl{k})); boxdots(k, v, col(ifelse(k == 1, 1, 2), :)); end
set(gca, 'XTick', 1:5, 'XTickLabel', gt); ylabel('Global log power 1-30 Hz'); title('D2 TDBRAIN adults (rest)');
% D3 and D4 from Stage 2b feature stores: subject mean of log variance over channels
for i = 1:2
    T = load(fullfile(ROOT, 'Stage2b_Features', ifelse(i == 1, 'D3_MENDELEY_features.mat', 'D4_BALLADEER_features.mat'))); F = T.F;
    [su, ~, k] = unique(F.subject); lab = accumarray(k, double(F.label), [], @(v) v(1)); lv = accumarray(k, mean(F.logvar, 2, 'omitnan'), [], @mean);
    if i == 2   % matched subset
        keep = matchGroups(find(lab == 1), find(lab == 0), accumarray(k, double(F.age), [], @(v) v(1)), accumarray(k, double(F.sex), [], @(v) v(1)), 1, 2026); m = false(size(su)); m([keep{1}; keep{2}]) = true; lab = lab(m); lv = lv(m);
    end
    subplot(1, 3, i + 1); hold on; grid on
    boxdots(1, lv(lab == 0), col(1, :)); boxdots(2, lv(lab == 1), col(2, :));
    set(gca, 'XTick', 1:2, 'XTickLabel', {'Healthy', 'ADHD'}); ylabel('Mean log variance over channels');
    title(ifelse(i == 1, 'D3 Mendeley adults (task)', 'D4 BALLADEER children, matched'));
end
exportBoth(fig, 'Fig6_amplitude_direction');

%% ================= FIGURE 7: Riemannian montage and variants
B = readCsv(fullfile(ROOT, 'Stage10_Riemannian', 'T_stage10b.csv'));
fig = figure('Units', 'centimeters', 'Position', [2 2 17 6], 'Color', 'w');
mont = {'frontal7', 'central9', 'all19'}; mLab = {'7 frontal', '9 central', '19 all'};
for i = 1:2
    subplot(1, 2, i); hold on; grid on; cohN = ifelse(i == 1, 'IEEE', 'BALLADEER');
    for v = 1:2
        var_ = ifelse(v == 1, 'full', 'corr'); b = nan(1, 3); s = b;
        for m = 1:3, for r = 1:numel(B.rows), R = B.rows{r}; if strcmp(R.cohort, cohN) && strcmp(R.montage, mont{m}) && strcmp(R.variant, var_) && strcmp(R.reference, 'native'), b(m) = str2double(R.BAL); s(m) = str2double(R.BAL_sd); end, end, end
        errorbar((1:3) + (v - 1.5) * 0.25, b, s, 'o', 'Color', col(v, :), 'MarkerFaceColor', col(v, :), 'MarkerSize', 5, 'CapSize', 3);
    end
    yline(50, 'k:'); if i == 1, yline(73.5, '--', 'Color', [0.4 0.4 0.4], 'Label', 'published nested-CV benchmark, 7 ch', 'FontSize', 6.5); end
    set(gca, 'XTick', 1:3, 'XTickLabel', mLab); ylim([40 90]); ylabel('Participant-level balanced accuracy (%)');
    title(ifelse(i == 1, 'D1 IEEE', 'D4 BALLADEER, matched, same montage')); if i == 1, legend({'Full tangent vector', 'Correlation (amplitude-free)'}, 'Location', 'northwest', 'FontSize', 6.5); end
end
exportBoth(fig, 'Fig7_riemannian');

%% ================= FIGURE 8: transfer matrix
% participant-level AUC for source -> target; feature sets: amplitude (Cz/F4 log power), phase-space (Cz/F4), Riemannian (19 ch children)
T3 = readCsv(fullfile(ROOT, 'Stage3_Results', 'T_transfer.csv')); T4 = readCsv(fullfile(ROOT, 'Stage4_Amplitude', 'T_stage4_transfer.csv')); T10 = readCsv(fullfile(ROOT, 'Stage10_Riemannian', 'T_stage10.csv'));
names = {'D1 IEEE', 'D2 TDBRAIN', 'D3 Mendeley', 'D4 BALLADEER'}; keys = {'D1_IEEE', 'D2m_TDBRAIN', 'D3_MENDELEY', 'D4m_BALLADEER'};
sets = {'Phase space (Cz,F4)', 'Amplitude (Cz,F4)', 'Riemannian (19 ch)'}; Mx = nan(4, 4, 3);
for r = 1:numel(T3.rows), R = T3.rows{r}; a = find(strcmp(keys, strrep(R.train, '_matched', ''))); b = find(strcmp(keys, strrep(R.test, '_matched', ''))); if ~isempty(a) && ~isempty(b), Mx(a, b, 1) = str2double(R.subject_AUC); end, end
for r = 1:numel(T4.rows), R = T4.rows{r}; if strcmp(R.featureset, 'LV_1_30'), a = find(strcmp(keys, R.train)); b = find(strcmp(keys, R.test)); if ~isempty(a) && ~isempty(b), Mx(a, b, 2) = str2double(R.subject_AUC); end, end, end
for r = 1:numel(T10.rows), R = T10.rows{r}; if startsWith(R.experiment, 'transfer_perDataset') && strcmp(R.variant, 'full'), if strcmp(R.cohort, 'IEEE->BALLADEER'), Mx(1, 4, 3) = str2double(R.AUC); elseif strcmp(R.cohort, 'BALLADEER->IEEE'), Mx(4, 1, 3) = str2double(R.AUC); end, end, end
fig = figure('Units', 'centimeters', 'Position', [2 2 17 5.5], 'Color', 'w');
for s = 1:3
    subplot(1, 3, s); imagesc(Mx(:, :, s), [0 1]); colormap(gca, parula); axis square
    set(gca, 'XTick', 1:4, 'XTickLabel', names, 'XTickLabelRotation', 30, 'YTick', 1:4, 'YTickLabel', names); xlabel('Test cohort'); if s == 1, ylabel('Training cohort'); end; title(sets{s});
    for a = 1:4, for b = 1:4, if ~isnan(Mx(a, b, s)), text(b, a, sprintf('%.2f', Mx(a, b, s)), 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', ifelse(Mx(a, b, s) > 0.6, 'k', 'w')); end, end, end
end
cb = colorbar('Position', [0.93 0.2 0.015 0.6]); cb.Label.String = 'Participant-level AUC';
exportBoth(fig, 'Fig8_transfer');
fprintf('Figures written to %s\n', OUT);

%% ======================================================== LOCAL FUNCTIONS
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
function exportPair(fig, base)
    exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector');
    exportgraphics(fig, [base '.png'], 'Resolution', 600);
end
function drawBox(x, y, w, h, txt, c)
    rectangle('Position', [x y w h], 'Curvature', 0.15, 'FaceColor', c, 'EdgeColor', [0.3 0.3 0.3]);
    text(x + w/2, y + h/2, txt, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 7.5);
end
function C = readCsv(p)
    txt = fileread(p); lines = strsplit(strtrim(txt), {'\r\n', '\n'}); hdr = strsplit(lines{1}, ',');
    rows = {};
    for i = 2:numel(lines)
        if isempty(strtrim(lines{i})), continue; end
        v = strsplit(lines{i}, ',', 'CollapseDelimiters', false); if numel(v) < numel(hdr), v(end+1:numel(hdr)) = {''}; end
        r = struct(); for j = 1:numel(hdr), r.(matlab.lang.makeValidName(hdr{j})) = v{j}; end; r.values = v; rows{end+1} = r; %#ok<AGROW>
    end
    C.header = hdr; C.rows = rows;
end
function v = pick(C, conds, key)
    v = NaN;
    for r = 1:numel(C.rows)
        ok = true; for k = 1:2:numel(conds), if ~strcmp(C.rows{r}.(matlab.lang.makeValidName(conds{k})), conds{k+1}), ok = false; break; end, end
        if ok, v = str2double(C.rows{r}.(matlab.lang.makeValidName(key))); return; end
    end
end
function boxdots(k, v, c)
    q = prctile(v, [25 50 75]); rectangle('Position', [k - 0.3, q(1), 0.6, max(q(3) - q(1), 1e-6)], 'FaceColor', [0.92 0.92 0.92], 'EdgeColor', 'k');
    plot([k - 0.3 k + 0.3], [q(2) q(2)], 'k', 'LineWidth', 1.5); scatter(k + 0.25 * (rand(numel(v), 1) - 0.5), v, 8, c, 'filled', 'MarkerFaceAlpha', 0.5);
    text(k, max(v) + 0.05 * range(v), sprintf('n=%d', numel(v)), 'HorizontalAlignment', 'center', 'FontSize', 6.5);
end
function pos = scalpPositions(chn)
    % approximate 2-D 10-10 positions (x right, y anterior), unit head radius
    map = containers.Map();
    map('Fp1') = [-0.31 0.95]; map('Fp2') = [0.31 0.95]; map('F7') = [-0.81 0.59]; map('F3') = [-0.55 0.55]; map('Fz') = [0 0.62]; map('F4') = [0.55 0.55]; map('F8') = [0.81 0.59];
    map('FC3') = [-0.5 0.3]; map('FCz') = [0 0.31]; map('FC4') = [0.5 0.3]; map('T7') = [-1 0]; map('C3') = [-0.5 0]; map('Cz') = [0 0]; map('C4') = [0.5 0]; map('T8') = [1 0];
    map('CP3') = [-0.5 -0.3]; map('CPz') = [0 -0.31]; map('CP4') = [0.5 -0.3]; map('P7') = [-0.81 -0.59]; map('P3') = [-0.55 -0.55]; map('Pz') = [0 -0.62]; map('P4') = [0.55 -0.55]; map('P8') = [0.81 -0.59];
    map('O1') = [-0.31 -0.95]; map('Oz') = [0 -1]; map('O2') = [0.31 -0.95];
    pos = zeros(numel(chn), 2); for i = 1:numel(chn), if isKey(map, chn{i}), pos(i, :) = map(chn{i}); end, end
end
function keep = matchGroups(pos, cand, age, sex, ratio, seed)
    pos = pos(:); cand = cand(:); age = age(:); sex = sex(:);
    rng(seed); pos = pos(randperm(numel(pos))); used = false(size(cand)); neg = [];
    for i = pos'
        for r = 1:ratio
            c = find(~used & sex(cand) == sex(i)); if isempty(c), c = find(~used); end; if isempty(c), break; end
            [~, j] = min(abs(age(cand(c)) - age(i))); used(c(j)) = true; neg(end+1) = cand(c(j)); %#ok<AGROW>
        end
    end
    keep = {pos(:), neg(:)};
end
function [cvScore, cvLab, othScore] = adhdModelScores(X, pos, neg, others, K, nTrees, seed)
    idx = [pos; neg]; y = [ones(numel(pos), 1); zeros(numel(neg), 1)]; rng(seed);
    folds = zeros(size(y)); for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
    cvScore = zeros(numel(idx), 1); othAcc = zeros(numel(others), 1);
    for f = 1:K
        tr = folds ~= f; te = folds == f; mu = mean(X(idx(tr), :)); sd = std(X(idx(tr), :)); sd(sd < 1e-12) = 1;
        mdl = TreeBagger(nTrees, (X(idx(tr), :) - mu) ./ sd, y(tr), 'Method', 'classification', 'Prior', 'Uniform');
        [~, sc] = predict(mdl, (X(idx(te), :) - mu) ./ sd); cvScore(te) = sc(:, strcmp(mdl.ClassNames, '1'));
        [~, so] = predict(mdl, (X(others, :) - mu) ./ sd); othAcc = othAcc + so(:, strcmp(mdl.ClassNames, '1'));
    end
    cvLab = y; othScore = othAcc / K;
end
