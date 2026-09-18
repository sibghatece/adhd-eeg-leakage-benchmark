%% stage7b_overlap_leakage.m
%  STAGE 7b - Overlapping windows: how near-duplicate samples inflate epoch-level CV.
%  From the Stage 1b epoch stores, each subject's kept epochs are concatenated and re-cut
%  into 2 s windows with 0 %, 50 %, 75 % and 90 % overlap. Features: per-channel log-variance
%  and relative band powers. Classifier: RF 100 trees, uniform prior. Two protocols:
%     P0  window-level random 5-fold (windows from the same subject, sharing samples, on both sides)
%     P3  subject-grouped 5-fold
%  Note: concatenating kept epochs can create a discontinuity at a boundary where an epoch was
%  rejected; this only affects windows that straddle such a boundary and is irrelevant to the
%  leakage mechanism being demonstrated.
%  Outputs: <OUT>\Stage7b_Report.txt, T_stage7b_overlap.csv, Stage7b_overlap.png

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage1b_Epochs'); OUT = fullfile(ROOT, 'Stage7b_Overlap');
R_REPEATS = 3; K = 5; SEED0 = 2026; RF_TREES = 100; MATCH_RATIO = 2; MIN_EP = 10;
OVERLAPS = [0 0.5 0.75 0.9]; WIN_SEC = 2;
BANDS = [0.5 4; 4 8; 8 13; 13 30];

if ~isfolder(OUT), mkdir(OUT); end
fid = fopen(fullfile(OUT, 'Stage7b_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 7b - OVERLAPPING-WINDOW LEAKAGE   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')), try, if isempty(gcp('nocreate')), parpool; end, catch, end, end

stores = {'D1_IEEE_epochs.mat', 'D1_IEEE_19ch'; 'D2_TDBRAIN_epochs.mat', 'D2m_TDBRAIN_26ch'; 'D3_MENDELEY_epochs.mat', 'D3_MENDELEY_2ch'};
fT = fopen(fullfile(OUT, 'T_stage7b_overlap.csv'), 'w');
fprintf(fT, 'cohort,featureset,overlap,nWindows,protocol,ACC,ACC_sd,BAL,BAL_sd,AUC,AUC_sd\n');
res = struct();
for d = 1:size(stores, 1)
    T = load(fullfile(IN, stores{d, 1})); S = T.S; fs = S.fs; nCh = size(S.X, 3);
    S = dropSparseS(S, MIN_EP); if d == 2, [S, msg] = matchS(S, MATCH_RATIO, SEED0); L('%s', msg); end
    [subj, ~, k] = unique(S.subject); labS = accumarray(k, S.label, [], @(v) v(1));
    L('\n=== %s : %d subjects, %d stored epochs', stores{d, 2}, numel(subj), numel(S.label));
    for ov = OVERLAPS
        step = max(1, round(WIN_SEC * fs * (1 - ov))); win = WIN_SEC * fs;
        % re-window per subject
        Xw = {}; yw = {}; gw = {};
        for i = 1:numel(subj)
            Z = S.X(S.subject == subj(i), :, :); Z = reshape(permute(Z, [2 1 3]), [], nCh);   % concatenated samples x ch
            starts = 1:step:(size(Z, 1) - win + 1); nW = numel(starts); W = zeros(nW, win, nCh, 'single');
            for w = 1:nW, W(w, :, :) = reshape(Z(starts(w) + (0:win - 1), :), [1 win nCh]); end
            Xw{end+1} = W; yw{end+1} = repmat(labS(i), nW, 1); gw{end+1} = repmat(subj(i), nW, 1); %#ok<AGROW>
        end
        Xw = cat(1, Xw{:}); y = cat(1, yw{:}); g = cat(1, gw{:}); nW = numel(y);
        % features
        LV = zeros(nW, nCh); BP = zeros(nW, 4 * nCh); hw = hamming(128);
        parfor w = 1:nW
            lv = zeros(1, nCh); bp = zeros(1, 4 * nCh);
            for c = 1:nCh
                x = double(squeeze(Xw(w, :, c)))'; lv(c) = log(var(x) + eps);
                [P, fr] = pwelch(x, hw, 64, 256, fs); tot = trapz(fr(fr >= 0.5 & fr <= 45), P(fr >= 0.5 & fr <= 45));
                for b = 1:4, m = fr >= BANDS(b, 1) & fr < BANDS(b, 2); bp((c-1)*4 + b) = trapz(fr(m), P(m)) / max(tot, eps); end
            end
            LV(w, :) = lv; BP(w, :) = bp;
        end
        for fsn = {'LogVar', 'BandPower'}
            X = ifelse(strcmp(fsn{1}, 'LogVar'), LV, BP);
            for prot = {'P0_windowCV', 'P3_subjectCV'}
                ms = [];
                for r = 1:R_REPEATS
                    rng(SEED0 + r);
                    if strcmp(prot{1}, 'P0_windowCV'), folds = randomFolds(y, K); else, folds = groupedFolds(g, y, K); end
                    P = zeros(nW, 3);
                    for f = 1:K
                        tr = folds ~= f; te = folds == f; mu = mean(X(tr, :)); sd = std(X(tr, :)); sd(sd < 1e-12) = 1;
                        rng(SEED0 + 100 * r + f);
                        mdl = TreeBagger(RF_TREES, (X(tr, :) - mu) ./ sd, y(tr), 'Method', 'classification', 'Prior', 'Uniform', 'OOBPrediction', 'off');
                        [lab, sc] = predict(mdl, (X(te, :) - mu) ./ sd); P(te, :) = [y(te), str2double(lab), sc(:, strcmp(mdl.ClassNames, '1'))];
                    end
                    ms = [ms, metrics(P)]; %#ok<AGROW>
                end
                L('  overlap %3.0f%%  %-9s %-13s windows=%6d  ACC %.1f +/- %.1f  BAL %.1f  AUC %.3f', 100 * ov, fsn{1}, prot{1}, nW, mean([ms.ACC]), std([ms.ACC]), mean([ms.BAL]), mean([ms.AUC]));
                fprintf(fT, '%s,%s,%.2f,%d,%s,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f\n', stores{d, 2}, fsn{1}, ov, nW, prot{1}, mean([ms.ACC]), std([ms.ACC]), mean([ms.BAL]), std([ms.BAL]), mean([ms.AUC]), std([ms.AUC]));
                res.(matlab.lang.makeValidName(stores{d, 2})).(fsn{1}).(prot{1})(OVERLAPS == ov) = mean([ms.ACC]);
            end
        end
    end
end
fclose(fT);

% figure: accuracy vs overlap
fig = figure('Color', 'w', 'Position', [100 100 1100 380]); cn = fieldnames(res);
for c = 1:numel(cn)
    subplot(1, numel(cn), c); hold on; grid on
    plot(100 * OVERLAPS, res.(cn{c}).LogVar.P0_windowCV, '-o', 'LineWidth', 1.5); plot(100 * OVERLAPS, res.(cn{c}).LogVar.P3_subjectCV, '-s', 'LineWidth', 1.5);
    plot(100 * OVERLAPS, res.(cn{c}).BandPower.P0_windowCV, '--o', 'LineWidth', 1.5); plot(100 * OVERLAPS, res.(cn{c}).BandPower.P3_subjectCV, '--s', 'LineWidth', 1.5);
    xlabel('window overlap (%)'); ylabel('accuracy (%)'); ylim([50 100]); title(strrep(cn{c}, '_', '\_'));
    if c == 1, legend({'LogVar, window CV', 'LogVar, subject CV', 'BandPower, window CV', 'BandPower, subject CV'}, 'Location', 'southeast'); end
end
saveas(fig, fullfile(OUT, 'Stage7b_overlap.png'));
L('\nDONE. Send Stage7b_Report.txt, T_stage7b_overlap.csv and Stage7b_overlap.png.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
function S = subsetS(S, m), for fn = {'subject', 'subjectID', 'label', 'age', 'sex'}, S.(fn{1}) = S.(fn{1})(m, :); end; S.X = S.X(m, :, :); end
function S = dropSparseS(S, minEp)
    [subj, ~, k] = unique(S.subject); cnt = accumarray(k, 1); bad = subj(cnt < minEp); if ~isempty(bad), S = subsetS(S, ~ismember(S.subject, bad)); end
end
function [S, msg] = matchS(S, ratio, seed)
    [subj, ia] = unique(S.subject); lab = double(S.label(ia)); age = double(S.age(ia)); sex = double(S.sex(ia));
    rng(seed); idxA = find(lab == 1); idxA = idxA(randperm(numel(idxA))); idxH = find(lab == 0); used = false(size(idxH)); chosen = [];
    for i = idxA'
        for rr = 1:ratio
            cand = find(~used & sex(idxH) == sex(i)); if isempty(cand), cand = find(~used); end; if isempty(cand), break; end
            [~, j] = min(abs(age(idxH(cand)) - age(i))); used(cand(j)) = true; chosen(end+1) = idxH(cand(j)); %#ok<AGROW>
        end
    end
    S = subsetS(S, ismember(S.subject, [subj(idxA); subj(chosen)]));
    msg = sprintf('TDBRAIN matched: ADHD %d vs HC %d', numel(idxA), numel(chosen));
end
function folds = randomFolds(y, K)
    folds = zeros(size(y)); for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
end
function folds = groupedFolds(g, y, K)
    [subj, ia] = unique(g); lab = y(ia); folds = zeros(size(g));
    for c = [0 1], s = subj(lab == c); s = s(randperm(numel(s))); fidx = mod((1:numel(s)) - 1, K) + 1; for i = 1:numel(s), folds(g == s(i)) = fidx(i); end, end
end
function m = metrics(P)
    y = P(:, 1); yh = P(:, 2); sc = P(:, 3);
    TP = nnz(y == 1 & yh == 1); TN = nnz(y == 0 & yh == 0); FP = nnz(y == 0 & yh == 1); FN = nnz(y == 1 & yh == 0);
    m.ACC = 100 * (TP + TN) / max(numel(y), 1); m.SEN = 100 * TP / max(TP + FN, 1); m.SPE = 100 * TN / max(TN + FP, 1); m.BAL = (m.SEN + m.SPE) / 2;
    try, [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc; catch, m.AUC = NaN; end
end
