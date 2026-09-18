%% stage4_amplitude_validation.m
%  STAGE 4 - Is the log-variance / absolute-power separation real?
%
%  Works from the Stage 1b epoch stores. For every epoch and channel it computes
%  (Welch PSD, 128-pt Hamming, 256-pt FFT):
%     LV_full     log total power 0.5-45 Hz             (= log-variance of the stored epoch)
%     LV_1_30     log total power 1-30 Hz               (muscle band excluded)
%     ABP         log absolute power delta/theta/alpha/beta   (4 per channel)
%     HF_30_45    log power 30-45 Hz                    (muscle proxy - if THIS classifies, worry)
%     RelBP       relative delta/theta/alpha/beta + theta/beta (5 per channel)
%
%  Then, per cohort (D1 IEEE 19ch, D2m TDBRAIN matched 26ch, D3 Mendeley 2ch):
%     A. 5-fold subject-grouped CV x R repeats, DT/kNN/SVM/RF, uniform prior, all features (no selection)
%        for LV_full, LV_1_30, ABP, HF_30_45, RelBP, and LV_1_30 with age regressed out (D2m only)
%     B. per-channel univariate subject-level AUC and Cohen's d for LV_1_30 (topography check)
%     C. label-permutation test for LV_1_30 / RF on D2m (N_PERM shuffles at subject level)
%     D. transfer D2m -> D3 and D3 -> D2m on the common Cz/F4 channels (LV_1_30 and ABP)
%
%  Outputs in <OUT>: Stage4_Report.txt, T_stage4_results.csv, T_channel_auc.csv, T_stage4_transfer.csv,
%                    Stage4_channel_auc.png, Stage4_permutation.png, Stage4_Results.mat

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage1b_Epochs'); OUT = fullfile(ROOT, 'Stage4_Amplitude');

R_REPEATS = 3; K_OUTER = 5; K_INNER = 3; SEED0 = 2026;
GRID.DT_MinLeaf = [1 5 20]; GRID.KNN_K = [5 15 35]; GRID.SVM_C = [0.1 1 10]; GRID.RF_MinLeaf = [1 5];
RF_TREES = 100; INNER_MAX_ROWS = 3000; SVM_MAX_ROWS = 6000;
TDB_MATCH_RATIO = 2; MIN_EPOCHS_PER_SUBJECT = 10;
N_PERM = 100; PERM_TREES = 50;
BANDS = [0.5 4; 4 8; 8 13; 13 30]; BAND_NAMES = {'delta','theta','alpha','beta'};
clfNames = {'DT', 'kNN', 'SVM', 'RF'};

if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage4_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 4 - AMPLITUDE FEATURE VALIDATION   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('repeats=%d outer=%d inner=%d | uniform prior | permutations=%d', R_REPEATS, K_OUTER, K_INNER, N_PERM);

%% ------------------------------------------------------------------ LOAD + FEATURES
stores = {'D1_IEEE_epochs.mat', 'D1_IEEE'; 'D2_TDBRAIN_epochs.mat', 'D2m_TDBRAIN'; 'D3_MENDELEY_epochs.mat', 'D3_MENDELEY'};
C = struct('name', {}, 'F', {});
for d = 1:3
    T = load(fullfile(IN, stores{d, 1})); S = T.S;
    F = amplitudeFeatures(S, BANDS, BAND_NAMES);
    F = dropSparse(F, MIN_EPOCHS_PER_SUBJECT);
    if d == 2, [F, msg] = matchTdbrain(F, TDB_MATCH_RATIO, SEED0); L('%s', msg); end
    [~, ia] = unique(F.subject);
    L('Cohort %-12s subjects=%3d (ADHD %3d / HC %3d) epochs=%5d channels=%d', stores{d, 2}, numel(ia), nnz(F.label(ia) == 1), nnz(F.label(ia) == 0), numel(F.label), numel(F.channels));
    C(end+1) = struct('name', stores{d, 2}, 'F', F); %#ok<SAGROW>
end

%% ------------------------------------------------------------------ A. CLASSIFICATION
fRes = fopen(fullfile(OUT, 'T_stage4_results.csv'), 'w');
fprintf(fRes, 'cohort,featureset,classifier,level,BAL,BAL_sd,ACC,ACC_sd,SEN,SEN_sd,SPE,SPE_sd,MCC,MCC_sd,AUC,AUC_sd\n');
results = struct();
for c = 1:numel(C)
    F = C(c).F; y = F.label; g = F.subject;
    sets = {'LV_full', F.LV_full; 'LV_1_30', F.LV_1_30; 'ABP', F.ABP; 'HF_30_45', F.HF; 'RelBP', F.RelBP};
    if strcmp(C(c).name, 'D2m_TDBRAIN'), sets(end+1, :) = {'LV_1_30_ageResid', F.LV_1_30}; end %#ok<SAGROW>
    for s = 1:size(sets, 1)
        X = sets{s, 2}; nF = size(X, 2); ageResid = endsWith(sets{s, 1}, 'ageResid');
        L('\n================================================================ %s | %s (%d features)', C(c).name, sets{s, 1}, nF);
        predE = cell(R_REPEATS, 4); predS = predE;
        for r = 1:R_REPEATS
            rng(SEED0 + r); folds = groupedStratifiedFolds(g, y, K_OUTER);
            for k = 1:4, predE{r, k} = []; predS{r, k} = []; end
            for f = 1:K_OUTER
                tr = folds ~= f; te = folds == f;
                Xtr = X(tr, :); ytr = y(tr); gtr = g(tr); Xte = X(te, :); yte = y(te); gte = g(te);
                if ageResid   % remove linear age effect estimated on the training fold
                    A = [ones(nnz(tr), 1), F.age(tr)]; beta = A \ Xtr;
                    Xtr = Xtr - A * beta; Xte = Xte - [ones(nnz(te), 1), F.age(te)] * beta;
                end
                mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd < 1e-12) = 1; Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
                for k = 1:4
                    [yh, sc] = trainPredict(clfNames{k}, Xtr, ytr, gtr, Xte, GRID, RF_TREES, K_INNER, SEED0 + r, INNER_MAX_ROWS, SVM_MAX_ROWS);
                    predE{r, k} = [predE{r, k}; yte, yh, sc]; predS{r, k} = [predS{r, k}; subjectVote(gte, yte, yh, sc)];
                end
            end
            for k = 1:4
                mE = metrics(predE{r, k}); mS = metrics(predS{r, k});
                L('  rep %d %-4s epoch BAL %.1f AUC %.3f | subject BAL %.1f SEN %.1f SPE %.1f AUC %.3f MCC %.3f', r, clfNames{k}, mE.BAL, mE.AUC, mS.BAL, mS.SEN, mS.SPE, mS.AUC, mS.MCC);
            end
        end
        for k = 1:4
            msE = cellfun(@metrics, predE(:, k)); msS = cellfun(@metrics, predS(:, k));
            writeRow(fRes, C(c).name, sets{s, 1}, clfNames{k}, 'epoch', msE); writeRow(fRes, C(c).name, sets{s, 1}, clfNames{k}, 'subject', msS);
            L('  MEAN  %-4s subject BAL %.1f +/- %.1f  AUC %.3f +/- %.3f', clfNames{k}, mean([msS.BAL]), std([msS.BAL]), mean([msS.AUC]), std([msS.AUC]));
        end
        results.(matlab.lang.makeValidName([C(c).name '_' sets{s, 1}])) = struct('predE', {predE}, 'predS', {predS});
    end
end
fclose(fRes);

%% ------------------------------------------------------------------ B. CHANNEL TOPOGRAPHY (LV_1_30)
L('\n================================================================ B. PER-CHANNEL SUBJECT-LEVEL AUC / COHEN d (LV_1_30)');
fCh = fopen(fullfile(OUT, 'T_channel_auc.csv'), 'w'); fprintf(fCh, 'cohort,channel,AUC,cohen_d,mean_ADHD,mean_HC,r_age\n');
fig = figure('Color', 'w', 'Position', [100 100 1200 800]);
for c = 1:numel(C)
    F = C(c).F; [~, ~, k] = unique(F.subject); lab = accumarray(k, F.label, [], @(v) v(1)); ageS = accumarray(k, F.age, [], @(v) v(1));
    M = zeros(max(k), numel(F.channels)); for j = 1:numel(F.channels), M(:, j) = accumarray(k, F.LV_1_30(:, j), [], @mean); end
    auc = zeros(1, numel(F.channels)); dd = auc; ra = nan(1, numel(F.channels));
    for j = 1:numel(F.channels)
        [~, ~, ~, auc(j)] = perfcurve(lab, M(:, j), 1);
        a = M(lab == 1, j); h = M(lab == 0, j); dd(j) = (mean(a) - mean(h)) / sqrt((var(a) + var(h)) / 2);
        if all(~isnan(ageS)), ra(j) = corr(M(:, j), ageS); end
        fprintf(fCh, '%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f\n', C(c).name, F.channels{j}, auc(j), dd(j), mean(a), mean(h), ra(j));
    end
    [~, o] = sort(auc, 'descend');
    L('  %s: channels by AUC: %s', C(c).name, strjoin(arrayfun(@(j) sprintf('%s %.2f (d=%.2f)', F.channels{j}, auc(j), dd(j)), o, 'UniformOutput', false), ', '));
    if all(~isnan(ra)), L('    correlation of LV_1_30 with age, median over channels r = %.2f (max |r| = %.2f)', median(ra), max(abs(ra))); end
    subplot(3, 1, c); bar(auc); set(gca, 'XTick', 1:numel(F.channels), 'XTickLabel', F.channels); ylim([0.3 1]); grid on
    hold on; yline(0.5, 'k--'); ylabel('subject-level AUC'); title(sprintf('%s - LV\\_1\\_30 per channel (ADHD > HC if d > 0)', strrep(C(c).name, '_', '\_')));
    for j = 1:numel(F.channels), if dd(j) < 0, text(j, auc(j) + 0.02, '-', 'HorizontalAlignment', 'center'); end, end
end
fclose(fCh); saveas(fig, fullfile(OUT, 'Stage4_channel_auc.png'));

%% ------------------------------------------------------------------ C. PERMUTATION TEST (D2m, LV_1_30, RF)
L('\n================================================================ C. PERMUTATION TEST (D2m TDBRAIN, LV_1_30, RF, %d permutations)', N_PERM);
F = C(strcmp({C.name}, 'D2m_TDBRAIN')).F; X = F.LV_1_30; y = F.label; g = F.subject;
obs = permRun(X, y, g, K_OUTER, PERM_TREES, SEED0);
[subj, ia] = unique(g); labS = y(ia); nullBal = zeros(N_PERM, 1); nullAuc = zeros(N_PERM, 1);
for p = 1:N_PERM
    rng(SEED0 + 5000 + p); perm = labS(randperm(numel(labS)));
    yp = zeros(size(y)); for i = 1:numel(subj), yp(g == subj(i)) = perm(i); end
    m = permRun(X, yp, g, K_OUTER, PERM_TREES, SEED0 + p); nullBal(p) = m.BAL; nullAuc(p) = m.AUC;
    if mod(p, 20) == 0, L('  %d permutations done (null BAL so far: mean %.1f, max %.1f)', p, mean(nullBal(1:p)), max(nullBal(1:p))); end
end
pBal = (1 + nnz(nullBal >= obs.BAL)) / (N_PERM + 1); pAuc = (1 + nnz(nullAuc >= obs.AUC)) / (N_PERM + 1);
L('  observed subject BAL %.1f (null mean %.1f, 95th pct %.1f)  p = %.4f', obs.BAL, mean(nullBal), prctile(nullBal, 95), pBal);
L('  observed subject AUC %.3f (null mean %.3f, 95th pct %.3f)  p = %.4f', obs.AUC, mean(nullAuc), prctile(nullAuc, 95), pAuc);
fig2 = figure('Color', 'w'); histogram(nullBal, 20); hold on; xline(obs.BAL, 'r', 'LineWidth', 2);
xlabel('subject-level balanced accuracy (%)'); ylabel('count'); title(sprintf('Permutation null (n=%d) vs observed, D2m LV\\_1\\_30 RF, p=%.3f', N_PERM, pBal));
saveas(fig2, fullfile(OUT, 'Stage4_permutation.png'));

%% ------------------------------------------------------------------ D. TRANSFER (Cz/F4)
L('\n================================================================ D. TRANSFER on Cz+F4 (RF uniform prior)');
fT = fopen(fullfile(OUT, 'T_stage4_transfer.csv'), 'w'); fprintf(fT, 'featureset,train,test,epoch_BAL,epoch_AUC,subject_BAL,subject_AUC,subject_MCC\n');
pairs = {'D2m_TDBRAIN', 'D3_MENDELEY'; 'D3_MENDELEY', 'D2m_TDBRAIN'; 'D2m_TDBRAIN', 'D1_IEEE'; 'D1_IEEE', 'D2m_TDBRAIN'};
for fsn = {'LV_1_30', 'ABP'}
    for p = 1:size(pairs, 1)
        A = C(strcmp({C.name}, pairs{p, 1})).F; B = C(strcmp({C.name}, pairs{p, 2})).F;
        Xa = pickCzF4(A, fsn{1}); Xb = pickCzF4(B, fsn{1});
        mu = mean(Xa, 1); sd = std(Xa, 0, 1); sd(sd < 1e-12) = 1; Xa = (Xa - mu) ./ sd; Xb = (Xb - mu) ./ sd;
        [yh, sc] = rfFixed(Xa, A.label, Xb, RF_TREES, 5, SEED0);
        mE = metrics([B.label, yh, sc]); mS = metrics(subjectVote(B.subject, B.label, yh, sc));
        L('  %-8s train %-12s -> test %-12s : epoch BAL %.1f AUC %.3f | subject BAL %.1f AUC %.3f MCC %.3f', fsn{1}, pairs{p, 1}, pairs{p, 2}, mE.BAL, mE.AUC, mS.BAL, mS.AUC, mS.MCC);
        fprintf(fT, '%s,%s,%s,%.2f,%.3f,%.2f,%.3f,%.3f\n', fsn{1}, pairs{p, 1}, pairs{p, 2}, mE.BAL, mE.AUC, mS.BAL, mS.AUC, mS.MCC);
    end
end
fclose(fT);
save(fullfile(OUT, 'Stage4_Results.mat'), 'results', 'nullBal', 'nullAuc', 'obs', '-v7.3');
L('\nDONE. Send Stage4_Report.txt, the two CSVs and the two PNGs.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end

function F = amplitudeFeatures(S, BANDS, BAND_NAMES)
    X = S.X; nEp = size(X, 1); nCh = size(X, 3); fs = S.fs; win = hamming(128);
    LVf = zeros(nEp, nCh); LV130 = LVf; HF = LVf; ABP = zeros(nEp, 4 * nCh); Rel = zeros(nEp, 5 * nCh);
    parfor e = 1:nEp
        rf = zeros(1, nCh); r130 = rf; rhf = rf; rabp = zeros(1, 4 * nCh); rrel = zeros(1, 5 * nCh);
        for c = 1:nCh
            x = double(squeeze(X(e, :, c)))';
            [P, fr] = pwelch(x, win, 64, 256, fs);
            bp = @(lo, hi) trapz(fr(fr >= lo & fr < hi), P(fr >= lo & fr < hi));
            tot = bp(0.5, 45.01); rf(c) = log(tot + eps); r130(c) = log(bp(1, 30) + eps); rhf(c) = log(bp(30, 45.01) + eps);
            ab = zeros(1, 4); for b = 1:4, ab(b) = bp(BANDS(b, 1), BANDS(b, 2)); end
            rabp((c-1)*4 + (1:4)) = log(ab + eps);
            rel = ab / max(tot, eps); rrel((c-1)*5 + (1:5)) = [rel, rel(2) / max(rel(4), eps)];
        end
        LVf(e, :) = rf; LV130(e, :) = r130; HF(e, :) = rhf; ABP(e, :) = rabp; Rel(e, :) = rrel;
    end
    F = struct('channels', {S.channels}, 'LV_full', LVf, 'LV_1_30', LV130, 'HF', HF, 'ABP', ABP, 'RelBP', Rel, ...
        'subject', double(S.subject(:)), 'subjectID', S.subjectID, 'label', double(S.label(:)), 'age', double(S.age(:)), 'sex', double(S.sex(:)));
    F.abpNames = {}; for c = 1:nCh, for b = 1:4, F.abpNames{end+1} = sprintf('%s_%s', S.channels{c}, BAND_NAMES{b}); end, end
end
function X = pickCzF4(F, setName)
    m = ismember(F.channels, {'Cz', 'F4'});
    switch setName
        case 'LV_1_30', X = F.LV_1_30(:, m);
        case 'ABP', idx = find(m); cols = []; for i = idx, cols = [cols, (i-1)*4 + (1:4)]; end; X = F.ABP(:, cols); %#ok<AGROW>
    end
end
function F = dropSparse(F, minEp)
    [subj, ~, k] = unique(F.subject); cnt = accumarray(k, 1); bad = subj(cnt < minEp); if isempty(bad), return; end
    m = ~ismember(F.subject, bad); F = subsetF(F, m);
end
function F = subsetF(F, m)
    for fn = {'LV_full', 'LV_1_30', 'HF', 'ABP', 'RelBP', 'subject', 'subjectID', 'label', 'age', 'sex'}, F.(fn{1}) = F.(fn{1})(m, :); end
end
function [Fm, msg] = matchTdbrain(F, ratio, seed)
    [subj, ia] = unique(F.subject); lab = F.label(ia); age = F.age(ia); sex = F.sex(ia);
    rng(seed); idxA = find(lab == 1); idxA = idxA(randperm(numel(idxA))); idxH = find(lab == 0); used = false(size(idxH)); chosen = [];
    for i = idxA'
        for rr = 1:ratio
            cand = find(~used & sex(idxH) == sex(i)); if isempty(cand), cand = find(~used); end; if isempty(cand), break; end
            [~, j] = min(abs(age(idxH(cand)) - age(i))); used(cand(j)) = true; chosen(end+1) = idxH(cand(j)); %#ok<AGROW>
        end
    end
    Fm = subsetF(F, ismember(F.subject, [subj(idxA); subj(chosen)]));
    msg = sprintf('TDBRAIN matched: ADHD %d (mean age %.1f, male %.0f%%) vs HC %d (mean age %.1f, male %.0f%%)', ...
        numel(idxA), mean(age(idxA)), 100 * mean(sex(idxA) == 1), numel(chosen), mean(age(chosen)), 100 * mean(sex(chosen) == 1));
end
function m = permRun(X, y, g, K, nTrees, seed)
    rng(seed); folds = groupedStratifiedFolds(g, y, K); P = [];
    for f = 1:K
        tr = folds ~= f; te = folds == f; mu = mean(X(tr, :)); sd = std(X(tr, :)); sd(sd < 1e-12) = 1;
        [yh, sc] = rfFixed((X(tr, :) - mu) ./ sd, y(tr), (X(te, :) - mu) ./ sd, nTrees, 5, seed);
        P = [P; subjectVote(g(te), y(te), yh, sc)]; %#ok<AGROW>
    end
    m = metrics(P);
end
function folds = groupedStratifiedFolds(g, y, K)
    [subj, ia] = unique(g); lab = y(ia); folds = zeros(size(g));
    for c = [0 1]
        s = subj(lab == c); s = s(randperm(numel(s))); fid = mod((1:numel(s)) - 1, K) + 1;
        for i = 1:numel(s), folds(g == s(i)) = fid(i); end
    end
end
function [Xs, ys, gs] = stratSub(X, y, g, maxRows)
    n = numel(y); if n <= maxRows, Xs = X; ys = y; gs = g; return; end
    keep = []; for c = [0 1], idx = find(y == c); idx = idx(randperm(numel(idx))); keep = [keep; idx(1:round(maxRows * numel(idx) / n))]; end %#ok<AGROW>
    Xs = X(keep, :); ys = y(keep); gs = g(keep);
end
function [yhat, score] = trainPredict(name, Xtr, ytr, gtr, Xte, GRID, nTrees, kInner, seed, maxRows, svmMax)
    rng(seed); [XtrG, ytrG, gtrG] = stratSub(Xtr, ytr, gtr, maxRows); inner = groupedStratifiedFolds(gtrG, ytrG, kInner);
    switch name, case 'DT', grid = GRID.DT_MinLeaf; case 'kNN', grid = GRID.KNN_K; case 'SVM', grid = GRID.SVM_C; case 'RF', grid = GRID.RF_MinLeaf; end
    bal = zeros(size(grid));
    for gi = 1:numel(grid)
        P = [];
        for f = 1:kInner
            tr = inner ~= f; te = inner == f;
            [yh, sc] = fitAndPredict(name, XtrG(tr, :), ytrG(tr), XtrG(te, :), grid(gi), nTrees, seed, svmMax); P = [P; ytrG(te), yh, sc]; %#ok<AGROW>
        end
        m = metrics(P); bal(gi) = m.BAL;
    end
    [~, gi] = max(bal); [yhat, score] = fitAndPredict(name, Xtr, ytr, Xte, grid(gi), nTrees, seed, svmMax);
end
function [yhat, score] = fitAndPredict(name, Xtr, ytr, Xte, hp, nTrees, seed, svmMax)
    rng(seed);
    switch name
        case 'DT',  mdl = fitctree(Xtr, ytr, 'MinLeafSize', hp, 'Prior', 'uniform'); [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'kNN', mdl = fitcknn(Xtr, ytr, 'NumNeighbors', hp, 'DistanceWeight', 'inverse', 'Prior', 'uniform'); [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'SVM', [Xtr, ytr] = stratSub(Xtr, ytr, ytr, svmMax);
                    mdl = fitcsvm(Xtr, ytr, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'BoxConstraint', hp, 'Prior', 'uniform'); [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'RF',  [yhat, score] = rfFixed(Xtr, ytr, Xte, nTrees, hp, seed);
    end
    yhat = double(yhat(:)); score = double(score(:));
end
function [yhat, score] = rfFixed(Xtr, ytr, Xte, nTrees, minLeaf, seed)
    rng(seed); mdl = TreeBagger(nTrees, Xtr, ytr, 'Method', 'classification', 'MinLeafSize', minLeaf, 'Prior', 'Uniform', 'OOBPrediction', 'off');
    [lab, sc] = predict(mdl, Xte); yhat = double(str2double(lab(:))); score = double(sc(:, strcmp(mdl.ClassNames, '1')));
end
function S = subjectVote(g, y, yhat, score)
    [~, ~, k] = unique(g); yt = accumarray(k, y, [], @(v) v(1)); ms = accumarray(k, score, [], @mean); vote = accumarray(k, yhat, [], @mean);
    yv = double(vote > 0.5); tie = vote == 0.5; yv(tie) = double(ms(tie) > 0.5); S = [yt, yv, ms];
end
function m = metrics(P)
    y = P(:, 1); yh = P(:, 2); sc = P(:, 3);
    TP = nnz(y == 1 & yh == 1); TN = nnz(y == 0 & yh == 0); FP = nnz(y == 0 & yh == 1); FN = nnz(y == 1 & yh == 0);
    m.ACC = 100 * (TP + TN) / max(numel(y), 1); m.SEN = 100 * TP / max(TP + FN, 1); m.SPE = 100 * TN / max(TN + FP, 1); m.BAL = (m.SEN + m.SPE) / 2;
    den = sqrt(double(TP + FP) * double(TP + FN) * double(TN + FP) * double(TN + FN)); m.MCC = ifelse(den > 0, (TP * TN - FP * FN) / den, 0);
    try, [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc; catch, m.AUC = NaN; end
end
function writeRow(fid, cohort, fset, clf, level, ms)
    fld = {'BAL', 'ACC', 'SEN', 'SPE', 'MCC', 'AUC'}; fprintf(fid, '%s,%s,%s,%s', cohort, fset, clf, level);
    for i = 1:numel(fld), v = arrayfun(@(m) m.(fld{i}), ms); fprintf(fid, ',%.3f,%.3f', mean(v, 'omitnan'), std(v, 'omitnan')); end
    fprintf(fid, '\n');
end
