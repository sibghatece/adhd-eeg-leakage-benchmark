%% stage9_master_benchmark.m
%  STAGE 9 - Master benchmark: every feature family x every dataset x two protocols.
%
%  Families (feature stores):
%     PSR, BandPower, LogVar               from Stage2b_Features\D*_features.mat
%     timedomain, entropy, nonlinear,      from Stage8_Features\<family>\D*_<family>.mat
%     wavelet, vmd, riemannian
%  Cohorts:
%     D1 IEEE (all), D2m TDBRAIN (adults, 1:2 age/sex-matched), D3 Mendeley (all),
%     D4m BALLADEER (1:1 age/sex-matched), D5 OSF (all; no age/sex available)
%     + secondary (SECONDARY = true): D2a TDBRAIN all adults, D4a BALLADEER all
%  Protocols:
%     P3  subject-grouped stratified 5-fold, R repeats; inside each training fold: median imputation,
%         z-scoring, |t|-ranking on subject means with K in K_GRID chosen by inner 3-fold grouped RF
%         balanced accuracy; classifiers in CLASSIFIERS with uniform prior and inner-CV hyperparameters.
%         Epoch-level and subject-level (majority vote) metrics.
%     P1  epoch-level random 5-fold (subject leakage), RF only, same selection inside the fold.
%  Outputs: <OUT>\Stage9_Report.txt, T_stage9_master.csv, T_stage9_summary.csv, Stage9_Results.mat

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
OUT = fullfile(ROOT, 'Stage9_Master');
R_REPEATS = 3; K_OUTER = 5; K_INNER = 3; SEED0 = 2026;
K_GRID = [10 30 100]; NO_SELECTION_BELOW = 60;
CLASSIFIERS = {'RF', 'SVM'};                % add 'kNN','DT' if wanted (slower)
GRID.SVM_C = [0.1 1 10]; GRID.RF_MinLeaf = [1 5]; GRID.KNN_K = [5 15 35]; GRID.DT_MinLeaf = [1 5 20];
RF_TREES = 100; INNER_MAX_ROWS = 3000; SVM_MAX_ROWS = 6000; MIN_EPOCHS_PER_SUBJECT = 10;
SECONDARY = true; RUN_P1 = true;
FAMILIES = {'PSR', 'BandPower', 'LogVar', 'timedomain', 'entropy', 'nonlinear', 'wavelet', 'vmd', 'riemannian'};
DATASETS = {'D1_IEEE', 'D2_TDBRAIN', 'D3_MENDELEY', 'D4_BALLADEER', 'D5_OSF'};

if ~isfolder(OUT), mkdir(OUT); end
CACHE = fullfile(OUT, 'cache'); if ~isfolder(CACHE), mkdir(CACHE); end   % one file per cohort x family; delete to recompute
fid = fopen(fullfile(OUT, 'Stage9_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 9 - MASTER BENCHMARK   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('repeats=%d outer=%d inner=%d | K grid %s | classifiers %s | uniform prior | primary metric balanced accuracy', R_REPEATS, K_OUTER, K_INNER, mat2str(K_GRID), strjoin(CLASSIFIERS, ','));

%% ---------------- cohort definitions (subject sets), built once from the Stage 2b stores
cohorts = struct('name', {}, 'dataset', {}, 'keep', {});
for d = 1:numel(DATASETS)
    F = loadStore(ROOT, 'PSR', DATASETS{d}); if isempty(F), L('!! %s: no Stage 2b store, skipped', DATASETS{d}); continue; end
    F = dropSparse(F, MIN_EPOCHS_PER_SUBJECT);
    [subj, ia] = unique(F.subject); lab = F.label(ia); age = F.age(ia); sex = F.sex(ia);
    switch DATASETS{d}
        case 'D2_TDBRAIN'
            keepM = matchSubjects(subj, lab, age, sex, 2, SEED0);
            cohorts(end+1) = struct('name', 'D2m_TDBRAIN', 'dataset', DATASETS{d}, 'keep', keepM); %#ok<SAGROW>
            if SECONDARY, cohorts(end+1) = struct('name', 'D2a_TDBRAIN', 'dataset', DATASETS{d}, 'keep', subj); end %#ok<SAGROW>
        case 'D4_BALLADEER'
            keepM = matchSubjects(subj, lab, age, sex, 1, SEED0);
            cohorts(end+1) = struct('name', 'D4m_BALLADEER', 'dataset', DATASETS{d}, 'keep', keepM); %#ok<SAGROW>
            if SECONDARY, cohorts(end+1) = struct('name', 'D4a_BALLADEER', 'dataset', DATASETS{d}, 'keep', subj); end %#ok<SAGROW>
        otherwise
            cohorts(end+1) = struct('name', DATASETS{d}, 'dataset', DATASETS{d}, 'keep', subj); %#ok<SAGROW>
    end
end
for c = 1:numel(cohorts)
    F = dropSparse(loadStore(ROOT, 'PSR', cohorts(c).dataset), MIN_EPOCHS_PER_SUBJECT); m = ismember(F.subject, cohorts(c).keep);
    [~, ia] = unique(F.subject(m)); lab = F.label(m); lab = lab(ia); ag = F.age(m); ag = ag(ia); sx = F.sex(m); sx = sx(ia);
    L('Cohort %-14s subjects=%3d (ADHD %3d / HC %3d) epochs=%5d | age ADHD %.1f / HC %.1f | male ADHD %.0f%% / HC %.0f%%', cohorts(c).name, numel(ia), ...
        nnz(lab == 1), nnz(lab == 0), nnz(m), mean(ag(lab == 1), 'omitnan'), mean(ag(lab == 0), 'omitnan'), 100 * mean(sx(lab == 1), 'omitnan'), 100 * mean(sx(lab == 0), 'omitnan'));
end

%% ---------------- main loop
fM = fopen(fullfile(OUT, 'T_stage9_master.csv'), 'w');
fprintf(fM, 'cohort,family,nFeatures,protocol,classifier,level,BAL,BAL_sd,ACC,ACC_sd,SEN,SEN_sd,SPE,SPE_sd,MCC,MCC_sd,AUC,AUC_sd\n');
summary = struct();
for c = 1:numel(cohorts)
    for fam = 1:numel(FAMILIES)
        F = loadStore(ROOT, FAMILIES{fam}, cohorts(c).dataset);
        if isempty(F), L('\n== %s | %s : no feature store, skipped', cohorts(c).name, FAMILIES{fam}); continue; end
        F = dropSparse(F, MIN_EPOCHS_PER_SUBJECT); m = ismember(F.subject, cohorts(c).keep);
        X = F.feat(m, :); y = F.label(m); g = F.subject(m); nF = size(X, 2);
        allNaN = all(isnan(X), 1); X = X(:, ~allNaN); nF = size(X, 2);
        L('\n================================================================ %s | %s (%d features, NaN %.1f%%)', cohorts(c).name, FAMILIES{fam}, nF, 100 * mean(isnan(X(:))));
        t0 = tic;
        cacheFile = fullfile(CACHE, sprintf('%s__%s.mat', cohorts(c).name, FAMILIES{fam}));
        if isfile(cacheFile)
            Cc = load(cacheFile); msE = Cc.msE; msS = Cc.msS; msE1 = Cc.msE1;
            for k = 1:numel(CLASSIFIERS)
                writeRow(fM, cohorts(c).name, FAMILIES{fam}, nF, 'P3_subjectCV', CLASSIFIERS{k}, 'epoch', msE{k});
                writeRow(fM, cohorts(c).name, FAMILIES{fam}, nF, 'P3_subjectCV', CLASSIFIERS{k}, 'subject', msS{k});
                L('  [cache] P3 %-4s subject BAL %.1f +/- %.1f  AUC %.3f', CLASSIFIERS{k}, mean([msS{k}.BAL]), std([msS{k}.BAL]), mean([msS{k}.AUC]));
                if strcmp(CLASSIFIERS{k}, 'RF'), summary.(matlab.lang.makeValidName(cohorts(c).name)).(FAMILIES{fam}) = [mean([msS{k}.BAL]), mean([msS{k}.AUC]), mean([msE{k}.BAL]), NaN]; end
            end
            if ~isempty(msE1)
                writeRow(fM, cohorts(c).name, FAMILIES{fam}, nF, 'P1_epochCV', 'RF', 'epoch', msE1);
                summary.(matlab.lang.makeValidName(cohorts(c).name)).(FAMILIES{fam})(4) = mean([msE1.BAL]);
                L('  [cache] P1 RF   epoch BAL %.1f', mean([msE1.BAL]));
            end
            continue;
        end
        % ---- P3 subject-grouped
        predE = cell(R_REPEATS, numel(CLASSIFIERS)); predS = predE; kCh = [];
        for r = 1:R_REPEATS
            rng(SEED0 + r); folds = groupedStratifiedFolds(g, y, K_OUTER);
            for k = 1:numel(CLASSIFIERS), predE{r, k} = []; predS{r, k} = []; end
            for f = 1:K_OUTER
                tr = folds ~= f; te = folds == f;
                [Xtr, Xte] = prepFold(X, tr, te);
                if nF >= NO_SELECTION_BELOW
                    order = subjectTRank(Xtr, y(tr), g(tr)); Kb = chooseK(Xtr, y(tr), g(tr), order, K_GRID, K_INNER, RF_TREES, INNER_MAX_ROWS, SEED0 + 10 * r + f);
                    sel = order(1:min(Kb, nF)); kCh(end+1) = Kb; %#ok<AGROW>
                else, sel = 1:nF; end
                for k = 1:numel(CLASSIFIERS)
                    [yh, sc] = trainPredict(CLASSIFIERS{k}, Xtr(:, sel), y(tr), g(tr), Xte(:, sel), GRID, RF_TREES, K_INNER, SEED0 + r, INNER_MAX_ROWS, SVM_MAX_ROWS);
                    predE{r, k} = [predE{r, k}; y(te), yh, sc]; predS{r, k} = [predS{r, k}; subjectVote(g(te), y(te), yh, sc)];
                end
            end
        end
        msEc = cell(1, numel(CLASSIFIERS)); msSc = msEc;
        for k = 1:numel(CLASSIFIERS)
            msE = cellfun(@metrics, predE(:, k)); msS = cellfun(@metrics, predS(:, k)); msEc{k} = msE; msSc{k} = msS;
            writeRow(fM, cohorts(c).name, FAMILIES{fam}, nF, 'P3_subjectCV', CLASSIFIERS{k}, 'epoch', msE);
            writeRow(fM, cohorts(c).name, FAMILIES{fam}, nF, 'P3_subjectCV', CLASSIFIERS{k}, 'subject', msS);
            L('  P3 %-4s epoch BAL %.1f AUC %.3f | subject BAL %.1f +/- %.1f  SEN %.1f SPE %.1f  AUC %.3f +/- %.3f  MCC %.3f', CLASSIFIERS{k}, ...
                mean([msE.BAL]), mean([msE.AUC]), mean([msS.BAL]), std([msS.BAL]), mean([msS.SEN]), mean([msS.SPE]), mean([msS.AUC]), std([msS.AUC]), mean([msS.MCC]));
            if strcmp(CLASSIFIERS{k}, 'RF'), summary.(matlab.lang.makeValidName(cohorts(c).name)).(FAMILIES{fam}) = [mean([msS.BAL]), mean([msS.AUC]), mean([msE.BAL])]; end
        end
        if ~isempty(kCh), L('  K chosen: %s', mat2str(kCh)); end
        % ---- P1 epoch-level random (leaky), RF
        if RUN_P1
            msE1 = [];
            for r = 1:R_REPEATS
                rng(SEED0 + r); folds = randomFolds(y, K_OUTER); P = zeros(numel(y), 3);
                for f = 1:K_OUTER
                    tr = folds ~= f; te = folds == f; [Xtr, Xte] = prepFold(X, tr, te);
                    if nF >= NO_SELECTION_BELOW, order = subjectTRank(Xtr, y(tr), g(tr)); sel = order(1:min(100, nF)); else, sel = 1:nF; end
                    [yh, sc] = rfFixed(Xtr(:, sel), y(tr), Xte(:, sel), RF_TREES, 5, SEED0 + r); P(te, :) = [y(te), yh, sc];
                end
                msE1 = [msE1, metrics(P)]; %#ok<AGROW>
            end
            writeRow(fM, cohorts(c).name, FAMILIES{fam}, nF, 'P1_epochCV', 'RF', 'epoch', msE1);
            L('  P1 RF   epoch BAL %.1f +/- %.1f  AUC %.3f   (inflation vs P3 epoch: %+.1f BAL points)', mean([msE1.BAL]), std([msE1.BAL]), mean([msE1.AUC]), mean([msE1.BAL]) - summary.(matlab.lang.makeValidName(cohorts(c).name)).(FAMILIES{fam})(3));
            summary.(matlab.lang.makeValidName(cohorts(c).name)).(FAMILIES{fam})(4) = mean([msE1.BAL]);
        end
        L('  (%.1f min)', toc(t0) / 60);
        if ~RUN_P1, msE1 = []; end
        msE = msEc; msS = msSc; save(cacheFile, 'msE', 'msS', 'msE1', 'nF');
    end
end
fclose(fM);

%% ---------------- summary table: subject-level BAL (P3, RF) per cohort x family, and P1-P3 inflation
fS = fopen(fullfile(OUT, 'T_stage9_summary.csv'), 'w');
cn = fieldnames(summary);
fprintf(fS, 'metric,cohort%s\n', sprintf(',%s', FAMILIES{:}));
for what = {'subjectBAL_P3_RF', 'subjectAUC_P3_RF', 'epochBAL_P3_RF', 'epochBAL_P1_RF'}
    idx = find(strcmp(what{1}, {'subjectBAL_P3_RF', 'subjectAUC_P3_RF', 'epochBAL_P3_RF', 'epochBAL_P1_RF'}));
    for c = 1:numel(cn)
        fprintf(fS, '%s,%s', what{1}, cn{c});
        for fam = 1:numel(FAMILIES)
            v = NaN; if isfield(summary.(cn{c}), FAMILIES{fam}) && numel(summary.(cn{c}).(FAMILIES{fam})) >= idx, v = summary.(cn{c}).(FAMILIES{fam})(idx); end
            fprintf(fS, ',%.3f', v);
        end
        fprintf(fS, '\n');
    end
end
fclose(fS);
L('\n=== SUBJECT-LEVEL BALANCED ACCURACY (P3, RF): rows = cohorts, columns = %s', strjoin(FAMILIES, ' '));
for c = 1:numel(cn)
    v = nan(1, numel(FAMILIES));
    for fam = 1:numel(FAMILIES), if isfield(summary.(cn{c}), FAMILIES{fam}), v(fam) = summary.(cn{c}).(FAMILIES{fam})(1); end, end
    L('  %-14s %s', cn{c}, sprintf('%6.1f', v));
end
save(fullfile(OUT, 'Stage9_Results.mat'), 'summary', 'cohorts', 'FAMILIES', '-v7.3');
L('\nDONE. Send Stage9_Report.txt, T_stage9_master.csv and T_stage9_summary.csv.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end

function F = loadStore(ROOT, family, dataset)
    % returns struct with feat, subject, label, age, sex (empty if the store does not exist)
    F = [];
    switch family
        case {'PSR', 'BandPower', 'LogVar'}
            p = fullfile(ROOT, 'Stage2b_Features', [dataset '_features.mat']); if ~isfile(p), return; end
            T = load(p); G = T.F;
            switch family, case 'PSR', X = G.psr; case 'BandPower', X = G.bp; case 'LogVar', X = G.logvar; end
        otherwise
            p = fullfile(ROOT, 'Stage8_Features', family, sprintf('%s_%s.mat', dataset, family)); if ~isfile(p), return; end
            T = load(p); G = T.F; X = G.feat;
    end
    F = struct('feat', double(X), 'subject', double(G.subject(:)), 'label', double(G.label(:)), 'age', double(G.age(:)), 'sex', double(G.sex(:)));
end
function F = dropSparse(F, minEp)
    if isempty(F), return; end
    [subj, ~, k] = unique(F.subject); cnt = accumarray(k, 1); bad = subj(cnt < minEp); if isempty(bad), return; end
    m = ~ismember(F.subject, bad); for fn = {'feat', 'subject', 'label', 'age', 'sex'}, F.(fn{1}) = F.(fn{1})(m, :); end
end
function keep = matchSubjects(subj, lab, age, sex, ratio, seed)
    % anchors = the SMALLER group (or ADHD if ratio > 1 and ADHD is smaller); each anchor gets `ratio` nearest-age same-sex partners
    rng(seed);
    if nnz(lab == 1) <= nnz(lab == 0), anchor = find(lab == 1); pool = find(lab == 0); else, anchor = find(lab == 0); pool = find(lab == 1); ratio = 1; end
    anchor = anchor(randperm(numel(anchor))); used = false(size(pool)); chosen = [];
    for i = anchor'
        for r = 1:ratio
            c = find(~used & sex(pool) == sex(i)); if isempty(c), c = find(~used); end; if isempty(c), break; end
            [~, j] = min(abs(age(pool(c)) - age(i))); used(c(j)) = true; chosen(end+1) = pool(c(j)); %#ok<AGROW>
        end
    end
    keep = [subj(anchor); subj(chosen)];
end
function [Xtr, Xte] = prepFold(X, tr, te)
    Xtr = X(tr, :); Xte = X(te, :);
    med = median(Xtr, 1, 'omitnan'); med(isnan(med)) = 0; Xtr = fillNaN(Xtr, med); Xte = fillNaN(Xte, med);
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd < 1e-12) = 1; Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
end
function X = fillNaN(X, med), [ii, jj] = find(isnan(X)); if isempty(ii), return; end; X(sub2ind(size(X), ii, jj)) = med(jj); end
function folds = groupedStratifiedFolds(g, y, K)
    [subj, ia] = unique(g); lab = y(ia); folds = zeros(size(g));
    for c = [0 1], s = subj(lab == c); s = s(randperm(numel(s))); fx = mod((1:numel(s)) - 1, K) + 1; for i = 1:numel(s), folds(g == s(i)) = fx(i); end, end
end
function folds = randomFolds(y, K)
    folds = zeros(size(y)); for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
end
function order = subjectTRank(X, y, g)
    [~, ~, k] = unique(g); lab = accumarray(k, y, [], @(v) v(1)); M = zeros(max(k), size(X, 2));
    for j = 1:size(X, 2), M(:, j) = accumarray(k, X(:, j), [], @mean); end
    [~, ~, ~, st] = ttest2(M(lab == 1, :), M(lab == 0, :)); t = abs(st.tstat); t(isnan(t)) = 0; [~, order] = sort(t, 'descend');
end
function Kbest = chooseK(X, y, g, order, Kgrid, kInner, nTrees, maxRows, seed)
    rng(seed); [Xs, ys, gs] = stratSub(X, y, g, maxRows); inner = groupedStratifiedFolds(gs, ys, kInner); bal = zeros(size(Kgrid));
    for q = 1:numel(Kgrid)
        sel = order(1:min(Kgrid(q), numel(order))); P = [];
        for f = 1:kInner, tr = inner ~= f; te = inner == f; [yh, sc] = rfFixed(Xs(tr, sel), ys(tr), Xs(te, sel), nTrees, 5, seed); P = [P; ys(te), yh, sc]; end %#ok<AGROW>
        m = metrics(P); bal(q) = m.BAL;
    end
    [~, q] = max(bal); Kbest = Kgrid(q);
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
        for f = 1:kInner, tr = inner ~= f; te = inner == f; [yh, sc] = fitAndPredict(name, XtrG(tr, :), ytrG(tr), XtrG(te, :), grid(gi), nTrees, seed, svmMax); P = [P; ytrG(te), yh, sc]; end %#ok<AGROW>
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
    den = sqrt(double(TP + FP) * double(TP + FN) * double(TN + FP) * double(TN + FN)); if den > 0, m.MCC = (TP * TN - FP * FN) / den; else, m.MCC = 0; end
    try, [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc; catch, m.AUC = NaN; end
end
function writeRow(fid, cohort, fam, nF, prot, clf, level, ms)
    fld = {'BAL', 'ACC', 'SEN', 'SPE', 'MCC', 'AUC'}; fprintf(fid, '%s,%s,%d,%s,%s,%s', cohort, fam, nF, prot, clf, level);
    for i = 1:numel(fld), v = [ms.(fld{i})]; fprintf(fid, ',%.3f,%.3f', mean(v, 'omitnan'), std(v, 'omitnan')); end; fprintf(fid, '\n');
end
