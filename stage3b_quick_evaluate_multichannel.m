%% stage3b_quick_evaluate_multichannel.m
%  STAGE 3b (QUICK PASS, ~1.5-2 h) - single repeat, four decisive cohort configurations.
%  For the manuscript numbers rerun stage3b_evaluate_multichannel.m (3 repeats, all configurations).
%
%  Cohorts (channel sets are column subsets of the stored features):
%     D1_IEEE_19ch, D1_IEEE_frontal7 (Fp1 F7 F3 Fz F4 F8 Fp2), D1_IEEE_CzF4,
%     D2m_TDBRAIN_26ch (adults, 1:2 age/sex-matched), D2m_TDBRAIN_CzF4,
%     D3_MENDELEY_2ch
%  Feature sets evaluated separately on every cohort:
%     PSR (EMD-EWT-PSR), BandPower (relative delta/theta/alpha/beta + theta/beta), LogVar, PSR+BandPower
%  Per outer fold (5-fold subject-grouped, stratified, R repeats), training fold only:
%     median imputation -> z-scoring -> rank features by |t| of subject-level means
%     -> keep top K, K chosen from K_GRID by inner 3-fold grouped RF balanced accuracy
%     -> DT / kNN / SVM / RF with uniform class prior, hyperparameter by inner grouped CV (balanced accuracy)
%  Metrics: epoch-level and subject-level (majority vote); BAL = balanced accuracy is primary.
%  Transfer (RF): PSR features on the common Cz/F4 channels between adult cohorts.

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage2b_Features'); OUT = fullfile(ROOT, 'Stage3b_Quick_Results');

R_REPEATS = 1; K_OUTER = 5; K_INNER = 3; SEED0 = 2026;
K_GRID = [10 30 100];           % number of features kept after t-ranking (chosen by inner CV)
NO_SELECTION_BELOW = 60;        % feature sets with fewer columns than this are used whole
GRID.DT_MinLeaf = [1 5 20]; GRID.KNN_K = [5 15 35]; GRID.SVM_C = [0.1 1 10]; GRID.RF_MinLeaf = [1 5];
RF_TREES = 100; INNER_MAX_ROWS = 3000; SVM_MAX_ROWS = 6000;
TDB_MATCH_RATIO = 2;
MIN_EPOCHS_PER_SUBJECT = 10;    % subjects with fewer kept epochs are excluded (bad recordings)
FRONTAL7 = {'Fp1','F7','F3','Fz','F4','F8','Fp2'};
RUN_TRANSFER = true;

if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage3b_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 3b QUICK PASS - MULTICHANNEL EVALUATION   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('repeats=%d outer=%d inner=%d | K grid %s | uniform class prior | primary metric = balanced accuracy', R_REPEATS, K_OUTER, K_INNER, mat2str(K_GRID));

F1 = loadF(fullfile(IN, 'D1_IEEE_features.mat')); F2 = loadF(fullfile(IN, 'D2_TDBRAIN_features.mat')); F3 = loadF(fullfile(IN, 'D3_MENDELEY_features.mat'));
[F1, d1] = dropSparseSubjects(F1, MIN_EPOCHS_PER_SUBJECT); [F2, d2] = dropSparseSubjects(F2, MIN_EPOCHS_PER_SUBJECT); [F3, d3] = dropSparseSubjects(F3, MIN_EPOCHS_PER_SUBJECT);
L('Subjects removed for < %d epochs: IEEE %s | TDBRAIN %s | MENDELEY %s', MIN_EPOCHS_PER_SUBJECT, d1, d2, d3);
[F2m, mlog] = matchTdbrain(F2, TDB_MATCH_RATIO, SEED0); L('%s', mlog);

cohorts = struct('name', {}, 'F', {}, 'chan', {});
cohorts(end+1) = struct('name', 'D1_IEEE_19ch',      'F', F1,  'chan', {F1.channels});
cohorts(end+1) = struct('name', 'D1_IEEE_frontal7',  'F', F1,  'chan', {FRONTAL7});
% cohorts(end+1) = struct('name', 'D1_IEEE_CzF4',      'F', F1,  'chan', {{'Cz','F4'}});   % covered by Stage 3
cohorts(end+1) = struct('name', 'D2m_TDBRAIN_26ch',  'F', F2m, 'chan', {F2m.channels});
% cohorts(end+1) = struct('name', 'D2m_TDBRAIN_CzF4',  'F', F2m, 'chan', {{'Cz','F4'}});   % covered by Stage 3
cohorts(end+1) = struct('name', 'D3_MENDELEY_2ch',   'F', F3,  'chan', {F3.channels});
featSets = {'PSR', 'BandPower', 'LogVar', 'PSR+BandPower'};
clfNames = {'DT', 'kNN', 'SVM', 'RF'};

for c = 1:numel(cohorts)
    F = cohorts(c).F; [~, ia] = unique(F.subject);
    L('Cohort %-20s subjects=%3d (ADHD %3d / HC %3d) epochs=%5d channels=%d', cohorts(c).name, numel(ia), ...
        nnz(F.label(ia) == 1), nnz(F.label(ia) == 0), numel(F.label), numel(cohorts(c).chan));
end

results = struct(); selFreq = struct();
fMain = fopen(fullfile(OUT, 'T_results.csv'), 'w');
fprintf(fMain, 'cohort,featureset,classifier,level,BAL,BAL_sd,ACC,ACC_sd,SEN,SEN_sd,SPE,SPE_sd,F1,F1_sd,MCC,MCC_sd,AUC,AUC_sd\n');

%% ------------------------------------------------------------------ MAIN
for c = 1:numel(cohorts)
    F = cohorts(c).F; cname = cohorts(c).name; y = F.label; g = F.subject;
    chMask = @(names) ismember(cellfun(@(s) strtok(s, '_'), names, 'UniformOutput', false), cohorts(c).chan);
    for fs = 1:numel(featSets)
        switch featSets{fs}
            case 'PSR',           X = F.psr(:, chMask(F.psrNames)); names = F.psrNames(chMask(F.psrNames));
            case 'BandPower',     X = F.bp(:, chMask(F.bpNames));  names = F.bpNames(chMask(F.bpNames));
            case 'LogVar',        X = F.logvar(:, chMask(F.lvNames)); names = F.lvNames(chMask(F.lvNames));
            case 'PSR+BandPower', X = [F.psr(:, chMask(F.psrNames)), F.bp(:, chMask(F.bpNames))]; names = [F.psrNames(chMask(F.psrNames)), F.bpNames(chMask(F.bpNames))];
        end
        nF = size(X, 2); tag = sprintf('%s | %s (%d features)', cname, featSets{fs}, nF);
        L('\n================================================================ %s', tag);
        predE = cell(R_REPEATS, numel(clfNames)); predS = predE; selCount = zeros(1, nF); kChosen = [];
        for r = 1:R_REPEATS
            rng(SEED0 + r); folds = groupedStratifiedFolds(g, y, K_OUTER);
            for k = 1:numel(clfNames), predE{r, k} = []; predS{r, k} = []; end
            for f = 1:K_OUTER
                tr = folds ~= f; te = folds == f;
                Xtr = X(tr, :); ytr = y(tr); gtr = g(tr); Xte = X(te, :); yte = y(te); gte = g(te);
                med = median(Xtr, 1, 'omitnan'); med(isnan(med)) = 0; Xtr = fillNaN(Xtr, med); Xte = fillNaN(Xte, med);
                mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd < 1e-12) = 1; Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
                if nF >= NO_SELECTION_BELOW
                    order = subjectTRank(Xtr, ytr, gtr);
                    Kbest = chooseK(Xtr, ytr, gtr, order, K_GRID, K_INNER, RF_TREES, INNER_MAX_ROWS, SEED0 + 10 * r + f);
                    sel = order(1:min(Kbest, nF)); kChosen(end+1) = Kbest; %#ok<AGROW>
                else
                    sel = 1:nF;
                end
                selCount(sel) = selCount(sel) + 1;
                for k = 1:numel(clfNames)
                    [yh, sc] = trainPredict(clfNames{k}, Xtr(:, sel), ytr, gtr, Xte(:, sel), GRID, RF_TREES, K_INNER, SEED0 + r, INNER_MAX_ROWS, SVM_MAX_ROWS);
                    predE{r, k} = [predE{r, k}; yte, yh, sc]; predS{r, k} = [predS{r, k}; subjectVote(gte, yte, yh, sc)];
                end
            end
            for k = 1:numel(clfNames)
                mE = metrics(predE{r, k}); mS = metrics(predS{r, k});
                L('  rep %d %-4s epoch BAL %.1f ACC %.1f AUC %.3f MCC %.3f | subject BAL %.1f ACC %.1f SEN %.1f SPE %.1f AUC %.3f MCC %.3f', ...
                    r, clfNames{k}, mE.BAL, mE.ACC, mE.AUC, mE.MCC, mS.BAL, mS.ACC, mS.SEN, mS.SPE, mS.AUC, mS.MCC);
            end
        end
        if ~isempty(kChosen), L('  K chosen per fold: %s', mat2str(kChosen)); end
        key = matlab.lang.makeValidName([cname '_' featSets{fs}]);
        results.(key) = struct('cohort', cname, 'featureSet', featSets{fs}, 'predE', {predE}, 'predS', {predS}, 'names', {names}, 'selCount', selCount);
        for k = 1:numel(clfNames)
            writeRow(fMain, cname, featSets{fs}, clfNames{k}, 'epoch',   cellfun(@metrics, predE(:, k)));
            writeRow(fMain, cname, featSets{fs}, clfNames{k}, 'subject', cellfun(@metrics, predS(:, k)));
        end
        if nF >= NO_SELECTION_BELOW
            [~, o] = sort(selCount, 'descend'); top = o(1:min(12, nnz(selCount)));
            L('  most selected: %s', strjoin(arrayfun(@(i) sprintf('%s(%d)', names{i}, selCount(i)), top, 'UniformOutput', false), ', '));
        end
    end
end
fclose(fMain);

%% ------------------------------------------------------------------ TRANSFER (PSR, Cz/F4, RF)
if RUN_TRANSFER
    L('\n================================================================ CROSS-COHORT TRANSFER (PSR on Cz+F4, RF uniform prior)');
    fT = fopen(fullfile(OUT, 'T_transfer.csv'), 'w');
    fprintf(fT, 'train,test,epoch_BAL,epoch_AUC,epoch_MCC,subject_BAL,subject_AUC,subject_MCC\n');
    getCzF4 = @(F) F.psr(:, ismember(cellfun(@(s) strtok(s, '_'), F.psrNames, 'UniformOutput', false), {'Cz','F4'}));
    pool = struct('name', {'D2m_TDBRAIN', 'D3_MENDELEY', 'D1_IEEE'}, 'F', {F2m, F3, F1});
    pairs = [1 2; 2 1; 1 3; 2 3; 3 1; 3 2];
    for p = 1:size(pairs, 1)
        A = pool(pairs(p, 1)); B = pool(pairs(p, 2));
        Xa = getCzF4(A.F); Xb = getCzF4(B.F);
        med = median(Xa, 1, 'omitnan'); med(isnan(med)) = 0; Xa = fillNaN(Xa, med); Xb = fillNaN(Xb, med);
        mu = mean(Xa, 1); sd = std(Xa, 0, 1); sd(sd < 1e-12) = 1; Xa = (Xa - mu) ./ sd; Xb = (Xb - mu) ./ sd;
        order = subjectTRank(Xa, A.F.label, A.F.subject); sel = order(1:min(30, numel(order)));
        [yh, sc] = rfFixed(Xa(:, sel), A.F.label, Xb(:, sel), RF_TREES, 5, SEED0);
        mE = metrics([B.F.label, yh, sc]); mS = metrics(subjectVote(B.F.subject, B.F.label, yh, sc));
        L('  train %-12s -> test %-12s : epoch BAL %.1f AUC %.3f MCC %.3f | subject BAL %.1f AUC %.3f MCC %.3f', ...
            A.name, B.name, mE.BAL, mE.AUC, mE.MCC, mS.BAL, mS.AUC, mS.MCC);
        fprintf(fT, '%s,%s,%.2f,%.3f,%.3f,%.2f,%.3f,%.3f\n', A.name, B.name, mE.BAL, mE.AUC, mE.MCC, mS.BAL, mS.AUC, mS.MCC);
    end
    fclose(fT);
end
save(fullfile(OUT, 'Stage3b_Results.mat'), 'results', 'GRID', '-v7.3');
L('\nDONE. Send Stage3b_Report.txt, T_results.csv and T_transfer.csv.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
function F = loadF(p)
    T = load(p); F = T.F; F.label = double(F.label(:)); F.subject = double(F.subject(:)); F.age = double(F.age(:)); F.sex = double(F.sex(:));
end
function X = fillNaN(X, med)
    [ii, jj] = find(isnan(X)); if isempty(ii), return; end
    X(sub2ind(size(X), ii, jj)) = med(jj);
end
function [F, msg] = dropSparseSubjects(F, minEp)
    [subj, ~, k] = unique(F.subject); cnt = accumarray(k, 1); bad = subj(cnt < minEp);
    if isempty(bad), msg = 'none'; else
        [~, ia] = unique(F.subject); ids = F.subjectID(ia); msg = strjoin(cellstr(ids(ismember(subj, bad)))', ', ');
        m = ~ismember(F.subject, bad);
        for fn = {'psr', 'bp', 'logvar', 'subject', 'subjectID', 'label', 'age', 'sex'}, F.(fn{1}) = F.(fn{1})(m, :); end
    end
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
    keep = [subj(idxA); subj(chosen)]; m = ismember(F.subject, keep); Fm = F;
    for fn = {'psr', 'bp', 'logvar', 'subject', 'subjectID', 'label', 'age', 'sex'}, Fm.(fn{1}) = F.(fn{1})(m, :); end
    msg = sprintf('TDBRAIN matched: ADHD %d (mean age %.1f, male %.0f%%) vs HC %d (mean age %.1f, male %.0f%%)', ...
        numel(idxA), mean(age(idxA)), 100 * mean(sex(idxA) == 1), numel(chosen), mean(age(chosen)), 100 * mean(sex(chosen) == 1));
end
function folds = groupedStratifiedFolds(g, y, K)
    [subj, ia] = unique(g); lab = y(ia); folds = zeros(size(g));
    for c = [0 1]
        s = subj(lab == c); s = s(randperm(numel(s))); fid = mod((1:numel(s)) - 1, K) + 1;
        for i = 1:numel(s), folds(g == s(i)) = fid(i); end
    end
end
function order = subjectTRank(X, y, g)
    [~, ~, k] = unique(g); lab = accumarray(k, y, [], @(v) v(1));
    M = zeros(max(k), size(X, 2));
    for j = 1:size(X, 2), M(:, j) = accumarray(k, X(:, j), [], @mean); end
    [~, ~, ~, st] = ttest2(M(lab == 1, :), M(lab == 0, :));
    t = abs(st.tstat); t(isnan(t)) = 0; [~, order] = sort(t, 'descend');
end
function Kbest = chooseK(X, y, g, order, Kgrid, kInner, nTrees, maxRows, seed)
    rng(seed); [Xs, ys, gs] = stratSub(X, y, g, maxRows); inner = groupedStratifiedFolds(gs, ys, kInner);
    bal = zeros(size(Kgrid));
    for q = 1:numel(Kgrid)
        sel = order(1:min(Kgrid(q), numel(order))); P = [];
        for f = 1:kInner
            tr = inner ~= f; te = inner == f;
            [yh, sc] = rfFixed(Xs(tr, sel), ys(tr), Xs(te, sel), nTrees, 5, seed); P = [P; ys(te), yh, sc]; %#ok<AGROW>
        end
        m = metrics(P); bal(q) = m.BAL;
    end
    [~, q] = max(bal); Kbest = Kgrid(q);
end
function [Xs, ys, gs] = stratSub(X, y, g, maxRows)
    n = numel(y); if n <= maxRows, Xs = X; ys = y; gs = g; return; end
    keep = [];
    for c = [0 1], idx = find(y == c); idx = idx(randperm(numel(idx))); keep = [keep; idx(1:round(maxRows * numel(idx) / n))]; end %#ok<AGROW>
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
    [~, gi] = max(bal);
    [yhat, score] = fitAndPredict(name, Xtr, ytr, Xte, grid(gi), nTrees, seed, svmMax);
end
function [yhat, score] = fitAndPredict(name, Xtr, ytr, Xte, hp, nTrees, seed, svmMax)
    rng(seed);
    switch name
        case 'DT',  mdl = fitctree(Xtr, ytr, 'MinLeafSize', hp, 'Prior', 'uniform'); [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'kNN', mdl = fitcknn(Xtr, ytr, 'NumNeighbors', hp, 'DistanceWeight', 'inverse', 'Prior', 'uniform'); [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'SVM'
            [Xtr, ytr] = stratSub(Xtr, ytr, ytr, svmMax);
            mdl = fitcsvm(Xtr, ytr, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'BoxConstraint', hp, 'Prior', 'uniform');
            [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'RF',  [yhat, score] = rfFixed(Xtr, ytr, Xte, nTrees, hp, seed);
    end
    yhat = double(yhat(:)); score = double(score(:));
end
function [yhat, score] = rfFixed(Xtr, ytr, Xte, nTrees, minLeaf, seed)
    rng(seed);
    mdl = TreeBagger(nTrees, Xtr, ytr, 'Method', 'classification', 'MinLeafSize', minLeaf, 'Prior', 'Uniform', 'OOBPrediction', 'off');
    [lab, sc] = predict(mdl, Xte); yhat = double(str2double(lab(:))); score = double(sc(:, strcmp(mdl.ClassNames, '1')));
end
function S = subjectVote(g, y, yhat, score)
    [~, ~, k] = unique(g); yt = accumarray(k, y, [], @(v) v(1)); ms = accumarray(k, score, [], @mean); vote = accumarray(k, yhat, [], @mean);
    yv = double(vote > 0.5); tie = vote == 0.5; yv(tie) = double(ms(tie) > 0.5); S = [yt, yv, ms];
end
function m = metrics(P)
    y = P(:, 1); yh = P(:, 2); sc = P(:, 3);
    TP = nnz(y == 1 & yh == 1); TN = nnz(y == 0 & yh == 0); FP = nnz(y == 0 & yh == 1); FN = nnz(y == 1 & yh == 0);
    m.ACC = 100 * (TP + TN) / max(numel(y), 1); m.SEN = 100 * TP / max(TP + FN, 1); m.SPE = 100 * TN / max(TN + FP, 1);
    m.BAL = (m.SEN + m.SPE) / 2; m.PPV = 100 * TP / max(TP + FP, 1); m.F1 = 200 * TP / max(2 * TP + FP + FN, 1);
    den = sqrt(double(TP + FP) * double(TP + FN) * double(TN + FP) * double(TN + FN)); m.MCC = ifelse(den > 0, (TP * TN - FP * FN) / den, 0);
    try, [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc; catch, m.AUC = NaN; end
end
function writeRow(fid, cohort, fset, clf, level, ms)
    fld = {'BAL', 'ACC', 'SEN', 'SPE', 'F1', 'MCC', 'AUC'};
    fprintf(fid, '%s,%s,%s,%s', cohort, fset, clf, level);
    for i = 1:numel(fld), v = arrayfun(@(m) m.(fld{i}), ms); fprintf(fid, ',%.3f,%.3f', mean(v, 'omitnan'), std(v, 'omitnan')); end
    fprintf(fid, '\n');
end
