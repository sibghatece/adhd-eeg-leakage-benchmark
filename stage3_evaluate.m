%% stage3_evaluate.m
%  STAGE 3 - Leakage-free evaluation on the Stage 2 feature stores.
%
%  Cohorts
%     D1  IEEE      children, all subjects
%     D2m TDBRAIN   adults (>= 18 y) with 1:2 age- and sex-matched healthy controls
%     D2a TDBRAIN   all adults (unmatched, secondary)
%     D3  MENDELEY  adults, all subjects
%
%  Main experiment (per cohort, per repeat):
%     5-fold subject-grouped CV, folds stratified by subject label.
%     Inside each TRAINING fold only:
%        median imputation of NaN -> z-scoring
%        -> AO and GWO wrapper selection (fitness: inner 3-fold grouped k-NN on a subsample)
%        -> intersection (fallback to union if too small)
%        -> subject-level two-sample test with Benjamini-Hochberg (q < 0.05) pruning
%        -> DT / k-NN / SVM-RBF / RF with hyperparameters chosen by inner 3-fold grouped CV
%     Test fold: epoch-level metrics and subject-level metrics (majority vote).
%
%  Also: log-variance baseline (2 features), ablations (RF), cross-cohort transfer (RF).
%
%  Outputs in <OUT>: Stage3_Report.txt, Stage3_Results.mat, and CSV tables
%     T_main.csv, T_subject.csv, T_baseline.csv, T_ablation.csv, T_transfer.csv,
%     T_selected_features.csv
%
%  Requires Statistics and Machine Learning Toolbox. No readtable/readcell used.

clear; clc; close all;

%% ------------------------------------------------------------------ CONFIG
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN   = fullfile(ROOT, 'Stage2_Features');
OUT  = fullfile(ROOT, 'Stage3_Results');

R_REPEATS   = 3;        % repeats of the outer 5-fold CV with different subject shuffles
K_OUTER     = 5;
K_INNER     = 3;
SEED0       = 2026;

% metaheuristic feature selection
POP         = 20;       % population size
ITERS       = 40;       % iterations
FIT_ALPHA   = 0.99;     % fitness = alpha*(1-acc) + (1-alpha)*(#selected/#features)
FIT_MAX_EPOCHS = 1500;  % subsample of training epochs used inside the fitness function
FIT_KNN_K   = 5;
MIN_INTERSECTION = 8;   % if AO n GWO is smaller than this, use the union
BH_Q        = 0.05;     % Benjamini-Hochberg threshold for the subject-level test
MIN_AFTER_TEST = 5;     % if fewer survive the test, keep the pre-test set

% classifiers and small inner grids
GRID.DT_MinLeaf  = [1 5 20];
GRID.KNN_K       = [5 15 35];
GRID.SVM_C       = [0.1 1 10];
GRID.RF_MinLeaf  = [1 5];
RF_TREES         = 100;
INNER_MAX_ROWS   = 3000;  % inner hyperparameter grid runs on at most this many training epochs (final fit uses all)

% TDBRAIN cohort
TDB_MIN_AGE = 18;
TDB_MATCH_RATIO = 2;    % healthy controls per ADHD subject

RUN_ABLATION = true;
RUN_TRANSFER = true;

%% ------------------------------------------------------------------ SETUP
if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage3_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 3 - EVALUATION   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('repeats=%d outer=%d inner=%d | AO/GWO pop=%d iters=%d alpha=%.2f fitness-subsample=%d | BH q=%.2f', ...
    R_REPEATS, K_OUTER, K_INNER, POP, ITERS, FIT_ALPHA, FIT_MAX_EPOCHS, BH_Q);

F1 = loadStore(fullfile(IN, 'D1_IEEE_features.mat'));
F2 = loadStore(fullfile(IN, 'D2_TDBRAIN_features.mat'));
F3 = loadStore(fullfile(IN, 'D3_MENDELEY_features.mat'));
featNames = F1.featNames;

%% ------------------------------------------------------------------ COHORTS
cohorts = struct('name', {}, 'F', {});
cohorts(end + 1) = struct('name', 'D1_IEEE', 'F', F1);
[F2m, F2a, matchLog] = buildTdbrainCohorts(F2, TDB_MIN_AGE, TDB_MATCH_RATIO, SEED0);
for q = 1:numel(matchLog), L('%s', matchLog{q}); end
cohorts(end + 1) = struct('name', 'D2m_TDBRAIN_matched', 'F', F2m);
cohorts(end + 1) = struct('name', 'D2a_TDBRAIN_alladults', 'F', F2a);
cohorts(end + 1) = struct('name', 'D3_MENDELEY', 'F', F3);
for c = 1:numel(cohorts)
    F = cohorts(c).F;
    [~, ia] = unique(F.subject);
    L('Cohort %-24s subjects=%3d (ADHD %3d / HC %3d)  epochs=%5d  mean age ADHD %.1f / HC %.1f', ...
        cohorts(c).name, numel(ia), nnz(F.label(ia) == 1), nnz(F.label(ia) == 0), numel(F.label), ...
        mean(F.age(ia(F.label(ia) == 1)), 'omitnan'), mean(F.age(ia(F.label(ia) == 0)), 'omitnan'));
end

clfNames = {'DT', 'kNN', 'SVM', 'RF'};
results = struct();

%% ------------------------------------------------------------------ MAIN EXPERIMENT
for c = 1:numel(cohorts)
    F = cohorts(c).F; cname = cohorts(c).name;
    L('\n================================================================ %s', cname);
    X = F.feat; y = F.label; g = F.subject; B = F.baseline;
    nF = size(X, 2);
    perRep = struct('epoch', {}, 'subject', {}, 'base_epoch', {}, 'base_subject', {}, 'sel', {}, 'abl', {});
    selCount = zeros(1, nF);
    for r = 1:R_REPEATS
        rng(SEED0 + r);
        folds = groupedStratifiedFolds(g, y, K_OUTER);
        L('--- repeat %d', r);
        predE = struct(); predS = struct();               % accumulate per classifier
        for k = 1:numel(clfNames), predE.(clfNames{k}) = []; predS.(clfNames{k}) = []; end
        baseE = struct(); baseS = struct();
        for k = 1:numel(clfNames), baseE.(clfNames{k}) = []; baseS.(clfNames{k}) = []; end
        ablNames = {'AO_only', 'GWO_only', 'Intersection_noTest', 'All304_noSel', 'Cz_only_noSel', 'F4_only_noSel', ...
                    'PSR2D_only_noSel', 'PSR3D_only_noSel'};
        ablE = struct(); ablS = struct();
        for k = 1:numel(ablNames), ablE.(ablNames{k}) = []; ablS.(ablNames{k}) = []; end
        selInfo = zeros(K_OUTER, 4);   % nAO, nGWO, nInter, nFinal

        for f = 1:K_OUTER
            tr = folds ~= f; te = folds == f;
            Xtr = X(tr, :); ytr = y(tr); gtr = g(tr);
            Xte = X(te, :); yte = y(te); gte = g(te);
            % ---- imputation + scaling from training fold only
            med = median(Xtr, 1, 'omitnan'); med(isnan(med)) = 0;
            Xtr = fillNaN(Xtr, med); Xte = fillNaN(Xte, med);
            mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd < 1e-12) = 1;
            Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;

            % ---- feature selection (training fold only)
            fitObj = makeFitness(Xtr, ytr, gtr, FIT_MAX_EPOCHS, K_INNER, FIT_KNN_K, FIT_ALPHA, SEED0 + 100 * r + f);
            maskAO  = aquilaOptimizer(fitObj, nF, POP, ITERS, SEED0 + 1000 * r + f);
            maskGWO = greyWolfOptimizer(fitObj, nF, POP, ITERS, SEED0 + 2000 * r + f);
            inter = maskAO & maskGWO;
            if nnz(inter) < MIN_INTERSECTION, inter = maskAO | maskGWO; usedUnion = true; else, usedUnion = false; end
            selTest = subjectLevelTest(Xtr, ytr, gtr, find(inter), BH_Q);
            if numel(selTest) < MIN_AFTER_TEST, selFinal = find(inter); else, selFinal = selTest; end
            selInfo(f, :) = [nnz(maskAO), nnz(maskGWO), nnz(maskAO & maskGWO), numel(selFinal)];
            selCount(selFinal) = selCount(selFinal) + 1;
            L('   fold %d: AO=%d GWO=%d inter=%d%s -> after test=%d', f, selInfo(f, 1), selInfo(f, 2), selInfo(f, 3), ...
                ifelse(usedUnion, ' (union used)', ''), selInfo(f, 4));

            % ---- classifiers with inner grid
            for k = 1:numel(clfNames)
                [yhat, score] = trainPredict(clfNames{k}, Xtr(:, selFinal), ytr, gtr, Xte(:, selFinal), GRID, RF_TREES, K_INNER, SEED0 + r, INNER_MAX_ROWS);
                predE.(clfNames{k}) = [predE.(clfNames{k}); yte, yhat, score];
                predS.(clfNames{k}) = [predS.(clfNames{k}); subjectVote(gte, yte, yhat, score)];
                % baseline: log-variance only, same classifier, default grid
                Btr = B(tr, :); Bte = B(te, :); bmu = mean(Btr); bsd = std(Btr); bsd(bsd < 1e-12) = 1;
                [yhb, scb] = trainPredict(clfNames{k}, (Btr - bmu) ./ bsd, ytr, gtr, (Bte - bmu) ./ bsd, GRID, RF_TREES, K_INNER, SEED0 + r, INNER_MAX_ROWS);
                baseE.(clfNames{k}) = [baseE.(clfNames{k}); yte, yhb, scb];
                baseS.(clfNames{k}) = [baseS.(clfNames{k}); subjectVote(gte, yte, yhb, scb)];
            end

            % ---- ablations (RF, fixed MinLeaf 5, no inner grid)
            if RUN_ABLATION
                is2D = ~cellfun(@isempty, regexp(featNames, '_(CEM|TPS|ICA|ISM|IAM|TMS|PDD|SDD|OCM|RDA|SAM|OTS|OAM|BDI|ECM)$'));
                isCz = startsWith(featNames, 'Cz_');
                ablSets = {find(maskAO), find(maskGWO), find(inter), 1:nF, find(isCz), find(~isCz), find(is2D), find(~is2D)};
                for a = 1:numel(ablNames)
                    cols = ablSets{a}; if isempty(cols), continue; end
                    [yha, sca] = rfFixed(Xtr(:, cols), ytr, Xte(:, cols), RF_TREES, 5, SEED0 + r);
                    ablE.(ablNames{a}) = [ablE.(ablNames{a}); yte, yha, sca];
                    ablS.(ablNames{a}) = [ablS.(ablNames{a}); subjectVote(gte, yte, yha, sca)];
                end
            end
        end
        rep = struct('epoch', predE, 'subject', predS, 'base_epoch', baseE, 'base_subject', baseS, ...
                     'sel', selInfo, 'abl', struct('epoch', ablE, 'subject', ablS));
        perRep(r) = rep;
        for k = 1:numel(clfNames)
            mE = metrics(predE.(clfNames{k})); mS = metrics(predS.(clfNames{k}));
            L('   %-4s epoch: ACC %.2f SEN %.2f SPE %.2f F1 %.2f MCC %.3f AUC %.3f | subject: ACC %.2f SEN %.2f SPE %.2f MCC %.3f AUC %.3f', ...
                clfNames{k}, mE.ACC, mE.SEN, mE.SPE, mE.F1, mE.MCC, mE.AUC, mS.ACC, mS.SEN, mS.SPE, mS.MCC, mS.AUC);
        end
    end
    results.(cname).perRep = perRep;
    results.(cname).selCount = selCount;
    results.(cname).nSubjects = numel(unique(g));
end

%% ------------------------------------------------------------------ CROSS-COHORT TRANSFER
transfer = [];
if RUN_TRANSFER
    L('\n================================================================ CROSS-COHORT TRANSFER (RF)');
    pairs = {'D2m_TDBRAIN_matched', 'D3_MENDELEY'; 'D3_MENDELEY', 'D2m_TDBRAIN_matched'; ...
             'D2m_TDBRAIN_matched', 'D1_IEEE'; 'D3_MENDELEY', 'D1_IEEE'; 'D1_IEEE', 'D3_MENDELEY'; 'D1_IEEE', 'D2m_TDBRAIN_matched'};
    % feature selection once per training cohort (on the full cohort)
    selByCohort = struct();
    for c = 1:numel(cohorts)
        if strcmp(cohorts(c).name, 'D2a_TDBRAIN_alladults'), continue; end
        F = cohorts(c).F; X = F.feat; y = F.label; g = F.subject;
        med = median(X, 1, 'omitnan'); med(isnan(med)) = 0; X = fillNaN(X, med);
        mu = mean(X, 1); sd = std(X, 0, 1); sd(sd < 1e-12) = 1; Xz = (X - mu) ./ sd;
        fitObj = makeFitness(Xz, y, g, FIT_MAX_EPOCHS, K_INNER, FIT_KNN_K, FIT_ALPHA, SEED0 + 7);
        mA = aquilaOptimizer(fitObj, size(X, 2), POP, ITERS, SEED0 + 77);
        mG = greyWolfOptimizer(fitObj, size(X, 2), POP, ITERS, SEED0 + 78);
        inter = mA & mG; if nnz(inter) < MIN_INTERSECTION, inter = mA | mG; end
        sel = subjectLevelTest(Xz, y, g, find(inter), BH_Q); if numel(sel) < MIN_AFTER_TEST, sel = find(inter); end
        selByCohort.(cohorts(c).name) = struct('sel', sel, 'med', med, 'mu', mu, 'sd', sd);
        L('  %-24s selected %d features for transfer', cohorts(c).name, numel(sel));
    end
    for p = 1:size(pairs, 1)
        A = cohorts(strcmp({cohorts.name}, pairs{p, 1})).F; Bc = cohorts(strcmp({cohorts.name}, pairs{p, 2})).F;
        sb = selByCohort.(pairs{p, 1});
        Xa = (fillNaN(A.feat, sb.med) - sb.mu) ./ sb.sd; Xb = (fillNaN(Bc.feat, sb.med) - sb.mu) ./ sb.sd;
        [yhat, score] = rfFixed(Xa(:, sb.sel), A.label, Xb(:, sb.sel), RF_TREES, 5, SEED0);
        mE = metrics([Bc.label, yhat, score]); mS = metrics(subjectVote(Bc.subject, Bc.label, yhat, score));
        L('  train %-24s -> test %-24s : epoch ACC %.2f AUC %.3f MCC %.3f | subject ACC %.2f AUC %.3f MCC %.3f', ...
            pairs{p, 1}, pairs{p, 2}, mE.ACC, mE.AUC, mE.MCC, mS.ACC, mS.AUC, mS.MCC);
        transfer = [transfer; struct('train', pairs{p, 1}, 'test', pairs{p, 2}, 'epoch', mE, 'subject', mS)]; %#ok<AGROW>
    end
end

%% ------------------------------------------------------------------ TABLES
L('\n================================================================ TABLES (mean +/- sd over %d repeats)', R_REPEATS);
fMain = fopen(fullfile(OUT, 'T_main.csv'), 'w'); fSubj = fopen(fullfile(OUT, 'T_subject.csv'), 'w');
fBase = fopen(fullfile(OUT, 'T_baseline.csv'), 'w'); fAbl = fopen(fullfile(OUT, 'T_ablation.csv'), 'w');
hdrLine = 'cohort,classifier,ACC,ACC_sd,SEN,SEN_sd,SPE,SPE_sd,PPV,PPV_sd,F1,F1_sd,MCC,MCC_sd,AUC,AUC_sd\n';
fprintf(fMain, hdrLine); fprintf(fSubj, hdrLine); fprintf(fBase, ['level,' hdrLine]); fprintf(fAbl, ['level,' hdrLine]);
cn = fieldnames(results);
for c = 1:numel(cn)
    R = results.(cn{c}).perRep;
    for k = 1:numel(clfNames)
        writeRow(fMain, cn{c}, clfNames{k}, arrayfun(@(rr) metrics(rr.epoch.(clfNames{k})), R));
        writeRow(fSubj, cn{c}, clfNames{k}, arrayfun(@(rr) metrics(rr.subject.(clfNames{k})), R));
        fprintf(fBase, 'epoch,'); writeRow(fBase, cn{c}, clfNames{k}, arrayfun(@(rr) metrics(rr.base_epoch.(clfNames{k})), R));
        fprintf(fBase, 'subject,'); writeRow(fBase, cn{c}, clfNames{k}, arrayfun(@(rr) metrics(rr.base_subject.(clfNames{k})), R));
    end
    if RUN_ABLATION
        an = fieldnames(R(1).abl.epoch);
        for a = 1:numel(an)
            if isempty(R(1).abl.epoch.(an{a})), continue; end
            fprintf(fAbl, 'epoch,'); writeRow(fAbl, cn{c}, an{a}, arrayfun(@(rr) metrics(rr.abl.epoch.(an{a})), R));
            fprintf(fAbl, 'subject,'); writeRow(fAbl, cn{c}, an{a}, arrayfun(@(rr) metrics(rr.abl.subject.(an{a})), R));
        end
    end
    % selected-feature frequency
    sc = results.(cn{c}).selCount; [~, o] = sort(sc, 'descend');
    L('  %s: most frequently selected features (of %d folds): %s', cn{c}, R_REPEATS * K_OUTER, ...
        strjoin(arrayfun(@(i) sprintf('%s(%d)', featNames{i}, sc(i)), o(1:min(15, nnz(sc))), 'UniformOutput', false), ', '));
end
fclose(fMain); fclose(fSubj); fclose(fBase); fclose(fAbl);
fSel = fopen(fullfile(OUT, 'T_selected_features.csv'), 'w');
fprintf(fSel, 'feature_index,feature_name%s\n', sprintf(',%s', cn{:}));
for i = 1:numel(featNames)
    fprintf(fSel, '%d,%s', i, featNames{i});
    for c = 1:numel(cn), fprintf(fSel, ',%d', results.(cn{c}).selCount(i)); end
    fprintf(fSel, '\n');
end
fclose(fSel);
if RUN_TRANSFER
    fT = fopen(fullfile(OUT, 'T_transfer.csv'), 'w');
    fprintf(fT, 'train,test,epoch_ACC,epoch_SEN,epoch_SPE,epoch_MCC,epoch_AUC,subject_ACC,subject_SEN,subject_SPE,subject_MCC,subject_AUC\n');
    for p = 1:numel(transfer)
        t = transfer(p);
        fprintf(fT, '%s,%s,%.2f,%.2f,%.2f,%.3f,%.3f,%.2f,%.2f,%.2f,%.3f,%.3f\n', t.train, t.test, ...
            t.epoch.ACC, t.epoch.SEN, t.epoch.SPE, t.epoch.MCC, t.epoch.AUC, t.subject.ACC, t.subject.SEN, t.subject.SPE, t.subject.MCC, t.subject.AUC);
    end
    fclose(fT);
end
save(fullfile(OUT, 'Stage3_Results.mat'), 'results', 'transfer', 'cohorts', 'featNames', 'GRID', '-v7.3');
L('\nDONE. Tables in %s. Send Stage3_Report.txt and the CSV files.', OUT);

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin)
    s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s);
end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end

function F = loadStore(p)
    T = load(p); F = T.F;
    F.label = double(F.label(:)); F.subject = double(F.subject(:)); F.age = double(F.age(:)); F.sex = double(F.sex(:));
end

function X = fillNaN(X, med)
    [ii, jj] = find(isnan(X)); if isempty(ii), return; end
    X(sub2ind(size(X), ii, jj)) = med(jj);
end

% --------------------------------------------------------------------------
function [Fm, Fa, logc] = buildTdbrainCohorts(F, minAge, ratio, seed)
    logc = {};
    [subj, ia] = unique(F.subject);
    lab = F.label(ia); age = F.age(ia); sex = F.sex(ia);
    adult = age >= minAge;
    logc{end + 1} = sprintf('TDBRAIN adults (>=%d y): ADHD %d, HC %d (excluded %d ADHD and %d HC under %d)', ...
        minAge, nnz(adult & lab == 1), nnz(adult & lab == 0), nnz(~adult & lab == 1), nnz(~adult & lab == 0), minAge);
    keepA = subj(adult);
    Fa = subsetStore(F, keepA);
    % greedy nearest-age same-sex matching, ratio controls per ADHD
    rng(seed);
    idxA = find(adult & lab == 1); idxH = find(adult & lab == 0);
    idxA = idxA(randperm(numel(idxA)));
    used = false(size(idxH)); chosen = [];
    for i = idxA'
        for rr = 1:ratio
            cand = find(~used & sex(idxH) == sex(i));
            if isempty(cand), cand = find(~used); end
            if isempty(cand), break; end
            [~, j] = min(abs(age(idxH(cand)) - age(i)));
            used(cand(j)) = true; chosen(end + 1) = idxH(cand(j)); %#ok<AGROW>
        end
    end
    keepM = [subj(idxA); subj(chosen)];
    Fm = subsetStore(F, keepM);
    logc{end + 1} = sprintf('TDBRAIN matched: ADHD %d (mean age %.1f, male %.0f%%) vs HC %d (mean age %.1f, male %.0f%%)', ...
        numel(idxA), mean(age(idxA)), 100 * mean(sex(idxA) == 1), numel(chosen), mean(age(chosen)), 100 * mean(sex(chosen) == 1));
end

function Fs = subsetStore(F, keepSubjects)
    m = ismember(F.subject, keepSubjects);
    Fs = F;
    fn = {'feat', 'baseline', 'subject', 'subjectID', 'label', 'age', 'sex', 'nIMF'};
    for k = 1:numel(fn), if isfield(Fs, fn{k}), Fs.(fn{k}) = F.(fn{k})(m, :); end, end
    if isfield(Fs, 'nMode'), Fs.nMode = F.nMode(m, :, :); end
end

% --------------------------------------------------------------------------
function folds = groupedStratifiedFolds(g, y, K)
    % assign subjects (groups) to K folds, balancing subject labels
    [subj, ia] = unique(g); lab = y(ia);
    folds = zeros(size(g));
    for c = [0 1]
        s = subj(lab == c); s = s(randperm(numel(s)));
        fid = mod((1:numel(s)) - 1, K) + 1;
        for i = 1:numel(s), folds(g == s(i)) = fid(i); end
    end
end

% --------------------------------------------------------------------------
function fitObj = makeFitness(X, y, g, maxEp, kInner, kNN, alpha, seed)
    % fixed subsample + fixed inner grouped folds so all evaluations are comparable
    rng(seed);
    n = size(X, 1);
    if n > maxEp
        keep = [];
        for c = [0 1]
            idx = find(y == c); idx = idx(randperm(numel(idx)));
            keep = [keep; idx(1:min(numel(idx), round(maxEp * numel(idx) / n)))]; %#ok<AGROW>
        end
    else
        keep = (1:n)';
    end
    Xs = X(keep, :); ys = y(keep); gs = g(keep);
    inner = groupedStratifiedFolds(gs, ys, kInner);
    fitObj = struct('X', Xs, 'y', ys, 'inner', inner, 'kInner', kInner, 'kNN', kNN, 'alpha', alpha, 'nF', size(X, 2));
end

function fval = fitness(mask, fo)
    sel = find(mask);
    if isempty(sel), fval = 1; return; end
    correct = 0;
    for f = 1:fo.kInner
        tr = fo.inner ~= f; te = fo.inner == f;
        mdl = fitcknn(fo.X(tr, sel), fo.y(tr), 'NumNeighbors', fo.kNN, 'Distance', 'euclidean');
        correct = correct + nnz(predict(mdl, fo.X(te, sel)) == fo.y(te));
    end
    acc = correct / numel(fo.y);
    fval = fo.alpha * (1 - acc) + (1 - fo.alpha) * numel(sel) / fo.nF;
end

% --------------------------------------------------------------------------
function best = aquilaOptimizer(fo, D, N, T, seed)
    % Aquila Optimizer (Abualigah et al., 2021), continuous search in [0,1]^D,
    % binary mask by thresholding at 0.5. Four update strategies:
    %  t <= 2T/3 : X1 expanded exploration, X2 narrowed exploration (Levy + spiral)
    %  t >  2T/3 : X3 expanded exploitation, X4 narrowed exploitation (QF)
    rng(seed);
    X = rand(N, D); fit = zeros(N, 1);
    for i = 1:N, fit(i) = fitness(X(i, :) >= 0.5, fo); end
    [bestFit, ib] = min(fit); Xbest = X(ib, :);
    alphaP = 0.1; deltaP = 0.1; LB = 0; UB = 1;
    for t = 1:T
        XM = mean(X, 1);
        G2 = 2 * (1 - t / T);
        for i = 1:N
            if t <= (2 / 3) * T
                if rand < 0.5
                    Xn = Xbest * (1 - t / T) + (XM - Xbest * rand);
                else
                    r1 = 10; U = 0.00565; omega = 0.005; D1 = 1:D;
                    rr = r1 + U * D1; theta = -omega * D1 + 3 * pi / 2;
                    xs = rr .* sin(theta); ys = rr .* cos(theta);
                    Xn = Xbest .* levyFlight(D) + X(randi(N), :) + (ys - xs) * rand;
                end
            else
                if rand < 0.5
                    Xn = (Xbest - XM) * alphaP - rand + ((UB - LB) * rand + LB) * deltaP;
                else
                    QF = t ^ ((2 * rand - 1) / (1 - T) ^ 2); G1 = 2 * rand - 1;
                    Xn = QF * Xbest - (G1 * X(i, :) * rand) - G2 * levyFlight(D) + rand * G1;
                end
            end
            Xn = min(max(Xn, LB), UB);
            fn = fitness(Xn >= 0.5, fo);
            if fn < fit(i), X(i, :) = Xn; fit(i) = fn; end
            if fn < bestFit, bestFit = fn; Xbest = Xn; end
        end
    end
    best = Xbest >= 0.5;
end

function s = levyFlight(D)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
    u = randn(1, D) * sigma; v = randn(1, D);
    s = 0.01 * u ./ abs(v) .^ (1 / beta);
end

function best = greyWolfOptimizer(fo, D, N, T, seed)
    % Grey Wolf Optimizer (Mirjalili et al., 2014), continuous in [0,1]^D,
    % binary mask by thresholding at 0.5.
    rng(seed);
    X = rand(N, D); fit = zeros(N, 1);
    for i = 1:N, fit(i) = fitness(X(i, :) >= 0.5, fo); end
    [~, o] = sort(fit); Xa = X(o(1), :); Xb = X(o(2), :); Xd = X(o(3), :);
    fa = fit(o(1)); fb = fit(o(2)); fd = fit(o(3));
    for t = 1:T
        a = 2 - 2 * t / T;
        for i = 1:N
            A1 = 2 * a * rand(1, D) - a; C1 = 2 * rand(1, D); X1 = Xa - A1 .* abs(C1 .* Xa - X(i, :));
            A2 = 2 * a * rand(1, D) - a; C2 = 2 * rand(1, D); X2 = Xb - A2 .* abs(C2 .* Xb - X(i, :));
            A3 = 2 * a * rand(1, D) - a; C3 = 2 * rand(1, D); X3 = Xd - A3 .* abs(C3 .* Xd - X(i, :));
            Xn = min(max((X1 + X2 + X3) / 3, 0), 1);
            fn = fitness(Xn >= 0.5, fo);
            X(i, :) = Xn; fit(i) = fn;
            if fn < fa, Xd = Xb; fd = fb; Xb = Xa; fb = fa; Xa = Xn; fa = fn;
            elseif fn < fb, Xd = Xb; fd = fb; Xb = Xn; fb = fn;
            elseif fn < fd, Xd = Xn; fd = fn;
            end
        end
    end
    best = Xa >= 0.5;
end

% --------------------------------------------------------------------------
function sel = subjectLevelTest(X, y, g, cand, q)
    % two-sample t-test on subject means (one value per subject), BH-corrected
    [subj, ~, k] = unique(g);
    lab = accumarray(k, y, [], @(v) v(1));
    M = zeros(numel(subj), numel(cand));
    for j = 1:numel(cand), M(:, j) = accumarray(k, X(:, cand(j)), [], @mean); end
    [~, p] = ttest2(M(lab == 1, :), M(lab == 0, :));
    p = p(:)'; m = numel(p); [ps, o] = sort(p);
    thr = (1:m) / m * q; kmax = find(ps <= thr, 1, 'last');
    if isempty(kmax), sel = []; else, sel = cand(sort(o(1:kmax))); end
end

% --------------------------------------------------------------------------
function [yhat, score] = trainPredict(name, Xtr, ytr, gtr, Xte, GRID, nTrees, kInner, seed, maxRows)
    % choose hyperparameter by inner grouped CV accuracy (on a stratified subsample of at
    % most maxRows epochs, subject-grouped), then fit on the full training fold
    rng(seed);
    n = numel(ytr);
    if n > maxRows
        keep = [];
        for c = [0 1]
            idx = find(ytr == c); idx = idx(randperm(numel(idx)));
            keep = [keep; idx(1:round(maxRows * numel(idx) / n))]; %#ok<AGROW>
        end
        XtrG = Xtr(keep, :); ytrG = ytr(keep); gtrG = gtr(keep);
    else
        XtrG = Xtr; ytrG = ytr; gtrG = gtr;
    end
    inner = groupedStratifiedFolds(gtrG, ytrG, kInner);
    switch name
        case 'DT',  grid = GRID.DT_MinLeaf;
        case 'kNN', grid = GRID.KNN_K;
        case 'SVM', grid = GRID.SVM_C;
        case 'RF',  grid = GRID.RF_MinLeaf;
    end
    accs = zeros(size(grid));
    for gi = 1:numel(grid)
        correct = 0;
        for f = 1:kInner
            tr = inner ~= f; te = inner == f;
            yh = fitAndPredict(name, XtrG(tr, :), ytrG(tr), XtrG(te, :), grid(gi), nTrees, seed);
            correct = correct + nnz(yh == ytrG(te));
        end
        accs(gi) = correct / numel(ytrG);
    end
    [~, gi] = max(accs);
    [yhat, score] = fitAndPredict(name, Xtr, ytr, Xte, grid(gi), nTrees, seed);
end

function [yhat, score] = fitAndPredict(name, Xtr, ytr, Xte, hp, nTrees, seed)
    rng(seed);
    switch name
        case 'DT'
            mdl = fitctree(Xtr, ytr, 'MinLeafSize', hp);
            [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'kNN'
            mdl = fitcknn(Xtr, ytr, 'NumNeighbors', hp, 'Distance', 'euclidean', 'DistanceWeight', 'inverse');
            [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'SVM'
            % RBF SVM scales badly with n; cap the training set at 6000 stratified epochs
            SVM_MAX = 6000;
            if numel(ytr) > SVM_MAX
                keep = [];
                for c = [0 1]
                    idx = find(ytr == c); idx = idx(randperm(numel(idx)));
                    keep = [keep; idx(1:round(SVM_MAX * numel(idx) / numel(ytr)))]; %#ok<AGROW>
                end
                Xtr = Xtr(keep, :); ytr = ytr(keep);
            end
            mdl = fitcsvm(Xtr, ytr, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'BoxConstraint', hp, 'Standardize', false);
            [yhat, sc] = predict(mdl, Xte); score = sc(:, mdl.ClassNames == 1);
        case 'RF'
            [yhat, score] = rfFixed(Xtr, ytr, Xte, nTrees, hp, seed);
    end
    yhat = double(yhat(:)); score = double(score(:));
end

function [yhat, score] = rfFixed(Xtr, ytr, Xte, nTrees, minLeaf, seed)
    rng(seed);
    mdl = TreeBagger(nTrees, Xtr, ytr, 'Method', 'classification', 'MinLeafSize', minLeaf, 'OOBPrediction', 'off');
    [lab, sc] = predict(mdl, Xte);
    yhat = str2double(lab); score = sc(:, strcmp(mdl.ClassNames, '1'));
    yhat = double(yhat(:)); score = double(score(:));
end

% --------------------------------------------------------------------------
function S = subjectVote(g, y, yhat, score)
    % one row per subject: [true label, majority-vote label, mean score]
    [subj, ~, k] = unique(g);
    yt = accumarray(k, y, [], @(v) v(1));
    ms = accumarray(k, score, [], @mean);
    vote = accumarray(k, yhat, [], @mean);
    yv = double(vote > 0.5); tie = vote == 0.5; yv(tie) = double(ms(tie) > 0.5);
    S = [yt, yv, ms];
    S = S(1:numel(subj), :);
end

function m = metrics(P)
    % P = [y, yhat, score]; positive class = 1 (ADHD)
    y = P(:, 1); yh = P(:, 2); sc = P(:, 3);
    TP = nnz(y == 1 & yh == 1); TN = nnz(y == 0 & yh == 0); FP = nnz(y == 0 & yh == 1); FN = nnz(y == 1 & yh == 0);
    m.ACC = 100 * (TP + TN) / max(numel(y), 1);
    m.SEN = 100 * TP / max(TP + FN, 1); m.SPE = 100 * TN / max(TN + FP, 1);
    m.PPV = 100 * TP / max(TP + FP, 1);
    m.F1  = 2 * TP / max(2 * TP + FP + FN, 1) * 100;
    den = sqrt(double(TP + FP) * double(TP + FN) * double(TN + FP) * double(TN + FN));
    m.MCC = ifelse(den > 0, (TP * TN - FP * FN) / den, 0);
    try
        [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc;
    catch
        m.AUC = NaN;
    end
end

function writeRow(fid, cohort, name, ms)
    fld = {'ACC', 'SEN', 'SPE', 'PPV', 'F1', 'MCC', 'AUC'};
    fprintf(fid, '%s,%s', cohort, name);
    for i = 1:numel(fld)
        v = arrayfun(@(m) m.(fld{i}), ms);
        fprintf(fid, ',%.3f,%.3f', mean(v, 'omitnan'), std(v, 'omitnan'));
    end
    fprintf(fid, '\n');
end
