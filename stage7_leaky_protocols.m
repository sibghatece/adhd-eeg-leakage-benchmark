%% stage7_leaky_protocols.m
%  STAGE 7 - How much of the published accuracy is evaluation leakage?
%  Same features (Stage 2b), same classifier (RF, 100 trees, uniform prior), three protocols:
%     P1  epoch-level random 5-fold, feature selection (t-rank top K) on ALL data     <- typical published setup
%     P2  epoch-level random 5-fold, feature selection inside each training fold      <- leaks subject identity only
%     P3  subject-grouped stratified 5-fold, selection inside each training fold      <- leakage-free (our protocol)
%  Reported at epoch level (what the literature reports) and, for P3, also at subject level.
%  Cohorts: D1 IEEE 19ch, D2m TDBRAIN 26ch matched adults, D3 Mendeley 2ch. Feature sets: PSR, BandPower, LogVar.
%  Outputs: <OUT>\Stage7_Report.txt, T_stage7_leakage.csv

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage2b_Features'); OUT = fullfile(ROOT, 'Stage7_Leakage');
R_REPEATS = 3; K = 5; SEED0 = 2026; RF_TREES = 100; TOPK = 100; MATCH_RATIO = 2; MIN_EP = 10;

if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage7_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 7 - LEAKY vs LEAKAGE-FREE PROTOCOLS   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('RF %d trees uniform prior | top-%d t-ranked features (when > %d available) | %d repeats', RF_TREES, TOPK, TOPK, R_REPEATS);

F1 = loadF(fullfile(IN, 'D1_IEEE_features.mat')); F2 = loadF(fullfile(IN, 'D2_TDBRAIN_features.mat')); F3 = loadF(fullfile(IN, 'D3_MENDELEY_features.mat'));
F1 = dropSparse(F1, MIN_EP); F2 = dropSparse(F2, MIN_EP); F3 = dropSparse(F3, MIN_EP);
[F2, msg] = matchTdbrain(F2, MATCH_RATIO, SEED0); L('%s', msg);
cohorts = struct('name', {'D1_IEEE_19ch', 'D2m_TDBRAIN_26ch', 'D3_MENDELEY_2ch'}, 'F', {F1, F2, F3});
featSets = {'PSR', 'BandPower', 'LogVar'};
protocols = {'P1_epochCV_selAll', 'P2_epochCV_selFold', 'P3_subjectCV_selFold'};

fT = fopen(fullfile(OUT, 'T_stage7_leakage.csv'), 'w');
fprintf(fT, 'cohort,featureset,protocol,level,ACC,ACC_sd,BAL,BAL_sd,AUC,AUC_sd,MCC,MCC_sd\n');
for c = 1:numel(cohorts)
    F = cohorts(c).F; y = F.label; g = F.subject;
    for s = 1:numel(featSets)
        switch featSets{s}, case 'PSR', X = F.psr; case 'BandPower', X = F.bp; case 'LogVar', X = F.logvar; end
        med = median(X, 1, 'omitnan'); med(isnan(med)) = 0; X = fillNaN(X, med);
        nF = size(X, 2); useSel = nF > TOPK;
        L('\n=== %s | %s (%d features)', cohorts(c).name, featSets{s}, nF);
        for p = 1:numel(protocols)
            mE = []; mS = [];
            for r = 1:R_REPEATS
                rng(SEED0 + r);
                if p == 3, folds = groupedStratifiedFolds(g, y, K); else, folds = randomFolds(y, K); end
                if p == 1 && useSel, selAll = subjectTRankAll(X, y, g, p); end
                P = zeros(numel(y), 3);
                for f = 1:K
                    tr = folds ~= f; te = folds == f;
                    if useSel
                        if p == 1, sel = selAll(1:TOPK); else, sel = subjectTRank(X(tr, :), y(tr), g(tr)); sel = sel(1:TOPK); end
                    else, sel = 1:nF; end
                    mu = mean(X(tr, sel)); sd = std(X(tr, sel)); sd(sd < 1e-12) = 1;
                    Xtr = (X(tr, sel) - mu) ./ sd; Xte = (X(te, sel) - mu) ./ sd;
                    rng(SEED0 + 100 * r + f);
                    mdl = TreeBagger(RF_TREES, Xtr, y(tr), 'Method', 'classification', 'Prior', 'Uniform', 'OOBPrediction', 'off');
                    [lab, sc] = predict(mdl, Xte); P(te, :) = [y(te), str2double(lab), sc(:, strcmp(mdl.ClassNames, '1'))];
                end
                mE = [mE, metrics(P)]; %#ok<AGROW>
                if p == 3, mS = [mS, metrics(subjectVote(g, P(:, 1), P(:, 2), P(:, 3)))]; end %#ok<AGROW>
            end
            L('  %-22s epoch ACC %.1f +/- %.1f  BAL %.1f  AUC %.3f  MCC %.3f', protocols{p}, mean([mE.ACC]), std([mE.ACC]), mean([mE.BAL]), mean([mE.AUC]), mean([mE.MCC]));
            writeRow(fT, cohorts(c).name, featSets{s}, protocols{p}, 'epoch', mE);
            if p == 3
                L('  %-22s subject ACC %.1f +/- %.1f  BAL %.1f  AUC %.3f  MCC %.3f', '', mean([mS.ACC]), std([mS.ACC]), mean([mS.BAL]), mean([mS.AUC]), mean([mS.MCC]));
                writeRow(fT, cohorts(c).name, featSets{s}, protocols{p}, 'subject', mS);
            end
        end
    end
end
fclose(fT);
L('\nDONE. Send Stage7_Report.txt and T_stage7_leakage.csv.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function F = loadF(p), T = load(p); F = T.F; F.label = double(F.label(:)); F.subject = double(F.subject(:)); F.age = double(F.age(:)); F.sex = double(F.sex(:)); end
function X = fillNaN(X, med), [ii, jj] = find(isnan(X)); if isempty(ii), return; end; X(sub2ind(size(X), ii, jj)) = med(jj); end
function F = subsetF(F, m), for fn = {'psr', 'bp', 'logvar', 'subject', 'subjectID', 'label', 'age', 'sex'}, F.(fn{1}) = F.(fn{1})(m, :); end, end
function F = dropSparse(F, minEp)
    [subj, ~, k] = unique(F.subject); cnt = accumarray(k, 1); bad = subj(cnt < minEp); if ~isempty(bad), F = subsetF(F, ~ismember(F.subject, bad)); end
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
    msg = sprintf('TDBRAIN matched: ADHD %d vs HC %d (mean age %.1f / %.1f)', numel(idxA), numel(chosen), mean(age(idxA)), mean(age(chosen)));
end
function folds = randomFolds(y, K)
    folds = zeros(size(y)); for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
end
function folds = groupedStratifiedFolds(g, y, K)
    [subj, ia] = unique(g); lab = y(ia); folds = zeros(size(g));
    for c = [0 1], s = subj(lab == c); s = s(randperm(numel(s))); fid = mod((1:numel(s)) - 1, K) + 1; for i = 1:numel(s), folds(g == s(i)) = fid(i); end, end
end
function order = subjectTRank(X, y, g)
    [~, ~, k] = unique(g); lab = accumarray(k, y, [], @(v) v(1)); M = zeros(max(k), size(X, 2));
    for j = 1:size(X, 2), M(:, j) = accumarray(k, X(:, j), [], @mean); end
    [~, ~, ~, st] = ttest2(M(lab == 1, :), M(lab == 0, :)); t = abs(st.tstat); t(isnan(t)) = 0; [~, order] = sort(t, 'descend');
end
function order = subjectTRankAll(X, y, g, ~), order = subjectTRank(X, y, g); end
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
function writeRow(fid, cohort, fset, prot, level, ms)
    fld = {'ACC', 'BAL', 'AUC', 'MCC'}; fprintf(fid, '%s,%s,%s,%s', cohort, fset, prot, level);
    for i = 1:numel(fld), v = [ms.(fld{i})]; fprintf(fid, ',%.3f,%.3f', mean(v), std(v)); end; fprintf(fid, '\n');
end
