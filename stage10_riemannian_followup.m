%% stage10_riemannian_followup.m
%  STAGE 10 - Does the Riemannian (tangent-space covariance) result on IEEE hold up?
%
%  Common montage: the 19 IEEE 10-20 channels, present in BALLADEER as
%     IEEE  : Fz Cz Pz C3 T3 C4 T4 Fp1 Fp2 F3 F4 F7 F8 P3 P4 T5 T6 O1 O2
%     BALLA : Fz Cz Pz C3 T7 C4 T8 Fp1 Fp2 F3 F4 F7 F8 P3 P4 P7 P8 O1 O2
%  Both re-referenced to the common average of these 19 channels (CAR-19) so that the
%  covariance geometry is comparable; the IEEE native reference is also evaluated.
%
%  Feature variants (per epoch, log-Euclidean tangent vectors of the 19x19 covariance):
%     full     upper triangle incl. diagonal (as in Stage 8f)                  190 features
%     diag     diagonal only (log-variance-like)                                19
%     offdiag  off-diagonal only (inter-channel structure, amplitude-scaled)   171
%     corr     covariance normalised to correlation before logm (amplitude-free) 171
%
%  A. IEEE: P3 subject-grouped 5-fold x 3 repeats, RF, each variant; permutation test on 'full' and 'corr'
%  B. BALLADEER 1:1 matched (43/43), same 19 channels, same variants; permutation test on 'full'
%  C. Transfer IEEE -> BALLADEER and BALLADEER -> IEEE (RF, uniform prior), each variant,
%     with (i) training-set z-scoring and (ii) label-free per-dataset z-scoring (removes gain offset)
%  Outputs: <OUT>\Stage10_Report.txt, T_stage10.csv

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage1b_Epochs'); OUT = fullfile(ROOT, 'Stage10_Riemannian');
R_REPEATS = 3; K = 5; SEED0 = 2026; RF_TREES = 100; MIN_EP = 10; N_PERM_IEEE = 200; N_PERM_BALLA = 100; PERM_TREES = 50;
IEEE19  = {'Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2'};
BALLA19 = {'Fz','Cz','Pz','C3','T7','C4','T8','Fp1','Fp2','F3','F4','F7','F8','P3','P4','P7','P8','O1','O2'};
VARIANTS = {'full', 'diag', 'offdiag', 'corr'};

if ~isfolder(OUT), mkdir(OUT); end
fid = fopen(fullfile(OUT, 'Stage10_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 10 - RIEMANNIAN FOLLOW-UP   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fT = fopen(fullfile(OUT, 'T_stage10.csv'), 'w');
fprintf(fT, 'experiment,cohort,variant,reference,level,BAL,BAL_sd,SEN,SPE,AUC,AUC_sd,MCC,perm_p\n');
if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')), try, if isempty(gcp('nocreate')), parpool; end, catch, end, end

%% ---------------- load and subset stores
T = load(fullfile(IN, 'D1_IEEE_epochs.mat')); S1 = dropSparseS(T.S, MIN_EP);
T = load(fullfile(IN, 'D4_BALLADEER_epochs.mat')); S4 = dropSparseS(T.S, MIN_EP);
[subj, ia] = unique(S4.subject); keep4 = matchSubjects(subj, double(S4.label(ia)), double(S4.age(ia)), double(S4.sex(ia)), 1, SEED0);
S4 = subsetS(S4, ismember(S4.subject, keep4));
i1 = cellfun(@(c) find(strcmpi(S1.channels, c), 1), IEEE19); i4 = cellfun(@(c) find(strcmpi(S4.channels, c), 1), BALLA19);
X1 = double(S1.X(:, :, i1)); X4 = double(S4.X(:, :, i4));
L('IEEE: %d subjects (ADHD %d), %d epochs | BALLADEER matched: %d subjects (ADHD %d), %d epochs | 19 common channels', ...
    numel(unique(S1.subject)), numel(unique(S1.subject(S1.label == 1))), size(X1, 1), numel(unique(S4.subject)), numel(unique(S4.subject(S4.label == 1))), size(X4, 1));

%% ---------------- features
refs = {'native', 'CAR19'};
F1 = struct(); F4 = struct();
for r = 1:2
    Xa = X1; Xb = X4;
    if strcmp(refs{r}, 'CAR19'), Xa = Xa - mean(Xa, 3, 'omitnan'); Xb = Xb - mean(Xb, 3, 'omitnan'); end
    F1.(refs{r}) = riemannFeatures(Xa); F4.(refs{r}) = riemannFeatures(Xb);
    L('features computed (%s)', refs{r});
end
y1 = double(S1.label(:)); g1 = double(S1.subject(:)); y4 = double(S4.label(:)); g4 = double(S4.subject(:));

%% ---------------- A/B. within-cohort CV per variant and reference
for coh = {'IEEE', 'BALLADEER19'}
    for r = 1:2
        for v = 1:numel(VARIANTS)
            if strcmp(coh{1}, 'IEEE'), X = F1.(refs{r}).(VARIANTS{v}); y = y1; g = g1; else, X = F4.(refs{r}).(VARIANTS{v}); y = y4; g = g4; end
            ms = []; mE = [];
            for rep = 1:R_REPEATS
                rng(SEED0 + rep); folds = groupedFolds(g, y, K); P = zeros(numel(y), 3);
                for f = 1:K
                    tr = folds ~= f; te = folds == f; [Xtr, Xte] = prep(X, tr, te);
                    [yh, sc] = rfFixed(Xtr, y(tr), Xte, RF_TREES, 5, SEED0 + rep); P(te, :) = [y(te), yh, sc];
                end
                ms = [ms, metrics(subjectVote(g, P(:, 1), P(:, 2), P(:, 3)))]; mE = [mE, metrics(P)]; %#ok<AGROW>
            end
            pp = NaN;
            if strcmp(refs{r}, 'CAR19') && (strcmp(VARIANTS{v}, 'full') || (strcmp(VARIANTS{v}, 'corr') && strcmp(coh{1}, 'IEEE')))
                nP = ifelse(strcmp(coh{1}, 'IEEE'), N_PERM_IEEE, N_PERM_BALLA);
                pp = permTest(X, y, g, K, PERM_TREES, SEED0, nP, mean([ms.BAL]));
            end
            L('  %-12s %-7s %-8s subject BAL %.1f +/- %.1f  SEN %.1f SPE %.1f  AUC %.3f +/- %.3f  MCC %.3f | epoch BAL %.1f AUC %.3f%s', coh{1}, VARIANTS{v}, refs{r}, ...
                mean([ms.BAL]), std([ms.BAL]), mean([ms.SEN]), mean([ms.SPE]), mean([ms.AUC]), std([ms.AUC]), mean([ms.MCC]), mean([mE.BAL]), mean([mE.AUC]), ifelse(isnan(pp), '', sprintf('  perm p=%.4f', pp)));
            fprintf(fT, 'withinCV,%s,%s,%s,subject,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%s\n', coh{1}, VARIANTS{v}, refs{r}, mean([ms.BAL]), std([ms.BAL]), mean([ms.SEN]), mean([ms.SPE]), mean([ms.AUC]), std([ms.AUC]), mean([ms.MCC]), num2str(pp));
        end
    end
end

%% ---------------- C. transfer on CAR-19
L('\n=== C. Transfer, CAR-19 features, RF uniform prior');
for v = 1:numel(VARIANTS)
    for dirn = 1:2
        if dirn == 1, Xa = F1.CAR19.(VARIANTS{v}); ya = y1; Xb = F4.CAR19.(VARIANTS{v}); yb = y4; gb = g4; nm = 'IEEE->BALLADEER'; else, Xa = F4.CAR19.(VARIANTS{v}); ya = y4; Xb = F1.CAR19.(VARIANTS{v}); yb = y1; gb = g1; nm = 'BALLADEER->IEEE'; end
        for sc = {'trainStats', 'perDataset'}
            [Xtr, Xte] = prep(Xa, true(size(ya)), true(size(ya)));                 % training-set imputation/scaling
            if strcmp(sc{1}, 'trainStats')
                med = median(Xa, 1, 'omitnan'); med(isnan(med)) = 0; Xt = fillNaN(Xb, med); Xt = (Xt - mean(fillNaN(Xa, med), 1)) ./ max(std(fillNaN(Xa, med), 0, 1), 1e-12);
            else
                med = median(Xb, 1, 'omitnan'); med(isnan(med)) = 0; Xt = fillNaN(Xb, med); Xt = (Xt - mean(Xt, 1)) ./ max(std(Xt, 0, 1), 1e-12);   % label-free, test-set own statistics
            end
            [yh, s] = rfFixed(Xtr, ya, Xt, RF_TREES, 5, SEED0);
            mE = metrics([yb, yh, s]); mS = metrics(subjectVote(gb, yb, yh, s));
            L('  %-16s %-7s %-10s epoch BAL %.1f AUC %.3f | subject BAL %.1f AUC %.3f MCC %.3f', nm, VARIANTS{v}, sc{1}, mE.BAL, mE.AUC, mS.BAL, mS.AUC, mS.MCC);
            fprintf(fT, 'transfer_%s,%s,%s,CAR19,subject,%.2f,,%.2f,%.2f,%.3f,,%.3f,\n', sc{1}, nm, VARIANTS{v}, mS.BAL, mS.SEN, mS.SPE, mS.AUC, mS.MCC);
        end
    end
end
fclose(fT);
L('\nDONE. Send Stage10_Report.txt and T_stage10.csv.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
function F = riemannFeatures(X)
    % X: epochs x samples x 19 (NaN channels allowed) -> struct of variants
    nEp = size(X, 1); nCh = size(X, 3); [iu, ju] = find(triu(true(nCh))); nF = numel(iu); w = ones(nF, 1); w(iu ~= ju) = sqrt(2);
    full = nan(nEp, nF); corrF = nan(nEp, nF);
    parfor e = 1:nEp
        Z = squeeze(X(e, :, :)); good = ~any(isnan(Z), 1); v = nan(nF, 1); vc = nan(nF, 1);
        if nnz(good) >= 2
            Zg = Z(:, good); ng = nnz(good); C = cov(Zg); C = C + 1e-3 * trace(C) / ng * eye(ng);
            Lg = logmSym(C); fullM = nan(nCh); fullM(good, good) = Lg; v = fullM(sub2ind([nCh nCh], iu, ju)) .* w;
            d = sqrt(diag(C)); R = C ./ (d * d'); R = R + 1e-3 * eye(ng); Lr = logmSym(R); fullR = nan(nCh); fullR(good, good) = Lr; vc = fullR(sub2ind([nCh nCh], iu, ju)) .* w;
        end
        full(e, :) = v'; corrF(e, :) = vc';
    end
    isDiag = (iu == ju)';
    F.full = full; F.diag = full(:, isDiag); F.offdiag = full(:, ~isDiag); F.corr = corrF(:, ~isDiag);
end
function Lg = logmSym(C), [V, D] = eig((C + C') / 2); Lg = V * diag(log(max(diag(D), eps))) * V'; end
function S = subsetS(S, m), S.X = S.X(m, :, :); for fn = {'subject', 'subjectID', 'label', 'age', 'sex'}, S.(fn{1}) = S.(fn{1})(m, :); end, end
function S = dropSparseS(S, minEp)
    [subj, ~, k] = unique(S.subject); cnt = accumarray(k, 1); bad = subj(cnt < minEp); if ~isempty(bad), S = subsetS(S, ~ismember(S.subject, bad)); end
end
function keep = matchSubjects(subj, lab, age, sex, ratio, seed)
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
function [Xtr, Xte] = prep(X, tr, te)
    Xtr = X(tr, :); Xte = X(te, :); med = median(Xtr, 1, 'omitnan'); med(isnan(med)) = 0; Xtr = fillNaN(Xtr, med); Xte = fillNaN(Xte, med);
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd < 1e-12) = 1; Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
end
function X = fillNaN(X, med), [ii, jj] = find(isnan(X)); if isempty(ii), return; end; X(sub2ind(size(X), ii, jj)) = med(jj); end
function folds = groupedFolds(g, y, K)
    [subj, ia] = unique(g); lab = y(ia); folds = zeros(size(g));
    for c = [0 1], s = subj(lab == c); s = s(randperm(numel(s))); fx = mod((1:numel(s)) - 1, K) + 1; for i = 1:numel(s), folds(g == s(i)) = fx(i); end, end
end
function p = permTest(X, y, g, K, nTrees, seed, nPerm, obs)
    [subj, ia] = unique(g); labS = y(ia); nullB = zeros(nPerm, 1);
    for q = 1:nPerm
        rng(seed + 7000 + q); perm = labS(randperm(numel(labS))); yp = zeros(size(y)); for i = 1:numel(subj), yp(g == subj(i)) = perm(i); end
        folds = groupedFolds(g, yp, K); P = zeros(numel(y), 3);
        for f = 1:K, tr = folds ~= f; te = folds == f; [Xtr, Xte] = prep(X, tr, te); [yh, sc] = rfFixed(Xtr, yp(tr), Xte, nTrees, 5, seed + q); P(te, :) = [yp(te), yh, sc]; end
        m = metrics(subjectVote(g, P(:, 1), P(:, 2), P(:, 3))); nullB(q) = m.BAL;
    end
    p = (1 + nnz(nullB >= obs)) / (nPerm + 1);
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
    m.SEN = 100 * TP / max(TP + FN, 1); m.SPE = 100 * TN / max(TN + FP, 1); m.BAL = (m.SEN + m.SPE) / 2;
    den = sqrt(double(TP + FP) * double(TP + FN) * double(TN + FP) * double(TN + FN)); if den > 0, m.MCC = (TP * TN - FP * FN) / den; else, m.MCC = 0; end
    try, [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc; catch, m.AUC = NaN; end
end
