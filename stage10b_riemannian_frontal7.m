%% stage10b_riemannian_frontal7.m
%  STAGE 10b - Fair comparison with the published nested-CV benchmark on IEEE (7 frontal channels).
%  Riemannian tangent-space features (log-Euclidean) on IEEE with three montages:
%     frontal7  Fp1 F7 F3 Fz F4 F8 Fp2      (28 features)     <- the published benchmark's montage
%     central9  Fz Cz Pz C3 C4 F3 F4 P3 P4  (45 features)     <- mid-size control
%     all19     (190 features)                                  <- Stage 10 result
%  Variants full / corr; native reference (as distributed) and CAR over the montage's channels.
%  P3 subject-grouped 5-fold x 5 repeats, RF uniform prior; permutation test (200) on frontal7/full/native.
%  Also runs the same montages on BALLADEER matched for the replication column.
%  Outputs: <OUT>\Stage10b_Report.txt, T_stage10b.csv

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage1b_Epochs'); OUT = fullfile(ROOT, 'Stage10_Riemannian');
R_REPEATS = 5; K = 5; SEED0 = 2026; RF_TREES = 100; MIN_EP = 10; N_PERM = 200; PERM_TREES = 50;
MONT.frontal7 = {'Fp1','F7','F3','Fz','F4','F8','Fp2'};
MONT.central9 = {'Fz','Cz','Pz','C3','C4','F3','F4','P3','P4'};
MONT.all19    = {'Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2'};
BALLA_MAP = containers.Map({'T3','T4','T5','T6'}, {'T7','T8','P7','P8'});   % IEEE name -> BALLADEER name

fid = fopen(fullfile(OUT, 'Stage10b_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 10b - RIEMANNIAN, MONTAGE COMPARISON   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('repeats=%d outer=%d RF uniform prior | permutations=%d on IEEE frontal7 full native', R_REPEATS, K, N_PERM);
fT = fopen(fullfile(OUT, 'T_stage10b.csv'), 'w'); fprintf(fT, 'cohort,montage,nCh,variant,reference,BAL,BAL_sd,SEN,SPE,AUC,AUC_sd,MCC,perm_p\n');
if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')), try, if isempty(gcp('nocreate')), parpool; end, catch, end, end

T = load(fullfile(IN, 'D1_IEEE_epochs.mat')); S1 = dropSparseS(T.S, MIN_EP);
T = load(fullfile(IN, 'D4_BALLADEER_epochs.mat')); S4 = dropSparseS(T.S, MIN_EP);
[subj, ia] = unique(S4.subject); S4 = subsetS(S4, ismember(S4.subject, matchSubjects(subj, double(S4.label(ia)), double(S4.age(ia)), double(S4.sex(ia)), 1, SEED0)));
cohorts = struct('name', {'IEEE', 'BALLADEER'}, 'S', {S1, S4});

mn = fieldnames(MONT);
for c = 1:numel(cohorts)
    S = cohorts(c).S; y = double(S.label(:)); g = double(S.subject(:));
    for m = 1:numel(mn)
        chans = MONT.(mn{m});
        if strcmp(cohorts(c).name, 'BALLADEER'), chans = cellfun(@(x) mapName(BALLA_MAP, x), chans, 'UniformOutput', false); end
        idx = cellfun(@(x) find(strcmpi(S.channels, x), 1), chans); X = double(S.X(:, :, idx));
        for ref = {'native', 'CAR'}
            Xr = X; if strcmp(ref{1}, 'CAR'), Xr = Xr - mean(Xr, 3, 'omitnan'); end
            F = riemannFeatures(Xr);
            for v = {'full', 'corr'}
                Xf = F.(v{1}); ms = [];
                for rep = 1:R_REPEATS
                    rng(SEED0 + rep); folds = groupedFolds(g, y, K); P = zeros(numel(y), 3);
                    for f = 1:K, tr = folds ~= f; te = folds == f; [Xtr, Xte] = prep(Xf, tr, te); [yh, sc] = rfFixed(Xtr, y(tr), Xte, RF_TREES, 5, SEED0 + rep); P(te, :) = [y(te), yh, sc]; end
                    ms = [ms, metrics(subjectVote(g, P(:, 1), P(:, 2), P(:, 3)))]; %#ok<AGROW>
                end
                pp = NaN;
                if strcmp(cohorts(c).name, 'IEEE') && strcmp(mn{m}, 'frontal7') && strcmp(v{1}, 'full') && strcmp(ref{1}, 'native')
                    pp = permTest(Xf, y, g, K, PERM_TREES, SEED0, N_PERM, mean([ms.BAL]));
                end
                L('  %-9s %-9s (%2d ch, %3d feat) %-4s %-6s subject BAL %.1f +/- %.1f  SEN %.1f SPE %.1f  AUC %.3f +/- %.3f  MCC %.3f%s', cohorts(c).name, mn{m}, numel(chans), size(Xf, 2), v{1}, ref{1}, ...
                    mean([ms.BAL]), std([ms.BAL]), mean([ms.SEN]), mean([ms.SPE]), mean([ms.AUC]), std([ms.AUC]), mean([ms.MCC]), ifelse(isnan(pp), '', sprintf('  perm p=%.4f', pp)));
                fprintf(fT, '%s,%s,%d,%s,%s,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%s\n', cohorts(c).name, mn{m}, numel(chans), v{1}, ref{1}, mean([ms.BAL]), std([ms.BAL]), mean([ms.SEN]), mean([ms.SPE]), mean([ms.AUC]), std([ms.AUC]), mean([ms.MCC]), num2str(pp));
            end
        end
    end
end
fclose(fT);
L('\nReference: published nested-CV benchmark on IEEE, 7 frontal channels: balanced accuracy 73.5%% (95%% CI 69.8-77.0), AUC 0.796, MCC 0.483.');
L('DONE. Send Stage10b_Report.txt.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
function n = mapName(M, x), if isKey(M, x), n = M(x); else, n = x; end, end
function F = riemannFeatures(X)
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
    isDiag = (iu == ju)'; F.full = full; F.corr = corrF(:, ~isDiag);
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
