%% stage10c_fill_tables.m
%  Fills the gaps in Tables 3 and 4.
%  Part A (Table 3): 200-shuffle permutation p for ADHD-vs-all-other-patients, any-patient-vs-healthy,
%                    depression-vs-healthy (TDBRAIN adults, 26 log powers, RF uniform prior, 5-fold, subject level).
%  Part B (Table 4): Riemannian variants full / diag / offdiag / corr for IEEE and matched BALLADEER on the
%                    7 frontal, 9 central and 19 channel montages (native reference), 5 repeats, and a
%                    100-shuffle permutation p for the FULL variant on every montage and cohort.
%  Outputs: <ROOT>\Stage10_Riemannian\Stage10c_Report.txt, T_stage10c_table3.csv, T_stage10c_table4.csv

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
OUT = fullfile(ROOT, 'Stage10_Riemannian'); CSV5 = fullfile(ROOT, 'Stage5_Metadata', 'T_stage5_subjects.csv'); IN = fullfile(ROOT, 'Stage1b_Epochs');
SEED0 = 2026; K = 5; RF_TREES = 200; N_PERM_A = 200; N_PERM_B = 100; PERM_TREES = 50; R_REPEATS = 5; MIN_EP = 10;
MONT.frontal7 = {'Fp1','F7','F3','Fz','F4','F8','Fp2'}; MONT.central9 = {'Fz','Cz','Pz','C3','C4','F3','F4','P3','P4'};
MONT.all19 = {'Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2'};
BMAP = containers.Map({'T3','T4','T5','T6'}, {'T7','T8','P7','P8'});

fid = fopen(fullfile(OUT, 'Stage10c_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 10c - TABLE GAPS   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')), try, if isempty(gcp('nocreate')), parpool; end, catch, end, end

%% ---------------- Part A: Table 3 permutation p values
txt = fileread(CSV5); lines = strsplit(strtrim(txt), {'\r\n', '\n'}); hdr = strsplit(lines{1}, ',');
grp = {}; age = []; sex = []; X = zeros(0, 26); iAge = find(strcmp(hdr, 'age'), 1); iSex = find(strcmp(hdr, 'sex'), 1);
for i = 2:numel(lines)
    f = strsplit(strtrim(lines{i}), ','); if numel(f) < 28, continue; end
    grp{end+1, 1} = strtrim(f{2}); age(end+1, 1) = str2double(f{iAge}); sex(end+1, 1) = str2double(f{iSex}); X(end+1, :) = cellfun(@str2double, f(end-25:end)); %#ok<SAGROW>
end
isHC = strcmp(grp, 'HC'); isA = strcmp(grp, 'ADHD'); isMDD = strcmp(grp, 'MDD'); isOther = ismember(grp, {'MDD', 'OCD', 'INSOMNIA'});
fA = fopen(fullfile(OUT, 'T_stage10c_table3.csv'), 'w'); fprintf(fA, 'experiment,n_pos,n_neg,observed_BAL,perm_p\n');
exps = {'ADHD_vs_otherPatients', find(isA), find(isOther); 'anyPatient_vs_HC', find(~isHC), find(isHC); 'MDD_vs_HC', find(isMDD), find(isHC)};
L('\n=== A. Table 3 permutation tests (%d shuffles)', N_PERM_A);
for e = 1:size(exps, 1)
    pos = exps{e, 2}; neg = exps{e, 3}; obs = mean(arrayfun(@(r) cvBal(X, pos, neg, K, RF_TREES, SEED0 + r), 1:5));
    p = permSubj(X, pos, neg, K, PERM_TREES, SEED0, N_PERM_A, obs);
    L('  %-24s n=%3d/%3d  observed BAL %.1f  perm p = %.4f', exps{e, 1}, numel(pos), numel(neg), obs, p);
    fprintf(fA, '%s,%d,%d,%.2f,%.4f\n', exps{e, 1}, numel(pos), numel(neg), obs, p);
end
fclose(fA);

%% ---------------- Part B: Table 4 missing variants and p values
T = load(fullfile(IN, 'D1_IEEE_epochs.mat')); S1 = dropSparseS(T.S, MIN_EP);
T = load(fullfile(IN, 'D4_BALLADEER_epochs.mat')); S4 = dropSparseS(T.S, MIN_EP);
[subj, ia] = unique(S4.subject); S4 = subsetS(S4, ismember(S4.subject, matchSubjects(subj, double(S4.label(ia)), double(S4.age(ia)), double(S4.sex(ia)), 1, SEED0)));
cohorts = struct('name', {'IEEE', 'BALLADEER'}, 'S', {S1, S4});
fB = fopen(fullfile(OUT, 'T_stage10c_table4.csv'), 'w'); fprintf(fB, 'cohort,montage,nCh,variant,BAL,BAL_sd,AUC,AUC_sd,perm_p_full\n');
mn = fieldnames(MONT);
L('\n=== B. Table 4: all variants, native reference, %d repeats; permutation on full (%d shuffles)', R_REPEATS, N_PERM_B);
for c = 1:numel(cohorts)
    S = cohorts(c).S; y = double(S.label(:)); g = double(S.subject(:));
    for m = 1:numel(mn)
        chans = MONT.(mn{m}); if strcmp(cohorts(c).name, 'BALLADEER'), chans = cellfun(@(x) mapName(BMAP, x), chans, 'UniformOutput', false); end
        idx = cellfun(@(x) find(strcmpi(S.channels, x), 1), chans); F = riemannFeatures(double(S.X(:, :, idx)));
        pFull = NaN;
        for v = {'full', 'diag', 'offdiag', 'corr'}
            Xf = F.(v{1}); ms = [];
            for rep = 1:R_REPEATS
                rng(SEED0 + rep); folds = groupedFolds(g, y, K); P = zeros(numel(y), 3);
                for f = 1:K, tr = folds ~= f; te = folds == f; [Xtr, Xte] = prep(Xf, tr, te); [yh, sc] = rfFixed(Xtr, y(tr), Xte, 100, 5, SEED0 + rep); P(te, :) = [y(te), yh, sc]; end
                ms = [ms, metrics(subjectVote(g, P(:, 1), P(:, 2), P(:, 3)))]; %#ok<AGROW>
            end
            if strcmp(v{1}, 'full'), pFull = permTest(Xf, y, g, K, PERM_TREES, SEED0, N_PERM_B, mean([ms.BAL])); end
            L('  %-9s %-9s %-7s BAL %.1f +/- %.1f  AUC %.3f +/- %.3f%s', cohorts(c).name, mn{m}, v{1}, mean([ms.BAL]), std([ms.BAL]), mean([ms.AUC]), std([ms.AUC]), ifelse(strcmp(v{1}, 'full'), sprintf('  perm p = %.3f', pFull), ''));
            fprintf(fB, '%s,%s,%d,%s,%.2f,%.2f,%.3f,%.3f,%s\n', cohorts(c).name, mn{m}, numel(chans), v{1}, mean([ms.BAL]), std([ms.BAL]), mean([ms.AUC]), std([ms.AUC]), ifelse(strcmp(v{1}, 'full'), sprintf('%.3f', pFull), ''));
        end
    end
end
fclose(fB);
L('\nDONE. Send Stage10c_Report.txt.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
function n = mapName(M, x), if isKey(M, x), n = M(x); else, n = x; end, end
function bal = cvBal(X, pos, neg, K, nTrees, seed)
    idx = [pos(:); neg(:)]; y = [ones(numel(pos), 1); zeros(numel(neg), 1)]; rng(seed);
    folds = zeros(size(y)); for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
    P = zeros(numel(idx), 3);
    for f = 1:K
        tr = folds ~= f; te = folds == f; mu = mean(X(idx(tr), :)); sd = std(X(idx(tr), :)); sd(sd < 1e-12) = 1;
        mdl = TreeBagger(nTrees, (X(idx(tr), :) - mu) ./ sd, y(tr), 'Method', 'classification', 'Prior', 'Uniform');
        [lab, sc] = predict(mdl, (X(idx(te), :) - mu) ./ sd); P(te, :) = [y(te), str2double(lab), sc(:, strcmp(mdl.ClassNames, '1'))];
    end
    m = metrics(P); bal = m.BAL;
end
function p = permSubj(X, pos, neg, K, nTrees, seed, nPerm, obs)
    idx = [pos(:); neg(:)]; y0 = [ones(numel(pos), 1); zeros(numel(neg), 1)]; nullB = zeros(nPerm, 1);
    for q = 1:nPerm
        rng(seed + 9000 + q); y = y0(randperm(numel(y0))); folds = zeros(size(y)); for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
        P = zeros(numel(idx), 3);
        for f = 1:K, tr = folds ~= f; te = folds == f; mu = mean(X(idx(tr), :)); sd = std(X(idx(tr), :)); sd(sd < 1e-12) = 1;
            mdl = TreeBagger(nTrees, (X(idx(tr), :) - mu) ./ sd, y(tr), 'Method', 'classification', 'Prior', 'Uniform');
            [lab, sc] = predict(mdl, (X(idx(te), :) - mu) ./ sd); P(te, :) = [y(te), str2double(lab), sc(:, strcmp(mdl.ClassNames, '1'))]; end
        m = metrics(P); nullB(q) = m.BAL;
    end
    p = (1 + nnz(nullB >= obs)) / (nPerm + 1);
end
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
    isDiag = (iu == ju)'; F.full = full; F.diag = full(:, isDiag); F.offdiag = full(:, ~isDiag); F.corr = corrF(:, ~isDiag);
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
    try, [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc; catch, m.AUC = NaN; end
end
