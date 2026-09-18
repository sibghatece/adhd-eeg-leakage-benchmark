%% stage6_specificity_test.m
%  STAGE 6 - Diagnostic specificity of the TDBRAIN amplitude classifier.
%  Input: Stage5_Metadata\T_stage5_subjects.csv (one row per subject: group, age, sex,
%         global and per-channel log power 1-30 Hz; HC, ADHD, MDD, OCD, INSOMNIA).
%
%  A. ADHD vs HC (1:2 age/sex-matched), 26-channel log power, RF uniform prior,
%     5-fold subject CV x R repeats. In every fold the model also scores ALL MDD, OCD and
%     INSOMNIA subjects (never in training). Report: CV sensitivity/specificity/AUC, and the
%     fraction of each other-patient group the ADHD model labels "ADHD".
%  B. ADHD vs MDD (1:2 matched) and ADHD vs all other patients, same protocol - should be ~chance
%     if the amplitude effect is not ADHD-specific.
%  C. Any-patient vs HC, same protocol - should look like ADHD vs HC.
%  D. Label permutation for A and B (subject-level, N_PERM).
%
%  Outputs: <OUT>\Stage6_Report.txt, T_stage6_results.csv, Stage6_scores.png

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
CSV = fullfile(ROOT, 'Stage5_Metadata', 'T_stage5_subjects.csv'); OUT = fullfile(ROOT, 'Stage6_Specificity');
R_REPEATS = 20; K = 5; SEED0 = 2026; RF_TREES = 200; MATCH_RATIO = 2; N_PERM = 200;

if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage6_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 6 - SPECIFICITY TEST   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('RF %d trees, uniform prior, %d-fold subject CV x %d repeats, matching 1:%d, permutations %d', RF_TREES, K, R_REPEATS, MATCH_RATIO, N_PERM);

%% ---------------- read the subject table (no readtable)
txt = fileread(CSV); lines = strsplit(strtrim(txt), {'\r\n', '\n'}); hdr = strsplit(strtrim(lines{1}), ',');
chCols = find(endsWith(hdr, '_lp')); assert(numel(chCols) == 26, 'expected 26 channel columns');
chNames = strrep(hdr(chCols), '_lp', '');
ci = @(name) find(strcmp(hdr, name), 1); iAge = ci('age'); iSex = ci('sex');
grp = {}; age = []; sex = []; X = zeros(0, 26); nSkipped = 0;
for i = 2:numel(lines)
    f = strsplit(strtrim(lines{i}), ',');
    if numel(f) < 28 || isempty(strtrim(lines{i})), nSkipped = nSkipped + 1; continue; end
    % channels are always the LAST 26 fields; group is always field 2; age/sex counted from the start
    grp{end+1, 1} = strtrim(f{2}); age(end+1, 1) = str2double(f{iAge}); sex(end+1, 1) = str2double(f{iSex}); %#ok<SAGROW>
    X(end+1, :) = cellfun(@str2double, f(end-25:end)); %#ok<SAGROW>
end
n = numel(grp); L('Rows read: %d, skipped (malformed): %d', n, nSkipped);
assert(all(isfinite(X(:))), 'non-numeric channel values found');
L('Subjects: %s', strjoin(arrayfun(@(g) sprintf('%s=%d', g{1}, nnz(strcmp(grp, g{1}))), unique(grp)', 'UniformOutput', false), ', '));

isHC = strcmp(grp, 'HC'); isADHD = strcmp(grp, 'ADHD'); isMDD = strcmp(grp, 'MDD'); isOCD = strcmp(grp, 'OCD'); isINS = strcmp(grp, 'INSOMNIA');
isOtherPt = isMDD | isOCD | isINS;

fRes = fopen(fullfile(OUT, 'T_stage6_results.csv'), 'w');
fprintf(fRes, 'experiment,n_pos,n_neg,BAL,BAL_sd,SEN,SEN_sd,SPE,SPE_sd,AUC,AUC_sd,MCC,MCC_sd,perm_p\n');

%% ---------------- A. ADHD vs HC, matched; score other patients
[posA, negA] = matchGroups(find(isADHD), find(isHC), age, sex, MATCH_RATIO, SEED0);
L('\n=== A. ADHD (n=%d, age %.1f) vs matched HC (n=%d, age %.1f)', numel(posA), mean(age(posA)), numel(negA), mean(age(negA)));
[mA, othScore, othLabel] = runCV(X, posA, negA, find(isOtherPt), K, R_REPEATS, RF_TREES, SEED0);
pA = permTest(X, posA, negA, K, RF_TREES, SEED0, N_PERM, mean([mA.BAL]));
reportExp(L, fRes, 'ADHD_vs_HC_matched', numel(posA), numel(negA), mA, pA);
oth = find(isOtherPt);
L('  Other patients scored by the ADHD-vs-HC model (fraction labelled ADHD, mean ADHD-probability):');
for g = {'MDD', 'OCD', 'INSOMNIA'}
    m = strcmp(grp(oth), g{1});
    L('    %-9s n=%3d  labelled ADHD %.1f%%   mean P(ADHD) %.3f', g{1}, nnz(m), 100 * mean(othLabel(m)), mean(othScore(m)));
end
L('  For reference, in CV the model labels ADHD as ADHD %.1f%% (sensitivity) and HC as ADHD %.1f%% (1-specificity)', mean([mA.SEN]), 100 - mean([mA.SPE]));

%% ---------------- B. ADHD vs MDD (matched) and ADHD vs all other patients
[posB, negB] = matchGroups(find(isADHD), find(isMDD), age, sex, MATCH_RATIO, SEED0);
L('\n=== B1. ADHD (n=%d, age %.1f) vs matched MDD (n=%d, age %.1f)', numel(posB), mean(age(posB)), numel(negB), mean(age(negB)));
mB = runCV(X, posB, negB, [], K, R_REPEATS, RF_TREES, SEED0);
pB = permTest(X, posB, negB, K, RF_TREES, SEED0, N_PERM, mean([mB.BAL]));
reportExp(L, fRes, 'ADHD_vs_MDD_matched', numel(posB), numel(negB), mB, pB);
L('\n=== B2. ADHD (n=%d) vs all other patients (n=%d, unmatched)', nnz(isADHD), nnz(isOtherPt));
mB2 = runCV(X, find(isADHD), find(isOtherPt), [], K, R_REPEATS, RF_TREES, SEED0);
reportExp(L, fRes, 'ADHD_vs_otherPatients', nnz(isADHD), nnz(isOtherPt), mB2, NaN);

%% ---------------- C. any patient vs HC
L('\n=== C. Any patient (n=%d) vs HC (n=%d)', nnz(~isHC), nnz(isHC));
mC = runCV(X, find(~isHC), find(isHC), [], K, R_REPEATS, RF_TREES, SEED0);
reportExp(L, fRes, 'anyPatient_vs_HC', nnz(~isHC), nnz(isHC), mC, NaN);
L('\n=== C2. MDD (n=%d) vs HC (n=%d)  [same-protocol comparison with A]', nnz(isMDD), nnz(isHC));
mC2 = runCV(X, find(isMDD), find(isHC), [], K, R_REPEATS, RF_TREES, SEED0);
reportExp(L, fRes, 'MDD_vs_HC', nnz(isMDD), nnz(isHC), mC2, NaN);
fclose(fRes);

%% ---------------- figure: ADHD-model scores by group
fig = figure('Color', 'w', 'Position', [100 100 800 450]);
[~, ~, ~, cvScoreA, cvLabelA] = runCV(X, posA, negA, [], K, 1, RF_TREES, SEED0);   % one repeat for the plot
data = [cvScoreA(cvLabelA == 0); cvScoreA(cvLabelA == 1); othScore(strcmp(grp(oth), 'MDD')); othScore(strcmp(grp(oth), 'OCD')); othScore(strcmp(grp(oth), 'INSOMNIA'))];
gid = [ones(nnz(cvLabelA == 0), 1); 2 * ones(nnz(cvLabelA == 1), 1); 3 * ones(nnz(strcmp(grp(oth), 'MDD')), 1); 4 * ones(nnz(strcmp(grp(oth), 'OCD')), 1); 5 * ones(nnz(strcmp(grp(oth), 'INSOMNIA')), 1)];
boxplot(data, gid, 'Labels', {'HC (CV)', 'ADHD (CV)', 'MDD', 'OCD', 'Insomnia'}); hold on; yline(0.5, 'k--');
ylabel('P(ADHD) from ADHD-vs-HC model'); title('TDBRAIN: ADHD-vs-HC amplitude model applied to other patient groups'); grid on
saveas(fig, fullfile(OUT, 'Stage6_scores.png'));
L('\nDONE. Send Stage6_Report.txt and Stage6_scores.png.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end

function [pos, neg] = matchGroups(pos, cand, age, sex, ratio, seed)
    rng(seed); pos = pos(randperm(numel(pos))); used = false(size(cand)); neg = [];
    for i = pos'
        for r = 1:ratio
            c = find(~used & sex(cand) == sex(i)); if isempty(c), c = find(~used); end; if isempty(c), break; end
            [~, j] = min(abs(age(cand(c)) - age(i))); used(c(j)) = true; neg(end+1) = cand(c(j)); %#ok<AGROW>
        end
    end
    neg = neg(:); pos = pos(:);
end

function [ms, othScore, othLabel, cvScore, cvLabel] = runCV(X, pos, neg, others, K, R, nTrees, seed)
    % subject-level K-fold CV of RF (uniform prior); others are scored by every fold's model
    idx = [pos; neg]; y = [ones(numel(pos), 1); zeros(numel(neg), 1)];
    ms = struct('SEN', {}, 'SPE', {}, 'BAL', {}, 'MCC', {}, 'AUC', {});   % same field order as metrics()
    othAcc = zeros(numel(others), 1); othCnt = 0;
    cvScore = zeros(numel(idx), 1); cvLabel = y;
    for r = 1:R
        rng(seed + r); folds = zeros(size(y));
        for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
        P = zeros(numel(idx), 3);
        for f = 1:K
            tr = folds ~= f; te = folds == f;
            mu = mean(X(idx(tr), :)); sd = std(X(idx(tr), :)); sd(sd < 1e-12) = 1;
            rng(seed + 100 * r + f);
            mdl = TreeBagger(nTrees, (X(idx(tr), :) - mu) ./ sd, y(tr), 'Method', 'classification', 'Prior', 'Uniform', 'OOBPrediction', 'off');
            [lab, sc] = predict(mdl, (X(idx(te), :) - mu) ./ sd); s1 = sc(:, strcmp(mdl.ClassNames, '1'));
            P(te, :) = [y(te), str2double(lab), s1];
            if ~isempty(others)
                [~, sco] = predict(mdl, (X(others, :) - mu) ./ sd); othAcc = othAcc + sco(:, strcmp(mdl.ClassNames, '1')); othCnt = othCnt + 1;
            end
        end
        ms(r) = metrics(P); cvScore = P(:, 3);
    end
    if isempty(others), othScore = []; othLabel = []; else, othScore = othAcc / othCnt; othLabel = double(othScore > 0.5); end
end

function p = permTest(X, pos, neg, K, nTrees, seed, nPerm, obsBal)
    idx = [pos; neg]; y0 = [ones(numel(pos), 1); zeros(numel(neg), 1)]; nullBal = zeros(nPerm, 1);
    for q = 1:nPerm
        rng(seed + 9000 + q); y = y0(randperm(numel(y0)));
        folds = zeros(size(y)); for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
        P = zeros(numel(idx), 3);
        for f = 1:K
            tr = folds ~= f; te = folds == f; mu = mean(X(idx(tr), :)); sd = std(X(idx(tr), :)); sd(sd < 1e-12) = 1;
            mdl = TreeBagger(round(nTrees / 2), (X(idx(tr), :) - mu) ./ sd, y(tr), 'Method', 'classification', 'Prior', 'Uniform', 'OOBPrediction', 'off');
            [lab, sc] = predict(mdl, (X(idx(te), :) - mu) ./ sd); P(te, :) = [y(te), str2double(lab), sc(:, strcmp(mdl.ClassNames, '1'))];
        end
        m = metrics(P); nullBal(q) = m.BAL;
    end
    p = (1 + nnz(nullBal >= obsBal)) / (nPerm + 1);
end

function reportExp(L, fid, name, nPos, nNeg, ms, permP)
    fld = {'BAL', 'SEN', 'SPE', 'AUC', 'MCC'}; v = cellfun(@(f) mean([ms.(f)]), fld); s = cellfun(@(f) std([ms.(f)]), fld);
    L('  %-24s n=%3d/%3d  BAL %.1f +/- %.1f  SEN %.1f  SPE %.1f  AUC %.3f +/- %.3f  MCC %.3f  perm p=%s', ...
        name, nPos, nNeg, v(1), s(1), v(2), v(3), v(4), s(4), v(5), num2str(permP, '%.4f'));
    fprintf(fid, '%s,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%s\n', name, nPos, nNeg, v(1), s(1), v(2), s(2), v(3), s(3), v(4), s(4), v(5), s(5), num2str(permP, '%.4f'));
end

function m = metrics(P)
    y = P(:, 1); yh = P(:, 2); sc = P(:, 3);
    TP = nnz(y == 1 & yh == 1); TN = nnz(y == 0 & yh == 0); FP = nnz(y == 0 & yh == 1); FN = nnz(y == 1 & yh == 0);
    m.SEN = 100 * TP / max(TP + FN, 1); m.SPE = 100 * TN / max(TN + FP, 1); m.BAL = (m.SEN + m.SPE) / 2;
    den = sqrt(double(TP + FP) * double(TP + FN) * double(TN + FP) * double(TN + FN)); if den > 0, m.MCC = (TP * TN - FP * FN) / den; else, m.MCC = 0; end
    try, [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc; catch, m.AUC = NaN; end
end
