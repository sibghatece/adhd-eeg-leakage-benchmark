%% stage8b_entropy.m
%  STAGE 8b - Entropy family (5 per channel), computed on the z-normalised epoch:
%     sample entropy (m=2, r=0.2), permutation entropy (m=3 and m=4, tau=1, normalised),
%     spectral entropy (0.5-45 Hz, normalised), fuzzy entropy (m=2, r=0.2, n=2)
%  Writes Stage8_Features\entropy\D*_entropy.mat

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
FAMILY = 'entropy'; PER_CH_NAMES = {'SampEn', 'PermEn3', 'PermEn4', 'SpecEn', 'FuzzyEn'};
s8_run(ROOT, FAMILY, PER_CH_NAMES, @featFamily);

function f = featFamily(x, fs)
    x = (x - mean(x)) / (std(x) + eps); N = numel(x);
    f = [sampEn(x, 2, 0.2), permEn(x, 3), permEn(x, 4), specEn(x, fs), fuzzyEn(x, 2, 0.2, 2)];
end
function se = sampEn(x, m, r)
    N = numel(x); r = r * std(x);
    B = countMatches(x, m, r, N); A = countMatches(x, m + 1, r, N);
    if A == 0 || B == 0, se = NaN; else, se = -log(A / B); end
end
function cnt = countMatches(x, m, r, N)
    n = N - m; E = zeros(n, m); for j = 1:m, E(:, j) = x(j:j + n - 1); end
    D = zeros(n, n); for j = 1:m, D = max(D, abs(E(:, j) - E(:, j)')); end
    D(1:n + 1:end) = Inf; cnt = nnz(D <= r) / 2;
end
function pe = permEn(x, m)
    N = numel(x) - m + 1; E = zeros(N, m); for j = 1:m, E(:, j) = x(j:j + N - 1); end
    [~, idx] = sort(E, 2); code = idx * (m .^ (0:m - 1))'; [~, ~, k] = unique(code); p = accumarray(k, 1) / N;
    pe = -sum(p .* log(p)) / log(factorial(m));
end
function se = specEn(x, fs)
    [P, fr] = pwelch(x, hamming(128), 64, 256, fs); m = fr >= 0.5 & fr <= 45; p = P(m) / sum(P(m)); p = p(p > 0);
    se = -sum(p .* log(p)) / log(nnz(m));
end
function fe = fuzzyEn(x, m, r, n)
    r = r * std(x); N = numel(x);
    phi = zeros(1, 2);
    for mm = [m, m + 1]
        L = N - mm; E = zeros(L, mm); for j = 1:mm, E(:, j) = x(j:j + L - 1); end
        E = E - mean(E, 2);
        D = zeros(L, L); for j = 1:mm, D = max(D, abs(E(:, j) - E(:, j)')); end
        Sim = exp(-(D / r) .^ n); Sim(1:L + 1:end) = 0;
        phi(mm - m + 1) = sum(Sim(:)) / (L * (L - 1));
    end
    if phi(1) <= 0 || phi(2) <= 0, fe = NaN; else, fe = log(phi(1)) - log(phi(2)); end
end

%% ======================================================== SHARED RUNNER (identical in all Stage 8 scripts)
function s8_run(ROOT, family, perChNames, featFun)
    IN = fullfile(ROOT, 'Stage1b_Epochs'); OUT = fullfile(ROOT, 'Stage8_Features', family);
    if ~isfolder(OUT), mkdir(OUT); end
    fid = fopen(fullfile(OUT, sprintf('Stage8_%s_Report.txt', family)), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
    L = @(varargin) logline(fid, varargin{:});
    L('STAGE 8 - family "%s"   %s', family, datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    L('features per channel (%d): %s', numel(perChNames), strjoin(perChNames, ', '));
    if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')), try, if isempty(gcp('nocreate')), parpool; end, catch, end, end
    stores = dir(fullfile(IN, 'D*_epochs.mat')); stores = stores(~contains({stores.name}, 'undetermined'));
    for d = 1:numel(stores)
        if isfile(fullfile(OUT, strrep(stores(d).name, '_epochs.mat', sprintf('_%s.mat', family)))), L('  %s already done - skipped', stores(d).name); continue; end
        T = load(fullfile(IN, stores(d).name)); S = T.S; X = S.X; nEp = size(X, 1); nCh = size(X, 3); fs = S.fs; nPer = numel(perChNames);
        featNames = cell(1, nPer * nCh);
        for c = 1:nCh, for j = 1:nPer, featNames{(c-1)*nPer + j} = sprintf('%s_%s', S.channels{c}, perChNames{j}); end, end
        feat = nan(nEp, nPer * nCh); errs = false(nEp, 1); t0 = tic;
        parfor e = 1:nEp
            row = nan(1, nPer * nCh); bad = false;
            for c = 1:nCh
                try, row((c-1)*nPer + (1:nPer)) = featFun(double(squeeze(X(e, :, c)))', fs); catch, bad = true; end
            end
            feat(e, :) = row; errs(e) = bad;
        end
        L('\n=== %s : %d epochs x %d ch -> %d features | %.1f min | epochs with errors %d | NaN %.2f%% | Inf %d', ...
            S.dataset, nEp, nCh, nPer * nCh, toc(t0) / 60, nnz(errs), 100 * mean(isnan(feat(:))), nnz(isinf(feat(:))));
        feat(isinf(feat)) = NaN;
        F = struct('dataset', S.dataset, 'family', family, 'feat', feat, 'featNames', {featNames}, 'channels', {S.channels}, ...
            'subject', double(S.subject(:)), 'subjectID', S.subjectID, 'label', double(S.label(:)), 'age', double(S.age(:)), 'sex', double(S.sex(:)), 'fs', fs, 'preproc', S.preproc);
        outName = strrep(stores(d).name, '_epochs.mat', sprintf('_%s.mat', family));
        save(fullfile(OUT, outName), 'F', '-v7.3'); L('  saved %s', fullfile(OUT, outName));
        % sanity: subject-level univariate AUC, top 5
        [~, ~, k] = unique(F.subject); lab = accumarray(k, F.label, [], @(v) v(1)); auc = zeros(1, size(feat, 2));
        for j = 1:size(feat, 2)
            m = accumarray(k, feat(:, j), [], @(v) mean(v, 'omitnan')); if all(isnan(m)) || std(m, 'omitnan') < 1e-12, continue; end
            try, [~, ~, ~, a] = perfcurve(lab(~isnan(m)), m(~isnan(m)), 1); auc(j) = max(a, 1 - a); catch, end
        end
        [~, o] = sort(auc, 'descend');
        L('  top-5 subject-level |AUC|: %s', strjoin(arrayfun(@(j) sprintf('%s %.2f', featNames{j}, auc(j)), o(1:min(5, end)), 'UniformOutput', false), ', '));
    end
    L('\nDONE.');
end
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
