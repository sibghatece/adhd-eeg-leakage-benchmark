%% stage8c_nonlinear.m
%  STAGE 8c - Nonlinear / fractal family (6 per channel), on the z-normalised epoch:
%     Higuchi FD (kmax=8), Katz FD, Petrosian FD, DFA exponent (boxes 4..64),
%     Hurst exponent (rescaled range), Lempel-Ziv complexity (median-binarised, normalised)
%  Writes Stage8_Features\nonlinear\D*_nonlinear.mat

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
FAMILY = 'nonlinear'; PER_CH_NAMES = {'HiguchiFD', 'KatzFD', 'PetrosianFD', 'DFA', 'Hurst', 'LZC'};
s8_run(ROOT, FAMILY, PER_CH_NAMES, @featFamily);

function f = featFamily(x, ~)
    x = (x - mean(x)) / (std(x) + eps);
    f = [higuchi(x, 8), katz(x), petrosian(x), dfa(x), hurstRS(x), lzc(x)];
end
function fd = higuchi(x, kmax)
    N = numel(x); Lk = zeros(1, kmax);
    for k = 1:kmax
        Lm = zeros(1, k);
        for m = 1:k
            idx = m:k:N; n = numel(idx); if n < 2, Lm(m) = NaN; continue; end
            Lm(m) = sum(abs(diff(x(idx)))) * (N - 1) / ((n - 1) * k) / k;
        end
        Lk(k) = mean(Lm, 'omitnan');
    end
    p = polyfit(log(1 ./ (1:kmax)), log(Lk), 1); fd = p(1);
end
function fd = katz(x)
    n = numel(x) - 1; L = sum(sqrt(1 + diff(x) .^ 2)); d = max(sqrt((1:n)' .^ 2 + (x(2:end) - x(1)) .^ 2));
    fd = log10(n) / (log10(n) + log10(d / L));
end
function fd = petrosian(x)
    N = numel(x); nd = nnz(diff(sign(diff(x))) ~= 0);
    fd = log10(N) / (log10(N) + log10(N / (N + 0.4 * nd)));
end
function a = dfa(x)
    y = cumsum(x - mean(x)); N = numel(y); boxes = unique(round(logspace(log10(4), log10(64), 8)));
    Fn = zeros(size(boxes));
    for b = 1:numel(boxes)
        n = boxes(b); nb = floor(N / n); res = zeros(nb, 1);
        for i = 1:nb
            seg = y((i - 1) * n + (1:n)); t = (1:n)'; p = polyfit(t, seg, 1); res(i) = mean((seg - polyval(p, t)) .^ 2);
        end
        Fn(b) = sqrt(mean(res));
    end
    p = polyfit(log(boxes), log(Fn), 1); a = p(1);
end
function h = hurstRS(x)
    N = numel(x); sizes = unique(round(logspace(log10(8), log10(N / 2), 6))); rs = zeros(size(sizes));
    for s = 1:numel(sizes)
        n = sizes(s); nb = floor(N / n); v = zeros(nb, 1);
        for i = 1:nb
            seg = x((i - 1) * n + (1:n)); z = cumsum(seg - mean(seg)); R = max(z) - min(z); Sd = std(seg);
            v(i) = R / max(Sd, eps);
        end
        rs(s) = mean(v);
    end
    p = polyfit(log(sizes), log(rs), 1); h = p(1);
end
function c = lzc(x)
    % Lempel-Ziv complexity (Kaspar & Schuster 1987), median-binarised, normalised by n/log2(n)
    s = x > median(x); n = numel(s);
    c = 1; l = 1; i = 0; k = 1; kmax = 1;
    while true
        if i + k > n || l + k > n, c = c + 1; break; end
        if s(i + k) == s(l + k)
            k = k + 1;
            if l + k > n, c = c + 1; break; end
        else
            if k > kmax, kmax = k; end
            i = i + 1;
            if i == l
                c = c + 1; l = l + kmax;
                if l + 1 > n, break; end
                i = 0; k = 1; kmax = 1;
            else
                k = 1;
            end
        end
    end
    c = c * log2(n) / n;
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
