%% stage8f_riemannian.m
%  STAGE 8f - Riemannian family: per-epoch channel covariance matrix mapped to the tangent space
%  by the matrix logarithm (log-Euclidean, reference-free), upper triangle vectorised with
%  off-diagonal terms scaled by sqrt(2). nCh*(nCh+1)/2 features per epoch (IEEE 190, TDBRAIN 351, Mendeley 3).
%  Shrinkage: C + 1e-3 * trace(C)/nCh * I. Writes Stage8_Features\riemannian\D*_riemannian.mat
%  (This family is per epoch, not per channel, so it has its own runner.)

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage1b_Epochs'); OUT = fullfile(ROOT, 'Stage8_Features', 'riemannian'); family = 'riemannian';
if ~isfolder(OUT), mkdir(OUT); end
fid = fopen(fullfile(OUT, 'Stage8_riemannian_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 8 - family "riemannian"   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')), try, if isempty(gcp('nocreate')), parpool; end, catch, end, end
stores = dir(fullfile(IN, 'D*_epochs.mat')); stores = stores(~contains({stores.name}, 'undetermined'));
for d = 1:numel(stores)
    if isfile(fullfile(OUT, strrep(stores(d).name, '_epochs.mat', '_riemannian.mat'))), L('  %s already done - skipped', stores(d).name); continue; end
    T = load(fullfile(IN, stores(d).name)); S = T.S; X = S.X; nEp = size(X, 1); nCh = size(X, 3);
    [iu, ju] = find(triu(true(nCh))); nF = numel(iu); featNames = cell(1, nF);
    for q = 1:nF, featNames{q} = sprintf('%s_%s', S.channels{iu(q)}, S.channels{ju(q)}); end
    w = ones(nF, 1); w(iu ~= ju) = sqrt(2);
    feat = nan(nEp, nF); t0 = tic;
    parfor e = 1:nEp
        Z = double(squeeze(X(e, :, :))); if size(Z, 2) ~= nCh, Z = Z'; end
        good = ~any(isnan(Z), 1); v = nan(nF, 1);
        if nnz(good) >= 2
            Zg = Z(:, good); ng = nnz(good);
            C = cov(Zg); C = C + 1e-3 * trace(C) / ng * eye(ng);
            [V, D] = eig((C + C') / 2); Lg = V * diag(log(max(diag(D), eps))) * V';
            full = nan(nCh); full(good, good) = Lg;
            v = full(sub2ind([nCh nCh], iu, ju)) .* w;
        end
        feat(e, :) = v';
    end
    L('\n=== %s : %d epochs x %d ch -> %d features | %.1f min | NaN %.2f%%', S.dataset, nEp, nCh, nF, toc(t0) / 60, 100 * mean(isnan(feat(:))));
    F = struct('dataset', S.dataset, 'family', family, 'feat', feat, 'featNames', {featNames}, 'channels', {S.channels}, ...
        'subject', double(S.subject(:)), 'subjectID', S.subjectID, 'label', double(S.label(:)), 'age', double(S.age(:)), 'sex', double(S.sex(:)), 'fs', S.fs, 'preproc', S.preproc);
    outName = strrep(stores(d).name, '_epochs.mat', '_riemannian.mat'); save(fullfile(OUT, outName), 'F', '-v7.3'); L('  saved %s', fullfile(OUT, outName));
    [~, ~, k] = unique(F.subject); lab = accumarray(k, F.label, [], @(v) v(1)); auc = zeros(1, nF);
    for j = 1:nF
        m = accumarray(k, feat(:, j), [], @mean); if std(m) < 1e-12, continue; end
        [~, ~, ~, a] = perfcurve(lab, m, 1); auc(j) = max(a, 1 - a);
    end
    [~, o] = sort(auc, 'descend');
    L('  top-5 subject-level |AUC|: %s', strjoin(arrayfun(@(j) sprintf('%s %.2f', featNames{j}, auc(j)), o(1:min(5, end)), 'UniformOutput', false), ', '));
end
L('\nDONE.');

function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
