%% stage8e_vmd.m
%  STAGE 8e - Variational mode decomposition family (16 per channel): vmd(x, 'NumIMFs', 4)
%     per mode (low -> high frequency): log energy, relative energy, Hjorth mobility, spectral centroid (Hz)
%  Requires Signal Processing Toolbox R2020a+. Writes Stage8_Features\vmd\D*_vmd.mat

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
FAMILY = 'vmd'; PER_CH_NAMES = [strcat('M', arrayfun(@num2str, repelem(1:4, 4), 'UniformOutput', false), '_', repmat({'logE', 'relE', 'mobility', 'centroid'}, 1, 4))];
s8_run(ROOT, FAMILY, PER_CH_NAMES, @featFamily);

function f = featFamily(x, fs)
    x = (x - mean(x)) / (std(x) + eps);
    imf = vmd(x, 'NumIMFs', 4);                       % samples x 4
    f = nan(1, 16); E = sum(imf .^ 2, 1); relE = E / max(sum(E), eps);
    for m = 1:size(imf, 2)
        u = imf(:, m); mob = sqrt(var(diff(u)) / max(var(u), eps));
        [P, fr] = pwelch(u, hamming(128), 64, 256, fs); cen = sum(fr .* P) / max(sum(P), eps);
        f((m-1)*4 + (1:4)) = [log(E(m) + eps), relE(m), mob, cen];
    end
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
