%% stage8d_wavelet.m
%  STAGE 8d - Discrete wavelet family (16 per channel): db4, 4 levels at 128 Hz
%     sub-bands D1 32-64, D2 16-32, D3 8-16, D4 4-8, A4 0-4 Hz;
%     per sub-band log energy, relative energy, std; plus Shannon wavelet entropy of relative energies
%  Requires Wavelet Toolbox. Writes Stage8_Features\wavelet\D*_wavelet.mat

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
FAMILY = 'wavelet'; PER_CH_NAMES = [strcat({'D1','D2','D3','D4','A4'}, '_logE'), strcat({'D1','D2','D3','D4','A4'}, '_relE'), strcat({'D1','D2','D3','D4','A4'}, '_std'), {'WaveletEntropy'}];
s8_run(ROOT, FAMILY, PER_CH_NAMES, @featFamily);

function f = featFamily(x, ~)
    [C, Lw] = wavedec(x, 4, 'db4');
    bands = {detcoef(C, Lw, 1), detcoef(C, Lw, 2), detcoef(C, Lw, 3), detcoef(C, Lw, 4), appcoef(C, Lw, 'db4', 4)};
    E = cellfun(@(c) sum(c .^ 2), bands); rel = E / max(sum(E), eps); sd = cellfun(@std, bands);
    p = rel(rel > 0); went = -sum(p .* log(p));
    f = [log(E + eps), rel, sd, went];
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
