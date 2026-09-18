%% stage2b_extract_features_multichannel.m
%  STAGE 2b - Feature extraction on the Stage 1b multichannel epoch stores.
%  For every epoch and every channel:
%     PSR block : z-normalise -> emd (4 IMFs) -> ewt (2 modes) -> 19 PSR features   (152 per channel)
%     BandPower : relative delta/theta/alpha/beta power + theta/beta ratio           (5 per channel)
%     LogVar    : log variance of the raw epoch                                       (1 per channel)
%
%  Output <OUT>\D*_features.mat with struct F:
%     F.psr [nEp x 152*nCh], F.psrNames ; F.bp [nEp x 5*nCh], F.bpNames ; F.logvar [nEp x nCh], F.lvNames
%     F.channels, F.subject, F.subjectID, F.label, F.age, F.sex, F.dataset

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage1b_Epochs'); OUT = fullfile(ROOT, 'Stage2b_Features');
st = dir(fullfile(IN, 'D*_epochs.mat')); st = st(~contains({st.name}, 'undetermined'));
STORES = [{st.name}', strrep({st.name}', '_epochs.mat', '_features.mat')];   % all stores D1..D5 (D4u excluded)
SKIP_EXISTING = true;   % set false to recompute stores that already have a feature file
N_IMF = 4; N_MODE = 2; TAU = 1; SCI_BINS = 8; DEM_K = 10;
FEAT19 = {'CEM','TPS','ICA','ISM','IAM','TMS','PDD','SDD','OCM','RDA','SAM','OTS','OAM','BDI','ECM','SCI','DEM','TBI','TTI'};
BANDS = [0.5 4; 4 8; 8 13; 13 30]; BAND_NAMES = {'delta','theta','alpha','beta'};

if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage2b_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 2b - MULTICHANNEL FEATURES   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

sigDir = fullfile(matlabroot, 'toolbox', 'signal', 'signal'); if isfolder(sigDir), addpath(sigDir); end
assert(contains(lower(which('emd')), 'toolbox') && contains(lower(which('ewt')), 'toolbox'), 'emd/ewt shadowed - rename your own emd.m/ewt.m');
haveParallel = license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
if haveParallel, try, if isempty(gcp('nocreate')), parpool; end, catch, end, end
cfg = struct('N_IMF', N_IMF, 'N_MODE', N_MODE, 'TAU', TAU, 'SCI_BINS', SCI_BINS, 'DEM_K', DEM_K);

for d = 1:size(STORES, 1)
    inPath = fullfile(IN, STORES{d, 1}); if ~isfile(inPath), L('!! missing %s', inPath); continue; end
    if SKIP_EXISTING && isfile(fullfile(OUT, STORES{d, 2})), L('\n=== %s already has features - skipped (SKIP_EXISTING)', STORES{d, 1}); continue; end
    T = load(inPath); S = T.S; X = S.X; nEp = size(X, 1); nCh = size(X, 3); fs = S.fs; ch = S.channels;
    nPer = 19 * N_MODE * N_IMF;
    psrNames = cell(1, nPer * nCh); bpNames = cell(1, 5 * nCh); lvNames = cell(1, nCh); q = 0;
    for c = 1:nCh
        for i = 1:N_IMF, for k = 1:N_MODE, for f = 1:19
            q = q + 1; psrNames{q} = sprintf('%s_IMF%d_W%d_%s', ch{c}, i, k, FEAT19{f});
        end, end, end
        for b = 1:4, bpNames{(c-1)*5 + b} = sprintf('%s_BP_%s', ch{c}, BAND_NAMES{b}); end
        bpNames{(c-1)*5 + 5} = sprintf('%s_BP_thetabeta', ch{c}); lvNames{c} = sprintf('%s_logvar', ch{c});
    end
    L('\n=== %s : %d epochs x %d ch -> %d PSR + %d BP + %d LV features', S.dataset, nEp, nCh, nPer * nCh, 5 * nCh, nCh);
    psr = nan(nEp, nPer * nCh); bp = nan(nEp, 5 * nCh); lv = nan(nEp, nCh); errFlag = false(nEp, 1);
    win = hamming(128); t0 = tic;
    parfor e = 1:nEp
        rowP = nan(1, nPer * nCh); rowB = nan(1, 5 * nCh); rowL = nan(1, nCh); err = false;
        for c = 1:nCh
            x = double(squeeze(X(e, :, c)))';
            if any(~isfinite(x)), continue; end                 % bad channel (NaN-marked in Stage 1c): leave features NaN
            rowL(c) = log(var(x) + eps);
            [P, fr] = pwelch(x, win, 64, 256, fs);
            tot = trapz(fr(fr >= 0.5 & fr <= 45), P(fr >= 0.5 & fr <= 45));
            rel = zeros(1, 4);
            for b = 1:4, m = fr >= BANDS(b, 1) & fr < BANDS(b, 2); rel(b) = trapz(fr(m), P(m)) / max(tot, eps); end
            rowB((c-1)*5 + (1:5)) = [rel, rel(2) / max(rel(4), eps)];
            xz = (x - mean(x)) / (std(x) + eps);
            [rf, e1] = extractEpochChannel(xz, cfg); err = err | e1;
            rowP((c-1)*nPer + (1:nPer)) = rf;
        end
        psr(e, :) = rowP; bp(e, :) = rowB; lv(e, :) = rowL; errFlag(e) = err;
    end
    L('  elapsed %.1f min; epochs with an emd/ewt error: %d; NaN fraction PSR %.2f%%', toc(t0) / 60, nnz(errFlag), 100 * mean(isnan(psr(:))));
    F = struct('dataset', S.dataset, 'channels', {ch}, 'psr', psr, 'psrNames', {psrNames}, 'bp', bp, 'bpNames', {bpNames}, ...
        'logvar', lv, 'lvNames', {lvNames}, 'subject', S.subject, 'subjectID', S.subjectID, 'label', S.label, 'age', S.age, 'sex', S.sex, ...
        'fs', fs, 'featureConfig', cfg, 'preproc', S.preproc);
    save(fullfile(OUT, STORES{d, 2}), 'F', '-v7.3'); L('  saved %s', fullfile(OUT, STORES{d, 2}));
end
L('\nDONE. Send Stage2b_Report.txt.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end

function [rf, err] = extractEpochChannel(x, cfg)
    nI = cfg.N_IMF; nM = cfg.N_MODE; err = false; rf = nan(1, 19 * nM * nI);
    try, imfs = emd(x, 'MaxNumIMF', nI, 'Display', 0); catch, imfs = zeros(numel(x), 0); err = true; end
    for i = 1:min(nI, size(imfs, 2))
        try, modes = ewt(imfs(:, i), 'MaxNumPeaks', nM); catch, modes = zeros(numel(x), 0); err = true; end
        modes = modes(:, 1:min(nM, size(modes, 2)));
        for k = 1:size(modes, 2)
            rf(((i - 1) * nM + (k - 1)) * 19 + (1:19)) = psrFeatures19(modes(:, k), cfg.TAU, cfg.SCI_BINS, cfg.DEM_K);
        end
    end
end

function f = psrFeatures19(s, tau, nBins, K)
    s = s(:); L = numel(s); f = nan(1, 19);
    if L < 3 * tau + 4 || ~all(isfinite(s)) || std(s) < 1e-12, return; end
    x = s(1:end - tau); y = s(1 + tau:end); N = numel(x); dx = diff(x); dy = diff(y); d2 = dx.^2 + dy.^2;
    f(1) = pi * sum(d2);
    x1 = x(1:N-2); x2 = x(2:N-1); x3 = x(3:N); y1 = y(1:N-2); y2 = y(2:N-1); y3 = y(3:N);
    A = 0.5 * abs(x1 .* (y2 - y3) + x2 .* (y3 - y1) + x3 .* (y1 - y2)); f(2) = sum(A);
    a = hypot(x2 - x1, y2 - y1); b = hypot(x3 - x2, y3 - y2); c = hypot(x1 - x3, y1 - y3); per = a + b + c;
    r = 2 * A ./ max(per, eps); f(3) = pi * sum(r.^2);
    Xc = (b .* x1 + c .* x2 + a .* x3) ./ max(per, eps); Yc = (b .* y1 + c .* y2 + a .* y3) ./ max(per, eps);
    f(4) = sum(hypot(diff(Xc), diff(Yc))); f(5) = sumAngles([Xc Yc]); f(6) = sum(d2);
    u = (x - y) / sqrt(2); v = (x + y) / sqrt(2); f(7) = sum(abs(u)); f(8) = sum(abs(v));
    Sxx = mean(x.^2); Syy = mean(y.^2); Sxy = mean(x .* y);
    Delta = sqrt(max((Sxx + Syy)^2 - 4 * (Sxx * Syy - Sxy^2), 0));
    ea = sqrt(3 * max(Sxx + Syy + Delta, 0)); eb = sqrt(3 * max(Sxx + Syy - Delta, 0));
    f(9) = 2 / (1 + sqrt(2)) * ea^2; f(10) = sum(hypot(x, y)); f(11) = sumAngles([x y]);
    f(12) = 0.5 * sum(abs(x(1:N-1) .* y(2:N) - x(2:N) .* y(1:N-1))); f(13) = sum(abs(u .* v));
    f(14) = pi * std(u) * std(v); f(15) = pi * ea * eb;
    Q = [s(1:end - 2 * tau), s(1 + tau:end - tau), s(1 + 2 * tau:end)]; M = size(Q, 1); B = nBins; idx = zeros(M, 3);
    for j = 1:3, lo = min(Q(:, j)); w = max(max(Q(:, j)) - lo, eps); idx(:, j) = min(floor((Q(:, j) - lo) / w * B) + 1, B); end
    lin = (idx(:, 1) - 1) * B * B + (idx(:, 2) - 1) * B + idx(:, 3);
    p = accumarray(lin, 1, [B^3 1]) / M; p = p(p > 0); f(16) = exp(-sum(p .* log2(p)));
    Kk = min(K, M - 2); dv = zeros(Kk, 1);
    for k = 1:Kk, dv(k) = log(mean(sqrt(sum((Q(1 + k:end, :) - Q(1:end - k, :)).^2, 2))) + eps); end
    f(17) = mean(dv);
    d1 = diff(Q, 1, 1); d2_ = diff(Q, 2, 1); d3 = diff(Q, 3, 1); n = min([size(d1, 1), size(d2_, 1), size(d3, 1)]);
    d1 = d1(1:n, :); d2_ = d2_(1:n, :); d3 = d3(1:n, :);
    cr = cross(d1, d2_, 2); ncr = sqrt(sum(cr.^2, 2)); n1 = sqrt(sum(d1.^2, 2)); ok = n1 > 1e-9;
    f(18) = mean(ncr(ok) ./ n1(ok).^3); ok2 = ncr > 1e-9;
    if any(ok2), f(19) = mean(sum(d1(ok2, :) .* cross(d2_(ok2, :), d3(ok2, :), 2), 2) ./ ncr(ok2).^2); else, f(19) = 0; end
end

function a = sumAngles(P)
    P1 = P(1:end - 1, :); P2 = P(2:end, :); dotp = sum(P1 .* P2, 2); nrm = sqrt(sum(P1.^2, 2)) .* sqrt(sum(P2.^2, 2)); ok = nrm > 1e-12;
    a = sum(acos(max(min(dotp(ok) ./ nrm(ok), 1), -1)));
end
