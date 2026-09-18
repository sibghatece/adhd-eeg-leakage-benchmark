%% stage2_extract_features.m
%  STAGE 2 - EMD -> EWT -> 2D/3D phase-space feature extraction, written from
%  scratch to match the manuscript equations. Runs on the Stage 1 epoch
%  stores and writes one feature store per dataset:
%
%     <OUT>\D1_IEEE_features.mat, D2_TDBRAIN_features.mat, D3_MENDELEY_features.mat
%     <OUT>\Stage2_Report.txt
%
%  Each feature store contains struct F with
%     F.feat      [nEpochs x 304] double  (NaN where a component was missing)
%     F.featNames {1 x 304} cell           e.g. 'Cz_IMF2_W1_ISM'
%     F.baseline  [nEpochs x 2]            log-variance of the raw (un-normalised) epoch, Cz and F4
%     F.subject, F.subjectID, F.label, F.age, F.sex  copied from Stage 1
%     F.nIMF, F.nMode  [nEpochs x 2] / [nEpochs x 2 x 4]  components actually obtained
%     F.dataset, F.preproc, F.featureConfig
%
%  Method (per epoch, per channel):
%     z-normalise the 2 s epoch (zero mean, unit variance)
%     -> emd(x, MaxNumIMF = 4)              first 4 IMFs
%     -> ewt(IMF_i, MaxNumIMF = 2)          first 2 modes of each IMF
%     -> 19 PSR features per mode:  15 from 2D PSR (tau = 1), 4 from 3D PSR (tau = 1)
%     = 19 x 2 modes x 4 IMFs x 2 channels = 304 features
%
%  Requires Signal Processing Toolbox (emd, ewt). parfor is used if a pool is
%  available; otherwise it silently runs serially.

clear; clc;

%% ------------------------------------------------------------------ CONFIG
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN   = fullfile(ROOT, 'Stage1_Epochs');
OUT  = fullfile(ROOT, 'Stage2_Features');

STORES = {'D1_IEEE_epochs.mat', 'D1_IEEE_features.mat'; ...
          'D2_TDBRAIN_epochs.mat', 'D2_TDBRAIN_features.mat'; ...
          'D3_MENDELEY_epochs.mat', 'D3_MENDELEY_features.mat'};

N_IMF   = 4;      % EMD IMFs kept
N_MODE  = 2;      % EWT modes kept per IMF
TAU     = 1;      % PSR delay (samples)
SCI_BINS = 8;     % bins per axis for the 3D histogram (Shannon complexity index)
DEM_K   = 10;     % divergence horizon (samples) for the divergence exponent metric
NORMALISE_EPOCH = true;   % per-epoch z-normalisation (removes gain differences between cohorts)

FEAT19 = {'CEM','TPS','ICA','ISM','IAM','TMS','PDD','SDD','OCM','RDA','SAM','OTS','OAM','BDI','ECM', ...
          'SCI','DEM','TBI','TTI'};

%% ------------------------------------------------------------------ SETUP
if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage2_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 2 - FEATURE EXTRACTION   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('EMD IMFs = %d, EWT modes = %d, tau = %d, SCI bins = %d, DEM K = %d, per-epoch z-norm = %d', ...
    N_IMF, N_MODE, TAU, SCI_BINS, DEM_K, NORMALISE_EPOCH);

% feature names, index = ((ch-1)*N_IMF*N_MODE + (i-1)*N_MODE + (k-1))*19 + f
chNames = {'Cz', 'F4'};
featNames = cell(1, 2 * N_IMF * N_MODE * 19); q = 0;
for ch = 1:2
    for i = 1:N_IMF
        for k = 1:N_MODE
            for f = 1:19
                q = q + 1; featNames{q} = sprintf('%s_IMF%d_W%d_%s', chNames{ch}, i, k, FEAT19{f});
            end
        end
    end
end
nFeat = numel(featNames);
L('Total features: %d', nFeat);

% --- make sure the TOOLBOX emd/ewt are the ones called (user copies of emd.m /
%     ewt.m elsewhere on the path shadow them and reject name-value arguments)
sigDir = fullfile(matlabroot, 'toolbox', 'signal', 'signal');
if isfolder(sigDir), addpath(sigDir); end
wEmd = which('emd'); wEwt = which('ewt');
L('emd resolves to: %s', wEmd);
L('ewt resolves to: %s', wEwt);
if ~contains(lower(wEmd), 'toolbox') || ~contains(lower(wEwt), 'toolbox')
    error('emd/ewt are shadowed by non-toolbox files. Remove the folder containing your own emd.m / ewt.m from the path and rerun.');
end

% --- self-test on one real epoch, NOT inside try/catch, so any error is visible
T0 = load(fullfile(IN, STORES{1, 1})); x0 = double(squeeze(T0.S.X(1, :, 1)))'; x0 = (x0 - mean(x0)) / std(x0);
imf0 = emd(x0, 'MaxNumIMF', N_IMF, 'Display', 0);
L('self-test emd: %d samples -> %d IMFs (size %s)', numel(x0), size(imf0, 2), mat2str(size(imf0)));
assert(size(imf0, 1) == numel(x0) && size(imf0, 2) >= 1, 'emd self-test failed');
mode0 = ewt(imf0(:, 1), 'MaxNumPeaks', N_MODE);
L('self-test ewt: IMF1 -> %d modes (size %s)', size(mode0, 2), mat2str(size(mode0)));
assert(size(mode0, 1) == numel(x0) && size(mode0, 2) >= 1, 'ewt self-test failed');
f0 = psrFeatures19(mode0(:, 1), TAU, SCI_BINS, DEM_K);
L('self-test features: %d values, %d NaN', numel(f0), nnz(isnan(f0)));
assert(~any(isnan(f0)), 'feature self-test produced NaN');
clear T0 x0 imf0 mode0 f0

haveParallel = license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
if haveParallel
    try, if isempty(gcp('nocreate')), parpool; end, catch, haveParallel = false; end
end
L('Parallel pool: %d', haveParallel);

cfg = struct('N_IMF', N_IMF, 'N_MODE', N_MODE, 'TAU', TAU, 'SCI_BINS', SCI_BINS, 'DEM_K', DEM_K, ...
             'NORMALISE_EPOCH', NORMALISE_EPOCH, 'FEAT19', {FEAT19});

%% ------------------------------------------------------------------ MAIN
for d = 1:size(STORES, 1)
    inPath = fullfile(IN, STORES{d, 1}); outPath = fullfile(OUT, STORES{d, 2});
    if ~isfile(inPath), L('!! missing %s - skipped', inPath); continue; end
    T = load(inPath); S = T.S;
    nEp = size(S.X, 1); nSamp = size(S.X, 2);
    L('\n=== %s : %d epochs x %d samples x 2 ch', S.dataset, nEp, nSamp);

    feat  = nan(nEp, nFeat);
    base  = nan(nEp, 2);
    nIMF  = zeros(nEp, 2);
    nMode = zeros(nEp, 2, N_IMF);
    X = S.X;   % broadcast to workers
    errCell = cell(nEp, 1);
    t0 = tic;

    parfor e = 1:nEp
        rowFeat = nan(1, nFeat); rowBase = nan(1, 2); rowIMF = zeros(1, 2); rowMode = zeros(1, 2, N_IMF); rowErr = '';
        for ch = 1:2
            x = double(squeeze(X(e, :, ch)))';                       % column
            rowBase(ch) = log(var(x) + eps);
            if cfg.NORMALISE_EPOCH
                x = (x - mean(x)) / (std(x) + eps);
            end
            [rf, ni, nm, em] = extractEpochChannel(x, cfg);
            if ~isempty(em), rowErr = em; end
            base0 = (ch - 1) * cfg.N_IMF * cfg.N_MODE * 19;
            rowFeat(base0 + (1:numel(rf))) = rf;
            rowIMF(ch) = ni; rowMode(1, ch, :) = reshape(nm, 1, 1, []);
        end
        feat(e, :) = rowFeat; base(e, :) = rowBase; nIMF(e, :) = rowIMF; nMode(e, :, :) = rowMode;
        errCell{e} = rowErr;
    end
    elapsed = toc(t0);
    nErr = nnz(~cellfun(@isempty, errCell));
    if nErr > 0
        firstErr = errCell{find(~cellfun(@isempty, errCell), 1)};
        L('  !! %d epochs raised errors. First error: %s', nErr, firstErr);
    end

    % report component availability
    L('  elapsed %.1f min  (%.1f ms/epoch)', elapsed / 60, 1000 * elapsed / nEp);
    for ch = 1:2
        L('  %s: epochs with < %d IMFs: %d (%.1f%%)', chNames{ch}, N_IMF, nnz(nIMF(:, ch) < N_IMF), 100 * mean(nIMF(:, ch) < N_IMF));
        for i = 1:N_IMF
            L('      IMF%d: epochs with < %d EWT modes: %d (%.1f%%)', i, N_MODE, nnz(nMode(:, ch, i) < N_MODE), 100 * mean(nMode(:, ch, i) < N_MODE));
        end
    end
    L('  NaN fraction in feature matrix: %.2f%%   non-finite (Inf) entries: %d', 100 * mean(isnan(feat(:))), nnz(isinf(feat(:))));
    L('  columns entirely NaN: %d', nnz(all(isnan(feat), 1)));

    F = struct('dataset', S.dataset, 'feat', feat, 'featNames', {featNames}, 'baseline', base, ...
        'subject', S.subject, 'subjectID', S.subjectID, 'label', S.label, 'age', S.age, 'sex', S.sex, ...
        'nIMF', nIMF, 'nMode', nMode, 'preproc', S.preproc, 'featureConfig', cfg, 'fs', S.fs, 'epochLen', S.epochLen);
    save(outPath, 'F', '-v7.3');
    L('  saved %s', outPath);
end
L('\nDONE. Send Stage2_Report.txt.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin)
    s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s);
end

% --------------------------------------------------------------------------
function [rf, nImfGot, nModeGot, errMsg] = extractEpochChannel(x, cfg)
    % x: column vector (one epoch, one channel). Returns 19*N_MODE*N_IMF features
    % ordered IMF -> mode -> feature; NaN where a component does not exist.
    nI = cfg.N_IMF; nM = cfg.N_MODE; errMsg = '';
    rf = nan(1, 19 * nM * nI); nModeGot = zeros(1, nI);
    try
        imfs = emd(x, 'MaxNumIMF', nI, 'Display', 0);        % samples x nIMF (may be fewer)
    catch ME
        imfs = zeros(numel(x), 0); errMsg = ['emd: ' ME.message];
    end
    nImfGot = size(imfs, 2);
    for i = 1:min(nI, nImfGot)
        try
            modes = ewt(imfs(:, i), 'MaxNumPeaks', nM);        % samples x nModes (may be fewer or more)
        catch ME
            modes = zeros(numel(x), 0); if isempty(errMsg), errMsg = ['ewt: ' ME.message]; end
        end
        modes = modes(:, 1:min(nM, size(modes, 2)));
        nModeGot(i) = size(modes, 2);
        for k = 1:min(nM, nModeGot(i))
            idx = ((i - 1) * nM + (k - 1)) * 19 + (1:19);
            rf(idx) = psrFeatures19(modes(:, k), cfg.TAU, cfg.SCI_BINS, cfg.DEM_K);
        end
    end
end

% --------------------------------------------------------------------------
function f = psrFeatures19(s, tau, nBins, K)
    % 19 phase-space features of a 1-D signal s (column), following the
    % manuscript definitions (Section 2.5) with corrected curvature/torsion.
    s = s(:); L = numel(s);
    f = nan(1, 19);
    if L < 3 * tau + 4 || ~all(isfinite(s)) || std(s) < 1e-12, return; end

    % ---------------- 2D PSR: p_n = (x_n, y_n) = (s(n), s(n+tau)), N points
    x = s(1:end - tau); y = s(1 + tau:end); N = numel(x);
    dx = diff(x); dy = diff(y);                                   % successive displacements
    % 1 CEM  circular enclosures metric  sum pi * d_n^2
    d2 = dx.^2 + dy.^2;
    f(1) = pi * sum(d2);
    % 2 TPS  triangular patches sum: area of triangles of 3 consecutive points
    x1 = x(1:N-2); x2 = x(2:N-1); x3 = x(3:N); y1 = y(1:N-2); y2 = y(2:N-1); y3 = y(3:N);
    A = 0.5 * abs(x1 .* (y2 - y3) + x2 .* (y3 - y1) + x3 .* (y1 - y2));
    f(2) = sum(A);
    % 3 ICA  inscribed circle aggregate: sum pi r_n^2, r_n = 2A_n / perimeter_n
    a = hypot(x2 - x1, y2 - y1); b = hypot(x3 - x2, y3 - y2); c = hypot(x1 - x3, y1 - y3);
    per = a + b + c; r = 2 * A ./ max(per, eps);
    f(3) = pi * sum(r.^2);
    % incenters  (weighted by opposite side lengths: side a opposite vertex 3, etc.)
    % vertices: P1 (x1,y1), P2 (x2,y2), P3 (x3,y3); side lengths opposite: |P2P3| = b, |P3P1| = c, |P1P2| = a
    Xc = (b .* x1 + c .* x2 + a .* x3) ./ max(per, eps);
    Yc = (b .* y1 + c .* y2 + a .* y3) ./ max(per, eps);
    % 4 ISM  incircle separation metric: sum of distances between consecutive incenters
    f(4) = sum(hypot(diff(Xc), diff(Yc)));
    % 5 IAM  incircle angular measure: sum of angles between consecutive incenter vectors
    f(5) = sumAngles([Xc Yc]);
    % 6 TMS  trajectory magnitude sum: sum of squared successive displacements
    f(6) = sum(d2);
    % 7 PDD / 8 SDD  distances from the 45 and 135 degree diagonals
    u = (x - y) / sqrt(2); v = (x + y) / sqrt(2);
    f(7) = sum(abs(u)); f(8) = sum(abs(v));
    % second moments for ellipse / octagon
    Sxx = mean(x.^2); Syy = mean(y.^2); Sxy = mean(x .* y);
    Delta = sqrt(max((Sxx + Syy)^2 - 4 * (Sxx * Syy - Sxy^2), 0));
    ea = sqrt(3 * max(Sxx + Syy + Delta, 0));                    % semi-major
    eb = sqrt(3 * max(Sxx + Syy - Delta, 0));                    % semi-minor
    % 9 OCM  octagonal coverage metric  (2 / (1 + sqrt 2)) * a^2
    f(9) = 2 / (1 + sqrt(2)) * ea^2;
    % 10 RDA  radial distance aggregate
    f(10) = sum(hypot(x, y));
    % 11 SAM  sequential angular measure: angles between consecutive position vectors
    f(11) = sumAngles([x y]);
    % 12 OTS  origin-based triangle sum: area of triangle (0, p_n, p_{n+1})
    f(12) = 0.5 * sum(abs(x(1:N-1) .* y(2:N) - x(2:N) .* y(1:N-1)));
    % 13 OAM  orthogonal area metric
    f(13) = sum(abs(u .* v));
    % 14 BDI  bivariate dispersion index  pi * sigma1 * sigma2
    f(14) = pi * std(u) * std(v);
    % 15 ECM  elliptical coverage measure  pi a b
    f(15) = pi * ea * eb;

    % ---------------- 3D PSR: q_n = (s(n), s(n+tau), s(n+2tau))
    Q = [s(1:end - 2 * tau), s(1 + tau:end - tau), s(1 + 2 * tau:end)]; M = size(Q, 1);
    % 16 SCI  Shannon complexity index  exp(-sum p log2 p) over a 3D histogram
    B = nBins; idx = zeros(M, 3);
    for j = 1:3
        lo = min(Q(:, j)); hi = max(Q(:, j)); w = max(hi - lo, eps);
        idx(:, j) = min(floor((Q(:, j) - lo) / w * B) + 1, B);
    end
    lin = (idx(:, 1) - 1) * B * B + (idx(:, 2) - 1) * B + idx(:, 3);
    p = accumarray(lin, 1, [B^3 1]) / M; p = p(p > 0);
    f(16) = exp(-sum(p .* log2(p)));
    % 17 DEM  divergence exponent metric: mean over k of log mean trajectory divergence
    Kk = min(K, M - 2); dv = zeros(Kk, 1);
    for k = 1:Kk
        dv(k) = log(mean(sqrt(sum((Q(1 + k:end, :) - Q(1:end - k, :)).^2, 2))) + eps);
    end
    f(17) = mean(dv);
    % 18 TBI  trajectory bending index: mean curvature |q' x q''| / |q'|^3
    % 19 TTI  trajectory twisting index: mean torsion q'.(q'' x q''') / |q' x q''|^2
    d1 = diff(Q, 1, 1); d2_ = diff(Q, 2, 1); d3 = diff(Q, 3, 1);
    n = min([size(d1, 1), size(d2_, 1), size(d3, 1)]);
    d1 = d1(1:n, :); d2_ = d2_(1:n, :); d3 = d3(1:n, :);
    cr = cross(d1, d2_, 2); ncr = sqrt(sum(cr.^2, 2)); n1 = sqrt(sum(d1.^2, 2));
    valid = n1 > 1e-9;
    f(18) = mean(ncr(valid) ./ n1(valid).^3);
    valid2 = ncr > 1e-9;
    if any(valid2)
        f(19) = mean(sum(d1(valid2, :) .* cross(d2_(valid2, :), d3(valid2, :), 2), 2) ./ ncr(valid2).^2);
    else
        f(19) = 0;
    end
end

function a = sumAngles(P)
    % sum of angles between consecutive row vectors of P (n x 2)
    P1 = P(1:end - 1, :); P2 = P(2:end, :);
    dotp = sum(P1 .* P2, 2); nrm = sqrt(sum(P1.^2, 2)) .* sqrt(sum(P2.^2, 2));
    ok = nrm > 1e-12;
    cs = max(min(dotp(ok) ./ nrm(ok), 1), -1);
    a = sum(acos(cs));
end
