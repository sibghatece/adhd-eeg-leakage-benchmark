%% stage1c_balladeer_epochs.m
%  STAGE 1c - BALLADEER (CGX 29-channel dry EEG, Slackline level 1) -> common epoch store.
%  Label from users_demographics.json field "diagnosed": yes -> 1 (ADHD), no -> 0 (HC);
%  "undetermined" subjects are written to a separate store (label = 1 there is NOT used).
%  NOTE: the "group" field is the treatment-trial arm (experimental/control), NOT case/control.
%
%  Per file: read CSV (fopen/textscan), keep 29 scalp channels, DC removal, BAD-CHANNEL detection
%  (robust amplitude outliers: channel MAD-std > BAD_HI x median or < BAD_LO x median, or flat),
%  common-average reference over GOOD channels only, bad channels set to NaN (imputed fold-wise
%  downstream), resample 500 -> 128 Hz, then the Stage 1b chain (0.5-45 Hz, 50 Hz notch, 2 s epochs,
%  robust multichannel rejection computed on good channels, cap 60 epochs/subject).
%  Subjects with more than MAX_BAD_CH bad channels are skipped.
%  Outputs: Stage1b_Epochs\D4_BALLADEER_epochs.mat, Stage1b_Epochs\D4u_BALLADEER_undetermined_epochs.mat,
%           Stage1b_Epochs\Stage1c_Report.txt

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
D4_ROOT = fullfile(ROOT, 'Dataset4_BALLADEER', 'extracted', 'BALLADEER ADHD DATASET');
OUT = fullfile(ROOT, 'Stage1b_Epochs'); LEVEL = 'SlacklineLvl1';
NOMINAL_FS = 500;
FS_TARGET = 128; EPOCH_SEC = 2; BP_BAND = [0.5 45]; NOTCH_HZ = 50;
MAX_EPOCHS_PER_SUBJECT = 60; MAD_K = 5; ABS_UV_LIMIT = 400; REJECT_CH_FRACTION = 0.15;
BAD_HI = 5; BAD_LO = 0.1; MAX_BAD_CH = 8;      % bad-channel rule (relative to the median channel MAD-std)
EEG_CH = {'AF7','Fpz','F7','Fz','T7','FC6','Fp1','F4','C4','Oz','CP6','Cz','PO8','CP5','O2','O1','P3','P4','P7','P8','Pz','PO7','T8','C3','Fp2','F3','F8','FC5','AF8'};

if ~isfolder(OUT), mkdir(OUT); end
fid = fopen(fullfile(OUT, 'Stage1c_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 1c - BALLADEER EPOCH STORE   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
[bBP, aBP] = butter(4, BP_BAND / (FS_TARGET / 2), 'bandpass'); [bN, aN] = iirnotch(NOTCH_HZ / (FS_TARGET / 2), (NOTCH_HZ / (FS_TARGET / 2)) / 35);
pre = sprintf('DC->CAR->resample %dHz->BP %g-%g->notch %d->%ds epochs->MAD%g (>%.0f%% ch) reject->cap %d', FS_TARGET, BP_BAND(1), BP_BAND(2), NOTCH_HZ, EPOCH_SEC, MAD_K, 100 * REJECT_CH_FRACTION, MAX_EPOCHS_PER_SUBJECT);

%% demographics
demo = jsondecode(fileread(fullfile(D4_ROOT, 'users_demographics.json')));
ids = arrayfun(@(r) char(string(r.user)), demo, 'UniformOutput', false);
diag = arrayfun(@(r) lower(char(string(r.diagnosed))), demo, 'UniformOutput', false);
ageV = arrayfun(@(r) double(r.age), demo); sexV = arrayfun(@(r) double(r.gender == 1), demo);   % 1 = male
L('demographics: %d subjects | diagnosed yes %d, no %d, undetermined %d', numel(ids), nnz(strcmp(diag, 'yes')), nnz(strcmp(diag, 'no')), nnz(strcmp(diag, 'undetermined')));

%% recordings
S  = newStore('BALLADEER', FS_TARGET, EPOCH_SEC, EEG_CH, pre);
Su = newStore('BALLADEER_undetermined', FS_TARGET, EPOCH_SEC, EEG_CH, pre);
si = 0; siu = 0; fsEst = [];
for i = 1:numel(ids)
    d = dir(fullfile(D4_ROOT, ids{i}, LEVEL, '**', '*_EEG_CGX_*.csv')); if isempty(d), continue; end
    fp = fullfile(d(1).folder, d(1).name);
    try
        [sig, fsFile] = readCgxCsv(fp, EEG_CH);
    catch ME
        L('  !! %s: %s', ids{i}, ME.message); continue;
    end
    fsEst(end+1) = fsFile; %#ok<AGROW>
    fsUse = NOMINAL_FS; if abs(fsFile - NOMINAL_FS) / NOMINAL_FS > 0.25, L('  ?? %s: timestamp-derived fs %.0f Hz differs from nominal %d; using nominal', ids{i}, fsFile, NOMINAL_FS); end
    sig = sig - mean(sig, 1);                                     % DC removal
    [bT, aT] = butter(2, [1 45] / (fsUse / 2), 'bandpass'); tmp = filtfilt(bT, aT, sig);
    chStd = 1.4826 * median(abs(tmp - median(tmp, 1)), 1); md = median(chStd);
    badCh = chStd > BAD_HI * md | chStd < BAD_LO * md | chStd < 1e-3;
    if nnz(badCh) > MAX_BAD_CH, L('  -- %s: %d bad channels (%s), skipped', ids{i}, nnz(badCh), strjoin(EEG_CH(badCh), ' ')); continue; end
    sig = sig - mean(sig(:, ~badCh), 2);                           % common average reference over good channels
    sig(:, badCh) = NaN;
    [ep, info] = preprocessToEpochs(sig, fsUse, FS_TARGET, EPOCH_SEC, struct('bBP', bBP, 'aBP', aBP, 'bN', bN, 'aN', aN), MAD_K, ABS_UV_LIMIT, REJECT_CH_FRACTION, MAX_EPOCHS_PER_SUBJECT, badCh);
    switch diag{i}
        case 'yes',  si = si + 1;  S  = appendEpochs(S,  ep, si,  ids{i}, 1, ageV(i), sexV(i));
        case 'no',   si = si + 1;  S  = appendEpochs(S,  ep, si,  ids{i}, 0, ageV(i), sexV(i));
        otherwise,   siu = siu + 1; Su = appendEpochs(Su, ep, siu, ids{i}, 1, ageV(i), sexV(i));
    end
    L('  %-7s %-12s age=%2d sex=%d  dur=%6.1fs  badCh=%d%s  total=%3d rejected=%3d kept=%3d', ids{i}, diag{i}, ageV(i), sexV(i), size(sig, 1) / fsUse, ...
        nnz(badCh), ifelse(any(badCh), [' (' strjoin(EEG_CH(badCh), ' ') ')'], ''), info.nTotal, info.nRejected, info.nKept);
end
L('timestamp-derived sampling rate across files: median %.1f Hz (min %.1f, max %.1f); nominal %d Hz used', median(fsEst), min(fsEst), max(fsEst), NOMINAL_FS);
S = finalizeStore(S); Su = finalizeStore(Su);
saveStore(S, fullfile(OUT, 'D4_BALLADEER_epochs.mat'), L); saveStore(Su, fullfile(OUT, 'D4u_BALLADEER_undetermined_epochs.mat'), L);
[~, ia] = unique(S.subject);
L('\nD4 BALLADEER: subjects=%d (ADHD %d / HC %d) epochs=%d | ADHD mean age %.1f male %.0f%% | HC mean age %.1f male %.0f%%', numel(ia), ...
    nnz(S.label(ia) == 1), nnz(S.label(ia) == 0), numel(S.label), mean(S.age(ia(S.label(ia) == 1))), 100 * mean(S.sex(ia(S.label(ia) == 1))), ...
    mean(S.age(ia(S.label(ia) == 0))), 100 * mean(S.sex(ia(S.label(ia) == 0))));
L('Undetermined store: %d subjects', numel(unique(Su.subject)));
L('\nDONE. Send Stage1c_Report.txt.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
function [sig, fsEst] = readCgxCsv(fp, wantCh)
    f = fopen(fp, 'r'); c = onCleanup(@() fclose(f));
    hdr = strsplit(strtrim(fgetl(f)), ','); nCol = numel(hdr);
    names = regexprep(hdr, '\(.*\)$', ''); names = strtrim(names);
    C = textscan(f, repmat('%f', 1, nCol), 'Delimiter', ',', 'CollectOutput', true, 'EmptyValue', NaN);
    M = C{1}; if size(M, 2) ~= nCol, error('column count mismatch'); end
    idx = zeros(1, numel(wantCh));
    for k = 1:numel(wantCh), j = find(strcmpi(names, wantCh{k}), 1); if isempty(j), error('channel %s not found', wantCh{k}); end; idx(k) = j; end
    sig = M(:, idx); ok = all(isfinite(sig), 2); sig = sig(ok, :);
    t = M(ok, find(strcmpi(names, 'timestamps'), 1)); fsEst = (numel(t) - 1) / max(t(end) - t(1), eps);
end
function S = newStore(name, fs, epochSec, channels, preproc)
    S = struct('dataset', name, 'fs', fs, 'epochLen', epochSec, 'channels', {channels}, 'preproc', preproc, ...
        'X', zeros(0, fs * epochSec, numel(channels), 'single'), 'subject', [], 'subjectID', strings(0, 1), 'label', [], 'age', [], 'sex', []);
end
function S = appendEpochs(S, ep, si, sid, label, age, sex)
    n = size(ep, 1); if n == 0, return; end
    S.X = cat(1, S.X, single(ep)); S.subject = [S.subject; repmat(si, n, 1)]; S.subjectID = [S.subjectID; repmat(string(sid), n, 1)];
    S.label = [S.label; repmat(label, n, 1)]; S.age = [S.age; repmat(age, n, 1)]; S.sex = [S.sex; repmat(sex, n, 1)];
end
function S = finalizeStore(S), if isempty(S.subject), S.epochsPerSubject = []; return; end; [~, ~, k] = unique(S.subject); S.epochsPerSubject = accumarray(k, 1); end
function saveStore(S, path, L), save(path, 'S', '-v7.3'); L('  saved %s  (X: %d x %d x %d)', path, size(S.X, 1), size(S.X, 2), size(S.X, 3)); end
function [ep, info] = preprocessToEpochs(sig, fsIn, fsOut, epochSec, filt, madK, absLimit, chFrac, maxEpochs, badCh)
    sig = double(sig); good = ~badCh; sig(:, badCh) = 0;           % filter/resample on zeros, restore NaN after
    sig = sig - mean(sig, 1);
    if fsIn ~= fsOut, [p, q] = rat(fsOut / fsIn); sig = resample(sig, p, q); end
    sig = filtfilt(filt.bBP, filt.aBP, sig); sig = filtfilt(filt.bN, filt.aN, sig);
    n = fsOut * epochSec; nEp = floor(size(sig, 1) / n); nCh = size(sig, 2); ep = zeros(nEp, n, nCh);
    for e = 1:nEp, ep(e, :, :) = reshape(sig((e - 1) * n + (1:n), :), [1 n nCh]); end
    ptp = squeeze(max(ep, [], 2) - min(ep, [], 2)); if nEp == 1, ptp = reshape(ptp, 1, []); end
    flag = false(nEp, nCh);
    for c = find(good), v = ptp(:, c); md = median(v); mad_ = 1.4826 * median(abs(v - md)); flag(:, c) = v > md + madK * max(mad_, eps) | v < 1e-6; end
    bad = sum(flag, 2) / max(nnz(good), 1) > chFrac;
    if ~isnan(absLimit), mx = squeeze(max(abs(ep), [], 2)); if nEp == 1, mx = reshape(mx, 1, []); end; bad = bad | any(mx(:, good) > absLimit, 2); end
    ep(:, :, badCh) = NaN;
    info.nTotal = nEp; info.nRejected = nnz(bad); ep = ep(~bad, :, :); if size(ep, 1) > maxEpochs, ep = ep(1:maxEpochs, :, :); end
    info.nKept = size(ep, 1);
end
