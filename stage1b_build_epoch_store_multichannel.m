%% stage1b_build_epoch_store_multichannel.m
%  STAGE 1b - Multichannel epoch stores (same preprocessing chain as Stage 1).
%
%     <OUT>\D1_IEEE_epochs.mat      children, all 19 channels
%     <OUT>\D2_TDBRAIN_epochs.mat   adults (>= 18 y) ADHD + HEALTHY, session 1, restEC, 26 EEG channels
%     <OUT>\D3_MENDELEY_epochs.mat  adults, cognitive challenge, Cz + F4
%     <OUT>\Stage1b_Report.txt
%
%  S.X is [nEpochs x 256 x nCh] single; S.channels lists the channel labels.
%  Multichannel epoch rejection: an epoch is dropped when more than
%  REJECT_CH_FRACTION of its channels fail the robust peak-to-peak rule, or any
%  channel exceeds ABS_UV_LIMIT (microvolt datasets only).

clear; clc; close all;

%% ------------------------------------------------------------------ CONFIG
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
OUT  = fullfile(ROOT, 'Stage1b_Epochs');
D1_ROOT = fullfile(ROOT, 'Dataset1_IEEE Dataport (Iran, 61 ADHD + 60 HC)');
D2_ROOT = fullfile(ROOT, 'Dataset2_TDBRAIN_Dataset_V3_1_Encr', 'TDBRAIN_Dataset_V3_1');
D2_XLSX = fullfile(ROOT, 'Dataset2_TDBRAIN_Dataset_V3_1_Encr', 'TDBRAIN_participants_V3.xlsx');
D3_ROOT = fullfile(ROOT, 'Dataset3_MENDELEY_DATA_EEG_ADHD_GCPDS', 'MENDELEY_DATA_EEG_ADHD_GCPDS');

FS_TARGET = 128; EPOCH_SEC = 2; BP_BAND = [0.5 45]; NOTCH_HZ = 50;
MAX_EPOCHS_PER_SUBJECT = 60; MAD_K = 5; ABS_UV_LIMIT = 400; REJECT_CH_FRACTION = 0.15;

D1_CHANNELS = {'Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2'};  % column order (IEEE DataPort)
D1_FS = 128;
D2_CHANNELS = {'Fp1','Fp2','F7','F3','Fz','F4','F8','FC3','FCz','FC4','T7','C3','Cz','C4','T8','CP3','CPz','CP4','P7','P3','Pz','P4','P8','O1','Oz','O2'};
D2_TASK = 'restEC'; D2_SESSION = 1; D2_MIN_AGE = 18;
D3_CHANNELS = {'Cz','F4'}; D3_CELL = 4; D3_FS = 256; D3_EXCLUDE_FADHD = 7;

%% ------------------------------------------------------------------ SETUP
if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage1b_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 1b - MULTICHANNEL EPOCH STORES   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
[bBP, aBP] = butter(4, BP_BAND / (FS_TARGET / 2), 'bandpass');
[bN, aN] = iirnotch(NOTCH_HZ / (FS_TARGET / 2), (NOTCH_HZ / (FS_TARGET / 2)) / 35);
filt = struct('bBP', bBP, 'aBP', aBP, 'bN', bN, 'aN', aN);
pre = sprintf('DC->resample %dHz->BP %g-%g->notch %d->%ds epochs->MAD%g (>%.0f%% ch) reject->cap %d', ...
    FS_TARGET, BP_BAND(1), BP_BAND(2), NOTCH_HZ, EPOCH_SEC, MAD_K, 100 * REJECT_CH_FRACTION, MAX_EPOCHS_PER_SUBJECT);

%% ================================================================ D1 IEEE
L('\n=== D1 IEEE, %d channels', numel(D1_CHANNELS));
S1 = newStore('IEEE', FS_TARGET, EPOCH_SEC, D1_CHANNELS, pre);
groups = {'ADHD_part1', 1; 'ADHD_part2', 1; 'Control_part1', 0; 'Control_part2', 0};
si = 0;
for g = 1:size(groups, 1)
    d = dir(fullfile(D1_ROOT, groups{g, 1}, '**', '*.mat'));
    for k = 1:numel(d)
        T = load(fullfile(d(k).folder, d(k).name)); fn = fieldnames(T); x = T.(fn{1});
        if size(x, 2) ~= 19, L('  !! %s: %d columns, skipped', d(k).name, size(x, 2)); continue; end
        si = si + 1;
        [ep, info] = preprocessToEpochs(x, D1_FS, FS_TARGET, EPOCH_SEC, filt, MAD_K, NaN, REJECT_CH_FRACTION, MAX_EPOCHS_PER_SUBJECT);
        S1 = appendEpochs(S1, ep, si, string(erase(d(k).name, '.mat')), groups{g, 2}, NaN, NaN);
        L('  %-10s label=%d  total=%3d rejected=%3d kept=%3d', d(k).name, groups{g, 2}, info.nTotal, info.nRejected, info.nKept);
    end
end
S1 = finalizeStore(S1); saveStore(S1, fullfile(OUT, 'D1_IEEE_epochs.mat'), L);

%% ================================================================ D2 TDBRAIN
L('\n=== D2 TDBRAIN adults >= %d, %s, %d EEG channels', D2_MIN_AGE, D2_TASK, numel(D2_CHANNELS));
[~, ~, raw] = xlsread(D2_XLSX, 1);
hdr = cellfun(@toStr, raw(1, :), 'UniformOutput', false);
col = @(n) find(strcmpi(hdr, n), 1);
rows = raw(2:end, :);
ids = cellfun(@toStr, rows(:, col('TDBRAIN_ID')), 'UniformOutput', false);
stat = cellfun(@(x) upper(strtrim(toStr(x))), rows(:, col('formal_status')), 'UniformOutput', false);
sess = cellfun(@toNum, rows(:, col('sessID'))); ages = cellfun(@toNum, rows(:, col('age'))); sexes = cellfun(@toNum, rows(:, col('gender')));
isA = strcmp(stat, 'ADHD'); isH = strcmp(stat, 'HEALTHY');
sel = find((isA | isH) & sess == D2_SESSION & ages >= D2_MIN_AGE);
L('  candidates: ADHD %d, HEALTHY %d', nnz(isA(sel)), nnz(isH(sel)));
S2 = newStore('TDBRAIN', FS_TARGET, EPOCH_SEC, D2_CHANNELS, pre);
si = 0; nMiss = 0;
for r = sel'
    fp = fullfile(D2_ROOT, ids{r}, sprintf('ses-%d', D2_SESSION), 'eeg', sprintf('%s_ses-%d_task-%s_eeg.bdf', ids{r}, D2_SESSION, D2_TASK));
    if ~isfile(fp), nMiss = nMiss + 1; continue; end
    try
        [sig, fsB] = readBdfChannels(fp, D2_CHANNELS);
    catch ME
        L('  !! %s: %s', ids{r}, ME.message); continue;
    end
    si = si + 1;
    [ep, info] = preprocessToEpochs(sig, fsB, FS_TARGET, EPOCH_SEC, filt, MAD_K, ABS_UV_LIMIT, REJECT_CH_FRACTION, MAX_EPOCHS_PER_SUBJECT);
    S2 = appendEpochs(S2, ep, si, string(ids{r}), double(isA(r)), ages(r), sexes(r));
    L('  %-14s %-7s age=%5.1f sex=%d  total=%3d rejected=%3d kept=%3d', ids{r}, stat{r}, ages(r), sexes(r), info.nTotal, info.nRejected, info.nKept);
end
L('  loaded %d subjects, %d without %s file', si, nMiss, D2_TASK);
S2 = finalizeStore(S2); saveStore(S2, fullfile(OUT, 'D2_TDBRAIN_epochs.mat'), L);

%% ================================================================ D3 MENDELEY
L('\n=== D3 MENDELEY cell %d, %d channels', D3_CELL, numel(D3_CHANNELS));
S3 = newStore('MENDELEY', FS_TARGET, EPOCH_SEC, D3_CHANNELS, pre);
files3 = {'FADHD', 1, 0; 'MADHD', 1, 1; 'FC', 0, 0; 'MC', 0, 1};
si = 0;
for g = 1:size(files3, 1)
    nm = files3{g, 1}; T = load(fullfile(D3_ROOT, [nm '.mat'])); A = T.(nm){D3_CELL};
    for s = 1:size(A, 1)
        if strcmp(nm, 'FADHD') && s == D3_EXCLUDE_FADHD, L('  -- FADHD subject %d excluded', s); continue; end
        sig = squeeze(A(s, :, :)); if size(sig, 2) ~= 2, sig = sig'; end
        si = si + 1;
        [ep, info] = preprocessToEpochs(sig, D3_FS, FS_TARGET, EPOCH_SEC, filt, MAD_K, ABS_UV_LIMIT, REJECT_CH_FRACTION, MAX_EPOCHS_PER_SUBJECT);
        S3 = appendEpochs(S3, ep, si, string(sprintf('%s_%02d', nm, s)), files3{g, 2}, NaN, files3{g, 3});
        L('  %-10s label=%d  total=%3d rejected=%3d kept=%3d', sprintf('%s_%02d', nm, s), files3{g, 2}, info.nTotal, info.nRejected, info.nKept);
    end
end
S3 = finalizeStore(S3); saveStore(S3, fullfile(OUT, 'D3_MENDELEY_epochs.mat'), L);

%% ================================================================ SUMMARY
L('\n=== SUMMARY');
for S = {S1, S2, S3}
    S = S{1}; [~, ia] = unique(S.subject);
    L('%-9s subjects=%3d (ADHD %3d / HC %3d) epochs=%5d channels=%d  epochs/subject min %d median %g max %d', ...
        S.dataset, numel(ia), nnz(S.label(ia) == 1), nnz(S.label(ia) == 0), numel(S.label), numel(S.channels), ...
        min(S.epochsPerSubject), median(S.epochsPerSubject), max(S.epochsPerSubject));
end
L('\nDONE. Send Stage1b_Report.txt.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function v = toNum(x)
    if isnumeric(x) || islogical(x), if isempty(x), v = NaN; else, v = double(x(1)); end
    elseif ischar(x) || isstring(x), v = str2double(strtrim(char(x))); if isempty(v), v = NaN; end
    else, v = NaN; end
end
function s = toStr(x)
    if isempty(x), s = ''; elseif ischar(x), s = strtrim(x); elseif isstring(x), s = strtrim(char(x(1)));
    elseif isnumeric(x) || islogical(x), if all(isnan(double(x(:)))), s = ''; else, s = num2str(x(1)); end
    else, s = ''; end
end
function S = newStore(name, fs, epochSec, channels, preproc)
    S = struct('dataset', name, 'fs', fs, 'epochLen', epochSec, 'channels', {channels}, 'preproc', preproc, ...
        'X', zeros(0, fs * epochSec, numel(channels), 'single'), 'subject', [], 'subjectID', strings(0, 1), 'label', [], 'age', [], 'sex', []);
end
function S = appendEpochs(S, ep, si, sid, label, age, sex)
    n = size(ep, 1); if n == 0, return; end
    S.X = cat(1, S.X, single(ep));
    S.subject = [S.subject; repmat(si, n, 1)]; S.subjectID = [S.subjectID; repmat(string(sid), n, 1)];
    S.label = [S.label; repmat(label, n, 1)]; S.age = [S.age; repmat(age, n, 1)]; S.sex = [S.sex; repmat(sex, n, 1)];
end
function S = finalizeStore(S)
    [~, ~, k] = unique(S.subject); S.epochsPerSubject = accumarray(k, 1);
end
function saveStore(S, path, L)
    save(path, 'S', '-v7.3');
    L('  saved %s  (X: %d x %d x %d single, %.0f MB)', path, size(S.X, 1), size(S.X, 2), size(S.X, 3), numel(S.X) * 4 / 1e6);
end
function [ep, info] = preprocessToEpochs(sig, fsIn, fsOut, epochSec, filt, madK, absLimit, chFrac, maxEpochs)
    sig = double(sig); sig = sig - mean(sig, 1);
    if fsIn ~= fsOut, [p, q] = rat(fsOut / fsIn); sig = resample(sig, p, q); end
    sig = filtfilt(filt.bBP, filt.aBP, sig); sig = filtfilt(filt.bN, filt.aN, sig);
    n = fsOut * epochSec; nEp = floor(size(sig, 1) / n); nCh = size(sig, 2);
    ep = zeros(nEp, n, nCh);
    for e = 1:nEp, ep(e, :, :) = reshape(sig((e - 1) * n + (1:n), :), [1 n nCh]); end
    ptp = squeeze(max(ep, [], 2) - min(ep, [], 2)); if nEp == 1, ptp = reshape(ptp, 1, []); end
    flag = false(nEp, nCh);
    for c = 1:nCh
        v = ptp(:, c); med = median(v); madv = 1.4826 * median(abs(v - med));
        flag(:, c) = v > med + madK * max(madv, eps) | v < 1e-6;
    end
    bad = mean(flag, 2) > chFrac;
    if ~isnan(absLimit), bad = bad | any(squeeze(max(abs(ep), [], 2)) > absLimit, 2); end
    info.nTotal = nEp; info.nRejected = nnz(bad);
    ep = ep(~bad, :, :); if size(ep, 1) > maxEpochs, ep = ep(1:maxEpochs, :, :); end
    info.nKept = size(ep, 1);
end
function [sig, fs] = readBdfChannels(fp, wantLabels)
    f = fopen(fp, 'r', 'ieee-le'); c = onCleanup(@() fclose(f));
    hdr = fread(f, 256, 'uint8=>uint8')';
    if ~(hdr(1) == 255 && strcmp(char(hdr(2:8)), 'BIOSEMI')), error('not a BDF header'); end
    h = char(hdr); headerBytes = str2double(h(185:192)); nRec = str2double(h(237:244)); durRec = str2double(h(245:252)); nSig = str2double(h(253:256));
    sh = fread(f, nSig * 256, 'uint8=>char')';
    labels = strings(1, nSig); physMin = zeros(1, nSig); physMax = physMin; digMin = physMin; digMax = physMin; nSamp = physMin; off = 0;
    for k = 1:nSig, labels(k) = strtrim(sh(off + (k-1)*16 + (1:16))); end; off = off + nSig * 16 + nSig * 80 + nSig * 8;
    for k = 1:nSig, physMin(k) = str2double(sh(off + (k-1)*8 + (1:8))); end; off = off + nSig * 8;
    for k = 1:nSig, physMax(k) = str2double(sh(off + (k-1)*8 + (1:8))); end; off = off + nSig * 8;
    for k = 1:nSig, digMin(k) = str2double(sh(off + (k-1)*8 + (1:8))); end; off = off + nSig * 8;
    for k = 1:nSig, digMax(k) = str2double(sh(off + (k-1)*8 + (1:8))); end; off = off + nSig * 8 + nSig * 80;
    for k = 1:nSig, nSamp(k) = str2double(sh(off + (k-1)*8 + (1:8))); end
    if nRec < 0, d = dir(fp); nRec = floor((d.bytes - headerBytes) / (3 * sum(nSamp))); end
    idx = zeros(1, numel(wantLabels));
    for w = 1:numel(wantLabels), m = find(strcmpi(labels, wantLabels{w}), 1); if isempty(m), error('channel %s missing', wantLabels{w}); end; idx(w) = m; end
    fsAll = nSamp / durRec; assert(numel(unique(fsAll(idx))) == 1); fs = fsAll(idx(1));
    fseek(f, headerBytes, 'bof'); S = sum(nSamp);
    raw = fread(f, [3, nRec * S], 'uint8=>double');
    if size(raw, 2) < nRec * S, nRec = floor(size(raw, 2) / S); raw = raw(:, 1:nRec * S); end
    val = raw(1, :) + 256 * raw(2, :) + 65536 * raw(3, :); val(val >= 2^23) = val(val >= 2^23) - 2^24;
    val = reshape(val, S, nRec); starts = [0 cumsum(nSamp(1:end-1))];
    sig = zeros(nSamp(idx(1)) * nRec, numel(idx));
    for w = 1:numel(idx)
        k = idx(w); block = val(starts(k) + (1:nSamp(k)), :);
        sig(:, w) = (block(:) - digMin(k)) * (physMax(k) - physMin(k)) / (digMax(k) - digMin(k)) + physMin(k);
    end
end
