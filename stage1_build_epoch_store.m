%% stage1_build_epoch_store.m
%  STAGE 1 - Unified loader + identical preprocessing for the three ADHD EEG
%  datasets. Produces one standardized epoch store per dataset:
%
%     <OUT>\D1_IEEE_epochs.mat      children, visual attention task
%     <OUT>\D2_TDBRAIN_epochs.mat   clinic cohort, eyes-closed rest, session 1
%     <OUT>\D3_MENDELEY_epochs.mat  adults, cognitive challenge (cell 4)
%     <OUT>\Stage1_Report.txt       per-subject epoch counts, rejections, checks
%     <OUT>\Stage1_GroupPSD.png     group-mean Welch PSD, Cz and F4, per dataset
%
%  Each store contains a struct S with:
%     S.X          [nEpochs x 256 x 2] double   (2 s epochs at 128 Hz; ch1 = Cz, ch2 = F4)
%     S.subject    [nEpochs x 1] double          numeric subject index within the dataset
%     S.subjectID  [nEpochs x 1] string          original identifier (file / row / cell index)
%     S.label      [nEpochs x 1] double          1 = ADHD, 0 = HC
%     S.age, S.sex [nEpochs x 1] double          NaN where unknown (sex: 1 = male, 0 = female)
%     S.dataset    char                          'IEEE' | 'TDBRAIN' | 'MENDELEY'
%     S.fs, S.epochLen, S.channels, S.preproc    provenance
%
%  Preprocessing chain (identical for all datasets):
%     DC removal -> resample to 128 Hz -> 0.5-45 Hz Butterworth (order 4, zero phase)
%     -> 50 Hz IIR notch -> non-overlapping 2 s epochs -> robust epoch rejection
%     (per subject, per channel: peak-to-peak > median + 5*MAD, or flat)
%     -> optional absolute amplitude limit for datasets in microvolts
%     -> keep at most MAX_EPOCHS_PER_SUBJECT epochs (first clean ones).
%
%  No readtable / readcell anywhere. Needs Signal Processing Toolbox for
%  resample, butter, filtfilt, iirnotch, pwelch.
%
%  Run once. Takes a few minutes (TDBRAIN: 284 BDF files of 6 MB).

clear; clc; close all;

%% ------------------------------------------------------------------ CONFIG
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
OUT  = fullfile(ROOT, 'Stage1_Epochs');

D1_ROOT = fullfile(ROOT, 'Dataset1_IEEE Dataport (Iran, 61 ADHD + 60 HC)');
D2_ROOT = fullfile(ROOT, 'Dataset2_TDBRAIN_Dataset_V3_1_Encr', 'TDBRAIN_Dataset_V3_1');
D2_XLSX = fullfile(ROOT, 'Dataset2_TDBRAIN_Dataset_V3_1_Encr', 'TDBRAIN_participants_V3.xlsx');
D3_ROOT = fullfile(ROOT, 'Dataset3_MENDELEY_DATA_EEG_ADHD_GCPDS', 'MENDELEY_DATA_EEG_ADHD_GCPDS');

FS_TARGET   = 128;          % common sampling rate (Hz)
EPOCH_SEC   = 2;            % epoch length (s)
BP_BAND     = [0.5 45];     % band-pass (Hz)
NOTCH_HZ    = 50;           % mains frequency in all three countries (Iran, NL, Iran)
MAX_EPOCHS_PER_SUBJECT = 60; % cap so long recordings do not dominate (TDBRAIN rest = 60)
MAD_K       = 5;            % robust rejection: ptp > median + MAD_K * MAD
ABS_UV_LIMIT = 400;         % absolute limit for datasets known to be in microvolts (NaN = off)

% --- Dataset 1: IEEE DataPort column order. Published order (please CONFIRM
%     against Channel_Labels.docx; the .ced file in the folder uses a
%     different order and is NOT the column order).
D1_CHANNEL_ORDER = {'Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2'};
D1_FS = 128;
D1_UNITS_KNOWN_UV = false;   % raw ADC integers with offset -> use robust rule only

% --- Dataset 2: TDBRAIN
D2_TASK      = 'restEC';     % eyes-closed rest (cleanest; 120 s)
D2_SESSION   = 1;
D2_ADHD_LABELS    = {'ADHD'};       % formal_status values treated as ADHD ('ADD' left out)
D2_HEALTHY_LABELS = {'HEALTHY'};
D2_UNITS_KNOWN_UV = true;

% --- Dataset 3: Mendeley
D3_CELL      = 4;            % cognitive challenge, Cz/F4, 45 s
D3_FS        = 256;
D3_EXCLUDE   = struct('FADHD', 7);  % corrupted subject per dataset notes
D3_UNITS_KNOWN_UV = true;

%% ------------------------------------------------------------------ SETUP
if ~isfolder(OUT), mkdir(OUT); end
logPath = fullfile(OUT, 'Stage1_Report.txt');
fidLog = fopen(logPath, 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});

L('STAGE 1 - EPOCH STORE BUILD   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('Target fs = %d Hz, epoch = %d s (%d samples), band = %g-%g Hz, notch = %d Hz', ...
    FS_TARGET, EPOCH_SEC, FS_TARGET * EPOCH_SEC, BP_BAND(1), BP_BAND(2), NOTCH_HZ);
L('Rejection: ptp > median + %g*MAD per subject/channel; abs limit %g uV where units known; cap %d epochs/subject', ...
    MAD_K, ABS_UV_LIMIT, MAX_EPOCHS_PER_SUBJECT);

% filters designed once at the target rate
[bBP, aBP] = butter(4, BP_BAND / (FS_TARGET / 2), 'bandpass');
[bN,  aN ] = iirnotch(NOTCH_HZ / (FS_TARGET / 2), (NOTCH_HZ / (FS_TARGET / 2)) / 35);
filt = struct('bBP', bBP, 'aBP', aBP, 'bN', bN, 'aN', aN);

preprocDesc = sprintf('DC->resample %dHz->BP %g-%g Hz butter4 filtfilt->notch %dHz->%ds epochs->MAD%g reject->cap %d', ...
    FS_TARGET, BP_BAND(1), BP_BAND(2), NOTCH_HZ, EPOCH_SEC, MAD_K, MAX_EPOCHS_PER_SUBJECT);

%% ================================================================ DATASET 1
L('\n=============================== DATASET 1 : IEEE DataPort (children)');
iCz = find(strcmpi(D1_CHANNEL_ORDER, 'Cz')); iF4 = find(strcmpi(D1_CHANNEL_ORDER, 'F4'));
L('Assumed column order: %s', strjoin(D1_CHANNEL_ORDER, ' '));
L('Using column %d (Cz) and column %d (F4).  ** CONFIRM against Channel_Labels.docx **', iCz, iF4);

groups = {'ADHD_part1', 1; 'ADHD_part2', 1; 'Control_part1', 0; 'Control_part2', 0};
S1 = newStore('IEEE', FS_TARGET, EPOCH_SEC, preprocDesc);
subjIdx = 0;
for g = 1:size(groups, 1)
    d = dir(fullfile(D1_ROOT, groups{g, 1}, '**', '*.mat'));
    for k = 1:numel(d)
        fp = fullfile(d(k).folder, d(k).name);
        T = load(fp); fn = fieldnames(T); x = T.(fn{1});
        if size(x, 2) ~= 19, L('  !! %s has %d columns, skipped', d(k).name, size(x, 2)); continue; end
        sig = x(:, [iCz iF4]);
        subjIdx = subjIdx + 1;
        [ep, info] = preprocessToEpochs(sig, D1_FS, FS_TARGET, EPOCH_SEC, filt, MAD_K, ...
            ifelse(D1_UNITS_KNOWN_UV, ABS_UV_LIMIT, NaN), MAX_EPOCHS_PER_SUBJECT);
        S1 = appendEpochs(S1, ep, subjIdx, string(erase(d(k).name, '.mat')), groups{g, 2}, NaN, NaN);
        L('  %-22s label=%d  dur=%6.1fs  epochs total=%3d rejected=%3d kept=%3d', ...
            d(k).name, groups{g, 2}, size(x, 1) / D1_FS, info.nTotal, info.nRejected, info.nKept);
    end
end
S1 = finalizeStore(S1);
saveStore(S1, fullfile(OUT, 'D1_IEEE_epochs.mat'), L);

%% ================================================================ DATASET 2
L('\n=============================== DATASET 2 : TDBRAIN (%s, session %d)', D2_TASK, D2_SESSION);
[~, ~, raw] = xlsread(D2_XLSX, 1);
hdr = cellfun(@toStr, raw(1, :), 'UniformOutput', false);
col = @(name) find(strcmpi(hdr, name), 1);
cID = col('TDBRAIN_ID'); cStat = col('formal_status'); cSess = col('sessID'); cAge = col('age'); cSex = col('gender');
assert(~isempty(cID) && ~isempty(cStat) && ~isempty(cSess), 'participants sheet: expected columns not found');
rows = raw(2:end, :);
ids   = cellfun(@toStr, rows(:, cID),   'UniformOutput', false);
stat  = cellfun(@(x) upper(strtrim(toStr(x))), rows(:, cStat), 'UniformOutput', false);
sess  = cellfun(@toNum, rows(:, cSess));
ages  = cellfun(@toNum, rows(:, cAge));
sexes = cellfun(@toNum, rows(:, cSex));

isADHD = ismember(stat, upper(D2_ADHD_LABELS));
isHC   = ismember(stat, upper(D2_HEALTHY_LABELS));
sel = find((isADHD | isHC) & sess == D2_SESSION);
L('participants rows: %d | session-%d rows with ADHD=%d, HEALTHY=%d', numel(ids), D2_SESSION, ...
    nnz(isADHD & sess == D2_SESSION), nnz(isHC & sess == D2_SESSION));

S2 = newStore('TDBRAIN', FS_TARGET, EPOCH_SEC, preprocDesc);
subjIdx = 0; nMissing = 0; nBadHdr = 0;
for r = sel'
    sid = ids{r};
    fp = fullfile(D2_ROOT, sid, sprintf('ses-%d', D2_SESSION), 'eeg', ...
        sprintf('%s_ses-%d_task-%s_eeg.bdf', sid, D2_SESSION, D2_TASK));
    if ~isfile(fp), nMissing = nMissing + 1; L('  -- %s: no %s file, skipped', sid, D2_TASK); continue; end
    try
        [sig, fsBdf, labels] = readBdfChannels(fp, {'Cz', 'F4'});
    catch ME
        nBadHdr = nBadHdr + 1; L('  !! %s: BDF read failed (%s)', sid, ME.message); continue;
    end
    subjIdx = subjIdx + 1;
    lbl = double(isADHD(r));
    [ep, info] = preprocessToEpochs(sig, fsBdf, FS_TARGET, EPOCH_SEC, filt, MAD_K, ...
        ifelse(D2_UNITS_KNOWN_UV, ABS_UV_LIMIT, NaN), MAX_EPOCHS_PER_SUBJECT);
    S2 = appendEpochs(S2, ep, subjIdx, string(sid), lbl, ages(r), sexes(r));
    L('  %-16s %-8s age=%5.1f sex=%d fs=%d  dur=%6.1fs  epochs total=%3d rejected=%3d kept=%3d', ...
        sid, stat{r}, ages(r), sexes(r), fsBdf, size(sig, 1) / fsBdf, info.nTotal, info.nRejected, info.nKept);
end
L('TDBRAIN: %d subjects loaded, %d without %s file, %d unreadable', subjIdx, nMissing, D2_TASK, nBadHdr);
S2 = finalizeStore(S2);
saveStore(S2, fullfile(OUT, 'D2_TDBRAIN_epochs.mat'), L);

%% ================================================================ DATASET 3
L('\n=============================== DATASET 3 : Mendeley adults (cell %d)', D3_CELL);
files3 = {'FADHD', 1, 0; 'MADHD', 1, 1; 'FC', 0, 0; 'MC', 0, 1};   % name, label, sex
S3 = newStore('MENDELEY', FS_TARGET, EPOCH_SEC, preprocDesc);
subjIdx = 0;
for g = 1:size(files3, 1)
    nm = files3{g, 1};
    T = load(fullfile(D3_ROOT, [nm '.mat'])); C = T.(nm);
    A = C{D3_CELL};                                  % subjects x samples x 2 (Cz, F4)
    L('  %s cell %d: %d subjects x %d samples x %d ch (%.1f s at %d Hz)', nm, D3_CELL, size(A, 1), size(A, 2), size(A, 3), size(A, 2) / D3_FS, D3_FS);
    for s = 1:size(A, 1)
        if isfield(D3_EXCLUDE, nm) && any(D3_EXCLUDE.(nm) == s)
            L('  -- %s subject %d excluded (documented as corrupted)', nm, s); continue;
        end
        sig = squeeze(A(s, :, :));                   % samples x 2
        if size(sig, 2) ~= 2, sig = sig'; end
        subjIdx = subjIdx + 1;
        [ep, info] = preprocessToEpochs(sig, D3_FS, FS_TARGET, EPOCH_SEC, filt, MAD_K, ...
            ifelse(D3_UNITS_KNOWN_UV, ABS_UV_LIMIT, NaN), MAX_EPOCHS_PER_SUBJECT);
        S3 = appendEpochs(S3, ep, subjIdx, string(sprintf('%s_%02d', nm, s)), files3{g, 2}, NaN, files3{g, 3});
        L('  %-10s label=%d sex=%d  epochs total=%3d rejected=%3d kept=%3d', ...
            sprintf('%s_%02d', nm, s), files3{g, 2}, files3{g, 3}, info.nTotal, info.nRejected, info.nKept);
    end
end
S3 = finalizeStore(S3);
saveStore(S3, fullfile(OUT, 'D3_MENDELEY_epochs.mat'), L);

%% ================================================================ SUMMARY + PSD CHECK
L('\n=============================== SUMMARY');
stores = {S1, S2, S3};
fig = figure('Color', 'w', 'Position', [100 100 1100 900]);
for d = 1:3
    S = stores{d};
    nS = numel(unique(S.subject));
    nA = numel(unique(S.subject(S.label == 1))); nH = numel(unique(S.subject(S.label == 0)));
    L('%-9s subjects=%3d (ADHD %3d / HC %3d)  epochs=%5d (ADHD %5d / HC %5d)  epochs/subject: min %d, median %g, max %d', ...
        S.dataset, nS, nA, nH, size(S.X, 1), nnz(S.label == 1), nnz(S.label == 0), ...
        min(S.epochsPerSubject), median(S.epochsPerSubject), max(S.epochsPerSubject));
    few = S.subjectsWithFewEpochs;
    if ~isempty(few), L('    subjects with < 10 epochs: %s', strjoin(string(few), ', ')); end
    % group-mean PSD
    for ch = 1:2
        subplot(3, 2, (d - 1) * 2 + ch); hold on;
        for lbl = [0 1]
            E = S.X(S.label == lbl, :, ch);
            [P, f] = pwelch(E', hamming(128), 64, 256, S.fs);   % columns = epochs
            Pm = mean(P, 2);
            plot(f, 10 * log10(Pm), 'LineWidth', 1.4);
        end
        xlim([0 45]); grid on; xlabel('Hz'); ylabel('dB');
        title(sprintf('%s - %s', S.dataset, S.channels{ch}));
        legend({'HC', 'ADHD'}, 'Location', 'northeast');
    end
end
saveas(fig, fullfile(OUT, 'Stage1_GroupPSD.png'));
savefig(fig, fullfile(OUT, 'Stage1_GroupPSD.fig'));
L('PSD figure written: %s', fullfile(OUT, 'Stage1_GroupPSD.png'));
L('\nChecks to make by eye on the PSD figure:');
L('  - TDBRAIN (eyes closed) must show a clear alpha peak near 8-12 Hz in both channels.');
L('  - No spike at 50 Hz in any panel (notch working); smooth roll-off above 45 Hz.');
L('  - IEEE/Mendeley (task) alpha is weaker but 1/f shape and no line noise expected.');
L('\nDONE. Send Stage1_Report.txt and Stage1_GroupPSD.png.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin)
    s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s);
end

function out = ifelse(c, a, b)
    if c, out = a; else, out = b; end
end

function v = toNum(x)
    % robust cell -> double for xlsread 'raw' cells (numbers, text, [], NaN)
    if isnumeric(x) || islogical(x)
        if isempty(x), v = NaN; else, v = double(x(1)); end
    elseif ischar(x) || isstring(x)
        v = str2double(strtrim(char(x))); if isempty(v), v = NaN; end
    else
        v = NaN;
    end
end

function s = toStr(x)
    % robust cell -> char for xlsread 'raw' cells (char, string, number, [], NaN)
    if isempty(x)
        s = '';
    elseif ischar(x)
        s = strtrim(x);
    elseif isstring(x)
        s = strtrim(char(x(1)));
    elseif isnumeric(x) || islogical(x)
        if all(isnan(double(x(:)))), s = ''; else, s = num2str(x(1)); end
    else
        s = '';
    end
end

function S = newStore(name, fs, epochSec, preproc)
    S = struct('dataset', name, 'fs', fs, 'epochLen', epochSec, 'channels', {{'Cz', 'F4'}}, ...
        'preproc', preproc, 'X', [], 'subject', [], 'subjectID', strings(0, 1), 'label', [], 'age', [], 'sex', []);
    S.X = zeros(0, fs * epochSec, 2);
end

function S = appendEpochs(S, ep, subjIdx, subjID, label, age, sex)
    n = size(ep, 1); if n == 0, return; end
    S.X = cat(1, S.X, ep);
    S.subject   = [S.subject;   repmat(subjIdx, n, 1)];
    S.subjectID = [S.subjectID; repmat(string(subjID), n, 1)];
    S.label     = [S.label;     repmat(label, n, 1)];
    S.age       = [S.age;       repmat(age, n, 1)];
    S.sex       = [S.sex;       repmat(sex, n, 1)];
end

function S = finalizeStore(S)
    [u, ~, k] = unique(S.subject);
    S.epochsPerSubject = accumarray(k, 1);
    S.subjectsWithFewEpochs = u(S.epochsPerSubject < 10);
    S.subjectLabel = accumarray(k, S.label, [], @(v) v(1));
end

function saveStore(S, path, L)
    save(path, 'S', '-v7.3');
    L('  saved %s  (X: %d x %d x %d, %.1f MB)', path, size(S.X, 1), size(S.X, 2), size(S.X, 3), numel(S.X) * 8 / 1e6);
end

% --------------------------------------------------------------------------
function [ep, info] = preprocessToEpochs(sig, fsIn, fsOut, epochSec, filt, madK, absLimit, maxEpochs)
    % sig: samples x 2 (Cz, F4) at fsIn. Returns ep: nEpochs x (fsOut*epochSec) x 2.
    sig = double(sig);
    sig = sig - mean(sig, 1);                                   % DC removal
    if fsIn ~= fsOut
        [p, q] = rat(fsOut / fsIn);
        sig = resample(sig, p, q);                              % anti-aliased
    end
    sig = filtfilt(filt.bBP, filt.aBP, sig);                    % band-pass
    sig = filtfilt(filt.bN,  filt.aN,  sig);                    % notch
    n = fsOut * epochSec;
    nEp = floor(size(sig, 1) / n);
    ep = zeros(nEp, n, 2);
    for e = 1:nEp
        ep(e, :, :) = reshape(sig((e - 1) * n + (1:n), :), [1 n 2]);
    end
    % robust rejection per channel
    ptp = squeeze(max(ep, [], 2) - min(ep, [], 2));             % nEp x 2
    if nEp == 1, ptp = ptp(:)'; end
    bad = false(nEp, 1);
    for ch = 1:2
        v = ptp(:, ch);
        med = median(v); madv = 1.4826 * median(abs(v - med));
        bad = bad | v > med + madK * max(madv, eps) | v < 1e-6;
    end
    if ~isnan(absLimit)
        bad = bad | any(max(abs(ep), [], 2) > absLimit, 3);
    end
    info.nTotal = nEp; info.nRejected = nnz(bad);
    ep = ep(~bad, :, :);
    if size(ep, 1) > maxEpochs, ep = ep(1:maxEpochs, :, :); end
    info.nKept = size(ep, 1);
end

% --------------------------------------------------------------------------
function [sig, fs, labelsOut] = readBdfChannels(fp, wantLabels)
    % Minimal BioSemi BDF (24-bit) reader. Returns samples x numel(wantLabels), in
    % physical units (uV), plus sampling rate (all requested channels must share it).
    f = fopen(fp, 'r', 'ieee-le'); c = onCleanup(@() fclose(f));
    hdr = fread(f, 256, 'uint8=>uint8')';
    if ~(hdr(1) == 255 && strcmp(char(hdr(2:8)), 'BIOSEMI')), error('not a BDF header'); end
    h = char(hdr);
    headerBytes = str2double(h(185:192));
    nRec = str2double(h(237:244)); durRec = str2double(h(245:252)); nSig = str2double(h(253:256));
    sh = fread(f, nSig * 256, 'uint8=>char')';
    labels = strings(1, nSig); physMin = zeros(1, nSig); physMax = physMin; digMin = physMin; digMax = physMin; nSamp = physMin;
    off = 0;
    for k = 1:nSig, labels(k) = strtrim(sh(off + (k-1)*16 + (1:16))); end;            off = off + nSig * 16;
    off = off + nSig * 80;                                                              % transducer
    off = off + nSig * 8;                                                               % physical dimension
    for k = 1:nSig, physMin(k) = str2double(sh(off + (k-1)*8 + (1:8))); end;         off = off + nSig * 8;
    for k = 1:nSig, physMax(k) = str2double(sh(off + (k-1)*8 + (1:8))); end;         off = off + nSig * 8;
    for k = 1:nSig, digMin(k)  = str2double(sh(off + (k-1)*8 + (1:8))); end;         off = off + nSig * 8;
    for k = 1:nSig, digMax(k)  = str2double(sh(off + (k-1)*8 + (1:8))); end;         off = off + nSig * 8;
    off = off + nSig * 80;                                                              % prefiltering
    for k = 1:nSig, nSamp(k)   = str2double(sh(off + (k-1)*8 + (1:8))); end
    if nRec < 0   % unknown record count: derive from file size
        d = dir(fp); nRec = floor((d.bytes - headerBytes) / (3 * sum(nSamp)));
    end
    idx = zeros(1, numel(wantLabels));
    for w = 1:numel(wantLabels)
        m = find(strcmpi(labels, wantLabels{w}), 1);
        if isempty(m), error('channel %s not found', wantLabels{w}); end
        idx(w) = m;
    end
    fsAll = nSamp / durRec;
    if numel(unique(fsAll(idx))) ~= 1, error('requested channels have different sampling rates'); end
    fs = fsAll(idx(1));
    % read all records at once
    fseek(f, headerBytes, 'bof');
    S = sum(nSamp);
    raw = fread(f, [3, nRec * S], 'uint8=>double');
    if size(raw, 2) < nRec * S, nRec = floor(size(raw, 2) / S); raw = raw(:, 1:nRec * S); end
    val = raw(1, :) + 256 * raw(2, :) + 65536 * raw(3, :);
    val(val >= 2^23) = val(val >= 2^23) - 2^24;
    val = reshape(val, S, nRec);                                % samples-in-record x records
    starts = [0 cumsum(nSamp(1:end-1))];
    sig = zeros(nSamp(idx(1)) * nRec, numel(idx));
    for w = 1:numel(idx)
        k = idx(w);
        block = val(starts(k) + (1:nSamp(k)), :);               % nSamp x nRec
        gain = (physMax(k) - physMin(k)) / (digMax(k) - digMin(k));
        sig(:, w) = (block(:) - digMin(k)) * gain + physMin(k);
    end
    labelsOut = labels(idx);
end
