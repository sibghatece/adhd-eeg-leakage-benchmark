%% stage1d_osf_vahid_epochs.m
%  STAGE 1d - OSF / Vahid et al. 2019 (144 children, 56-channel CSD, 0.5-20 Hz, 1.5 s flanker-task trials)
%  -> common epoch store. Each trial becomes one epoch (resampled 256 -> 128 Hz: 192 samples).
%  y_stim: [subject index within group, isControl, isSubtype1(ADD), isSubtype2(ADHD-C)] - verified below.
%  Labels: controls 0; subtype1 and subtype2 -> 1; S.subtype keeps 0 / 1 / 2 for secondary analyses.
%  Channel labels: 60 locations are given but the data have 56 channels with no documented mapping,
%  so channels are named CSD01..CSD56 (within-dataset use only). Age and sex are not distributed.
%  Preprocessing on already-cleaned trials: DC removal, 0.5-45 Hz band-pass and 50 Hz notch (no-ops
%  on 0.5-20 Hz data, applied for protocol uniformity), robust per-trial rejection, cap 60 trials/subject.
%  Output: Stage1b_Epochs\D5_OSF_epochs.mat, Stage1b_Epochs\Stage1d_Report.txt

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
D5 = fullfile(ROOT, 'Dataset5_OSF_Vahid', 'data'); OUT = fullfile(ROOT, 'Stage1b_Epochs');
FS_IN = 256; FS_TARGET = 128; MAX_TRIALS_PER_SUBJECT = 60; MAD_K = 5; REJECT_CH_FRACTION = 0.15;
BP_BAND = [0.5 45]; NOTCH_HZ = 50; EXPECTED_N = [44 52 48];

fid = fopen(fullfile(OUT, 'Stage1d_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 1d - OSF/VAHID EPOCH STORE   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

T = load(fullfile(D5, 'y_stim.mat')); y = T.y_stim; T = load(fullfile(D5, 'sub_name_stim.mat')); names = T.sub_name_stim;
subjIdx = y(:, 1); grpFlag = y(:, 2:4); assert(all(sum(grpFlag, 2) == 1), 'group flags are not one-hot');
[~, grp] = max(grpFlag, [], 2);                                   % 1 = controls, 2 = subtype1, 3 = subtype2
for g = 1:3
    L('group %d: %d trials, %d subjects (max index %d), %d names in sub_name_stim', g, nnz(grp == g), numel(unique(subjIdx(grp == g))), max(subjIdx(grp == g)), numel(names{g}));
    assert(max(subjIdx(grp == g)) == EXPECTED_N(g), 'group/column order does not match the expected 44/52/48 - check y_stim');
end
[bBP, aBP] = butter(4, BP_BAND / (FS_TARGET / 2), 'bandpass'); [bN, aN] = iirnotch(NOTCH_HZ / (FS_TARGET / 2), (NOTCH_HZ / (FS_TARGET / 2)) / 35);

% global subject id: 1..44 controls, 45..96 subtype1, 97..144 subtype2
offset = [0 cumsum(EXPECTED_N(1:2))]; gsub = subjIdx + offset(grp)';
nSubj = sum(EXPECTED_N); label = double(grp > 1); subtype = grp - 1;

% pass over the seven data files, collecting trials per subject (in file order)
files = dir(fullfile(D5, 'd*.mat')); [~, o] = sort(arrayfun(@(f) sscanf(f.name, 'd%d.mat'), files)); files = files(o);
kept = cell(nSubj, 1); nTot = zeros(nSubj, 1); nRej = zeros(nSubj, 1); row0 = 0; chanN = [];
for f = 1:numel(files)
    T = load(fullfile(files(f).folder, files(f).name)); fn = fieldnames(T); D = T.(fn{1});   % trials x ch x samples
    n = size(D, 1); rows = row0 + (1:n); row0 = row0 + n; chanN = size(D, 2);
    L('  %s: %d trials x %d ch x %d samples', files(f).name, n, size(D, 2), size(D, 3));
    for i = unique(gsub(rows))'
        have = 0; if ~isempty(kept{i}), have = size(kept{i}, 1); end
        if have >= 2 * MAX_TRIALS_PER_SUBJECT, continue; end            % process up to 2x cap so rejection leaves enough
        r = rows(gsub(rows) == i); r = r(1:min(end, 2 * MAX_TRIALS_PER_SUBJECT - have));
        X = double(D(r - row0 + n, :, :));                             % local rows within this file
        [p, q] = rat(FS_TARGET / FS_IN); ep = [];
        for t = 1:size(X, 1)
            sig = squeeze(X(t, :, :))'; sig = sig - mean(sig, 1);        % samples x ch
            sig = resample(sig, p, q);                                    % 385 -> 193 samples
            sig = filtfilt(bBP, aBP, sig); sig = filtfilt(bN, aN, sig);
            if isempty(ep), ep = zeros(size(X, 1), size(sig, 1), size(sig, 2)); end
            ep(t, :, :) = reshape(sig, [1 size(sig)]);
        end
        prev = kept{i}; if isempty(prev), prev = zeros(0, size(ep, 2), size(ep, 3)); end
        kept{i} = cat(1, prev, ep); nTot(i) = nTot(i) + size(ep, 1);
    end
    clear D T
end
assert(row0 == size(y, 1), 'trial count mismatch between data files and y_stim');

% robust rejection per subject (on all its trials), then cap
S = struct('dataset', 'OSF', 'fs', FS_TARGET, 'epochLen', size(kept{1}, 2) / FS_TARGET, ...
    'channels', {arrayfun(@(c) sprintf('CSD%02d', c), 1:chanN, 'UniformOutput', false)}, ...
    'preproc', sprintf('CSD trials 0.5-20 Hz (source) -> DC -> resample %dHz -> BP %g-%g -> notch %d -> MAD%g (>%.0f%% ch) reject -> cap %d', FS_TARGET, BP_BAND(1), BP_BAND(2), NOTCH_HZ, MAD_K, 100 * REJECT_CH_FRACTION, MAX_TRIALS_PER_SUBJECT), ...
    'X', zeros(0, size(kept{1}, 2), chanN, 'single'), 'subject', [], 'subjectID', strings(0, 1), 'label', [], 'age', [], 'sex', [], 'subtype', []);
for i = 1:nSubj
    ep = kept{i}; if isempty(ep), L('  !! subject %d has no trials', i); continue; end
    ptp = squeeze(max(ep, [], 2) - min(ep, [], 2)); if size(ep, 1) == 1, ptp = reshape(ptp, 1, []); end
    flag = false(size(ptp));
    for c = 1:chanN, v = ptp(:, c); md = median(v); mad_ = 1.4826 * median(abs(v - md)); flag(:, c) = v > md + MAD_K * max(mad_, eps) | v < 1e-9; end
    bad = mean(flag, 2) > REJECT_CH_FRACTION; nRej(i) = nnz(bad); ep = ep(~bad, :, :); ep = ep(1:min(end, MAX_TRIALS_PER_SUBJECT), :, :);
    k = size(ep, 1); g = find(i > offset, 1, 'last'); lab = double(g > 1);
    S.X = cat(1, S.X, single(ep)); S.subject = [S.subject; repmat(i, k, 1)];
    S.subjectID = [S.subjectID; repmat(string(sprintf('G%d_%02d', g, i - offset(g))), k, 1)];
    S.label = [S.label; repmat(lab, k, 1)]; S.age = [S.age; nan(k, 1)]; S.sex = [S.sex; nan(k, 1)]; S.subtype = [S.subtype; repmat(g - 1, k, 1)];
    L('  subject %3d (%s) label=%d subtype=%d  trials total=%3d rejected=%3d kept=%3d', i, S.subjectID(end), lab, g - 1, nTot(i), nRej(i), k);
end
[~, ~, kk] = unique(S.subject); S.epochsPerSubject = accumarray(kk, 1);
save(fullfile(OUT, 'D5_OSF_epochs.mat'), 'S', '-v7.3');
[~, ia] = unique(S.subject);
L('\nD5 OSF: subjects=%d (ADHD %d [ADD %d, ADHD-C %d] / HC %d) epochs=%d, %d samples (%.2f s) x %d channels', numel(ia), ...
    nnz(S.label(ia) == 1), nnz(S.subtype(ia) == 1), nnz(S.subtype(ia) == 2), nnz(S.label(ia) == 0), numel(S.label), size(S.X, 2), size(S.X, 2) / FS_TARGET, size(S.X, 3));
L('saved %s', fullfile(OUT, 'D5_OSF_epochs.mat'));
L('\nDONE. Send Stage1d_Report.txt.');

function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
