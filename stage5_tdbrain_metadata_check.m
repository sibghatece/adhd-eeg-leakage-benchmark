%% stage5_tdbrain_metadata_check.m
%  STAGE 5 - Is the TDBRAIN amplitude difference diagnosis or acquisition?
%
%  Per subject: global log power 1-30 Hz (mean over the 26 channels), from the
%  Stage 1b adult store (ADHD + HEALTHY). Joined with the participants sheet:
%  Dataset (study membership, e.g. ADHD_NF), sessSeason, sessTime, indication, education.
%  Additionally reads restEC session-1 recordings of OTHER adult clinical groups
%  (MDD, OCD, INSOMNIA) directly from the BDF files and computes the same value,
%  to test "ADHD vs healthy" against "any clinical group vs healthy".
%
%  Tests reported (Welch t, Cohen d, AUC):
%     ADHD(ADHD_NF study) vs ADHD(other)    <- if large: amplitude tracks study, not diagnosis
%     ADHD(ADHD_NF) vs HC,  ADHD(other) vs HC
%     MDD vs HC, OCD vs HC, INSOMNIA vs HC  <- if these look like ADHD vs HC: it is clinic-vs-healthy
%     season / time-of-day effects within HC and within ADHD
%
%  Outputs: <OUT>\Stage5_Report.txt, <OUT>\T_stage5_subjects.csv, <OUT>\Stage5_groups.png

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
OUT = fullfile(ROOT, 'Stage5_Metadata');
STORE = fullfile(ROOT, 'Stage1b_Epochs', 'D2_TDBRAIN_epochs.mat');
D2_ROOT = fullfile(ROOT, 'Dataset2_TDBRAIN_Dataset_V3_1_Encr', 'TDBRAIN_Dataset_V3_1');
D2_XLSX = fullfile(ROOT, 'Dataset2_TDBRAIN_Dataset_V3_1_Encr', 'TDBRAIN_participants_V3.xlsx');
OTHER_GROUPS = {'MDD', 'OCD', 'INSOMNIA'}; MIN_AGE = 18; MAX_OTHER_PER_GROUP = 120;
CHANNELS = {'Fp1','Fp2','F7','F3','Fz','F4','F8','FC3','FCz','FC4','T7','C3','Cz','C4','T8','CP3','CPz','CP4','P7','P3','Pz','P4','P8','O1','Oz','O2'};
FS_TARGET = 128; EPOCH_SEC = 2; MAD_K = 5; ABS_UV = 400; CH_FRAC = 0.15; MAX_EP = 60;

if ~isfolder(OUT), mkdir(OUT); end
fidLog = fopen(fullfile(OUT, 'Stage5_Report.txt'), 'w', 'n', 'UTF-8'); assert(fidLog > 0);
cleanupLog = onCleanup(@() fclose(fidLog));
L = @(varargin) logline(fidLog, varargin{:});
L('STAGE 5 - TDBRAIN METADATA CHECK   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

%% ---------------- participants sheet
[~, ~, raw] = xlsread(D2_XLSX, 1);
hdr = cellfun(@toStr, raw(1, :), 'UniformOutput', false); col = @(n) find(strcmpi(hdr, n), 1); rows = raw(2:end, :);
P.id = cellfun(@toStr, rows(:, col('TDBRAIN_ID')), 'UniformOutput', false);
P.status = cellfun(@(x) upper(strtrim(toStr(x))), rows(:, col('formal_status')), 'UniformOutput', false);
P.dataset = cellfun(@(x) strtrim(toStr(x)), rows(:, col('Dataset')), 'UniformOutput', false);
P.season = cellfun(@(x) strtrim(toStr(x)), rows(:, col('sessSeason')), 'UniformOutput', false);
P.time = cellfun(@(x) strtrim(toStr(x)), rows(:, col('sessTime')), 'UniformOutput', false);
P.indication = cellfun(@(x) strtrim(toStr(x)), rows(:, col('indication')), 'UniformOutput', false);
P.sess = cellfun(@toNum, rows(:, col('sessID'))); P.age = cellfun(@toNum, rows(:, col('age'))); P.sex = cellfun(@toNum, rows(:, col('gender')));
P.edu = cellfun(@toNum, rows(:, col('education')));

%% ---------------- ADHD + HC from the Stage 1b store
T = load(STORE); S = T.S; assert(isequal(S.channels(:)', CHANNELS));
[subj, ia] = unique(S.subject); n = numel(subj);
G = struct('id', {}, 'group', {}, 'lv', {}, 'lvCh', {}, 'age', {}, 'sex', {}, 'dataset', {}, 'season', {}, 'time', {}, 'indication', {}, 'edu', {}, 'nEp', {});
win = hamming(128);
for i = 1:n
    m = S.subject == subj(i); X = S.X(m, :, :); lvCh = zeros(1, numel(CHANNELS));
    for c = 1:numel(CHANNELS)
        v = zeros(nnz(m), 1);
        for e = 1:nnz(m), [Pw, fr] = pwelch(double(squeeze(X(e, :, c)))', win, 64, 256, S.fs); v(e) = log(trapz(fr(fr >= 1 & fr < 30), Pw(fr >= 1 & fr < 30)) + eps); end
        lvCh(c) = mean(v);
    end
    id = char(S.subjectID(ia(i))); r = find(strcmp(P.id, id) & P.sess == 1, 1);
    G(end+1) = struct('id', id, 'group', ifelse(S.label(ia(i)) == 1, 'ADHD', 'HC'), 'lv', mean(lvCh), 'lvCh', lvCh, 'age', S.age(ia(i)), 'sex', S.sex(ia(i)), ...
        'dataset', P.dataset{r}, 'season', P.season{r}, 'time', P.time{r}, 'indication', P.indication{r}, 'edu', P.edu(r), 'nEp', nnz(m)); %#ok<SAGROW>
end
L('Store: %d subjects (ADHD %d, HC %d)', n, nnz(strcmp({G.group}, 'ADHD')), nnz(strcmp({G.group}, 'HC')));

%% ---------------- other clinical groups straight from BDF
[bBP, aBP] = butter(4, [0.5 45] / (FS_TARGET / 2), 'bandpass'); [bN, aN] = iirnotch(50 / 64, (50 / 64) / 35);
for gi = 1:numel(OTHER_GROUPS)
    sel = find(strcmp(P.status, OTHER_GROUPS{gi}) & P.sess == 1 & P.age >= MIN_AGE); sel = sel(1:min(numel(sel), MAX_OTHER_PER_GROUP));
    L('Reading %s: %d candidates', OTHER_GROUPS{gi}, numel(sel)); nOK = 0;
    for r = sel'
        fp = fullfile(D2_ROOT, P.id{r}, 'ses-1', 'eeg', sprintf('%s_ses-1_task-restEC_eeg.bdf', P.id{r})); if ~isfile(fp), continue; end
        try, [sig, fsB] = readBdfChannels(fp, CHANNELS); catch, continue; end
        sig = sig - mean(sig, 1); [p, q] = rat(FS_TARGET / fsB); sig = resample(sig, p, q); sig = filtfilt(bBP, aBP, sig); sig = filtfilt(bN, aN, sig);
        ne = floor(size(sig, 1) / (FS_TARGET * EPOCH_SEC)); if ne < 10, continue; end
        ep = reshape(sig(1:ne * FS_TARGET * EPOCH_SEC, :), FS_TARGET * EPOCH_SEC, ne, []); ep = permute(ep, [2 1 3]);
        ptp = squeeze(max(ep, [], 2) - min(ep, [], 2)); flag = false(size(ptp));
        for c = 1:size(ptp, 2), v = ptp(:, c); md = median(v); mad_ = 1.4826 * median(abs(v - md)); flag(:, c) = v > md + MAD_K * max(mad_, eps); end
        bad = mean(flag, 2) > CH_FRAC | any(squeeze(max(abs(ep), [], 2)) > ABS_UV, 2); ep = ep(~bad, :, :); ep = ep(1:min(end, MAX_EP), :, :);
        if size(ep, 1) < 10, continue; end
        lvCh = zeros(1, numel(CHANNELS));
        for c = 1:numel(CHANNELS)
            v = zeros(size(ep, 1), 1);
            for e = 1:size(ep, 1), [Pw, fr] = pwelch(squeeze(ep(e, :, c))', win, 64, 256, FS_TARGET); v(e) = log(trapz(fr(fr >= 1 & fr < 30), Pw(fr >= 1 & fr < 30)) + eps); end
            lvCh(c) = mean(v);
        end
        nOK = nOK + 1;
        G(end+1) = struct('id', P.id{r}, 'group', OTHER_GROUPS{gi}, 'lv', mean(lvCh), 'lvCh', lvCh, 'age', P.age(r), 'sex', P.sex(r), ...
            'dataset', P.dataset{r}, 'season', P.season{r}, 'time', P.time{r}, 'indication', P.indication{r}, 'edu', P.edu(r), 'nEp', size(ep, 1)); %#ok<SAGROW>
    end
    L('  %s loaded: %d', OTHER_GROUPS{gi}, nOK);
end

%% ---------------- subject table
fS = fopen(fullfile(OUT, 'T_stage5_subjects.csv'), 'w');
fprintf(fS, 'id,group,dataset,season,time,indication,age,sex,education,nEpochs,global_logpower_1_30,%s\n', strjoin(strcat(CHANNELS, '_lp'), ','));
for i = 1:numel(G), fprintf(fS, '%s,%s,%s,%s,%s,%s,%.1f,%d,%g,%d,%.4f,%s\n', G(i).id, G(i).group, G(i).dataset, G(i).season, G(i).time, strrep(G(i).indication, ',', ';'), G(i).age, G(i).sex, G(i).edu, G(i).nEp, G(i).lv, strjoin(arrayfun(@(v) sprintf('%.4f', v), G(i).lvCh, 'UniformOutput', false), ',')); end
fclose(fS);

%% ---------------- comparisons
grp = {G.group}'; lv = [G.lv]'; ds = {G.dataset}'; age = [G.age]'; sex = [G.sex]'; season = {G.season}'; tod = {G.time}';
isADHD = strcmp(grp, 'ADHD'); isHC = strcmp(grp, 'HC'); nf = contains(ds, 'ADHD_NF');
L('\n=== Global log power 1-30 Hz by group (mean +/- sd, n)');
for gname = [{'HC', 'ADHD'}, OTHER_GROUPS]
    m = strcmp(grp, gname{1}); if ~any(m), continue; end
    L('  %-9s %6.3f +/- %.3f  (n=%3d, age %.1f, male %.0f%%)', gname{1}, mean(lv(m)), std(lv(m)), nnz(m), mean(age(m)), 100 * mean(sex(m) == 1));
end
L('\n=== Within ADHD: study membership');
L('  ADHD in ADHD_NF study: n=%d, other ADHD: n=%d', nnz(isADHD & nf), nnz(isADHD & ~nf));
cmp(L, lv, isADHD & nf, isADHD & ~nf, 'ADHD(ADHD_NF) vs ADHD(other)');
cmp(L, lv, isADHD & nf, isHC, 'ADHD(ADHD_NF) vs HC');
cmp(L, lv, isADHD & ~nf, isHC, 'ADHD(other)   vs HC');
cmp(L, lv, isADHD, isHC, 'ADHD(all)     vs HC');
L('\n=== Other clinical groups vs HC (if these match ADHD vs HC, the effect is clinic-vs-healthy)');
for gi = 1:numel(OTHER_GROUPS), m = strcmp(grp, OTHER_GROUPS{gi}); if any(m), cmp(L, lv, m, isHC, sprintf('%-8s vs HC', OTHER_GROUPS{gi})); end, end
for gi = 1:numel(OTHER_GROUPS), m = strcmp(grp, OTHER_GROUPS{gi}); if any(m), cmp(L, lv, isADHD, m, sprintf('ADHD vs %s', OTHER_GROUPS{gi})); end, end
L('\n=== HC by Dataset column (are the healthy controls from one study?)');
[u, ~, k] = unique(ds(isHC)); cnt = accumarray(k, 1); for i = 1:numel(u), L('  HC dataset "%s": n=%d', u{i}, cnt(i)); end
L('\n=== Season / time of day within groups (one-way ANOVA p on global log power)');
for gname = {'HC', 'ADHD'}
    m = strcmp(grp, gname{1});
    ok = m & ~strcmpi(season, 'nan') & ~cellfun(@isempty, season); if nnz(ok) > 10, p = anova1(lv(ok), season(ok), 'off'); L('  %-5s season p = %.3f  (%s)', gname{1}, p, groupMeans(lv(ok), season(ok))); end
    ok = m & ~strcmpi(tod, 'nan') & ~cellfun(@isempty, tod); if nnz(ok) > 10, p = anova1(lv(ok), tod(ok), 'off'); L('  %-5s time   p = %.3f  (%s)', gname{1}, p, groupMeans(lv(ok), tod(ok))); end
end
L('\n=== Age and sex vs global log power (all subjects): r_age = %.3f (p=%.3f), male-female d = %.2f', ...
    corr(lv, age), corrP(lv, age), (mean(lv(sex == 1)) - mean(lv(sex == 0))) / sqrt((var(lv(sex == 1)) + var(lv(sex == 0))) / 2));

%% ---------------- figure
fig = figure('Color', 'w', 'Position', [100 100 900 500]);
labels = [{'HC', 'ADHD (NF study)', 'ADHD (other)'}, OTHER_GROUPS]; masks = [{isHC, isADHD & nf, isADHD & ~nf}, arrayfun(@(g) strcmp(grp, g{1}), OTHER_GROUPS, 'UniformOutput', false)];
data = []; gid = [];
for i = 1:numel(masks), data = [data; lv(masks{i})]; gid = [gid; repmat(i, nnz(masks{i}), 1)]; end %#ok<AGROW>
boxplot(data, gid, 'Labels', labels); ylabel('global log power 1-30 Hz (mean of 26 ch)'); grid on; title('TDBRAIN adults, session 1, eyes closed');
saveas(fig, fullfile(OUT, 'Stage5_groups.png'));
L('\nDONE. Send Stage5_Report.txt and Stage5_groups.png.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
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
function cmp(L, lv, mA, mB, name)
    a = lv(mA); b = lv(mB); if numel(a) < 3 || numel(b) < 3, L('  %-32s insufficient n', name); return; end
    [~, p, ~, st] = ttest2(a, b, 'Vartype', 'unequal'); d = (mean(a) - mean(b)) / sqrt((var(a) + var(b)) / 2);
    [~, ~, ~, auc] = perfcurve([ones(numel(a), 1); zeros(numel(b), 1)], [a; b], 1);
    L('  %-32s n=%3d/%3d  diff=%+.3f  d=%+.2f  AUC=%.3f  t=%+.2f  p=%.2e', name, numel(a), numel(b), mean(a) - mean(b), d, auc, st.tstat, p);
end
function s = groupMeans(v, g)
    [u, ~, k] = unique(g); s = strjoin(arrayfun(@(i) sprintf('%s %.2f (n=%d)', u{i}, mean(v(k == i)), nnz(k == i)), 1:numel(u), 'UniformOutput', false), ', ');
end
function p = corrP(a, b), [~, p] = corr(a, b); end
function [sig, fs] = readBdfChannels(fp, wantLabels)
    f = fopen(fp, 'r', 'ieee-le'); c = onCleanup(@() fclose(f));
    hdr = fread(f, 256, 'uint8=>uint8')'; if ~(hdr(1) == 255 && strcmp(char(hdr(2:8)), 'BIOSEMI')), error('not BDF'); end
    h = char(hdr); headerBytes = str2double(h(185:192)); nRec = str2double(h(237:244)); durRec = str2double(h(245:252)); nSig = str2double(h(253:256));
    sh = fread(f, nSig * 256, 'uint8=>char')'; labels = strings(1, nSig); physMin = zeros(1, nSig); physMax = physMin; digMin = physMin; digMax = physMin; nSamp = physMin; off = 0;
    for k = 1:nSig, labels(k) = strtrim(sh(off + (k-1)*16 + (1:16))); end; off = off + nSig * (16 + 80 + 8);
    for k = 1:nSig, physMin(k) = str2double(sh(off + (k-1)*8 + (1:8))); end; off = off + nSig * 8;
    for k = 1:nSig, physMax(k) = str2double(sh(off + (k-1)*8 + (1:8))); end; off = off + nSig * 8;
    for k = 1:nSig, digMin(k) = str2double(sh(off + (k-1)*8 + (1:8))); end; off = off + nSig * 8;
    for k = 1:nSig, digMax(k) = str2double(sh(off + (k-1)*8 + (1:8))); end; off = off + nSig * (8 + 80);
    for k = 1:nSig, nSamp(k) = str2double(sh(off + (k-1)*8 + (1:8))); end
    if nRec < 0, d = dir(fp); nRec = floor((d.bytes - headerBytes) / (3 * sum(nSamp))); end
    idx = zeros(1, numel(wantLabels)); for w = 1:numel(wantLabels), idx(w) = find(strcmpi(labels, wantLabels{w}), 1); end
    fs = nSamp(idx(1)) / durRec; fseek(f, headerBytes, 'bof'); Stot = sum(nSamp);
    raw = fread(f, [3, nRec * Stot], 'uint8=>double'); if size(raw, 2) < nRec * Stot, nRec = floor(size(raw, 2) / Stot); raw = raw(:, 1:nRec * Stot); end
    val = raw(1, :) + 256 * raw(2, :) + 65536 * raw(3, :); val(val >= 2^23) = val(val >= 2^23) - 2^24; val = reshape(val, Stot, nRec); starts = [0 cumsum(nSamp(1:end-1))];
    sig = zeros(nSamp(idx(1)) * nRec, numel(idx));
    for w = 1:numel(idx), k = idx(w); block = val(starts(k) + (1:nSamp(k)), :); sig(:, w) = (block(:) - digMin(k)) * (physMax(k) - physMin(k)) / (digMax(k) - digMin(k)) + physMin(k); end
end
