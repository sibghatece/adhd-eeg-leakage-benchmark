%% stage9b_eegnet.m
%  STAGE 9b - EEGNet-8,2 (Lawhern et al., 2018) on raw epochs, MATLAB Deep Learning Toolbox.
%  Cohorts as in Stage 9 (D1, D2m matched, D3, D4m matched, D5; secondary D2a/D4a optional).
%  Protocols: P3 subject-grouped 5-fold (15 % of training subjects held out for early stopping),
%             P1 epoch-level random 5-fold (same network; the leaky comparison).
%  Input scaling: per-channel division by the training-fold robust std (amplitude information kept,
%  cohort-specific gain removed); NaN channels (bad electrodes) set to zero.
%  Class imbalance handled with class-weighted cross-entropy.
%  Outputs: <OUT>\Stage9b_Report.txt, T_stage9b_eegnet.csv. Results are cached per cohort x protocol.
%
%  Runtime on CPU is the limit: roughly 15-40 min per fold on large cohorts. Defaults below
%  (1 repeat, 30 epochs, primary cohorts) are an overnight job. Set USE_GPU = true if available.

clear; clc; close all;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
IN = fullfile(ROOT, 'Stage1b_Epochs'); OUT = fullfile(ROOT, 'Stage9b_EEGNet');
R_REPEATS = 1; K_OUTER = 5; SEED0 = 2026; MIN_EPOCHS_PER_SUBJECT = 10;
MAX_EPOCHS = 30; MINI_BATCH = 64; LR = 1e-3; VAL_FRACTION = 0.15; PATIENCE = 6;
F1 = 8; D = 2; F2 = 16; KERN_T = 64; DROPOUT = 0.25;
SECONDARY = false; RUN_P1 = true; USE_GPU = false;
DATASETS = {'D1_IEEE', 'D2_TDBRAIN', 'D3_MENDELEY', 'D4_BALLADEER', 'D5_OSF'};

assert(license('test', 'Neural_Network_Toolbox'), 'Deep Learning Toolbox is required for Stage 9b.');
if ~isfolder(OUT), mkdir(OUT); end
CACHE = fullfile(OUT, 'cache'); if ~isfolder(CACHE), mkdir(CACHE); end
fid = fopen(fullfile(OUT, 'Stage9b_Report.txt'), 'w', 'n', 'UTF-8'); assert(fid > 0); cl = onCleanup(@() fclose(fid));
L = @(varargin) logline(fid, varargin{:});
L('STAGE 9b - EEGNet   %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L('EEGNet-%d,%d  F2=%d kernel=%d dropout=%.2f | epochs=%d batch=%d lr=%g | repeats=%d outer=%d | GPU=%d', F1, D, F2, KERN_T, DROPOUT, MAX_EPOCHS, MINI_BATCH, LR, R_REPEATS, K_OUTER, USE_GPU);
fT = fopen(fullfile(OUT, 'T_stage9b_eegnet.csv'), 'w');
fprintf(fT, 'cohort,protocol,level,BAL,BAL_sd,ACC,ACC_sd,SEN,SEN_sd,SPE,SPE_sd,MCC,MCC_sd,AUC,AUC_sd\n');

for d = 1:numel(DATASETS)
    p = fullfile(IN, [DATASETS{d} '_epochs.mat']); if ~isfile(p), L('!! %s missing', p); continue; end
    T = load(p); S = T.S; S = dropSparse(S, MIN_EPOCHS_PER_SUBJECT);
    [subj, ia] = unique(S.subject); lab = double(S.label(ia)); age = double(S.age(ia)); sex = double(S.sex(ia));
    cohortList = {};
    switch DATASETS{d}
        case 'D2_TDBRAIN',   cohortList(end+1, :) = {'D2m_TDBRAIN', matchSubjects(subj, lab, age, sex, 2, SEED0)}; if SECONDARY, cohortList(end+1, :) = {'D2a_TDBRAIN', subj}; end
        case 'D4_BALLADEER', cohortList(end+1, :) = {'D4m_BALLADEER', matchSubjects(subj, lab, age, sex, 1, SEED0)}; if SECONDARY, cohortList(end+1, :) = {'D4a_BALLADEER', subj}; end
        otherwise,           cohortList(end+1, :) = {DATASETS{d}, subj};
    end
    for c = 1:size(cohortList, 1)
        m = ismember(S.subject, cohortList{c, 2});
        X = double(S.X(m, :, :)); y = double(S.label(m)); g = double(S.subject(m));
        X(isnan(X)) = 0; nCh = size(X, 3); nT = size(X, 2);
        Ximg = permute(X, [3 2 4 1]);                                  % [C x T x 1 x N]
        clear X
        L('\n================================================================ %s : %d subjects, %d epochs, %d ch x %d samples', cohortList{c, 1}, numel(unique(g)), numel(y), nCh, nT);
        prots = {'P3_subjectCV'}; if RUN_P1, prots{end+1} = 'P1_epochCV'; end %#ok<AGROW>
        for pr = 1:numel(prots)
            cacheFile = fullfile(CACHE, sprintf('%s__%s.mat', cohortList{c, 1}, prots{pr}));
            if isfile(cacheFile), Cc = load(cacheFile); msE = Cc.msE; msS = Cc.msS; L('  [cache] %s', prots{pr});
            else
                msE = []; msS = [];
                for r = 1:R_REPEATS
                    rng(SEED0 + r);
                    if strcmp(prots{pr}, 'P3_subjectCV'), folds = groupedStratifiedFolds(g, y, K_OUTER); else, folds = randomFolds(y, K_OUTER); end
                    P = zeros(numel(y), 3);
                    for f = 1:K_OUTER
                        t0 = tic; tr = find(folds ~= f); te = find(folds == f);
                        % validation split (by subject for P3, by epoch for P1)
                        if strcmp(prots{pr}, 'P3_subjectCV')
                            trSubj = unique(g(tr)); trSubj = trSubj(randperm(numel(trSubj))); nv = max(2, round(VAL_FRACTION * numel(trSubj)));
                            va = tr(ismember(g(tr), trSubj(1:nv))); trn = tr(~ismember(g(tr), trSubj(1:nv)));
                        else
                            tr = tr(randperm(numel(tr))); nv = round(VAL_FRACTION * numel(tr)); va = tr(1:nv); trn = tr(nv + 1:end);
                        end
                        % fold-internal scaling: per-channel robust std of the training epochs
                        sc = zeros(1, nCh);
                        for ch = 1:nCh, v = Ximg(ch, :, 1, trn); sc(ch) = 1.4826 * median(abs(v(:) - median(v(:)))) + eps; end
                        scale = @(idx) Ximg(:, :, 1, idx) ./ sc';
                        Xtr = scale(trn); Xva = scale(va); Xte = scale(te);
                        ytr = categorical(y(trn), [0 1]); yva = categorical(y(va), [0 1]);
                        w = [numel(ytr) / (2 * nnz(y(trn) == 0)), numel(ytr) / (2 * nnz(y(trn) == 1))];
                        layers = eegnetLayers(nCh, nT, F1, D, F2, KERN_T, DROPOUT, w);
                        opts = trainingOptions('adam', 'MaxEpochs', MAX_EPOCHS, 'MiniBatchSize', MINI_BATCH, 'InitialLearnRate', LR, ...
                            'Shuffle', 'every-epoch', 'ValidationData', {Xva, yva}, 'ValidationFrequency', max(1, floor(numel(ytr) / MINI_BATCH)), ...
                            'ValidationPatience', PATIENCE, 'OutputNetwork', 'best-validation-loss', 'Verbose', false, 'Plots', 'none', ...
                            'ExecutionEnvironment', ifelse(USE_GPU, 'gpu', 'cpu'));
                        net = trainNetwork(Xtr, ytr, layers, opts);
                        [yh, scores] = classify(net, Xte, 'MiniBatchSize', 256);
                        P(te, :) = [y(te), double(yh == '1'), scores(:, 2)];
                        L('    %s rep %d fold %d: %d train / %d val / %d test epochs, %.1f min', prots{pr}, r, f, numel(trn), numel(va), numel(te), toc(t0) / 60);
                    end
                    msE = [msE, metrics(P)]; %#ok<AGROW>
                    if strcmp(prots{pr}, 'P3_subjectCV'), msS = [msS, metrics(subjectVote(g, P(:, 1), P(:, 2), P(:, 3)))]; end %#ok<AGROW>
                end
                save(cacheFile, 'msE', 'msS');
            end
            writeRow(fT, cohortList{c, 1}, prots{pr}, 'epoch', msE);
            L('  %-13s epoch BAL %.1f +/- %.1f  AUC %.3f  MCC %.3f', prots{pr}, mean([msE.BAL]), std([msE.BAL]), mean([msE.AUC]), mean([msE.MCC]));
            if ~isempty(msS)
                writeRow(fT, cohortList{c, 1}, prots{pr}, 'subject', msS);
                L('  %-13s subject BAL %.1f +/- %.1f  SEN %.1f SPE %.1f  AUC %.3f  MCC %.3f', '', mean([msS.BAL]), std([msS.BAL]), mean([msS.SEN]), mean([msS.SPE]), mean([msS.AUC]), mean([msS.MCC]));
            end
        end
        clear Ximg
    end
end
fclose(fT);
L('\nDONE. Send Stage9b_Report.txt and T_stage9b_eegnet.csv.');

%% ======================================================== LOCAL FUNCTIONS
function logline(fid, fmt, varargin), s = sprintf(fmt, varargin{:}); fprintf('%s\n', s); fprintf(fid, '%s\n', s); end
function out = ifelse(c, a, b), if c, out = a; else, out = b; end, end
function layers = eegnetLayers(C, T, F1, D, F2, kT, drop, classW)
    layers = [
        imageInputLayer([C T 1], 'Normalization', 'none', 'Name', 'in')
        convolution2dLayer([1 kT], F1, 'Padding', 'same', 'Name', 'temporal')
        batchNormalizationLayer('Name', 'bn1')
        groupedConvolution2dLayer([C 1], D, 'channel-wise', 'Name', 'depthwise')       % spatial filters per temporal filter
        batchNormalizationLayer('Name', 'bn2')
        eluLayer('Name', 'elu1')
        averagePooling2dLayer([1 4], 'Stride', [1 4], 'Name', 'pool1')
        dropoutLayer(drop, 'Name', 'drop1')
        groupedConvolution2dLayer([1 16], 1, 'channel-wise', 'Padding', 'same', 'Name', 'sep_depth')   % separable conv: depthwise
        convolution2dLayer(1, F2, 'Name', 'sep_point')                                                   % ... then pointwise
        batchNormalizationLayer('Name', 'bn3')
        eluLayer('Name', 'elu2')
        averagePooling2dLayer([1 8], 'Stride', [1 8], 'Name', 'pool2')
        dropoutLayer(drop, 'Name', 'drop2')
        fullyConnectedLayer(2, 'Name', 'fc')
        softmaxLayer('Name', 'sm')
        classificationLayer('Name', 'out', 'Classes', categorical([0 1]), 'ClassWeights', classW(:))];
end
function S = dropSparse(S, minEp)
    [subj, ~, k] = unique(S.subject); cnt = accumarray(k, 1); bad = subj(cnt < minEp); if isempty(bad), return; end
    m = ~ismember(S.subject, bad); S.X = S.X(m, :, :); for fn = {'subject', 'subjectID', 'label', 'age', 'sex'}, S.(fn{1}) = S.(fn{1})(m, :); end
end
function keep = matchSubjects(subj, lab, age, sex, ratio, seed)
    rng(seed);
    if nnz(lab == 1) <= nnz(lab == 0), anchor = find(lab == 1); pool = find(lab == 0); else, anchor = find(lab == 0); pool = find(lab == 1); ratio = 1; end
    anchor = anchor(randperm(numel(anchor))); used = false(size(pool)); chosen = [];
    for i = anchor'
        for r = 1:ratio
            c = find(~used & sex(pool) == sex(i)); if isempty(c), c = find(~used); end; if isempty(c), break; end
            [~, j] = min(abs(age(pool(c)) - age(i))); used(c(j)) = true; chosen(end+1) = pool(c(j)); %#ok<AGROW>
        end
    end
    keep = [subj(anchor); subj(chosen)];
end
function folds = groupedStratifiedFolds(g, y, K)
    [subj, ia] = unique(g); lab = y(ia); folds = zeros(size(g));
    for c = [0 1], s = subj(lab == c); s = s(randperm(numel(s))); fx = mod((1:numel(s)) - 1, K) + 1; for i = 1:numel(s), folds(g == s(i)) = fx(i); end, end
end
function folds = randomFolds(y, K)
    folds = zeros(size(y)); for c = [0 1], s = find(y == c); s = s(randperm(numel(s))); folds(s) = mod((1:numel(s)) - 1, K) + 1; end
end
function S = subjectVote(g, y, yhat, score)
    [~, ~, k] = unique(g); yt = accumarray(k, y, [], @(v) v(1)); ms = accumarray(k, score, [], @mean); vote = accumarray(k, yhat, [], @mean);
    yv = double(vote > 0.5); tie = vote == 0.5; yv(tie) = double(ms(tie) > 0.5); S = [yt, yv, ms];
end
function m = metrics(P)
    y = P(:, 1); yh = P(:, 2); sc = P(:, 3);
    TP = nnz(y == 1 & yh == 1); TN = nnz(y == 0 & yh == 0); FP = nnz(y == 0 & yh == 1); FN = nnz(y == 1 & yh == 0);
    m.ACC = 100 * (TP + TN) / max(numel(y), 1); m.SEN = 100 * TP / max(TP + FN, 1); m.SPE = 100 * TN / max(TN + FP, 1); m.BAL = (m.SEN + m.SPE) / 2;
    den = sqrt(double(TP + FP) * double(TP + FN) * double(TN + FP) * double(TN + FN)); if den > 0, m.MCC = (TP * TN - FP * FN) / den; else, m.MCC = 0; end
    try, [~, ~, ~, auc] = perfcurve(y, sc, 1); m.AUC = auc; catch, m.AUC = NaN; end
end
function writeRow(fid, cohort, prot, level, ms)
    fld = {'BAL', 'ACC', 'SEN', 'SPE', 'MCC', 'AUC'}; fprintf(fid, '%s,%s,%s', cohort, prot, level);
    for i = 1:numel(fld), v = [ms.(fld{i})]; fprintf(fid, ',%.3f,%.3f', mean(v, 'omitnan'), std(v, 'omitnan')); end; fprintf(fid, '\n');
end
