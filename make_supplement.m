%% make_supplement.m
%  Builds Supplementary_Material.docx (S1-S6) for the ADHD EEG benchmark paper,
%  entirely in MATLAB using Word automation (actxserver). Windows + Microsoft Word required.
%  No Python, no extra toolboxes. Equations in S1 are inserted as native Word equations
%  from their LaTeX via Word's OMML builder, so they are editable in Word.
%
%  Reads:  <ROOT>\Stage9_Master\T_stage9_master.csv        (S2, S3)
%          <ROOT>\Stage4_Amplitude\T_channel_auc.csv       (S4)
%  Writes: <ROOT>\Supplementary_Material.docx
%
%  Run make_supp_figures.m first to obtain figures S1 and S5 and the top-20 list for S5.

clear; clc;
ROOT = 'C:\Users\Admin\Documents\WROK_WORK_WORK\Next_phase_Mental_Disorders\3_Problem_3_ADHD_ThreeDatasets+Paper';
S9  = fullfile(ROOT, 'Stage9_Master', 'T_stage9_master.csv');
S4CH = fullfile(ROOT, 'Stage4_Amplitude', 'T_channel_auc.csv');
OUTDOC = fullfile(ROOT, 'Supplementary_Material.docx');

FAM_ORDER = {'BandPower','LogVar','timedomain','entropy','nonlinear','wavelet','vmd','PSR','riemannian'};
FAM_LAB = containers.Map({'BandPower','LogVar','timedomain','entropy','nonlinear','wavelet','vmd','PSR','riemannian'}, ...
    {'Spectral (band power)','Amplitude (log variance)','Time domain / Hjorth','Entropy','Fractal / nonlinear','Wavelet','VMD','EMD-EWT phase space','Riemannian covariance'});
PRIMARY = {'D1_IEEE','D2m_TDBRAIN','D3_MENDELEY','D4m_BALLADEER','D5_OSF'};
SECOND  = {'D2a_TDBRAIN','D4a_BALLADEER'};

% inline numeric formatter: accepts numeric or char, returns formatted string or the text as-is
nf = @(v, f) local_nf(v, f);

%% ---- Word application
word = actxserver('Word.Application'); word.Visible = 0;
doc = word.Documents.Add; sel = word.Selection;
sel.Font.Name = 'Times New Roman'; sel.Font.Size = 10;

titleP(sel, 'Supplementary Material');
italicCenter(sel, 'Why EEG classifiers for ADHD do not generalise: a leakage-controlled benchmark of ten method families across five public datasets');
sel.TypeParagraph;

%% ================= S1 feature definitions
heading(sel, 'S1. Feature definitions');
body(sel, ['Equations are given in the notation of the main text. Every feature is computed per epoch and per channel unless stated. x denotes the epoch of one channel, N its length; a dot denotes the first difference; a bar denotes the mean.']);
feats = {
 'Spectral relative band power', 'R_{c,b} = \frac{\int_{f_1}^{f_2} P_c(f)\,df}{\int_{0.5}^{45} P_c(f)\,df}', 'P_c(f) is the Welch power spectral density. Bands: delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30 Hz, plus the theta/beta ratio. Five features per channel.'
 'Log variance (amplitude)', 'LV = \log(\mathrm{var}(x)+\varepsilon)', 'The simplest amplitude descriptor and the main baseline. One feature per channel.'
 'Hjorth mobility', 'M = \sqrt{\mathrm{var}(\dot{x})/\mathrm{var}(x)}', ''
 'Hjorth complexity', 'C = \sqrt{\mathrm{var}(\ddot{x})/\mathrm{var}(\dot{x})}/M', ''
 'Skewness and kurtosis', '\gamma_1 = \frac{\frac{1}{N}\sum_n (x_n-\bar{x})^3}{\sigma^3},\quad \gamma_2 = \frac{\frac{1}{N}\sum_n (x_n-\bar{x})^4}{\sigma^4}', ''
 'Zero-crossing rate', 'ZCR = \frac{1}{N-1}\sum_n \mathbf{1}[\mathrm{sign}(x_{n+1})\neq \mathrm{sign}(x_n)]', ''
 'Line length', 'LL = \frac{1}{N-1}\sum_n |x_{n+1}-x_n|', ''
 'Sample entropy', 'SampEn = -\ln(A/B)', 'B and A are template-pair counts of length m and m+1 within Chebyshev distance r times the standard deviation; m=2, r=0.2, on the standardised epoch.'
 'Permutation entropy', 'PE = -\frac{1}{\log m!}\sum_\pi p(\pi)\log p(\pi)', 'p(pi) is the relative frequency of ordinal pattern pi of order m; orders 3 and 4.'
 'Spectral entropy', 'SE = -\frac{1}{\log K}\sum_k p_k \log p_k', 'p_k is the normalised power over the K bins in 0.5-45 Hz.'
 'Fuzzy entropy', 'FuzzyEn = \ln \phi^{m} - \ln \phi^{m+1}', 'phi computed with an exponential membership exp(-(d/r)^n); m=2, r=0.2, n=2.'
 'Higuchi fractal dimension', 'HFD = -\,\mathrm{slope}[\log L(k)\ \mathrm{vs}\ \log(1/k)]', 'L(k) is the mean curve length at scale k, k up to 8.'
 'Katz fractal dimension', 'KFD = \frac{\log_{10} n}{\log_{10} n + \log_{10}(d/L)}', 'L total length, d maximum distance from the first point, n=N-1.'
 'Petrosian fractal dimension', 'PFD = \frac{\log_{10} N}{\log_{10} N + \log_{10}(N/(N+0.4 N_\Delta))}', 'N_Delta is the number of sign changes of the first difference.'
 'Detrended fluctuation exponent', '\alpha = \mathrm{slope}[\log F(n)\ \mathrm{vs}\ \log n]', 'F(n) is the rms of the detrended integrated signal over boxes of size n = 4 to 64.'
 'Hurst exponent (rescaled range)', 'H = \mathrm{slope}[\log (R/S)_n\ \mathrm{vs}\ \log n]', ''
 'Lempel-Ziv complexity', 'LZC = c(N)\,\frac{\log_2 N}{N}', 'c(N) is the number of distinct patterns in the median-binarised sequence (Kaspar-Schuster).'
 'Wavelet sub-band energy', 'E_j = \sum_i w_{j,i}^2,\qquad r_j = E_j/\sum_j E_j', 'db4, four levels; log energy, relative energy and standard deviation per sub-band, plus wavelet entropy of the r_j.'
 'Variational mode decomposition', '\min_{u_k,\omega_k} \sum_{k=1}^{4} \left\| \partial_t\left[\left(\delta(t)+\frac{j}{\pi t}\right) * u_k(t)\right] e^{-j\omega_k t} \right\|_2^2 \ \ \mathrm{s.t.}\ \sum_k u_k = x', 'Per mode: log energy, relative energy, Hjorth mobility, spectral centroid.'
 'Empirical mode decomposition', 'x(t) = \sum_{i=1}^{4} \mathrm{IMF}_i(t) + r(t)', 'First four intrinsic mode functions, each split into two empirical wavelet modes; 19 phase-space descriptors per mode (main text eq. 7).'
 'Riemannian tangent space', '\mathbf{L} = \mathbf{V}\,\mathrm{diag}(\log\lambda)\,\mathbf{V}^{T},\quad \mathbf{S} = \mathbf{V}\,\mathrm{diag}(\lambda)\,\mathbf{V}^{T}', 'S is the regularised channel covariance; the upper triangle of L with off-diagonals scaled by sqrt(2) forms the feature vector.'
};
for i = 1:size(feats, 1)
    p = sel.Range; sel.Font.Bold = 1; sel.TypeText([feats{i, 1} '.  ']); sel.Font.Bold = 0;
    insertEquation(sel, feats{i, 2});
    sel.TypeParagraph;
    if ~isempty(feats{i, 3})
        sel.Font.Italic = 1; sel.Font.Size = 8.5; sel.TypeText(['    ' feats{i, 3}]); sel.Font.Italic = 0; sel.Font.Size = 10; sel.TypeParagraph;
    end
end
sel.TypeParagraph;

%% ================= S2 complete metrics
R9 = readCsvStruct(S9);
heading(sel, 'S2. Complete leakage-free metrics (protocol P3)');
body(sel, 'Participant-level balanced accuracy (BAL), sensitivity (SEN), specificity (SPE), AUC and MCC, mean over three repetitions, for every feature family, cohort and classifier under the leakage-free protocol. RF, random forest; SVM, support vector machine.');
if ~isempty(R9)
    hdr = {'Cohort','Family','Clf','BAL','SEN','SPE','AUC','MCC'}; rows = {};
    for c = 1:numel(PRIMARY)
        for f = 1:numel(FAM_ORDER)
            for clf = {'RF','SVM'}
                r = pickRow(R9, PRIMARY{c}, FAM_ORDER{f}, 'P3_subjectCV', clf{1}, 'subject');
                if ~isempty(r)
                    rows(end+1, :) = {strrep(PRIMARY{c}, '_', ' '), FAM_LAB(FAM_ORDER{f}), clf{1}, ...
                        nf(r.BAL, '%.1f'), nf(r.SEN, '%.1f'), nf(r.SPE, '%.1f'), nf(r.AUC, '%.3f'), nf(r.MCC, '%.3f')}; %#ok<SAGROW>
                end
            end
        end
    end
    makeTable(word, sel, hdr, rows);
else
    body(sel, '[T_stage9_master.csv not found under ROOT\Stage9_Master.]');
end
sel.TypeParagraph;

%% ================= S3 unmatched secondary cohorts
heading(sel, 'S3. Unmatched secondary cohorts');
body(sel, 'Participant-level balanced accuracy and AUC (random forest, P3) for the two unmatched cohorts: all TDBRAIN adults (56 ADHD vs 176 healthy) and all BALLADEER children (80 ADHD vs 43 healthy). Reported for completeness; the matched cohorts in the main text are the primary analysis.');
if ~isempty(R9)
    hdr = {'Cohort','Family','BAL','AUC'}; rows = {};
    for c = 1:numel(SECOND)
        for f = 1:numel(FAM_ORDER)
            r = pickRow(R9, SECOND{c}, FAM_ORDER{f}, 'P3_subjectCV', 'RF', 'subject');
            if ~isempty(r), rows(end+1, :) = {strrep(SECOND{c}, '_', ' '), FAM_LAB(FAM_ORDER{f}), nf(r.BAL, '%.1f'), nf(r.AUC, '%.3f')}; end %#ok<SAGROW>
        end
    end
    makeTable(word, sel, hdr, rows);
end
sel.TypeParagraph;

%% ================= S4 topography
heading(sel, 'S4. Age, sex and topography analyses (TDBRAIN)');
body(sel, 'Per-channel participant-level AUC and Cohen d of 1-30 Hz log power in the matched TDBRAIN adults. Cohen d < 0 means the ADHD group had lower power. The uniform sign across channels underlies the case-control interpretation in the main text.');
R4 = readCsvStruct(S4CH);
if ~isempty(R4)
    hdr = {'Channel','AUC','Cohen d','mean ADHD','mean HC'}; rows = {};
    idx = find(arrayfun(@(x) startsWith(char(string(x.cohort)), 'D2'), R4));
    aucs = arrayfun(@(x) toNum(x.AUC), R4(idx)); [~, ord] = sort(aucs, 'descend');
    idx = idx(ord); idx = idx(:).';                  % force a row of scalar indices
    for q = idx
        rr = R4(q);
        rows(end+1, :) = {char(string(rr.channel)), nf(rr.AUC, '%.3f'), nf(rr.cohen_d, '%.2f'), nf(rr.mean_ADHD, '%.2f'), nf(rr.mean_HC, '%.2f')}; %#ok<SAGROW>
    end
    makeTable(word, sel, hdr, rows);
else
    body(sel, '[T_channel_auc.csv not found under ROOT\Stage4_Amplitude.]');
end
sel.TypeParagraph;

%% ================= S5 note
heading(sel, 'S5. Single-feature separability on the Mendeley dataset (D3)');
body(sel, 'Figure S5 shows the distribution of participant-level |AUC| over every single feature on D3; the strongest reach 0.97-0.99. Paste the top-twenty list printed by make_supp_figures.m below, then insert Figure S5. A separation of this size from one feature is the signature of a recording-condition difference between the two groups rather than a neural effect.');
sel.TypeParagraph;

%% ================= S6 NERVE-ML mapping
heading(sel, 'S6. Mapping of the study to the NERVE-ML checklist');
body(sel, 'The NERVE-ML checklist (reference [32] in the main text) sets reproducibility and validity requirements for machine learning in neural engineering. The table maps each area to where it is addressed in this study.');
nerve = {
 'Data handling: split by subject; non-independence of segments controlled', 'Section 2.5 (P3 participant-grouped folds); figures 2 and 3 quantify the effect of not doing so'
 'Parameter selection performed inside training folds only', 'Section 2.5: fold-internal imputation, scaling, t-ranking, and inner-CV choice of K and hyperparameters'
 'Performance metrics appropriate, with uncertainty and a chance level', 'Section 2.5: balanced accuracy, sensitivity, specificity, AUC, MCC; SD over three repeats; permutation null (eq. 10)'
 'Class imbalance addressed', 'Section 2.5: uniform class prior for every classifier; balanced accuracy as the primary metric'
 'Confounds identified and controlled', 'Sections 2.3, 2.6 and 2.7: age/sex matching; muscle band, age regression, topography, metadata; diagnostic specificity'
 'External or cross-cohort validation', 'Section 2.8 and figure 8: transfer between cohorts; positive-control cohort (D4)'
 'Conclusions scoped to the population and protocol tested', 'Sections 4 and 5: claims limited to these datasets and protocols'
 'Reproducibility: code and data available', 'Section 2.9: released pipeline and result tables; stores regenerated from the public downloads'
};
makeTable(word, sel, {'NERVE-ML area', 'Where addressed'}, nerve);

%% ---- save and close
doc.SaveAs2(OUTDOC, 16);   % 16 = wdFormatDocumentDefault (.docx)
doc.Close; word.Quit;
fprintf('Written %s\n', OUTDOC);

%% ======================================================== LOCAL FUNCTIONS
function titleP(sel, txt)
    sel.ParagraphFormat.Alignment = 1; sel.Font.Bold = 1; sel.Font.Size = 14; sel.TypeText(txt);
    sel.Font.Bold = 0; sel.Font.Size = 10; sel.ParagraphFormat.Alignment = 0; sel.TypeParagraph;
end
function italicCenter(sel, txt)
    sel.ParagraphFormat.Alignment = 1; sel.Font.Italic = 1; sel.Font.Size = 11; sel.TypeText(txt);
    sel.Font.Italic = 0; sel.Font.Size = 10; sel.ParagraphFormat.Alignment = 0; sel.TypeParagraph;
end
function heading(sel, txt)
    sel.Font.Bold = 1; sel.Font.Size = 12; sel.TypeText(txt); sel.Font.Bold = 0; sel.Font.Size = 10; sel.TypeParagraph;
end
function body(sel, txt)
    sel.TypeText(txt); sel.TypeParagraph;
end
function insertEquation(sel, latex)
    % Insert a native Word equation built from LaTeX (Word 2016+ accepts LaTeX in the equation builder)
    try
        rng = sel.Range; eq = sel.OMaths.Add(rng);
        eq.Range.Text = latex; eq.Range.OMaths.BuildUp;   % converts linear/LaTeX to professional form
        sel.Collapse(0);
    catch
        % fallback: plain text if the Word build does not accept LaTeX
        sel.Font.Name = 'Cambria Math'; sel.TypeText(latex); sel.Font.Name = 'Times New Roman';
    end
end
function makeTable(word, sel, hdr, rows)
    nR = size(rows, 1) + 1; nC = numel(hdr);
    tbl = sel.Tables.Add(sel.Range, nR, nC);
    tbl.Borders.Enable = 1; tbl.Range.Font.Size = 8.5; tbl.Range.Font.Name = 'Times New Roman';
    tbl.Rows.Item(1).Range.Font.Bold = 1;
    for j = 1:nC, tbl.Cell(1, j).Range.Text = hdr{j}; end
    for i = 1:size(rows, 1)
        for j = 1:nC, tbl.Cell(i + 1, j).Range.Text = rows{i, j}; end
    end
    sel.Start = tbl.Range.End; sel.Collapse(0); sel.TypeParagraph;
end
function S = readCsvStruct(path)
    S = [];
    if ~isfile(path), return; end
    fid = fopen(path, 'r'); hdr = strsplit(strtrim(fgetl(fid)), ',');
    C = textscan(fid, repmat('%s', 1, numel(hdr)), 'Delimiter', ',', 'CollectOutput', true); fclose(fid);
    C = C{1};
    for r = 1:size(C, 1)
        for j = 1:numel(hdr)
            key = matlab.lang.makeValidName(hdr{j}); v = C{r, j}; num = str2double(v);
            if ~isnan(num) && ~isempty(v), S(r).(key) = num; else, S(r).(key) = v; end %#ok<AGROW>
        end
    end
end
function r = pickRow(S, cohort, family, protocol, clf, level)
    r = [];
    for i = 1:numel(S)
        if strcmp(S(i).cohort, cohort) && strcmp(S(i).family, family) && strcmp(S(i).protocol, protocol) && strcmp(S(i).classifier, clf) && strcmp(S(i).level, level)
            r = S(i); return;
        end
    end
end
function out = local_nf(v, f)
    if ischar(v) || isstring(v), x = str2double(v); else, x = v; end
    if isnumeric(x) && isscalar(x) && ~isnan(x), out = sprintf(f, x); else, out = char(string(v)); end
end
function y = toNum(v)
    if ischar(v) || isstring(v), y = str2double(v); else, y = v; end
    if isempty(y) || ~isscalar(y), y = NaN; end
end
