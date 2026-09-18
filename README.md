# A leakage-controlled benchmark of EEG-based ADHD classification

Code for the paper:

**Why EEG classifiers for ADHD do not generalise: a leakage-controlled benchmark of ten method families across five public datasets.**
Sibghatullah I. Khan et al., submitted to *Journal of Neural Engineering*, 2026.

This repository releases the complete, staged MATLAB pipeline and the evaluation
protocol used in the paper, so that any feature family or classifier can be added
and compared on the same five public datasets under the same leakage-free protocol.

## Author and contact

**Dr. Sibghatullah I. Khan**
Sreenidhi Institute of Science and Technology, Hyderabad, India
Email: sibghatikhan@gmail.com
Phone / WhatsApp: +91 90118 81353

## Licence

Released under the **MIT License** (see `LICENSE`): free to use, modify and
distribute with attribution. If you use this code, please cite the paper
(`CITATION.cff`).

## What this code does

A single harmonised pipeline is applied to five public ADHD EEG datasets
(617 participants; children and adults; task and resting; clinic and research
recruited). Nine hand-crafted feature families and one convolutional network
(EEGNet) are evaluated under a leakage-free, participant-grouped, nested
cross-validation protocol, and under the segment-level and overlapping-window
protocols common in the literature. Other diagnostic groups in one clinical
database provide a specificity test, and a research-recruited cohort recorded
under one protocol serves as a positive control.

## Data (not included)

The datasets are **not** redistributed here. Download them from the original
sources and place each under the working directory named as below. The scripts
regenerate every intermediate store from these raw downloads.

| ID | Dataset | Source |
|----|---------|--------|
| D1 | IEEE DataPort ADHD/control children | https://doi.org/10.21227/rzfh-zn36 |
| D2 | TDBRAIN | https://brainclinics.com/resources (Sci. Data 9:333) |
| D3 | Mendeley adults ADHD | https://doi.org/10.17632/6k4g25fhzg.1 |
| D4 | BALLADEER | Figshare (Sci. Data, 2026) |
| D5 | OSF flanker-task children | https://osf.io/6594x |

Set the working-directory path (the `ROOT` variable) at the top of each script
to the folder that contains the dataset subfolders.

## Requirements

- MATLAB R2023b or later
- Toolboxes: Signal Processing, Wavelet, Statistics and Machine Learning,
  Deep Learning (for EEGNet, `stage9b_eegnet.m` only), Parallel Computing (optional, for speed)
- Microsoft Word (Windows) for the supplementary document; `make_supplement.m` drives Word directly, no Python needed

## How to run (in order)

The pipeline is organised as numbered stages. Each writes intermediate `.mat`
stores that later stages read, so any table or figure can be regenerated from the
raw downloads.

1. **Epoch stores.** `stage1b_build_epoch_store_multichannel.m` (D1, D2, D3),
   `stage1c_balladeer_epochs.m` (D4), `stage1d_osf_vahid_epochs.m` (D5).
2. **Features.** `stage2b_extract_features_multichannel.m` (spectral, amplitude,
   EMD-EWT phase space), then `stage8a`..`stage8f` (time domain, entropy, fractal,
   wavelet, VMD, Riemannian).
3. **Leakage-free benchmark.** `stage9_master_benchmark.m` (all families, three
   protocols) and `stage9b_eegnet.m` (deep-learning arm).
4. **Leakage demonstrations.** `stage7_leaky_protocols.m` and
   `stage7b_overlap_leakage.m`.
5. **Confound and specificity.** `stage4_amplitude_validation.m`,
   `stage5_tdbrain_metadata_check.m`, `stage6_specificity_test.m`.
6. **Covariance follow-up.** `stage10_riemannian_followup.m`,
   `stage10b_riemannian_frontal7.m`, `stage10c_fill_tables.m`.
7. **Figures and supplement.** `make_figures.m`, `make_supp_figures.m`,
   run `make_supplement.m` (writes `Supplementary_Material.docx`).

Random seeds are fixed for every fold assignment, subsampling, permutation and
network initialisation, so results are reproducible.

## Repository layout

```
code/     all MATLAB stage scripts and the figure/supplement generators
figures/  publication figures (PDF/PNG) once generated
docs/     notes
LICENSE   MIT
CITATION.cff
```

## Note on reproducibility

The harmonised epoch stores cannot be redistributed (they derive from datasets
under their own licences), but they are regenerated exactly from the public
downloads by stages 1b-1d. Everything downstream is deterministic given the fixed
seeds.
