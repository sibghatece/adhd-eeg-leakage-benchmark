# Notes

- Paths: each stage has a ROOT variable at the top pointing to the dataset
  working directory. Change it once per machine.
- MATLAB environment: if you have your own emd.m / ewt.m on the path they can
  shadow the toolbox versions; the feature scripts check for this and stop with
  a clear message.
- The `stage1_*`, `stage2_*`, `stage3_*` (non-b) scripts are the earlier
  two-channel versions kept for provenance; the multichannel `*b` scripts are the
  ones used in the paper.
- CSV outputs of every stage are the machine-readable record of all results and
  are consumed by make_figures.m and make_supplement.py.
