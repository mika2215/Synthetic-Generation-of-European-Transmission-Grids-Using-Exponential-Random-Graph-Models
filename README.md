# Synthetic Generation of European Transmission Grids Using ERGMs

Code for the bachelor's thesis *Synthetic Generation of European Transmission Grids Using
Exponential Random Graph Models* (Karlsruhe Institute of Technology, 2026).

The code builds graphs of the national transmission grids of 29 European countries from the
PyPSA-Eur network, fits exponential random graph models (ERGMs) to 15 of them with the
equilibrium-expectation (EE) algorithm, samples synthetic grids from the fitted models and
compares them with the real grids, including their robustness to edge removal.

## Repository structure

| Folder | Thesis | Content |
|---|---|---|
| `common/` | all | Shared code: graph construction, statistics and change statistics, EE algorithm, sampler, sample spacing, robustness, held-out statistics |
| `01_descriptive/` | Ch. 2 (examples), Ch. 3 | Descriptors of the 29 country grids, Tables 1–2, Figures 1–7 |
| `02_selection/` | Sec. 4.4, 5.1 | LASSO pre-filter, forward selection, stability tests, Figure 8 |
| `03_estimation/` | Sec. 4.2, 5.2.1, App. B | EE fits (final and base model), Table 5, Figures 10, 15, 16 |
| `04_sampling/` | Sec. 4.1 | Calibration of the sample spacing, drawing of the synthetic grids |
| `05_evaluation/` | Sec. 4.5, 5.1–5.3, App. A | Fitted terms, held-out statistics, robustness, Figures 9, 11–14, Tables 6–9 |
| `results/` | | Small result files of the thesis (see below) |
| `figures/` | | All figures as used in the thesis |

Every folder contains one notebook (`*.ipynb`) that produces the figures and tables of its
chapter from the stored results, with all outputs saved, so that they can be read directly on
GitHub. Every script starts with a short description of what it does, its inputs and outputs.

## Installation and data

```
pip install -r requirements.txt
```

The raw data are the tables `buses.csv`, `lines.csv` and `transformers.csv` of the prebuilt
electricity network for PyPSA-Eur based on OpenStreetMap data, version 0.7 (Xiong et al. 2025,
DOI [10.5281/zenodo.18619025](https://doi.org/10.5281/zenodo.18619025)). Place them in
`data/pypsa_eur/` or set the environment variable `ERGM_GRIDS_DATA` to their folder. Large
intermediate files (EE trajectories, samples) are written to `output/` (or `ERGM_GRIDS_OUTPUT`);
neither folder is part of the repository. The synthetic grids (about 1 GB) are not part of the
repository.

## Running the analysis

All commands are run from the repository root. Runtimes refer to one CPU core.

| Step | Command | Runtime |
|---|---|---|
| Descriptive analysis | `python 01_descriptive/descriptive_metrics.py`, then `tables_descriptive.py` and the `fig*.py` scripts | minutes |
| Term selection, stage 1 | `python 02_selection/prefilter_lasso.py --variants original grid-extended no-reweight` | ~5 min |
| Term selection, stage 2 | `python 02_selection/forward_selection.py` | several hours |
| Stability tests | `python 02_selection/stability_tests.py` | ~1 h |
| EE fits | `python 03_estimation/fit_models.py --model final` and `--model base`; Serbia: `--countries RS --alpha 0.0002` | 2 min (AL) to hours (ES) per country |
| Trajectories | `python 03_estimation/export_trajectories.py` | seconds |
| Spacing calibration | `python 04_sampling/calibrate_spacing.py` | ~8 h |
| Sampling | `python 04_sampling/draw_samples.py --model final` and `--model base` | hours |
| Evaluation | `python 05_evaluation/fitted_terms_tau.py`, `evaluate_heldout.py`, `robustness_curves.py`, `degree_distribution_samples.py` | ~1–2 h |
| Figures and tables | `fig*.py`, `table*.py` in each folder, or the notebooks | minutes |

The figure and table scripts only read `results/`, so they run without refitting or resampling.

## Where each result of the thesis comes from

| Thesis | Script | Stored result |
|---|---|---|
| Fig. 1 (Austria example) | `01_descriptive/fig1_austria_example.py` | `results/descriptive/austria_sample_599.pkl` |
| Fig. 2 (Portugal, edge removal) | `01_descriptive/fig2_edge_removal_example.py` | — |
| Fig. 3 (Europe map) | `01_descriptive/fig3_europe_map.py` | — |
| Figs. 4, 6, 7; Tables 1–2; pan-European bridges | `01_descriptive/descriptive_metrics.py`, `tables_descriptive.py`, `fig4/6/7_*.py` | `results/descriptive/`, `results/tables/table1*, table2*` |
| Fig. 5 (degree distributions) | `01_descriptive/fig5_degree_distribution.py` | — |
| LASSO pre-filter (Sec. 4.4) and its footnote | `02_selection/prefilter_lasso.py` | `results/selection/lasso_coefficients.csv` |
| Forward selection (Sec. 5.1) | `02_selection/forward_selection.py` | `results/selection/forward_selection_trace_thesis.csv` |
| sigma² and AKS stability (Sec. 5.1) | `02_selection/stability_tests.py --part b` | `results/selection/reduced_model_sweeps.csv` |
| Fig. 8 (t1 sweep) | `02_selection/stability_tests.py --part a`, `fig_t1_sweep.py` | `results/selection/t1_sweep_thesis.csv` |
| Score 0.137 of the final model (Sec. 5.1) | `02_selection/score_final_model.py` | — |
| beta_bar, Table 5 | `03_estimation/fit_models.py`, `table_beta_bar.py` | `results/beta_bar/`, `results/tables/table5*` |
| Figs. 10, 15, 16 (trajectories) | `03_estimation/export_trajectories.py`, `fig_trajectories.py` | `results/trajectories/` |
| Sample spacing s(n) (Sec. 4.1) | `04_sampling/calibrate_spacing.py`, `check_spacing.py` | `results/sampling/` |
| Fig. 9 (base vs. final model) | `05_evaluation/evaluate_heldout.py`, `fig9_base_vs_final.py` | `results/evaluation/heldout_per_statistic.csv` |
| Fig. 11 (degree distribution of samples) | `05_evaluation/degree_distribution_samples.py`, `fig11_*.py` | `results/evaluation/degree_distribution_*.npz` |
| Figs. 12–13 (z vs. n) | `05_evaluation/fig12_13_z_vs_n.py` | `results/evaluation/heldout_per_statistic.csv` |
| Fig. 14 (S(f), BA/DE/ES) | `05_evaluation/robustness_curves.py`, `fig14_*.py` | `results/evaluation/robustness_*` |
| Tables 6–7 (held-out statistics, R; Sections 5.2.2, 5.3) | `05_evaluation/evaluate_heldout.py`, `tables_appendix.py` | `results/evaluation/heldout_per_statistic.csv`, `results/tables/table6*, table7*` |
| Tables 8–9 (fitted terms, Appendix A) | `05_evaluation/fitted_terms_tau.py`, `tables_appendix.py` | `results/evaluation/tau_fitted_terms.csv`, `results/tables/table8*, table9*` |

## Use of AI tools

The code was written with the support of the AI coding assistant Claude Code (Anthropic), as
described in the statement on the use of artificial intelligence in the thesis. The research
design, the analyses and all decisions on content are the author's.
