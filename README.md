# Generative Design of Improved Unspecific Peroxygenases

Code, data, model checkpoints and results for the paper,

**Generative Machine Learning Harnesses Droplet Microfluidics to Design
Improved Enzymes Beyond the Screen**.
Pradeep M. Nair, Daniel M. Steinberg, Tiago Resende, Angelica Faith Suarez,
Helen Power, Cheng Soon Ong, Vijay Sairam, Shelly Hsiao-Ying Cheng, Lorivie
Fragata, Alessandro Serafini, Victor Oh, Say Hwa Tan, Robert E. Speight, Akbar
K. Vahidi.

Preprint: bioRxiv 2025.11.02.685536;
doi: https://doi.org/10.1101/2025.11.02.685536. The preprint was posted under
the earlier title "Ultrahigh throughput screening to train generative protein
models for engineering specificity into unspecific peroxygenases".

This repository contains what is needed to reproduce the results reported
in the paper, and nothing more.

The generative models are variational search distributions (VSD), trained on
ultrahigh throughput screening (uHTS) data of an unspecific peroxygenase
(*AbrUPO*) to propose variants with improved specificity for styrene oxide
over phenylacetaldehyde.


## Installation

Requires Python 3.10 or newer. From the root of this repository:

    python -m venv .env
    source .env/bin/activate
    pip install -e .

This installs the [VSD](https://github.com/csiro-funml/variationalsearch)
package (tag `ICLR`) from GitHub, along with PyTorch and the usual scientific
Python stack.

Training the transformer model in a reasonable time needs a GPU. Everything in
`analyse_results.ipynb` runs on CPU in a few minutes, apart from the
permutation tests (see `FAST_R` below).


## Reproducing the paper results

Both notebooks are run from the `upo` directory, and read data using paths
relative to it:

    cd upo

### Analysis of the validation experiments

`analyse_results.ipynb` produces the result tables and figures in the paper. It
loads the published LSTM checkpoints from `results/lstm` and the experimental
measurements from `data`, so it can be run on its own without retraining
anything:

    jupyter notebook analyse_results.ipynb

It writes its tables back into `results/lstm`:

| File | Contents |
| --- | --- |
| `selection.csv` | the R- and G-series variant selections |
| `all_results.csv` | per-variant measurements joined to model scores and ranks |
| `val_results.csv` | the original 10-variant validation set with model scores |
| `res_corr.csv`, `val_corr.csv` | Spearman correlations between model scores and measurements |
| `top5mean.csv`, `allmean.csv` | R vs. G group means with Brunner-Munzel and Kolmogorov-Smirnov p-values |
| `ROC_results.csv`, `PRC_results.csv` | bootstrapped ROC and precision-recall summaries |

Figures are displayed inline in the notebook.

Two settings near the top of the notebook are worth knowing about:

- `FAST_R = False` computes Spearman p-values by permutation test. This is the
  setting used for the paper, and it is slow. Setting it to `True` uses the
  asymptotic approximation instead, which is much faster but more biased.
- `RESPATH = Path("../results/lstm")` selects which set of model checkpoints to
  score with. The LSTM was used to select the final validation variants, so
  this is the path the paper's numbers come from.

### Training the generative models

`upo.ipynb` contains the uHTS data analysis and trains the LSTM- and
transformer-based variational search distributions, including the model
finetuning alternative to VSD refinement:

    jupyter notebook upo.ipynb

Select the model by setting `GEN_CLASS` to `LSTMProposal` or
`DTransformerProposal` near the top of the notebook. The output directory
follows that choice:

    RESPATH = Path("../results") / ("lstm" if GEN_CLASS is LSTMProposal else "tfm")

so a transformer run cannot overwrite the LSTM results.

**Each training stage is skipped if its checkpoint already exists.** Since
`RESPATH` points at the published checkpoints, running this notebook as-is
will *load* them rather than retrain, and only recompute the analysis around
them. That is what you want to reproduce the paper.

To genuinely retrain, point `RESPATH` at an empty directory of your own, then
point `RESPATH` in `analyse_results.ipynb` at the same place to score the
result. Alternatively, delete the `*.pt` files from `results/<model>` to
retrain in place — training is seeded (`SEED = 139`), but exact
reproducibility across different hardware and library versions is not
guaranteed, so expect `git diff` to show small numerical changes.

`upo.sh` is the Slurm batch script used to train the transformer on a GPU
cluster. It runs the notebook headlessly via `jupyter nbconvert`. The module
load line is specific to our cluster and will need adjusting.

### What reproduces exactly, and what does not

Running `upo.ipynb` against the published checkpoints reproduces the following
byte-for-byte (modulo a UTF-8 BOM, see below):

- `ranked_variants.csv` and `targets_spearman_corr.csv` — the uHTS data
  processing and rank aggregation are fully deterministic.
- The variants, measurements, `Prior_ll`, `Vdist_ll` and `Rank` columns of
  `20_variants_scored.csv`. The proposal models carry no dropout, so scoring
  a loaded checkpoint is exact.

The `CPE_ll` and `Acq` columns of `20_variants_scored.csv`, and the
correlations in `validation_spearman_corr.csv` that derive from them, are
deterministic on a re-run but **differ from the published values** by up to
about 2 nats.

The class probability estimator uses dropout (`dropoutp=0.1`). Scoring it is
only deterministic with the model in eval mode, which `fit_cpe` leaves it in
after training but which the checkpoint-loading path did not previously set.
The published column is statistically consistent with a single dropout-active
draw from `results/lstm/cpe.pt`, the checkpoint shipped here: it sits a median
0.8 standard deviations from the dropout mean, where a real draw is that
extreme about 40% of the time. The published values are therefore a one-off
noisy sample rather than the surrogate's actual scores, and are not
reproducible by anyone, ourselves included. The eval-mode values that a re-run
now produces are the correct ones.

This affects only the two columns above and the correlations derived from
them. `analyse_results.ipynb`, which produces the result tables and figures in
the paper, drops `CPE_ll` and `Acq` and correlates the measurements against
the proposal log-likelihoods (`VSD_LSTM`, `FT_LSTM`, `Prior_LSTM`) instead,
all of which reproduce exactly. The variant ranking is likewise unaffected:
`Rank` comes from the uHTS rank aggregation.

**`generated_upo.csv` does not reproduce.** Sequence generation samples from
the variational distribution, and `torch.manual_seed(SEED)` is set once at the
top of the notebook. How many random numbers have been drawn by the time
generation runs therefore depends on how much training happened first — and
when the checkpoints already exist, training is skipped. A re-run produces a
different, equally valid sample of roughly 4400 sequences rather than the
published one.

The published `results/lstm/generated_upo.csv` is the canonical set: it is the
one the reported variant selections were made from, and the one
`analyse_results.ipynb` reads. Do not regenerate it if you want to reproduce
the paper's numbers.

Two smaller caveats:

- `20_variants_scored.csv` gains a `Finetuned_ll` column on a re-run. The
  published file predates the finetuning baseline, so the regenerated file is
  a superset of it.
- The committed `ranked_variants.csv`, `generated_upo.csv` and
  `targets_spearman_corr.csv` have no UTF-8 BOM, while the notebook now writes
  with `encoding="utf-8-sig"`. A re-run will show a one-byte header diff on
  those files. This is cosmetic.

### Figure 1

`figure1.py` draws the illustrative fitness-landscape schematic:

    python figure1.py

It writes `figure1.svg` and `figure1.png` next to itself. The landscape is a
synthetic mixture model, not experimental data.


## Repository layout

- `data` — the uHTS and validation experimental data.
  - `CSIRO_UPO_Data_package_2024.xlsx` — the uHTS screen used to train the
    models.
  - `AbrUPO validation.xlsx` — the original 10-variant validation experiment.
  - `UPO for CSIRO 20250827.xlsx` — the R- and G-series validation experiment.
  - `subset_ptm_results_*.csv` — protein-model baseline scores for the
    validation variants.
- `results` — model checkpoints and outputs, one directory per generative
  model.
  - `results/lstm` — the LSTM results used for the final variant selection and
    for every model-derived number in the paper. `prior_LSTMProposal.pt` is the
    prior fit to the uHTS data, `vard_LSTMProposal.pt` the VSD posterior,
    `finetuned_LSTMProposal.pt` the finetuning baseline, and `cpe.pt` the class
    probability estimator surrogate.
  - `results/tfm` — a subset of results from the transformer model, provided
    for comparison. There is no finetuned transformer checkpoint.
- `upo` — the notebooks and supporting code.
  - `utils.py` — sequence and mutation encoding helpers.
  - `vis.py` — plotting helpers for the training notebook.
  - `roc_boot.py` — bootstrapped ROC and precision-recall curves.


## Licence

The source code is licensed under the Apache License, Version 2.0; see
`LICENSE`.

The data in `data/` and the checkpoints and results in `results/` are licensed
under the Creative Commons Attribution 4.0 International License; see
`LICENSE-DATA`.

If you use this work, please cite the paper above.
