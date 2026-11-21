## DP-FZOO: Differentially Private Fast Zeroth-Order Optimization

DP-FZOO is a differentially private version of [FZOO](https://arxiv.org/abs/2506.09034) (Fast Zeroth-Order Optimizer),
built on the [DPZero](https://arxiv.org/abs/2310.09639) RoBERTa code, which is in turn based on [MeZO](https://github.com/princeton-nlp/MeZO).
Each step uses `N` random Rademacher directions, clips every per-example finite difference at `C/sqrt(N)`,
adds Gaussian noise to the batch average, and normalizes the step by the standard deviation of the `N` noisy values.
The `N` perturbed forward passes can be run as one batched pass, optionally with fused Triton kernels,
and the update can optionally use Nesterov momentum (post-processing, so it costs no extra privacy).
The code is for fine-tuning RoBERTa-large on SST-2, SST-5, SNLI, MNLI, RTE and TREC.



### Installation

This code is tested on Linux (WSL) with an RTX 5070 (12 GB), `python 3.9`, and the DPZero stack:
`torch` (CUDA build for your GPU), `transformers==4.28.1`, and `opacus==1.4.0`.
Triton is installed together with `torch` and is only needed for the fused kernels
(everything also runs without it, see `ZO_FUSED=0` below).

Create the environment with the commands below (the same file as DPZero's `environments.yml`):
```bash
conda env create -n dpzero -f environments.yml
conda activate dpzero
```
If your GPU needs a newer CUDA/PyTorch than the one in `environments.yml` (e.g. RTX 50-series),
install a matching `torch` build from https://pytorch.org after creating the environment.

### Preparations

The datasets can be found [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar).
Download and extract them to `./data/original`, and create the k-shot splits, by running:
```bash
cd data
bash prepare_datasets.sh
cd ..
```
This creates `data/k-shot-1k-test/<TASK>/<K>-<SEED>` for seeds 13, 21, 42, 87, 100 (K = 16 and 512).

### Layout

```
src/trainer_fzoo.py      DP-FZOO trainer (sampling, clipping, noise, normalization, update, momentum)
src/fzoo_step.py         clipping / privatization / update helpers
src/zo_parallel.py       batched forward over the N directions
src/zo_kernels.py        fused Triton kernels (perturb, update, Nesterov update)
examples/                run scripts (DP-FZOO, DPZero)
```

### Examples

Run DP-FZOO (batched forward, fused kernels, N = 4, batch size 32, 10,000 steps) on SST-2, seed 42:
```bash
FZOO_PARALLEL=1 TASKS="SST-2" SEEDS="42" bash examples/fzoo_parallel_all.sh
```
Other tasks: change `TASKS` to `sst-5`, `SNLI`, `MNLI`, `RTE`, or `trec`. Several tasks and seeds
(the default is `SEEDS="13 21 42 87 100"`) can be given as a space-separated list.
Logs and a `summary.txt` (dev accuracy, test accuracy, test loss per run) are written to `LOG_DIR`
(default `logs_fzoo_best_accuracy_2`).

Common options (environment variables):
```bash
# privacy budget (default 2.0), clip threshold C (default 200), delta (default 1e-5)
DPZERO_PRIVACY_EPS=6.0 DPZERO_THRESHOLD=200.0 bash examples/fzoo_parallel_all.sh

# Nesterov momentum with beta = 0.9 (0 = off)
FZOO_MOMENTUM=0.9 bash examples/fzoo_parallel_all.sh

# number of directions and batch size
FZOO_N=4 BS=32 bash examples/fzoo_parallel_all.sh

# fused Triton kernels off (plain PyTorch batched forward), or sequential trainer
ZO_FUSED=0 bash examples/fzoo_parallel_all.sh
FZOO_PARALLEL=0 bash examples/fzoo_parallel_all.sh

# quick 50-step timing run / print the plan without running / re-print the summary from existing logs
STEP=50 EVAL_STEP=50 SEEDS="42" bash examples/fzoo_parallel_all.sh
DRY_RUN=1 bash examples/fzoo_parallel_all.sh
SUMMARY_ONLY=1 bash examples/fzoo_parallel_all.sh
```
The trainer prints one `DP-FZOO config:` line at the start of each run listing the options in effect;
check it to make sure they were applied.

Compare against the DPZero baseline (two forward passes per step) with the same hyperparameters:
```bash
bash run_models.sh examples/dpzero.sh
```

### Acknowledgements

Based on [DPZero](https://arxiv.org/abs/2310.09639) and [MeZO](https://github.com/princeton-nlp/MeZO). The batched forward is adapted from the FZOO authors' parallelization script (written for OPT) and modified for RoBERTa.

### References

* Malladi et al., *Fine-Tuning Language Models with Just Forward Passes* (MeZO), NeurIPS 2023. arXiv:2305.17333
* Dang et al., *FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed*, 2025. arXiv:2506.09034
* Zhang et al., *DPZero: Private Fine-Tuning of Language Models without Backpropagation*, ICML 2024. arXiv:2310.09639
