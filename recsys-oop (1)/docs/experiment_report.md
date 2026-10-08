# Experimental Report

## 1. Setup
| Item | Value |
|---|---|
| Data | **Synthetic movie-like dataset** (`SyntheticLoader`, 600 users × 400 items, ~23.5k interactions, genre-based taste + Zipf popularity). *Generated because the build environment had no internet access.* |
| Split | per-user random hold-out, 20 %, seed-controlled (`RandomSplitter`) |
| Positive feedback | rating ≥ 4 |
| Candidates | all items the user has not seen in training (full ranking) |
| Metrics | Recall@10, NDCG@10, AUC (same code for every model) |
| Repetitions | 5 seeds (data, split and model seed all = 0…4); mean ± std |
| Ours | `PopularityRecommender`; `MatrixFactorizationRecommender` (BPR, 32 factors, lr 0.05, reg 0.01, 30 epochs, batch 256) |
| Benchmark | Cornac `BPR`, `MF` through `CornacRecommender` adapter (same Evaluator) |

## 2. Results — our models (synthetic data, 5 seeds)
| Model | Recall@10 | NDCG@10 | AUC | Train time (s) |
|---|---|---|---|---|
| Popularity (ours) | 0.1398 ± 0.0021 | 0.1211 ± 0.0041 | 0.6717 ± 0.0058 | 0.002 |
| MF-BPR (ours) | 0.3123 ± 0.0056 | 0.2963 ± 0.0066 | 0.8946 ± 0.0017 | 1.59 |

Single run with seed 42: see `results/results_synthetic.md`.

## 3. Comparison with Cornac — **NOT YET RUN**
| Model | Source | Recall@10 | NDCG@10 | AUC | Train time |
|---|---|---|---|---|---|
| Popularity | ours | _run on MovieLens_ | | | |
| MF-BPR | ours | | | | |
| BPR | Cornac | | | | |
| MF | Cornac | | | | |

Cornac is not installed in the environment where this project was built, so these rows are empty.
To fill them:
```bash
pip install cornac
# download MovieLens 100K and unzip to data/ml-100k
python scripts/run_experiment.py --movielens data/ml-100k --out results/results_ml100k.md
```
The script adds the Cornac rows automatically. The adapter code path is unit-tested against a
stub `cornac` module (`tests/test_cornac_adapter.py`), but it has **not** been run against the
real library — if Cornac's API differs (e.g. parameter names such as `k`, `max_iter`,
`lambda_reg`), adjust the defaults in `CornacBenchmark`.

## 4. Analysis (synthetic data)
- MF-BPR more than doubles Popularity on Recall/NDCG and lifts AUC from ~0.67 to ~0.89. This is expected: the synthetic users have genre-specific taste that a non-personalised model cannot capture.
- Standard deviations are small (< 0.007) across seeds, so the gap is not noise.
- Caveat: the synthetic data was designed with latent structure that MF can learn, so these numbers **say nothing about performance on real data**. On MovieLens, popularity is typically a much stronger baseline and the gap is smaller.

## 5. Fairness checklist for the Cornac comparison
- [x] same dataset object, same train/test split (one `RandomSplitter` call)
- [x] same positive threshold; Cornac gets binary implicit feedback
- [x] same candidates (unseen items) and same metric implementation
- [x] fixed seeds, hyper-parameters recorded here
- [ ] hyper-parameter tuning for both sides (not done; defaults only)
- [ ] results on real data (to be filled in)
