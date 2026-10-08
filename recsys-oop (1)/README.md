# recsys-oop — Object-Oriented Recommendation System

A small recommender system built from scratch with OOP in Python
(data layer → model → evaluation → service), benchmarked against
[Cornac](https://github.com/PreferredAI/cornac) (used as-is, never modified).

## Install
```bash
pip install -r requirements.txt      # numpy, scipy
pip install cornac                   # optional: only for the Cornac benchmark
```

## Quick start
```python
from recsys.data import MovieLensLoader
from recsys.service import RecommendationService, ServiceConfig

dataset = MovieLensLoader("data/ml-100k").load()          # or SyntheticLoader().load()
service = RecommendationService(ServiceConfig(model_name="mf")).train(dataset)
for r in service.recommend("196", k=5):
    print(r.item_id, r.title, round(r.score, 3))
```
Offline demos (no download needed):
- `python scripts/demo_service.py` — scripted demo (normal + error cases)
- `python scripts/interactive_demo.py` — **type your own input**: recommend for an existing user,
  or create a new user by entering movies you liked (`m12:5, m40:4`), then get personalised top-K.
  Add `--movielens data/ml-100k` to use real data.

Adding a new user from code:
```python
service.add_user("alice", {"m1": 5, "m2": 4})   # registers the user and retrains
service.recommend("alice", k=5)
```

## Run the experiment (ours vs Cornac)
```bash
# download + unzip MovieLens 100K from https://grouplens.org/datasets/movielens/100k/
python scripts/run_experiment.py --movielens data/ml-100k
python scripts/run_experiment.py                  # synthetic data, works offline
```
Cornac rows (BPR, MF) are added automatically when `cornac` is installed.
Results are written to `results/results.md`.

## Tests
```bash
python -m unittest discover -s tests -t .        # 83 tests, no extra dependency
```
(They are also pytest-compatible: `pytest tests`.)

## Layout
| Path | Layer | Content |
|---|---|---|
| `src/recsys/data` | Data | `User/Item/Interaction`, `Dataset`, `IdMapper`, loaders, splitters |
| `src/recsys/models` | Model | `BaseRecommender`, `PopularityRecommender`, `MatrixFactorizationRecommender` (BPR), `ModelFactory` |
| `src/recsys/evaluation` | Evaluation | `Recall@K`, `NDCG@K`, `AUC`, `Evaluator` |
| `src/recsys/service` | Service | `ServiceConfig`, `RecommendationService` (facade) |
| `src/recsys/benchmark` | Benchmark | `CornacRecommender` adapter, `CornacBenchmark` |
| `src/recsys/exceptions.py` | Common | `RecsysError` hierarchy |

## Error handling summary
| Situation | Exception |
|---|---|
| unknown user / item | `UnknownUserError` / `UnknownItemError` (service falls back to popularity for unknown users unless `cold_start_fallback=false`) |
| `k <= 0`, non-integer `k` | `InvalidParameterError` |
| empty / malformed / duplicate / too-small data | `DataError` |
| bad hyper-parameters, unknown model, bad config file | `ConfigError` |
| recommend/evaluate before training | `ModelNotFittedError` |

All inherit from `RecsysError`.

## Docs
`docs/design_report.md`, `docs/experiment_report.md`, `docs/uml/class_diagram.md`

## Team / course checklist
- [ ] Add instructor GitHub account `lthoang` as collaborator
- [ ] Fill team names in `docs/design_report.md`
