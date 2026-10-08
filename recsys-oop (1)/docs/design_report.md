# Design Report

Team: _<fill in names / student IDs>_  ·  Domain: movies (MovieLens; synthetic data for offline use)

## 1. Goals
A small, modular, testable recommender where the algorithm, data source, split strategy
and metrics can each be replaced without touching the rest.

## 2. Architecture (4 layers + benchmark)
```
Data  ->  Model  ->  Evaluation          Service (facade over Data+Model)
                                         Benchmark (adapter around Cornac)
```
Dependencies only point downward: `service` uses `models`/`data`; `evaluation` uses `models`/`data`;
`data` knows nothing about models. See `docs/uml/class_diagram.md`.

| Layer | Responsibility | Main classes |
|---|---|---|
| Data | represent and load data, split train/test | `User`, `Item`, `Interaction`, `Dataset`, `IdMapper`, `BaseLoader`, `BaseSplitter` |
| Model | learn + score + rank | `BaseRecommender`, `PopularityRecommender`, `MatrixFactorizationRecommender` |
| Evaluation | measure ranking quality | `BaseMetric`, `RecallAtK`, `NDCGAtK`, `AUC`, `Evaluator` |
| Service | simple top-K API, config | `RecommendationService`, `ServiceConfig` |
| Benchmark | compare with Cornac | `CornacRecommender`, `CornacBenchmark` |

## 3. OOP principles → where they appear
| Principle | Evidence in code |
|---|---|
| **Encapsulation** | `Dataset` hides index arrays and CSR matrices behind id-based methods; MF keeps `_P/_Q/_b` private; entities are frozen dataclasses validated in `__post_init__`; `score()` returns a copy. |
| **Abstraction** | `BaseRecommender`, `BaseMetric`, `BaseLoader`, `BaseSplitter` are ABCs. `RecommendationService` exposes only `train()` and `recommend()`. |
| **Composition** | `Dataset` has `IdMapper`s; `Evaluator` has a list of metrics; `RecommendationService` has a model, a config and a fallback model. |
| **Inheritance / polymorphism** | All models/metrics/loaders/splitters are interchangeable. `CornacRecommender` extends `BaseRecommender`, so the same `Evaluator` runs on our and Cornac's models. `tests/test_models.py` runs one shared contract (`ModelContract`) against every model (Liskov check). |
| **Separation of responsibilities** | Loaders don't split, splitters don't train, models don't evaluate, metrics don't know models (they only see ranked ids). |
| **Exception handling** | One hierarchy rooted at `RecsysError`; validation at the boundary (entities, config, `k`); errors carry actionable messages. |

## 4. Design patterns used
- **Template Method** — `BaseRecommender.fit/rank/recommend` do validation, seen-item masking and sorting; subclasses implement only `_fit` and `_score_user`. `BaseLoader.load` / `BaseSplitter.split` likewise.
- **Factory + registry** — `@register_model("mf")` + `ModelFactory.create(name, **params)`; adding a model needs no change elsewhere.
- **Strategy** — splitters, metrics and models are swappable strategies.
- **Facade** — `RecommendationService`.
- **Adapter** — `CornacRecommender` wraps Cornac behind our interface; Cornac is not modified.

## 5. Key design decisions
1. **Dataset fixes the user/item universe.** Train and test share it (`with_interactions`), so indices are consistent and a user with no training data is still *known* (cold user) instead of crashing.
2. **Implicit feedback.** `rating >= rating_threshold` (default 4.0) is positive; *all* interactions count as "seen" and are excluded from recommendations.
3. **Metrics operate on plain rankings** (list of ids vs. set of relevant ids), not on models, so they are trivially unit-testable with hand-computed values.
4. **Unknown user policy.** Models raise `UnknownUserError`; the *service* decides to fall back to popularity (configurable). Policy lives in the facade, mechanism in the model.
5. **Empty-input policy for metrics.** Empty ranking → 0.0; empty relevant set → `InvalidParameterError` (the `Evaluator` skips such users and reports how many); AUC with no negatives → NaN, ignored by `nanmean`.
6. **MF uses BPR loss in pure NumPy** (mini-batch SGD with negative sampling, item bias). Chosen because the task is ranking and because Cornac's BPR is the natural benchmark.
7. **Failure leaves a clean state.** `fit()` marks the model unfitted before training, so a failed fit cannot leave a half-trained model that looks usable.
8. **No third-party test framework needed** (`unittest`), tests are also pytest-compatible.

## 6. Error handling matrix
See README. Each row is covered by at least one test (`tests/test_*.py`, 83 tests).

## 7. Limitations / future work
- Random per-user hold-out; no validation set or hyper-parameter search.
- Evaluation ranks *all* unseen items (full ranking), which is O(items) per user.
- MF training is vectorised NumPy; much slower than Cornac's compiled implementations on large data.
- Only implicit-feedback models; no side-information models yet.

## 8. Work split
| Member | Main | Review |
|---|---|---|
| A | Data layer | Service |
| B | Model layer | Data |
| C | Evaluation + Cornac benchmark + experiment report | Model |
| D | Service, config, exceptions, README, tests | Evaluation |
