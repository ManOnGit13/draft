"""Run our models (and Cornac, if installed) on the same split + metrics.

  python scripts/run_experiment.py                         # synthetic data
  python scripts/run_experiment.py --movielens ./data/ml-100k
"""
import _bootstrap  # noqa: F401
import argparse, json, os, time

from recsys.benchmark import CornacBenchmark, cornac_available
from recsys.data import MovieLensLoader, RandomSplitter, SyntheticLoader
from recsys.evaluation import AUC, Evaluator, NDCGAtK, RecallAtK
from recsys.models import ModelFactory

p = argparse.ArgumentParser()
p.add_argument("--movielens", help="path to an extracted MovieLens folder")
p.add_argument("--variant", default="100k", choices=["100k", "1m"])
p.add_argument("--k", type=int, default=10)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="results/results.md")
a = p.parse_args()

dataset = (MovieLensLoader(a.movielens, a.variant).load() if a.movielens
           else SyntheticLoader(seed=a.seed).load())
source = f"MovieLens {a.variant}" if a.movielens else "synthetic (offline)"
train, test = RandomSplitter(0.2, seed=a.seed).split(dataset)
print(f"Data: {source} {dataset.summary()}\nTrain={train.n_interactions} Test={test.n_interactions}")

evaluator = Evaluator([RecallAtK(a.k), NDCGAtK(a.k), AUC()])
rows = []
for label, name, params in [("Popularity (ours)", "popularity", {}),
                            ("MF-BPR (ours)", "mf", {"n_factors": 32, "n_epochs": 30,
                                                     "learning_rate": 0.05, "reg": 0.01,
                                                     "seed": a.seed})]:
    model = ModelFactory.create(name, **params)
    t0 = time.perf_counter(); model.fit(train); fit_s = time.perf_counter() - t0
    res = evaluator.evaluate(model, test, label)
    res.seconds = fit_s
    rows.append(res)

if cornac_available():
    rows += CornacBenchmark(evaluator).run(train, test)
else:
    print("NOTE: cornac not installed -> Cornac rows skipped (pip install cornac)")

names = list(rows[0].scores)
lines = [f"Dataset: {source}; K={a.k}; seed={a.seed}; users evaluated={rows[0].n_users_evaluated}", "",
         "| Model | " + " | ".join(names) + " | Train time (s) |",
         "|---|" + "---|" * (len(names) + 1)]
for r in rows:
    lines.append(f"| {r.model} | " + " | ".join(f"{r.scores[n]:.4f}" for n in names)
                 + f" | {r.seconds:.2f} |")
text = "\n".join(lines)
print("\n" + text)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
