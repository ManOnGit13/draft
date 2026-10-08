"""Demo of the RecommendationService facade (MovieLens 100K)."""
import _bootstrap  # noqa: F401
from recsys.data import MovieLensLoader
from recsys.exceptions import RecsysError
from recsys.service import RecommendationService, ServiceConfig

USER = "196"   # user có thật trong MovieLens 100K

service = RecommendationService(ServiceConfig(model_name="mf", model_params={"n_epochs": 20}))
try:
    service.recommend(USER)
except RecsysError as e:
    print(f"[expected error before training] {type(e).__name__}: {e}")

service.train(MovieLensLoader("data/ml-100k").load())
print(f"\nTop-5 for user {USER}:")
for r in service.recommend(USER, k=5):
    print(f"  {r.item_id:>5}  {r.title:<40} score={r.score:.3f}")

print("\nUnknown user 'ghost' (cold-start fallback):")
for r in service.recommend("ghost", k=3):
    print(f"  {r.item_id:>5}  {r.title:<40} fallback={r.is_fallback}")

for bad_k in (0, -1, "ten"):
    try:
        service.recommend(USER, k=bad_k)
    except RecsysError as e:
        print(f"\nk={bad_k!r} -> {type(e).__name__}: {e}")