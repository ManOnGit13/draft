"""Interactive demo: type user ids / create new users while the program runs.

  python scripts/interactive_demo.py                          # synthetic data
  python scripts/interactive_demo.py --movielens data/ml-100k # real MovieLens
"""
import _bootstrap  # noqa: F401
import argparse

from recsys.data import MovieLensLoader, SyntheticLoader
from recsys.exceptions import RecsysError
from recsys.service import RecommendationService, ServiceConfig


def show(recs):
    for n, r in enumerate(recs, 1):
        tag = "  (phổ biến, chưa cá nhân hóa)" if r.is_fallback else ""
        print(f"  {n:>2}. [{r.item_id}] {r.title}  score={r.score:.3f}{tag}")


def ask_k(default=5):
    raw = input(f"Số phim muốn nhận k [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return raw          # let the service report InvalidParameterError


def parse_ratings(text):
    """'m10:5, m25:4, m3' -> {'m10': 5.0, 'm25': 4.0, 'm3': 5.0}"""
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        item, _, rating = part.partition(":")
        out[item.strip()] = float(rating) if rating.strip() else 5.0
    return out


def existing_user(service):
    uid = input("Nhập user id (ví dụ u0): ").strip()
    k = ask_k()
    show(service.recommend(uid, k))
    if not service.has_user(uid):
        print("  -> User này chưa có trong dữ liệu nên chỉ nhận được phim phổ biến. "
              "Chọn mục 2 để tạo user mới.")


def new_user(service):
    uid = input("Đặt tên user mới: ").strip()
    print("Một số phim phổ biến để bạn chọn:")
    show(service.popular_items(10))
    print("Nhập các phim bạn đã xem theo dạng  id:điểm  (điểm 1-5, bỏ trống = 5).")
    print("Ví dụ:  m12:5, m40:4, m7:2")
    try:
        ratings = parse_ratings(input("> "))
    except ValueError:
        print("Sai định dạng điểm, điểm phải là số.")
        return
    k = ask_k()
    print("Đang huấn luyện lại với user mới...")
    service.add_user(uid, ratings)
    print(f"Gợi ý cho {uid}:")
    show(service.recommend(uid, k))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--movielens")
    p.add_argument("--variant", default="100k")
    a = p.parse_args()
    dataset = MovieLensLoader(a.movielens, a.variant).load() if a.movielens \
        else SyntheticLoader().load()
    print("Đang huấn luyện mô hình...")
    service = RecommendationService(ServiceConfig("mf", {"n_epochs": 20})).train(dataset)
    ids = service.user_ids()
    print(f"Sẵn sàng. Có {len(ids)} user, ví dụ: {', '.join(ids[:5])}")

    actions = {"1": existing_user, "2": new_user}
    while True:
        print("\n1) Gợi ý cho user có sẵn   2) Tạo user mới   3) Thoát")
        choice = input("Chọn: ").strip()
        if choice == "3":
            break
        if choice not in actions:
            print("Chọn 1, 2 hoặc 3.")
            continue
        try:
            actions[choice](service)
        except RecsysError as exc:                      # expected, user-facing errors
            print(f"Lỗi ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    main()
