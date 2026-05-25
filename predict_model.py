import argparse
import json
from pathlib import Path

from joblib import load
from surprise import dump as surprise_dump


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "artifacts" / "svd_model.joblib"
LEGACY_MODEL = ROOT / "artifacts" / "svd_model.pkl"
DEFAULT_USER_ID = 1331154


def load_saved_predictions(user_id: int) -> dict | None:
    path = ROOT / "artifacts" / f"predictions_user_{user_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_saved_predictions(user_id: int, movie_ids: list[int] | None) -> bool:
    saved = load_saved_predictions(user_id)
    if saved is None:
        return False

    recommendations = saved.get("recommendations", [])
    if movie_ids:
        wanted = set(movie_ids)
        recommendations = [
            item for item in recommendations if int(item["movie_id"]) in wanted
        ]

    print(
        json.dumps(
            {
                "source": "saved_prediction_artifact",
                "message": (
                    "svd_model.joblib is not available yet, so these values come from "
                    "artifacts/predictions_user_1331154.json."
                ),
                "predictions": recommendations,
            },
            indent=2,
        )
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ratings from a dumped Surprise model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--user-id", type=int, default=DEFAULT_USER_ID)
    parser.add_argument(
        "--movie-ids",
        help=(
            "Comma-separated movie ids. If omitted, saved top predictions are shown "
            "when the dumped model is not available."
        ),
    )
    args = parser.parse_args()

    movie_ids = (
        [int(item.strip()) for item in args.movie_ids.split(",") if item.strip()]
        if args.movie_ids
        else None
    )

    model_path = args.model
    if not model_path.exists() and model_path == DEFAULT_MODEL and LEGACY_MODEL.exists():
        model_path = LEGACY_MODEL

    if not model_path.exists():
        if print_saved_predictions(args.user_id, movie_ids):
            return
        raise SystemExit(
            f"Model file not found: {args.model}\n"
            "Run dump_model.py after placing combined_data_1.txt next to it."
        )

    if model_path.suffix == ".pkl":
        _, model = surprise_dump.load(str(model_path))
    else:
        model = load(model_path)
    if not movie_ids:
        raise SystemExit("--movie-ids is required when svd_model.joblib is available.")

    predictions = [
        {
            "movie_id": movie_id,
            "estimate_score": round(float(model.predict(args.user_id, movie_id).est), 6),
        }
        for movie_id in movie_ids
    ]
    print(json.dumps({"predictions": predictions}))


if __name__ == "__main__":
    main()
