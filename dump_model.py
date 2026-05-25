import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from surprise import Dataset, Reader, SVD


ROOT = Path(__file__).resolve().parent
DEFAULT_RATINGS = ROOT / "combined_data_1.txt"
DEFAULT_TITLES = ROOT / "movie_titles.csv"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_ratings(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        header=None,
        names=["Cust_Id", "Rating"],
        usecols=[0, 1],
        dtype={"Cust_Id": "string", "Rating": "float32"},
    )

    movie_markers = raw["Rating"].isna()
    raw["Movie_Id"] = (
        raw["Cust_Id"]
        .where(movie_markers)
        .ffill()
        .str.replace(":", "", regex=False)
        .astype("int32")
    )

    ratings = raw.loc[~movie_markers].copy()
    ratings["Cust_Id"] = ratings["Cust_Id"].astype("int32")
    ratings["Movie_Id"] = ratings["Movie_Id"].astype("int32")
    return ratings


def trim_sparse_entities(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    movie_summary = ratings.groupby("Movie_Id")["Rating"].agg(["count"])
    movie_benchmark = round(movie_summary["count"].quantile(0.6), 0)
    drop_movie_list = movie_summary[movie_summary["count"] < movie_benchmark].index

    cust_summary = ratings.groupby("Cust_Id")["Rating"].agg(["count"])
    cust_benchmark = round(cust_summary["count"].quantile(0.6), 0)
    drop_cust_list = cust_summary[cust_summary["count"] < cust_benchmark].index

    trimmed = ratings.loc[
        ~ratings["Movie_Id"].isin(drop_movie_list)
        & ~ratings["Cust_Id"].isin(drop_cust_list)
    ].copy()
    return trimmed, drop_movie_list


def train_svd(ratings: pd.DataFrame, train_rows: int | None) -> SVD:
    train_df = ratings[["Cust_Id", "Movie_Id", "Rating"]]
    if train_rows:
        train_df = train_df.head(train_rows)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df, reader)
    trainset = data.build_full_trainset()

    model = SVD(random_state=42)
    model.fit(trainset)
    return model


def build_predictions(
    model: SVD,
    ratings: pd.DataFrame,
    dropped_movies: pd.Index,
    titles_path: Path,
    user_id: int,
    limit: int,
    exclude_rated: bool,
) -> list[dict]:
    titles = pd.read_csv(
        titles_path,
        encoding="latin",
        header=None,
        usecols=[0, 1, 2],
        names=["Movie_Id", "Year", "Name"],
    )
    candidates = titles.loc[~titles["Movie_Id"].isin(dropped_movies)].copy()

    if exclude_rated:
        rated_movies = ratings.loc[ratings["Cust_Id"] == user_id, "Movie_Id"].unique()
        candidates = candidates.loc[~candidates["Movie_Id"].isin(rated_movies)].copy()

    candidates["Estimate_Score"] = candidates["Movie_Id"].apply(
        lambda movie_id: model.predict(user_id, int(movie_id)).est
    )
    candidates = candidates.sort_values("Estimate_Score", ascending=False).head(limit)

    predictions = []
    for _, row in candidates.iterrows():
        est = float(row["Estimate_Score"])
        year = None if pd.isna(row["Year"]) else int(row["Year"])
        predictions.append(
            {
                "movie_id": int(row["Movie_Id"]),
                "title": str(row["Name"]),
                "year": year,
                "estimate_score": round(est, 6),
                "score": round((est / 5.0) * 100),
                "reason": f"SVD predicted {est:.2f}/5 for user {user_id}",
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and dump the Netflix SVD model used by the web app."
    )
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--titles", type=Path, default=DEFAULT_TITLES)
    parser.add_argument("--user-id", type=int, default=1331154)
    parser.add_argument("--train-rows", type=int, default=100000)
    parser.add_argument("--prediction-limit", type=int, default=50)
    parser.add_argument("--exclude-rated", action="store_true")
    args = parser.parse_args()

    if not args.ratings.exists():
        saved_prediction = ARTIFACTS_DIR / f"predictions_user_{args.user_id}.json"
        message = (
            f"Ratings file not found: {args.ratings}\n\n"
            "To train and dump the real Surprise SVD model, download/extract the "
            "Netflix Prize ratings file named combined_data_1.txt and place it here:\n"
            f"  {ROOT / 'combined_data_1.txt'}\n\n"
            "Then run:\n"
            f"  python dump_model.py --ratings combined_data_1.txt --user-id {args.user_id}\n"
        )
        if saved_prediction.exists():
            message += (
                "\nThe web page can still use the saved notebook prediction artifact:\n"
                f"  {saved_prediction}\n"
            )
        raise SystemExit(message)
    if not args.titles.exists():
        raise FileNotFoundError(f"Movie title file not found: {args.titles}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("Loading Netflix ratings...")
    ratings = load_ratings(args.ratings)

    print("Applying notebook trimming rules...")
    trimmed, dropped_movies = trim_sparse_entities(ratings)

    print("Training Surprise SVD model...")
    model = train_svd(trimmed, args.train_rows)

    model_path = ARTIFACTS_DIR / "svd_model.joblib"
    dump(model, model_path)

    print(f"Building predictions for user {args.user_id}...")
    predictions = build_predictions(
        model=model,
        ratings=trimmed,
        dropped_movies=dropped_movies,
        titles_path=args.titles,
        user_id=args.user_id,
        limit=args.prediction_limit,
        exclude_rated=args.exclude_rated,
    )

    prediction_path = ARTIFACTS_DIR / f"predictions_user_{args.user_id}.json"
    payload = {
        "model": "surprise.SVD",
        "source_notebook": "Netflix_.ipynb",
        "user_id": args.user_id,
        "train_rows": args.train_rows,
        "prediction_limit": args.prediction_limit,
        "exclude_rated": args.exclude_rated,
        "recommendations": predictions,
    }
    prediction_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest = {
        "model_path": str(model_path.relative_to(ROOT)),
        "model_library": "joblib",
        "model_class": "surprise.SVD",
        "latest_prediction_path": str(prediction_path.relative_to(ROOT)),
        "default_user_id": args.user_id,
        "train_rows": args.train_rows,
    }
    (ARTIFACTS_DIR / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Saved model: {model_path}")
    print(f"Saved predictions: {prediction_path}")


if __name__ == "__main__":
    main()
