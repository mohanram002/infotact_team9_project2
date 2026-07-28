import argparse
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

DATA_PATH = Path(__file__).resolve().parent / "notebook" / "clean_house_data.csv"

FEATURE_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat",
    "long", "sqft_living15", "sqft_lot15", "house_age", "renovated"
]


def load_dataset():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    return data


def build_training_frame(data):
    X = data[FEATURE_COLS].copy()
    X["sale_year"] = data["date"].dt.year
    X["sale_month"] = data["date"].dt.month
    X = pd.get_dummies(X, columns=["zipcode"], drop_first=True)
    y = data["price"].copy()
    return X, y


def build_default_payload(data):
    row = data.iloc[0].to_dict()
    return {
        "bedrooms": int(row["bedrooms"]),
        "bathrooms": float(row["bathrooms"]),
        "sqft_living": int(row["sqft_living"]),
        "sqft_lot": int(row["sqft_lot"]),
        "floors": float(row["floors"]),
        "waterfront": int(row["waterfront"]),
        "view": int(row["view"]),
        "condition": int(row["condition"]),
        "grade": int(row["grade"]),
        "sqft_above": int(row["sqft_above"]),
        "sqft_basement": int(row["sqft_basement"]),
        "yr_built": int(row["yr_built"]),
        "yr_renovated": int(row["yr_renovated"]),
        "zipcode": int(row["zipcode"]),
        "lat": float(row["lat"]),
        "long": float(row["long"]),
        "sqft_living15": int(row["sqft_living15"]),
        "sqft_lot15": int(row["sqft_lot15"]),
        "house_age": int(row["house_age"]),
        "renovated": int(row["renovated"]),
        "date": row["date"].strftime("%Y%m%dT%H%M%S"),
    }


def prepare_single_input(payload, training_columns):
    row = pd.DataFrame([payload])
    row["date"] = pd.to_datetime(row["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    row["sale_year"] = row["date"].dt.year
    row["sale_month"] = row["date"].dt.month
    row = pd.get_dummies(row, columns=["zipcode"], drop_first=True)
    return row.reindex(columns=training_columns, fill_value=0)


def main():
    data = load_dataset()
    default_payload = build_default_payload(data)

    parser = argparse.ArgumentParser(description="Predict a single house price with the trained XGBoost model")
    parser.add_argument("--bedrooms", type=int, default=default_payload["bedrooms"])
    parser.add_argument("--bathrooms", type=float, default=default_payload["bathrooms"])
    parser.add_argument("--sqft_living", type=int, default=default_payload["sqft_living"])
    parser.add_argument("--sqft_lot", type=int, default=default_payload["sqft_lot"])
    parser.add_argument("--floors", type=float, default=default_payload["floors"])
    parser.add_argument("--waterfront", type=int, default=default_payload["waterfront"])
    parser.add_argument("--view", type=int, default=default_payload["view"])
    parser.add_argument("--condition", type=int, default=default_payload["condition"])
    parser.add_argument("--grade", type=int, default=default_payload["grade"])
    parser.add_argument("--sqft_above", type=int, default=default_payload["sqft_above"])
    parser.add_argument("--sqft_basement", type=int, default=default_payload["sqft_basement"])
    parser.add_argument("--yr_built", type=int, default=default_payload["yr_built"])
    parser.add_argument("--yr_renovated", type=int, default=default_payload["yr_renovated"])
    parser.add_argument("--zipcode", type=int, default=default_payload["zipcode"])
    parser.add_argument("--lat", type=float, default=default_payload["lat"])
    parser.add_argument("--long", type=float, default=default_payload["long"])
    parser.add_argument("--sqft_living15", type=int, default=default_payload["sqft_living15"])
    parser.add_argument("--sqft_lot15", type=int, default=default_payload["sqft_lot15"])
    parser.add_argument("--house_age", type=int, default=default_payload["house_age"])
    parser.add_argument("--renovated", type=int, default=default_payload["renovated"])
    parser.add_argument("--date", default=default_payload["date"])
    args = parser.parse_args()

    payload = {
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "sqft_living": args.sqft_living,
        "sqft_lot": args.sqft_lot,
        "floors": args.floors,
        "waterfront": args.waterfront,
        "view": args.view,
        "condition": args.condition,
        "grade": args.grade,
        "sqft_above": args.sqft_above,
        "sqft_basement": args.sqft_basement,
        "yr_built": args.yr_built,
        "yr_renovated": args.yr_renovated,
        "zipcode": args.zipcode,
        "lat": args.lat,
        "long": args.long,
        "sqft_living15": args.sqft_living15,
        "sqft_lot15": args.sqft_lot15,
        "house_age": args.house_age,
        "renovated": args.renovated,
        "date": args.date,
    }

    X, y = build_training_frame(data)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    single_input = prepare_single_input(payload, X.columns.tolist())
    prediction = model.predict(single_input)[0]

    print("Single input prediction:")
    print(payload)
    print(f"Predicted price: ${prediction:,.2f}")


if __name__ == "__main__":
    main()
