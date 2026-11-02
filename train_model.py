"""
train_model.py
End-to-end Machine Learning training pipeline for Used Car Price Prediction.
Uses Scikit-Learn Pipeline with ColumnTransformer, OneHotEncoder, StandardScaler,
SimpleImputer, and RandomForestRegressor.
Evaluates using MAE, RMSE, and R2 score, and serializes the pipeline using Joblib.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure reproducibility
RANDOM_STATE = 42

def train_and_evaluate(data_path="data/used_cars.csv", model_dir="models"):
    print("=" * 65)
    print("USED CAR PRICE PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 65)
    
    # 1. Load the dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
        
    print(f"\n[1/7] Loading dataset from: {data_path}...")
    df = pd.read_csv(data_path)
    print(f"      Total records loaded: {len(df):,}")
    print(f"      Dataset columns: {list(df.columns)}")
    
    # Check for target column
    target_col = "selling_price"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # 2. Separate input features and target
    print("\n[2/7] Separating feature matrix (X) and target vector (y)...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Define categorical and numerical features
    categorical_features = ["brand", "fuel", "seller_type", "transmission", "owner"]
    numerical_features = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
    
    print(f"      Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"      Categorical features ({len(categorical_features)}): {categorical_features}")

    # 3. Train-test split (80:20 ratio)
    print("\n[3/7] Splitting dataset into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"      Training samples: {len(X_train):,}")
    print(f"      Testing samples : {len(X_test):,}")

    # 4. Construct Preprocessing Pipeline
    # Numeric transformer: impute missing values with median, scale features
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical transformer: impute missing with most frequent, One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    # 5. Build full model pipeline with RandomForestRegressor
    print("\n[4/7] Initializing Random Forest Regressor & Preprocessing Pipeline...")
    rf_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf_model)
    ])

    # 6. Train the model
    print("\n[5/7] Fitting full pipeline on training set...")
    full_pipeline.fit(X_train, y_train)
    print("      Model training completed successfully!")

    # 7. Evaluate the model on test set
    print("\n[6/7] Evaluating model performance on held-out test set...")
    y_pred_train = full_pipeline.predict(X_train)
    y_pred_test = full_pipeline.predict(X_test)

    # Evaluation metrics
    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    rmse = float(root_mean_squared_error(y_test, y_pred_test))
    mean_target = float(y_test.mean())
    mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100)

    print("\n" + "=" * 45)
    print("        ACTUAL EVALUATION METRICS        ")
    print("=" * 45)
    print(f"  R-squared (R2) Score (Test) : {test_r2:.4f} ({test_r2 * 100:.2f}%)")
    print(f"  R-squared (R2) Score (Train): {train_r2:.4f} ({train_r2 * 100:.2f}%)")
    print(f"  Mean Absolute Error (MAE)   : Rs. {mae:,.2f}")
    print(f"  Root Mean Sq. Error (RMSE)  : Rs. {rmse:,.2f}")
    print(f"  Mean Absolute % Error (MAPE): {mape:.2f}%")
    print(f"  Mean Selling Price (Test)   : Rs. {mean_target:,.2f}")
    print("=" * 45)

    # Extract Feature Importances
    preprocessor_fitted = full_pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numerical_features + cat_feature_names
    
    regressor = full_pipeline.named_steps["regressor"]
    importances = getattr(regressor, "feature_importances_", None)
    if importances is None:
        raise AttributeError("Trained regressor does not expose feature_importances_")
    
    # Aggregate importance by original feature category for intuitive display
    feature_importance_list = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    ]
    
    # Top original feature importance aggregation
    original_feature_imp = {}
    for item in feature_importance_list:
        orig = item["feature"].split("_")[0]
        original_feature_imp[orig] = original_feature_imp.get(orig, 0.0) + item["importance"]
    original_feature_imp = [
        {"feature": k, "importance": round(v, 4)}
        for k, v in sorted(original_feature_imp.items(), key=lambda x: x[1], reverse=True)
    ]

    # Sample test predictions for interactive dashboard comparison (first 25 test samples)
    test_sample_df = X_test.copy().head(30)
    test_sample_df["actual_price"] = y_test.head(30).values
    test_sample_df["predicted_price"] = np.round(y_pred_test[:30], 2)
    test_sample_df["error"] = np.round(test_sample_df["actual_price"] - test_sample_df["predicted_price"], 2)
    test_sample_records = test_sample_df.to_dict(orient="records")

    # 8. Save Pipeline & Metadata
    os.makedirs(model_dir, exist_ok=True)
    pipeline_path = os.path.join(model_dir, "car_price_pipeline.joblib")
    metrics_path = os.path.join(model_dir, "metrics.json")
    metadata_path = os.path.join(model_dir, "metadata.json")

    print(f"\n[7/7] Saving pipeline and artifacts to '{model_dir}' directory...")
    joblib.dump(full_pipeline, pipeline_path)
    print(f"      Model pipeline saved: {pipeline_path}")

    # Metrics JSON
    metrics_payload = {
        "r2_score": round(test_r2, 4),
        "train_r2_score": round(train_r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "algorithm": "RandomForestRegressor",
        "n_estimators": 120,
        "random_state": RANDOM_STATE,
        "feature_importances": original_feature_imp,
        "detailed_importances": feature_importance_list[:15],
        "test_samples": test_sample_records
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"      Evaluation metrics saved: {metrics_path}")

    # Metadata JSON (Unique categories, bounds for input validation)
    metadata_payload = {
        "brands": sorted([str(b) for b in df["brand"].dropna().unique()]),
        "fuel_types": sorted([str(f) for f in df["fuel"].dropna().unique()]),
        "seller_types": sorted([str(s) for s in df["seller_type"].dropna().unique()]),
        "transmission_types": sorted([str(t) for t in df["transmission"].dropna().unique()]),
        "owner_types": ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
        "seat_options": sorted([int(s) for s in df["seats"].dropna().unique()]),
        "numerical_bounds": {
            "year": {"min": int(df["year"].min()), "max": int(df["year"].max()), "default": 2018},
            "km_driven": {"min": 500, "max": 300000, "default": 45000},
            "mileage": {"min": round(float(df["mileage"].min()), 1), "max": round(float(df["mileage"].max()), 1), "default": 19.5},
            "engine": {"min": int(df["engine"].min()), "max": int(df["engine"].max()), "default": 1197},
            "max_power": {"min": round(float(df["max_power"].min()), 1), "max": round(float(df["max_power"].max()), 1), "default": 83.0},
            "seats": {"min": int(df["seats"].min()), "max": int(df["seats"].max()), "default": 5}
        },
        "target_summary": {
            "min": float(df[target_col].min()),
            "max": float(df[target_col].max()),
            "mean": round(float(df[target_col].mean()), 2),
            "median": round(float(df[target_col].median()), 2)
        }
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=4)
    print(f"      Metadata and validation bounds saved: {metadata_path}")

    print("\nPipeline execution complete! Model is ready for deployment.\n")
    return full_pipeline, metrics_payload

if __name__ == "__main__":
    train_and_evaluate()
