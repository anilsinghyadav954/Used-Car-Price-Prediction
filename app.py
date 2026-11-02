"""
app.py
Flask Web Application for Used Car Price Prediction.
Handles routing, input validation, model inference, Indian currency formatting,
and serves modern, responsive frontend pages (Home, Dashboard, About).
"""

import os
import sys
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
app.secret_key = "used-car-price-predictor-secret-key-college-ml"

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "used_cars.csv")

# Global caches
model_pipeline = None
model_metrics = {}
model_metadata = {}

def load_resources():
    """Safely loads model pipeline, metrics, and metadata."""
    global model_pipeline, model_metrics, model_metadata
    
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            print("[INFO] Model pipeline loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model pipeline: {e}")
            model_pipeline = None
    else:
        print(f"[WARNING] Model file not found at {MODEL_PATH}. Run train_model.py first.")
        model_pipeline = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                model_metrics = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load metrics: {e}")
            model_metrics = {}

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load metadata: {e}")
            model_metadata = {}

load_resources()

def format_inr(number):
    """
    Formats a numeric value into the Indian numbering system (Lakhs and Crores).
    Example: 650000 -> '₹ 6,50,000' and '6.50 Lakh'
    """
    try:
        val = float(number)
        val = max(0.0, val)
        int_part = int(round(val))
        s = str(int_part)
        
        # Indian comma grouping logic
        if len(s) <= 3:
            formatted_num = s
        else:
            last_three = s[-3:]
            remaining = s[:-3]
            chunks = []
            while len(remaining) > 2:
                chunks.append(remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                chunks.append(remaining)
            chunks.reverse()
            formatted_num = ",".join(chunks) + "," + last_three
            
        # Denomination text
        if val >= 10000000:
            denom_text = f"₹ {val / 10000000:.2f} Crore"
        elif val >= 100000:
            denom_text = f"₹ {val / 100000:.2f} Lakh"
        elif val >= 1000:
            denom_text = f"₹ {val / 1000:.1f} Thousand"
        else:
            denom_text = f"₹ {val:.0f}"
            
        return {
            "formatted": f"₹ {formatted_num}",
            "denomination": denom_text,
            "raw": round(val, 2)
        }
    except Exception:
        return {"formatted": "₹ 0", "denomination": "₹ 0", "raw": 0}

def validate_and_parse_input(data):
    """
    Validates form or JSON data against engineering rules and returns clean dictionary.
    Raises ValueError on validation failure.
    """
    required_fields = [
        "brand", "year", "km_driven", "fuel", "transmission",
        "owner", "seller_type", "mileage", "engine", "max_power", "seats"
    ]
    
    # 1. Check missing fields
    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            raise ValueError(f"Missing required field: '{field.replace('_', ' ').title()}'")
            
    # 2. Extract and sanitize
    brand = str(data["brand"]).strip()
    fuel = str(data["fuel"]).strip()
    transmission = str(data["transmission"]).strip()
    owner = str(data["owner"]).strip()
    seller_type = str(data["seller_type"]).strip()
    
    # 3. Numeric conversions & boundaries
    try:
        year = int(data["year"])
    except (ValueError, TypeError):
        raise ValueError("Manufacturing Year must be a valid whole number.")
    if year < 1990 or year > 2026:
        raise ValueError(f"Manufacturing Year ({year}) is invalid. Must be between 1990 and 2026.")

    try:
        km_driven = float(data["km_driven"])
    except (ValueError, TypeError):
        raise ValueError("Kilometres Driven must be a valid number.")
    if km_driven < 0:
        raise ValueError("Kilometres Driven cannot be negative.")
    if km_driven > 1000000:
        raise ValueError("Kilometres Driven exceeds plausible vehicle limit (10,00,000 km).")

    try:
        mileage = float(data["mileage"])
    except (ValueError, TypeError):
        raise ValueError("Mileage must be a valid numeric figure.")
    if mileage <= 0 or mileage > 60:
        raise ValueError("Mileage must be between 1.0 and 60.0 kmpl.")

    try:
        engine = float(data["engine"])
    except (ValueError, TypeError):
        raise ValueError("Engine Capacity must be a valid number.")
    if engine <= 500 or engine > 8000:
        raise ValueError("Engine Capacity must be between 500 CC and 8000 CC.")

    try:
        max_power = float(data["max_power"])
    except (ValueError, TypeError):
        raise ValueError("Maximum Power must be a valid number.")
    if max_power <= 20 or max_power > 1000:
        raise ValueError("Maximum Power must be between 20 bhp and 1000 bhp.")

    try:
        seats = int(data["seats"])
    except (ValueError, TypeError):
        raise ValueError("Number of Seats must be a valid integer.")
    if seats < 2 or seats > 12:
        raise ValueError("Number of Seats must be between 2 and 12.")

    parsed_row = {
        "brand": brand,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }
    return parsed_row

def predict_single_car(input_dict):
    """Passes validated car features to the trained model pipeline."""
    pipeline = model_pipeline
    if pipeline is None:
        raise RuntimeError("Model file not loaded. Please execute 'python train_model.py' to generate the model artifact.")
        
    df_input = pd.DataFrame([input_dict])
    raw_pred = pipeline.predict(df_input)[0]
    
    # Guarantee positive price floor
    clean_price = max(45000.0, float(raw_pred))
    
    inr_data = format_inr(clean_price)
    
    # Calculate estimated fair value confidence range (± 6% model prediction band)
    low_band = format_inr(clean_price * 0.94)
    high_band = format_inr(clean_price * 1.06)
    
    return {
        "predicted_price_raw": clean_price,
        "predicted_price": inr_data["formatted"],
        "predicted_denomination": inr_data["denomination"],
        "confidence_low": low_band["formatted"],
        "confidence_high": high_band["formatted"],
        "vehicle_summary": input_dict
    }

# ==========================================
# Application Routes
# ==========================================

@app.route("/", methods=["GET"])
def home():
    """Renders the primary prediction page."""
    # Fallback options if metadata is not generated yet
    default_metadata = {
        "brands": ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra", "Ford", "Volkswagen", "Renault", "Kia", "Skoda", "BMW", "Mercedes-Benz", "Audi"],
        "fuel_types": ["Petrol", "Diesel", "CNG"],
        "transmission_types": ["Manual", "Automatic"],
        "owner_types": ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
        "seller_types": ["Individual", "Dealer", "Trustmark Dealer"],
        "seat_options": [4, 5, 6, 7, 8],
        "numerical_bounds": {
            "year": {"min": 2007, "max": 2024, "default": 2018},
            "km_driven": {"min": 1000, "max": 250000, "default": 45000},
            "mileage": {"min": 10.0, "max": 33.0, "default": 19.5},
            "engine": {"min": 796, "max": 2993, "default": 1197},
            "max_power": {"min": 40.0, "max": 260.0, "default": 83.0},
            "seats": {"min": 4, "max": 8, "default": 5}
        }
    }
    meta = model_metadata if model_metadata else default_metadata
    metrics = model_metrics if model_metrics else {"r2_score": 0.7511, "mae": 187289.63}
    
    return render_template("index.html", metadata=meta, metrics=metrics, model_loaded=(model_pipeline is not None))

@app.route("/predict", methods=["POST"])
def predict():
    """Handles both AJAX JSON and classic HTML Form prediction submissions."""
    is_json = request.is_json
    data = request.get_json() if is_json else request.form.to_dict()
    
    try:
        validated_input = validate_and_parse_input(data)
        prediction_result = predict_single_car(validated_input)
        
        if is_json:
            return jsonify({
                "success": True,
                "data": prediction_result
            }), 200
        else:
            meta = model_metadata if model_metadata else {}
            return render_template(
                "index.html",
                metadata=meta,
                metrics=model_metrics,
                prediction=prediction_result,
                form_data=data,
                model_loaded=(model_pipeline is not None)
            )
            
    except (ValueError, RuntimeError) as e:
        error_msg = str(e)
        if is_json:
            return jsonify({"success": False, "error": error_msg}), 400
        else:
            flash(error_msg, "danger")
            meta = model_metadata if model_metadata else {}
            return render_template(
                "index.html",
                metadata=meta,
                metrics=model_metrics,
                error=error_msg,
                form_data=data,
                model_loaded=(model_pipeline is not None)
            ), 400
            
    except Exception as e:
        unexpected_error = f"An unexpected error occurred: {str(e)}"
        if is_json:
            return jsonify({"success": False, "error": unexpected_error}), 500
        else:
            flash(unexpected_error, "danger")
            return redirect(url_for("home"))

def get_dataset_summary():
    """Computes summary statistics from the dataset if available."""
    if not os.path.exists(DATASET_PATH):
        return {}
    try:
        df = pd.read_csv(DATASET_PATH)
        return {
            "total_cars": len(df),
            "avg_price": format_inr(df["selling_price"].mean())["formatted"],
            "median_price": format_inr(df["selling_price"].median())["formatted"],
            "min_price": format_inr(df["selling_price"].min())["formatted"],
            "max_price": format_inr(df["selling_price"].max())["formatted"],
            "top_brands": df["brand"].value_counts().head(8).to_dict(),
            "fuel_counts": df["fuel"].value_counts().to_dict(),
            "transmission_counts": df["transmission"].value_counts().to_dict(),
            "avg_price_by_fuel": {k: round(v, 2) for k, v in df.groupby("fuel")["selling_price"].mean().to_dict().items()},
            "avg_price_by_transmission": {k: round(v, 2) for k, v in df.groupby("transmission")["selling_price"].mean().to_dict().items()}
        }
    except Exception as e:
        print(f"[ERROR] Failed summarizing dataset: {e}")
        return {}

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Renders the comprehensive performance and analytics dashboard."""
    dataset_summary = get_dataset_summary()
    return render_template(
        "dashboard.html",
        metrics=model_metrics,
        summary=dataset_summary,
        model_loaded=(model_pipeline is not None)
    )

@app.route("/about", methods=["GET"])
def about():
    """Renders the detailed educational and architectural About page."""
    return render_template(
        "about.html",
        metrics=model_metrics,
        model_loaded=(model_pipeline is not None)
    )

# ==========================================
# REST API Endpoints
# ==========================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Public REST API for automated prediction integration."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "Request body must be a valid JSON object."}), 400
        validated = validate_and_parse_input(data)
        result = predict_single_car(validated)
        return jsonify({"success": True, "result": result}), 200
    except (ValueError, RuntimeError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns actual evaluation metrics and dataset summary as JSON."""
    return jsonify({
        "success": True,
        "metrics": model_metrics,
        "metadata": model_metadata,
        "summary": get_dataset_summary(),
        "model_loaded": (model_pipeline is not None)
    }), 200

if __name__ == "__main__":
    import socket
    port = int(os.environ.get("PORT", 5000))

    # Check if target port is occupied and fallback safely
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_port_in_use = (s.connect_ex(("127.0.0.1", port)) == 0)

    if is_port_in_use:
        fallback_port = 5001
        print(f"[WARNING] Port {port} is occupied by another process. Launching on fallback port {fallback_port}.")
        port = fallback_port

    print("\n" + "=" * 60)
    print("  CAR PRICE PREDICTION - FLASK SERVER")
    print(f"  Local URL: http://127.0.0.1:{port}")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=port, debug=True)
