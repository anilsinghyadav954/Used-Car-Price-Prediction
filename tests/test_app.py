"""
test_app.py
Comprehensive automated test suite for Used Car Price Prediction Flask Web Application.
Uses pytest to verify routes, validation logic, model predictions, and error handling.
"""

import os
import sys
import pytest

# Add parent directory to path so app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as flask_app
from app import app, format_inr, validate_and_parse_input, predict_single_car

@pytest.fixture
def client():
    """Configures test client for Flask application."""
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client

@pytest.fixture
def valid_car_payload():
    """Returns a realistic, valid car dictionary for testing."""
    return {
        "brand": "Maruti",
        "year": 2019,
        "km_driven": 45000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "owner": "First Owner",
        "seller_type": "Individual",
        "mileage": 21.4,
        "engine": 1197,
        "max_power": 82.0,
        "seats": 5
    }

# ==========================================
# 1. Page Load Tests
# ==========================================

def test_home_page_loads(client):
    """Verifies that the Home / Prediction page loads with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"AutoValuate ML" in response.data
    assert b"Predict Selling Price" in response.data

def test_about_page_loads(client):
    """Verifies that the About & Architecture page loads with HTTP 200."""
    response = client.get("/about")
    assert response.status_code == 200
    assert b"System Architecture" in response.data
    assert b"Random Forest Regressor" in response.data

def test_dashboard_page_loads(client):
    """Verifies that the Dashboard page loads with HTTP 200."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Performance &amp; Dataset Dashboard" in response.data or b"Dashboard" in response.data
    assert b"Test Set R" in response.data

def test_api_metrics_endpoint(client):
    """Verifies that /api/metrics returns valid JSON with metrics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "metrics" in data
    assert "r2_score" in data["metrics"]

# ==========================================
# 2. Prediction & Model Inference Tests
# ==========================================

def test_valid_json_prediction(client, valid_car_payload):
    """Verifies that submitting valid JSON to /predict produces a numeric prediction."""
    response = client.post("/predict", json=valid_car_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "predicted_price_raw" in data["data"]
    assert isinstance(data["data"]["predicted_price_raw"], (int, float))
    assert data["data"]["predicted_price_raw"] > 0
    assert "₹" in data["data"]["predicted_price"]

def test_valid_form_prediction(client, valid_car_payload):
    """Verifies that classic HTML form submission to /predict re-renders page with result."""
    response = client.post("/predict", data=valid_car_payload)
    assert response.status_code == 200
    assert b"Valuation Result" in response.data
    assert b"Estimated Resale Value" in response.data

def test_prediction_is_non_negative(valid_car_payload):
    """Verifies that prediction logic guarantees non-negative prices."""
    result = predict_single_car(valid_car_payload)
    assert result["predicted_price_raw"] >= 45000.0
    assert "₹" in result["predicted_price"]

def test_api_predict_endpoint(client, valid_car_payload):
    """Verifies that /api/predict REST endpoint operates correctly."""
    response = client.post("/api/predict", json=valid_car_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "result" in data
    assert data["result"]["predicted_price_raw"] > 0

# ==========================================
# 3. Input Validation & Boundary Tests
# ==========================================

def test_negative_kilometres_rejected(client, valid_car_payload):
    """Verifies that negative odometer readings are strictly rejected."""
    bad_payload = valid_car_payload.copy()
    bad_payload["km_driven"] = -5000
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "cannot be negative" in data["error"].lower()

def test_invalid_manufacturing_year_past(client, valid_car_payload):
    """Verifies that year before 1990 is rejected."""
    bad_payload = valid_car_payload.copy()
    bad_payload["year"] = 1982
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "invalid" in data["error"].lower() or "year" in data["error"].lower()

def test_invalid_manufacturing_year_future(client, valid_car_payload):
    """Verifies that future years beyond 2026 are rejected."""
    bad_payload = valid_car_payload.copy()
    bad_payload["year"] = 2035
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "invalid" in data["error"].lower() or "year" in data["error"].lower()

def test_missing_required_input_handled(client, valid_car_payload):
    """Verifies that missing essential inputs trigger descriptive error messages."""
    incomplete_payload = valid_car_payload.copy()
    del incomplete_payload["brand"]
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "missing" in data["error"].lower()

def test_invalid_seats_handled(client, valid_car_payload):
    """Verifies that extreme seat numbers are rejected."""
    bad_payload = valid_car_payload.copy()
    bad_payload["seats"] = 25
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "seats" in data["error"].lower()

def test_unseen_categorical_value_handled(client, valid_car_payload):
    """Verifies that unseen categorical value doesn't crash the pipeline (OneHotEncoder handle_unknown='ignore')."""
    unseen_payload = valid_car_payload.copy()
    unseen_payload["brand"] = "Tesla"
    unseen_payload["fuel"] = "Electric"
    response = client.post("/predict", json=unseen_payload)
    # The pipeline handles unknown categories gracefully and returns a valid prediction
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["data"]["predicted_price_raw"] > 0

def test_validate_and_parse_input_direct(valid_car_payload):
    """Directly verifies validate_and_parse_input function with clean types."""
    parsed = validate_and_parse_input(valid_car_payload)
    assert parsed["brand"] == "Maruti"
    assert parsed["year"] == 2019
    assert parsed["km_driven"] == 45000.0
    assert parsed["seats"] == 5

def test_invalid_engine_handled(client, valid_car_payload):
    """Verifies that out-of-bound engine capacity is rejected."""
    bad_payload = valid_car_payload.copy()
    bad_payload["engine"] = 12000
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "engine" in data["error"].lower()

def test_invalid_power_handled(client, valid_car_payload):
    """Verifies that out-of-bound max power is rejected."""
    bad_payload = valid_car_payload.copy()
    bad_payload["max_power"] = 1500
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "power" in data["error"].lower()

def test_invalid_mileage_handled(client, valid_car_payload):
    """Verifies that impossible mileage is rejected."""
    bad_payload = valid_car_payload.copy()
    bad_payload["mileage"] = 95.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "mileage" in data["error"].lower()

def test_api_metrics_includes_summary(client):
    """Verifies that /api/metrics returns dataset summary for frontend charts."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "summary" in data
    assert "top_brands" in data["summary"]
    assert "fuel_counts" in data["summary"]

# ==========================================
# 4. Missing Model File Handling Test
# ==========================================

def test_missing_model_file_handled(monkeypatch, valid_car_payload):
    """Verifies that attempting prediction without a loaded model raises informative RuntimeError."""
    monkeypatch.setattr(flask_app, "model_pipeline", None)
    with pytest.raises(RuntimeError) as exc_info:
        flask_app.predict_single_car(valid_car_payload)
    assert "not loaded" in str(exc_info.value).lower()

# ==========================================
# 5. Helper Function Tests
# ==========================================

def test_format_inr():
    """Verifies the Indian numbering formatting helper."""
    # 6.5 Lakh
    res1 = format_inr(650000)
    assert res1["formatted"] == "₹ 6,50,000"
    assert "Lakh" in res1["denomination"]

    # 1.2 Crore
    res2 = format_inr(12000000)
    assert res2["formatted"] == "₹ 1,20,00,000"
    assert "Crore" in res2["denomination"]

    # 45 Thousand
    res3 = format_inr(45000)
    assert res3["formatted"] == "₹ 45,000"
    assert "Thousand" in res3["denomination"]

