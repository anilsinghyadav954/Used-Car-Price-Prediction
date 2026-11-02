# 🚗 Used Car Price Prediction Using Machine Learning

> A complete, production-grade, and academically rigorous Machine Learning engineering assignment project developed in Python. Designed for undergraduate (B.Tech / B.E.) Computer Science & Engineering, Data Science, and AI/ML coursework.

---

## 📌 1. Project Overview

The **Used Car Price Prediction System** provides prospective buyers, sellers, and automotive dealerships with an unbiased, data-backed estimate of fair market resale value in **Indian Rupees (₹)**.

Automobile pricing is inherently non-linear and governed by complex interactions among mechanical specifications (engine capacity, peak horsepower, fuel economy), temporal decay (vehicle age, depreciation rate), operational history (odometer mileage, previous owners), and transaction channels (direct individual sale vs. certified dealerships).

This project implements an end-to-end Machine Learning solution using **Scikit-Learn's `RandomForestRegressor`** embedded within a full **ColumnTransformer** preprocessing pipeline, served through a modern **Flask** web application with interactive **Chart.js** analytics.

---

## ✨ 2. Key Features

- **End-to-End Scikit-Learn Pipeline**: Integrates `SimpleImputer`, `StandardScaler`, `OneHotEncoder(handle_unknown='ignore')`, and `RandomForestRegressor` in a unified pipeline.
- **Robust Input Validation**: Strict boundary checks on manufacturing year (1990–2026), non-negative odometer readings, plausible engine specs, and sanitization of user inputs.
- **Real-Time Valuation in Indian Currency**: Formatted output in Indian numbering system (`₹ 6,45,000` / `₹ 6.45 Lakh`) with a fair market confidence band (±6%).
- **Interactive 1-Click Presets**: Pre-configured test vehicles (*Maruti Swift, Hyundai Creta, Honda City AT, BMW 3 Series*) allowing evaluators to verify the pipeline with a single click.
- **Analytics & Model Verification Dashboard**:
  - Transparent test set metrics: **$R^2$ Score (75.11%)**, **MAE (₹ 1.87 Lakh)**, **RMSE (₹ 3.12 Lakh)**.
  - Interactive Chart.js charts: Feature importance weights, fuel distribution, brand representation, and actual vs. predicted test comparisons.
  - Held-out test set prediction table displaying exact residual errors.
- **College Viva & Architecture Page**:
  - Detailed 5-phase ML lifecycle workflow.
  - Mathematical formulation of Random Forest regression & ensemble variance reduction.
  - Curated answers to frequent oral examination questions.
- **Automated Test Suite**: 21 unit and integration tests using `pytest` covering routes, validations, model predictions, and edge-cases.

---

## 🛠️ 3. Technologies Used

| Layer | Technologies |
|---|---|
| **Core Language** | Python 3.10+ (tested on Python 3.14 on Windows) |
| **Machine Learning** | Scikit-Learn (`RandomForestRegressor`, `Pipeline`, `ColumnTransformer`) |
| **Data Processing** | Pandas, NumPy |
| **Model Persistence**| Joblib |
| **Web Backend** | Flask (WSGI Web Framework) |
| **Frontend UI** | HTML5, Modern Vanilla CSS (Design System: Dark Green & Lime), JavaScript (ES6+ AJAX) |
| **Data Visualization**| Chart.js (HTML5 Canvas) |
| **Automated Testing** | Pytest |

---

## 📂 4. Project Folder Structure

```
Used-Car-Price-Prediction/
├── app.py                      # Flask web application & REST API routes
├── train_model.py              # ML pipeline training, evaluation & artifact serialization
├── prepare_dataset.py          # Reproducible data preparation & statistical generation
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive documentation & setup instructions
├── .gitignore                  # Git cache and temporary artifact exclusions
├── data/
│   └── used_cars.csv           # 3,200 curated automotive records with 11 features
├── models/
│   ├── car_price_pipeline.joblib # Trained end-to-end model pipeline
│   ├── metrics.json            # Actual test set evaluation metrics & feature importances
│   └── metadata.json           # Validated categories & feature boundaries
├── templates/
│   ├── index.html              # Home page with valuation form & live results
│   ├── dashboard.html          # Performance analytics & interactive Chart.js graphs
│   └── about.html              # Architecture documentation, ML theory & viva Q&A
├── static/
│   ├── css/
│   │   └── style.css           # Custom dark green & subtle lime responsive styling
│   └── js/
│       └── main.js             # AJAX submission, preset handlers & chart loaders
└── tests/
    └── test_app.py             # 21 automated pytest unit & integration tests
```

---

## 💻 5. Windows Installation & Setup Guide

### Prerequisites
1. **Windows 10 or Windows 11**
2. **Python 3.10 or higher** installed. Make sure to check the box:  
   ☑️ **"Add Python to PATH"** during installation.

To verify Python installation, open **PowerShell** or **Command Prompt**:
```powershell
python --version
pip --version
```

---

### Step 1: Open the Project Directory
Navigate to the project root folder:
```powershell
cd "c:\Users\HP\Desktop\Car Price Prediction"
```

---

### Step 2: Create a Virtual Environment (Recommended)
Creating an isolated virtual environment prevents library conflicts:
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# (If on Command Prompt cmd.exe, use):
# venv\Scripts\activate.bat
```

> **PowerShell Execution Policy Tip**: If PowerShell shows an error regarding running scripts, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Step 3: Install Required Dependencies
Install all required libraries using `requirements.txt`:
```powershell
pip install -r requirements.txt
```

---

### Step 4: Prepare the Dataset (Optional / Already Included)
The project comes with a curated dataset at `data/used_cars.csv`. To regenerate it from scratch:
```powershell
python prepare_dataset.py
```

---

### Step 5: Train the Machine Learning Model
Train the Scikit-Learn `RandomForestRegressor` and generate model artifacts:
```powershell
python train_model.py
```

**Expected Training Output:**
```
=================================================================
USED CAR PRICE PREDICTION - MODEL TRAINING PIPELINE
=================================================================
[1/7] Loading dataset from: data/used_cars.csv...
      Total records loaded: 3,200
[2/7] Separating feature matrix (X) and target vector (y)...
[3/7] Splitting dataset into 80% Training and 20% Testing sets...
      Training samples: 2,560 | Testing samples: 640
[4/7] Initializing Random Forest Regressor & Preprocessing Pipeline...
[5/7] Fitting full pipeline on training set...
[6/7] Evaluating model performance on held-out test set...
=============================================
        ACTUAL EVALUATION METRICS        
=============================================
  R-squared (R2) Score (Test) : 0.7511 (75.11%)
  R-squared (R2) Score (Train): 0.9403 (94.03%)
  Mean Absolute Error (MAE)   : Rs. 187,289.63
  Root Mean Sq. Error (RMSE)  : Rs. 311,948.35
  Mean Absolute % Error (MAPE): 36.75%
=============================================
[7/7] Saving pipeline and artifacts to 'models' directory...
      Model pipeline saved: models\car_price_pipeline.joblib
      Evaluation metrics saved: models\metrics.json
      Metadata and validation bounds saved: models\metadata.json
```

---

### Step 6: Run Automated Tests
Verify that all unit tests, boundary validations, routes, and model predictions pass:
```powershell
pytest tests/test_app.py -v
```
*(All 21 test cases should report **PASSED**).*

---

### Step 7: Launch the Flask Web Application
Start the local development server:
```powershell
python app.py
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🌐 6. Application Pages & Navigation

1. **Prediction Portal (`/`)**:
   - Fill out the 11 vehicle parameters or click a preset button (*Maruti Swift, Hyundai Creta, Honda City, BMW 3 Series*).
   - Click **Predict Selling Price** to obtain an instantaneous estimated price in ₹, fair market confidence band, and vehicle recap.
2. **Analytics Dashboard (`/dashboard`)**:
   - Inspect actual model performance cards ($R^2$, MAE, RMSE, MAPE).
   - Explore interactive Chart.js visualizations (Feature Importance weights, Brand distributions, Fuel share, Actual vs. Predicted comparison).
   - View held-out test set records and prediction error residuals.
3. **Architecture & Viva Documentation (`/about`)**:
   - End-to-end Machine Learning life cycle explanation.
   - Random Forest Regressor mathematical background.
   - Comprehensive answers for college viva and project defense.

---

## 📡 7. REST API Usage

The application also exposes programmatic JSON endpoints for integration:

### Predict Price Endpoint
**`POST /api/predict`**

**Request Body:**
```json
{
  "brand": "Hyundai",
  "year": 2021,
  "km_driven": 35000,
  "fuel": "Diesel",
  "transmission": "Manual",
  "owner": "First Owner",
  "seller_type": "Dealer",
  "mileage": 18.5,
  "engine": 1493,
  "max_power": 113.4,
  "seats": 5
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "result": {
    "predicted_price_raw": 1045200.0,
    "predicted_price": "₹ 10,45,200",
    "predicted_denomination": "₹ 10.45 Lakh",
    "confidence_low": "₹ 9,82,488",
    "confidence_high": "₹ 11,07,912"
  }
}
```

---

## ⚠️ 8. Dataset Limitations & Academic Scope

1. **Accidental & Flood History**: Resale value in real markets is significantly penalized by accidental impact or flood history; these physical conditions cannot be determined solely through tabular vehicle metrics.
2. **State & RTO Road Tax Variations**: Road taxes vary between 6% to 18% across Indian states (e.g., Delhi, Maharashtra, Karnataka), leading to regional price premiums for identical models.
3. **Service Log Documentation**: Full dealership service history records (MASS / HASS) warrant a pricing premium not captured in basic mechanical specifications.
4. **Educational Classification**: This application is built for academic instruction, laboratory demonstration, and viva presentation; predictions represent statistical approximations rather than binding market appraisals.

---

## ❓ 9. Troubleshooting & Common Errors

| Error | Root Cause | Solution |
|---|---|---|
| `ImportError: DLL load failed while importing _partitioner` | Multiprocessing lock under Windows Application Control policy | `train_model.py` is configured with `n_jobs=1` to avoid child subprocess DLL restrictions. |
| `UnicodeEncodeError: 'charmap' codec can't encode character` | Windows console code page (cp1252) unable to render the Indian Rupee symbol `₹` | `sys.stdout.reconfigure(encoding='utf-8')` is integrated into all python scripts. |
| `FileNotFoundError: models/car_price_pipeline.joblib` | Model has not been trained yet | Execute `python train_model.py` before starting `app.py`. |
| `Address already in use (Port 5000)` | Another process is occupying port 5000 | Terminate the existing process or change port in `app.py` (`app.run(port=5001)`). |

---

## 👨‍🎓 10. Project Submission Metadata
- **Project Title**: Used Car Price Prediction Using Machine Learning
- **Course**: B.Tech Final Year / Semester Mini-Project in Machine Learning
- **Department**: Computer Science & Engineering / Artificial Intelligence & Data Science
- **Core Algorithm**: Scikit-Learn Random Forest Regression
