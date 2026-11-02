"""
prepare_dataset.py
Generates an educational, statistically grounded used car dataset 
modeled on real-world Indian automotive resale trends (Cardekho/CarWale distribution patterns).
Features include Brand, Year, KM Driven, Fuel, Transmission, Owner, Seller Type, Mileage, Engine, Max Power, Seats, and Selling Price.
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set fixed seed for perfect reproducibility
np.random.seed(42)

CAR_PROFILES = [
    # Maruti Suzuki
    {"brand": "Maruti", "models": ["Alto", "WagonR", "Swift", "Baleno", "Dzire", "Brezza", "Ertiga"],
     "fuel_dist": ["Petrol", "Diesel", "CNG"], "fuel_p": [0.65, 0.20, 0.15],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.75, 0.25],
     "engine_range": (796, 1462), "power_range": (47.0, 103.0), "mileage_range": (18.0, 31.0),
     "seats_dist": [5, 7], "seats_p": [0.85, 0.15], "base_price_range": (350000, 1150000)},
    
    # Hyundai
    {"brand": "Hyundai", "models": ["Santro", "Grand i10", "i20", "Venue", "Creta", "Verna"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.60, 0.40],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.65, 0.35],
     "engine_range": (998, 1497), "power_range": (68.0, 138.0), "mileage_range": (16.0, 24.0),
     "seats_dist": [5], "seats_p": [1.0], "base_price_range": (450000, 1600000)},
     
    # Honda
    {"brand": "Honda", "models": ["Brio", "Amaze", "Jazz", "City", "WR-V", "Civic"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.70, 0.30],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.55, 0.45],
     "engine_range": (1198, 1799), "power_range": (88.0, 140.0), "mileage_range": (15.5, 25.5),
     "seats_dist": [5], "seats_p": [1.0], "base_price_range": (550000, 1800000)},

    # Tata
    {"brand": "Tata", "models": ["Tiago", "Tigor", "Altroz", "Nexon", "Harrier", "Safari"],
     "fuel_dist": ["Petrol", "Diesel", "CNG"], "fuel_p": [0.50, 0.40, 0.10],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.70, 0.30],
     "engine_range": (1199, 1956), "power_range": (84.0, 168.0), "mileage_range": (16.0, 23.5),
     "seats_dist": [5, 7], "seats_p": [0.80, 0.20], "base_price_range": (500000, 2200000)},

    # Toyota
    {"brand": "Toyota", "models": ["Glanza", "Etios", "Yaris", "Innova Crysta", "Fortuner"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.35, 0.65],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.50, 0.50],
     "engine_range": (1197, 2755), "power_range": (82.0, 201.0), "mileage_range": (12.0, 23.0),
     "seats_dist": [5, 7, 8], "seats_p": [0.45, 0.45, 0.10], "base_price_range": (650000, 3600000)},

    # Mahindra
    {"brand": "Mahindra", "models": ["Bolero", "XUV300", "Scorpio", "XUV700", "Thar"],
     "fuel_dist": ["Diesel", "Petrol"], "fuel_p": [0.80, 0.20],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.70, 0.30],
     "engine_range": (1497, 2184), "power_range": (75.0, 197.0), "mileage_range": (13.0, 20.0),
     "seats_dist": [5, 6, 7], "seats_p": [0.40, 0.10, 0.50], "base_price_range": (600000, 2300000)},

    # Ford
    {"brand": "Ford", "models": ["Figo", "Aspire", "EcoSport", "Endeavour"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.40, 0.60],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.65, 0.35],
     "engine_range": (1194, 1996), "power_range": (95.0, 168.0), "mileage_range": (13.5, 23.0),
     "seats_dist": [5, 7], "seats_p": [0.85, 0.15], "base_price_range": (480000, 3200000)},

    # Volkswagen
    {"brand": "Volkswagen", "models": ["Polo", "Vento", "Taigun", "Virtus"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.70, 0.30],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.50, 0.50],
     "engine_range": (999, 1498), "power_range": (75.0, 148.0), "mileage_range": (15.0, 21.0),
     "seats_dist": [5], "seats_p": [1.0], "base_price_range": (550000, 1700000)},

    # Renault
    {"brand": "Renault", "models": ["Kwid", "Triber", "Kiger", "Duster"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.85, 0.15],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.70, 0.30],
     "engine_range": (799, 1461), "power_range": (53.0, 108.0), "mileage_range": (16.0, 25.0),
     "seats_dist": [5, 7], "seats_p": [0.80, 0.20], "base_price_range": (380000, 1100000)},

    # Kia
    {"brand": "Kia", "models": ["Sonet", "Seltos", "Carens", "Carnival"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.55, 0.45],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.45, 0.55],
     "engine_range": (1197, 2199), "power_range": (82.0, 197.0), "mileage_range": (14.0, 24.0),
     "seats_dist": [5, 6, 7], "seats_p": [0.75, 0.10, 0.15], "base_price_range": (750000, 2600000)},

    # Skoda
    {"brand": "Skoda", "models": ["Rapid", "Kushaq", "Slavia", "Octavia"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.80, 0.20],
     "trans_dist": ["Manual", "Automatic"], "trans_p": [0.45, 0.55],
     "engine_range": (999, 1984), "power_range": (108.0, 187.0), "mileage_range": (14.5, 21.5),
     "seats_dist": [5], "seats_p": [1.0], "base_price_range": (700000, 2500000)},

    # Luxury Brands: BMW, Mercedes-Benz, Audi
    {"brand": "BMW", "models": ["3 Series", "5 Series", "X1", "X3", "X5"],
     "fuel_dist": ["Diesel", "Petrol"], "fuel_p": [0.60, 0.40],
     "trans_dist": ["Automatic"], "trans_p": [1.0],
     "engine_range": (1995, 2993), "power_range": (188.0, 261.0), "mileage_range": (11.0, 18.5),
     "seats_dist": [5], "seats_p": [1.0], "base_price_range": (3500000, 7500000)},

    {"brand": "Mercedes-Benz", "models": ["C-Class", "E-Class", "GLA", "GLC"],
     "fuel_dist": ["Diesel", "Petrol"], "fuel_p": [0.65, 0.35],
     "trans_dist": ["Automatic"], "trans_p": [1.0],
     "engine_range": (1950, 2987), "power_range": (192.0, 255.0), "mileage_range": (11.5, 17.0),
     "seats_dist": [5], "seats_p": [1.0], "base_price_range": (3800000, 8000000)},

    {"brand": "Audi", "models": ["A4", "A6", "Q3", "Q5"],
     "fuel_dist": ["Petrol", "Diesel"], "fuel_p": [0.60, 0.40],
     "trans_dist": ["Automatic"], "trans_p": [1.0],
     "engine_range": (1984, 2967), "power_range": (188.0, 248.0), "mileage_range": (12.0, 17.5),
     "seats_dist": [5], "seats_p": [1.0], "base_price_range": (3400000, 7200000)}
]

def generate_car_record():
    profile = np.random.choice(CAR_PROFILES, p=[
        0.20, 0.16, 0.09, 0.11, 0.10, 0.09, 0.04, 0.04, 0.04, 0.05, 0.03, 0.02, 0.02, 0.01
    ])
    
    brand = profile["brand"]
    
    # Manufacturing Year: 2007 to 2024 (weighted towards 2014-2022)
    years = np.arange(2007, 2025)
    year_weights = np.linspace(1, 10, len(years))
    year_weights /= year_weights.sum()
    year = int(np.random.choice(years, p=year_weights))
    
    age = 2026 - year
    
    # Kilometres Driven: Correlated with age (average 9,000 - 15,000 km per year)
    base_km = age * np.random.uniform(7000, 16000)
    km_driven = int(np.clip(base_km + np.random.normal(0, 5000), 4000, 240000))
    # Round to nearest 500
    km_driven = int(round(km_driven / 500.0) * 500)
    
    # Categoricals
    fuel = np.random.choice(profile["fuel_dist"], p=profile["fuel_p"])
    transmission = np.random.choice(profile["trans_dist"], p=profile["trans_p"])
    
    # Owner: Correlated with age
    if age <= 3:
        owner_p = [0.85, 0.13, 0.02, 0.00]
    elif age <= 7:
        owner_p = [0.55, 0.35, 0.08, 0.02]
    elif age <= 12:
        owner_p = [0.25, 0.45, 0.22, 0.08]
    else:
        owner_p = [0.10, 0.35, 0.35, 0.20]
    owner = np.random.choice(["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"], p=owner_p)
    
    seller_type = np.random.choice(["Individual", "Dealer", "Trustmark Dealer"], p=[0.60, 0.32, 0.08])
    
    # Technical specs
    engine = int(round(np.random.uniform(*profile["engine_range"])))
    # Max power strongly correlated with engine size within profile range
    power_frac = (engine - profile["engine_range"][0]) / max(1, (profile["engine_range"][1] - profile["engine_range"][0]))
    power_min, power_max = profile["power_range"]
    max_power = round(power_min + power_frac * (power_max - power_min) + np.random.normal(0, 3.0), 1)
    max_power = max(35.0, max_power)
    
    # Mileage: inversely related to engine and power
    mileage_min, mileage_max = profile["mileage_range"]
    mileage = round(mileage_max - power_frac * (mileage_max - mileage_min) + np.random.normal(0, 1.2), 1)
    mileage = float(np.clip(mileage, 8.5, 33.5))
    
    seats = int(np.random.choice(profile["seats_dist"], p=profile["seats_p"]))
    
    # Price Generation Modeling (Indian Market Depreciation Curve)
    # Original Base Price (approximate new showroom price at that year)
    new_price = np.random.uniform(*profile["base_price_range"])
    # Adjust for engine and transmission
    if transmission == "Automatic":
        new_price *= 1.12
    if fuel == "Diesel":
        new_price *= 1.08
    elif fuel == "CNG":
        new_price *= 1.04
        
    # Depreciation factor based on age (12-16% per year early on, slowing down)
    deprec_rate = 0.125
    deprec_factor = np.exp(-deprec_rate * age)
    
    # Mileage/KM impact: -1% per 15,000 km
    km_penalty = max(0.65, 1.0 - (km_driven / 300000.0) * 0.40)
    
    # Owner impact
    owner_multiplier = {
        "First Owner": 1.00,
        "Second Owner": 0.88,
        "Third Owner": 0.76,
        "Fourth & Above Owner": 0.65
    }[owner]
    
    # Seller type margin
    seller_multiplier = {
        "Individual": 0.97,
        "Dealer": 1.03,
        "Trustmark Dealer": 1.08
    }[seller_type]
    
    # Brand retention bonus
    brand_retention = {
        "Toyota": 1.15,
        "Maruti": 1.08,
        "Hyundai": 1.04,
        "Honda": 1.03,
        "Kia": 1.04,
        "Tata": 1.02,
        "Mahindra": 1.03,
        "BMW": 0.90,
        "Mercedes-Benz": 0.92,
        "Audi": 0.88,
        "Ford": 0.92,
        "Volkswagen": 0.95,
        "Skoda": 0.94,
        "Renault": 0.90
    }.get(brand, 1.0)
    
    # Final price computation with realistic noise
    estimated_price = new_price * deprec_factor * km_penalty * owner_multiplier * seller_multiplier * brand_retention
    # Add random market noise (+/- 6%)
    market_noise = np.random.normal(1.0, 0.06)
    selling_price = estimated_price * market_noise
    
    # Minimum floor price for functional used car
    selling_price = max(65000.0, selling_price)
    # Round to clean nearest 1000
    selling_price = int(round(selling_price / 1000.0) * 1000)
    
    return {
        "brand": brand,
        "year": year,
        "selling_price": selling_price,
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

def main():
    print("Generating authentic educational used cars dataset...")
    num_samples = 3200
    records = [generate_car_record() for _ in range(num_samples)]
    df = pd.DataFrame(records)
    
    # Add a few realistic missing values (0.5%) to test and demonstrate real-world preprocessing
    # in mileage, engine, max_power, seats
    nan_indices = np.random.choice(df.index, size=16, replace=False)
    for idx in nan_indices[:5]:
        df.loc[idx, "mileage"] = np.nan
    for idx in nan_indices[5:10]:
        df.loc[idx, "engine"] = np.nan
    for idx in nan_indices[10:14]:
        df.loc[idx, "max_power"] = np.nan
    for idx in nan_indices[14:]:
        df.loc[idx, "seats"] = np.nan

    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "used_cars.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset successfully created and saved at: {csv_path}")
    print(f"Total records: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print("\nSummary Statistics of Target (Selling Price in ₹):")
    print(df["selling_price"].describe())
    print("\nBrand Distribution:")
    print(df["brand"].value_counts())

if __name__ == "__main__":
    main()
