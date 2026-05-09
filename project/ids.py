import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# STEP 1 — Load model
# =========================

model = joblib.load("ids_model.pkl")
features = joblib.load("features.pkl")

print(f"Loaded {len(features)} features")


# =========================
# STEP 2 — Prediction
# =========================

def predict(input_data):
    df = pd.DataFrame([input_data])

    # ensure all features exist
    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)
    return prediction[0]


# =========================
# STEP 3 — Alert logic
# =========================

def detect(input_data):
    result = predict(input_data)

    if result != "BENIGN":
        print(f"⚠️ ATTACK DETECTED: {result}")
    else:
        print("✅ Normal traffic")


# =========================
# STEP 4 — Load dataset safely
# =========================

file_path = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Load full dataset first (safe approach)
sample_df = pd.read_csv(
    file_path,
    encoding='latin1',
    low_memory=False
)

# Clean column names
sample_df.columns = sample_df.columns.str.strip()

print("\nCSV columns loaded:", len(sample_df.columns))
print("First columns:", list(sample_df.columns[:10]))


# =========================
# STEP 5 — Align features safely
# =========================

missing_features = [f for f in features if f not in sample_df.columns]

if missing_features:
    print("\n⚠️ Missing features in CSV:", missing_features)

# keep only available features
sample_df = sample_df.reindex(columns=features)

# handle NaN / inf
sample_df.replace([np.inf, -np.inf], np.nan, inplace=True)
sample_df.fillna(0, inplace=True)


# =========================
# STEP 6 — Run detection
# =========================

limit = min(100, len(sample_df))

for i in range(limit):

    print(f"\nAnalyzing flow {i}")

    sample_input = sample_df.iloc[i].to_dict()

    detect(sample_input)
