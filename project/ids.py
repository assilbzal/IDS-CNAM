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

chunk_size = 1000

for chunk in pd.read_csv(
    file_path,
    encoding='latin1',
    low_memory=False,
    chunksize=chunk_size
):

    chunk.columns = chunk.columns.str.strip()

    # align features safely
    chunk = chunk.reindex(columns=features, fill_value=0)

    # clean values
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.fillna(0, inplace=True)

    for i in range(min(100, len(chunk))):

        print(f"\nAnalyzing flow {i}")

        sample_input = chunk.iloc[i].to_dict()

        detect(sample_input)
