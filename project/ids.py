import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# STEP 1 — Load model + features


model = joblib.load("ids_model.pkl")
features = joblib.load("features.pkl")


# STEP 2 — Prediction function


def predict(input_data):

    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Ensure same feature order used during training
    df = df[features]

    # Predict
    prediction = model.predict(df)

    return prediction[0]


# STEP 3 — Alert logic


def detect(input_data):

    result = predict(input_data)

    if result != "BENIGN":
        print(f"⚠️ ATTACK DETECTED: {result}")
    else:
        print("✅ Normal traffic")


# STEP 4 — Load real sample traffic


sample_df = pd.read_csv(
    "C:\\Users\\USER\\OneDrive\\Desktop\\dr.Natasha\\ids paper\\project\\GeneratedLabelledFlows\\TrafficLabelling\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    encoding='latin1',
    low_memory=False
)

# Clean columns
sample_df.columns = sample_df.columns.str.strip()

# Remove invalid values
sample_df.replace([np.inf, -np.inf], np.nan, inplace=True)
sample_df.dropna(inplace=True)

# Keep only trained features
for i in range(len(sample_df)):  #for i in range(100):

    print(f"\nAnalyzing flow {i}")

    sample_input = sample_df[features].iloc[i].to_dict()

    detect(sample_input)
