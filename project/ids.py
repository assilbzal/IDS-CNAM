import logging
import joblib
import pandas as pd
import numpy as np
import warnings
from scapy.all import sniff

# =========================
# LOGGING CONFIG
# =========================

logging.basicConfig(
    filename="alerts.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# LOAD MODEL + FEATURES
# =========================

model = joblib.load("ids_model.pkl")
features = joblib.load("features.pkl")

print(f"[INFO] Loaded model with {len(features)} features")

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(input_data):
    """
    Map raw extracted_features.csv row → ML feature space (55 features)
    """

    feature_dict = {f: 0 for f in features}

    # map YOUR real extracted values
    feature_dict["length"] = input_data.get("length", 0)
    feature_dict["proto"] = input_data.get("proto", 0)
    feature_dict["time"] = input_data.get("time", 0)

    # optional derived features (VERY IMPORTANT IMPROVEMENT)
    feature_dict["src_bytes"] = input_data.get("length", 0)
    feature_dict["dst_bytes"] = 0
    feature_dict["duration"] = 0

    return feature_dict

# =========================
# PREDICTION
# =========================

def predict(feature_dict):

    df = pd.DataFrame([feature_dict])

    # enforce correct order
    df = df.reindex(columns=features, fill_value=0)

    # clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    prediction = model.predict(df)

    return prediction[0]

# =========================
# DETECTION ENGINE
# =========================

def detect(feature_dict):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([feature_dict])

        # Ensure correct feature order
        df = df[features]

        # Predict
        result = model.predict(df)[0]

        if result != "BENIGN":
            alert = f"⚠️ ATTACK DETECTED: {result}"
            print(alert)
            logging.warning(alert)
        else:
            print("✅ Normal Traffic")
            logging.info("Normal Traffic")

    except Exception as e:
        print("Error in detect():", e)

        
# =========================
# TEST / CSV MODE (TEMPORARY BRIDGE)
# =========================


def process_packet(pkt):

    feature_dict = {f: 0 for f in features}
    try:
        data = {
            "length": len(pkt),
            "proto": pkt.proto if hasattr(pkt, "proto") else 0,
            "time": pkt.time,
            "src_bytes": len(pkt),
            "dst_bytes": 0
        }

        # optional improvements
        if pkt.haslayer("IP"):
            data["src_ip"] = pkt["IP"].src
            data["dst_ip"] = pkt["IP"].dst

        if pkt.haslayer("TCP"):
            data["sport"] = pkt["TCP"].sport
            data["dport"] = pkt["TCP"].dport

        detect(data)

    except Exception as e:
        print(e)



"""
def run_csv_mode(file_path="traffic_features.csv"):

    chunk_size = 1000

    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):

            chunk.columns = chunk.columns.str.strip()

            for i in range(min(100, len(chunk))):

                raw_input = chunk.iloc[i].to_dict()

                detect(raw_input)

    except FileNotFoundError:
        print("❌ CSV file not found. Make sure traffic_features.csv exists.")
"""
# =========================
# MAIN ENTRY POINT
# =========================

if __name__ == "__main__":

    print("🚀 REAL-TIME IDS STARTED")

    sniff(prn=process_packet, store=False)

    print("🛑 IDS STOPPED")
