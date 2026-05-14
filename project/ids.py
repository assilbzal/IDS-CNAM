import logging
import joblib
import pandas as pd
import numpy as np
import warnings
from scapy.all import sniff, IP, TCP, UDP

# =========================
# LOGGING CONFIG
# =========================
logging.basicConfig(
    filename="alerts.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

warnings.filterwarnings("ignore")

# =========================
# LOAD MODEL + FEATURES
# =========================
model = joblib.load("ids_model.pkl")
features = joblib.load("features.pkl")

print(f"[INFO] Loaded model with {len(features)} features")

# =========================
# FEATURE MAPPING
# =========================
def extract_features(packet_data):
    """
    Build full feature vector matching trained model (55 features)
    Missing values → 0
    """

    feature_dict = {f: 0 for f in features}

    # basic mappings we actually have
    feature_dict["Flow Duration"] = packet_data.get("time", 0)
    feature_dict["Protocol"] = packet_data.get("proto", 0)

    feature_dict["Source Port"] = packet_data.get("sport", 0)
    feature_dict["Destination Port"] = packet_data.get("dport", 0)

    feature_dict["Total Length of Fwd Packets"] = packet_data.get("length", 0)
    feature_dict["Total Fwd Packets"] = 1

    feature_dict["Flow Bytes/s"] = packet_data.get("length", 0)

    return feature_dict


# =========================
# DETECTION ENGINE
# =========================
def detect(feature_dict):
    try:
        df = pd.DataFrame([feature_dict])

        # IMPORTANT FIX (prevents your crash)
        df = df.reindex(columns=features, fill_value=0)

        # clean
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

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
# PACKET PROCESSOR
# =========================
def process_packet(pkt):
    try:
        packet_data = {
            "length": len(pkt),
            "time": pkt.time
        }

        if IP in pkt:
            packet_data["proto"] = pkt[IP].proto

        if TCP in pkt:
            packet_data["sport"] = pkt[TCP].sport
            packet_data["dport"] = pkt[TCP].dport

        elif UDP in pkt:
            packet_data["sport"] = pkt[UDP].sport
            packet_data["dport"] = pkt[UDP].dport
            packet_data["proto"] = 17

        else:
            packet_data["proto"] = 0

        # FINAL PIPELINE (IMPORTANT FIX)
        feature_dict = extract_features(packet_data)
        detect(feature_dict)

    except Exception as e:
        print("Packet error:", e)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("🚀 REAL-TIME IDS STARTED")

    sniff(prn=process_packet, store=False)

    print("🛑 IDS STOPPED")
