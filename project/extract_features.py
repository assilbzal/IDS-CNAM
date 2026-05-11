from scapy.all import rdpcap
import pandas as pd

packets = rdpcap("traffic.pcap")

data = []

for pkt in packets:
    try:
        data.append({
            "length": len(pkt),
            "proto": pkt.proto if hasattr(pkt, "proto") else 0,
            "time": pkt.time
        })
    except:
        pass

df = pd.DataFrame(data)

df.to_csv("traffic_features.csv", index=False)

print("✅ Features saved to traffic_features.csv")