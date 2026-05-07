import pandas as pd
import glob
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Merging all CSVs
files = glob.glob("C:\\Users\\USER\\OneDrive\\Desktop\\dr.Natasha\\ids paper\\project\\GeneratedLabelledFlows\\TrafficLabelling\\*.csv")

df_list = [pd.read_csv(f, encoding='latin1', low_memory=False) for f in files]
df = pd.concat(df_list, ignore_index=True)

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

print("Before cleaning:", df.shape)  #(3119345, 85)

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN
df.dropna(inplace=True)

print("After cleaning:", df.shape)

# Encode labels 
print(df['Label'].value_counts())
''' 
before cleaning 
Label
BENIGN                        2273097
DoS Hulk                       231073
PortScan                       158930
DDoS                           128027
DoS GoldenEye                   10293
FTP-Patator                      7938
SSH-Patator                      5897
DoS slowloris                    5796
DoS Slowhttptest                 5499
Bot                              1966
Web Attack  Brute Force         1507
Web Attack  XSS                  652
Infiltration                       36
Web Attack  Sql Injection         21
Heartbleed                         11 
'''


# Take random sample of dataset
df = df.sample(n=500000, random_state=42)

print("After sampling:", df.shape)
print(df['Label'].value_counts())


# Drop useless columns
X = df.drop(columns=[
    'Label',
    'Flow ID',
    'Source IP',
    'Destination IP',
    'Timestamp'
])

attack_types = df['Label'].unique()
results = {}

all_rows = []

for attack in attack_types:
    
    print(f"\nProcessing label: {attack}")
    
    # Create binary labels
    # 1 = current attack
    # 0 = all other traffic
    y = (df['Label'] == attack).astype(int)
    
    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Get feature importance
    importances = model.feature_importances_
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    feature_importance = feature_importance.reset_index(drop=True)
    
    # Store top 10 features
    top10 = feature_importance.head(10)
    results[attack] = top10
    
    print(feature_importance.head(5))

    for _, row in top10.iterrows():
        all_rows.append({
            "Attack": attack,
            "Feature": row["Feature"],
            "Importance": row["Importance"]
        })

# Save final results
final_df = pd.DataFrame(all_rows)
final_df.to_csv("all_attacks_top_features.csv", index=False)
print("Done! File saved as all_attacks_top_features.csv")

