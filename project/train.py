import pandas as pd
import glob
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Load selected features
selected_df = pd.read_csv("all_attacks_top_features.csv")
selected_features = selected_df["Feature"].unique().tolist()

print(f"Total selected features: {len(selected_features)}")

# Load dataset
files = glob.glob("C:\\Users\\USER\\OneDrive\\Desktop\\dr.Natasha\\ids paper\\project\\GeneratedLabelledFlows\\TrafficLabelling\\*.csv")
df_list = [pd.read_csv(f, encoding='latin1', low_memory=False) for f in files]
df = pd.concat(df_list, ignore_index=True)

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Original dataset shape:", df.shape)

# Sampling
sample_size = 100000
sampled_parts = []

for label in df["Label"].unique():
    class_data = df[df["Label"] == label]

    n_samples = min(
        len(class_data),
        max(100, int(sample_size * len(class_data) / len(df)))
    )

    sampled_parts.append(class_data.sample(n=n_samples, random_state=42))

df = pd.concat(sampled_parts, ignore_index=True)

print("Sampled dataset shape:", df.shape)

# Filter classes
df = df.groupby("Label").filter(lambda x: len(x) >= 50)

X = df[selected_features]
y = df["Label"]

# Split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# MODEL
name = "RF"

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

print(f"Running {name}...")

start_time = time.time()

# Train
model.fit(X_train, y_train)

# Validation
y_val_pred = model.predict(X_val)

val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

# Test
y_test_pred = model.predict(X_test)

test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

end_time = time.time()
execution_time = end_time - start_time

# Results
results_df = pd.DataFrame([[
    name,
    val_precision,
    val_recall,
    val_f1,
    test_precision,
    test_recall,
    test_f1,
    execution_time
]], columns=[
    "Algorithm",
    "Validation Precision",
    "Validation Recall",
    "Validation F1",
    "Test Precision",
    "Test Recall",
    "Test F1",
    "Execution Time"
])

print(results_df)


# SAVE MODEL + FEATURES
joblib.dump(model, "ids_model.pkl")
joblib.dump(selected_features, "features.pkl")

print("Model + features saved successfully")