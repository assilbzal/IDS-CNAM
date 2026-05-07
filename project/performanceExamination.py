import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# Step 1: Load selected features

# Load output from featureSelection.py
selected_df = pd.read_csv("all_attacks_top_features.csv")


# Get unique features

# Cause some features may repeat across multiple attacks
selected_features = selected_df["Feature"].unique().tolist()

#print("\nFinal reduced feature set:\n")
#for feature in selected_features:
#    print(feature)

print(f"\nTotal selected features: {len(selected_features)}")

# Step 2: Reload original dataset

files = glob.glob("C:\\Users\\USER\\OneDrive\\Desktop\\dr.Natasha\\ids paper\\project\\GeneratedLabelledFlows\\TrafficLabelling\\*.csv")

df_list = [pd.read_csv(f, encoding='latin1', low_memory=False) for f in files]
df = pd.concat(df_list, ignore_index=True)

df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Original dataset shape:", df.shape)

# Keep a representative subset
sample_size = 100000
sampled_parts = []

for label in df["Label"].unique():
    class_data = df[df["Label"] == label]

    n_samples = min(
        len(class_data),
        max(
            100,
            int(sample_size * len(class_data) / len(df))
        )
    )

    sampled_parts.append(
        class_data.sample(
            n=n_samples,
            random_state=42
        )
    )

df = pd.concat(sampled_parts, ignore_index=True)

print("Sampled dataset shape:", df.shape)
print("Label distribution after sampling:")
print(df["Label"].value_counts())


# Step 3: Keep only selected features

df = df.groupby("Label").filter(lambda x: len(x) >= 50)

X = df[selected_features]
y = df['Label']


# Step 4: Split dataset into training/testing

# First split: 80% temp , 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Second split: from the 80%, create 60% train and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,   # 25% of 80% = 20%
    random_state=42,
    stratify=y_temp
)

# Step 5: Define the 7 algorithms

models = {
    "KNN": KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
    ),
    "RF": RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
    ),
    "ID3": DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
    n_estimators=50,
    random_state=42
    ),
    "MLP": MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=100,
    random_state=42
    ),
    "Naive-Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(
    reg_param=0.5,
    store_covariance=False
    )
}

# Step 6: Train, Validate and Test + Measure execution time

results = []

for name, model in models.items():
    print(f"Running {name}...")

    start_time = time.time()

    # Training
    model.fit(X_train, y_train)

    # Validation
    y_val_pred = model.predict(X_val)

    val_precision = precision_score(
        y_val,
        y_val_pred,
        average='weighted',
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        y_val_pred,
        average='weighted',
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        y_val_pred,
        average='weighted',
        zero_division=0
    )

    # Testing
    y_test_pred = model.predict(X_test)

    test_precision = precision_score(
        y_test,
        y_test_pred,
        average='weighted',
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        y_test_pred,
        average='weighted',
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        y_test_pred,
        average='weighted',
        zero_division=0
    )

    end_time = time.time()
    execution_time = end_time - start_time

    results.append([
        name,
        val_precision,
        val_recall,
        val_f1,
        test_precision,
        test_recall,
        test_f1,
        execution_time
    ])

# Step 7: Create Table 4

results_df = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Validation Precision",
        "Validation Recall",
        "Validation F1",
        "Test Precision",
        "Test Recall",
        "Test F1",
        "Execution Time"
    ]
)

print(results_df)

results_df.to_csv("table4_results.csv", index=False)