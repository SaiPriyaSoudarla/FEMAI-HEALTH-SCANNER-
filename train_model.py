import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_excel("PCOS_data_without_infertility.xlsx", sheet_name=1)

# Drop unnecessary columns
cols_to_drop = [c for c in df.columns if "Sl" in c or "File" in c or "Unnamed" in c]
df = df.drop(columns=cols_to_drop, errors='ignore')

# Identify target column
# === Set the target column manually ===
TARGET_COL_NAME = "PCOS (Y/N)"  # <-- this is correct based on your dataset

if TARGET_COL_NAME not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL_NAME}' not found! Available columns: {list(df.columns)}")

target = TARGET_COL_NAME



# Split features and target
X = df.drop(columns=[target])
y = df[target]

# Convert all feature columns to numeric (coerce errors to NaN)
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values (after coercing)
X = X.fillna(X.median(numeric_only=True))


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, pred))

# Save model
with open("modelfinal_new.pkl", "wb") as f:
    pickle.dump(model, f)

print("MODEL TRAINED SUCCESSFULLY!")
print("Saved as modelfinal_new.pkl")
