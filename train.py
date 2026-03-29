import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# ================= LOAD DATASET =================
df = pd.read_csv("housing_dataset_1200.csv")   # 👈 use new dataset
df = df.dropna()

# ================= HANDLE CATEGORICAL DATA =================
# Automatically convert ALL object columns (including location)
df = pd.get_dummies(df, drop_first=True)

# ================= SPLIT DATA =================
X = df.drop("price", axis=1)
y = df["price"]

# Optional (for checking accuracy)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= TRAIN MODEL =================
model = LinearRegression()
model.fit(X_train, y_train) 

# ================= MODEL ACCURACY =================
score = model.score(X_test, y_test)
print(f"✅ Model Accuracy (R² Score): {round(score * 100, 2)}%")

# ================= SAVE MODEL =================
with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

print("✅ Model trained and saved to model.pkl")