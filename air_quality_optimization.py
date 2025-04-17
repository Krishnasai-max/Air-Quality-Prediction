import pandas as pd
import numpy as np
import optuna
import zipfile
import requests
import os
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ---------------------- 1️⃣ Download and Extract Dataset ----------------------

zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
zip_path = "AirQualityUCI.zip"
csv_filename = "AirQualityUCI.csv"

print("📥 Downloading dataset...")
response = requests.get(zip_url)
with open(zip_path, "wb") as file:
    file.write(response.content)

print("📂 Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extract(csv_filename)

os.remove(zip_path)

# ---------------------- 2️⃣ Load and Preprocess Data ----------------------

print("📊 Loading dataset...")
df = pd.read_csv(csv_filename, sep=';', decimal=',', na_values=-200, low_memory=False)
df.drop(columns=["Date", "Time"], inplace=True, errors='ignore')
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(axis=1, how='all', inplace=True)

print(f"🔍 Data Shape Before Imputation: {df.shape}")

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(f"✅ Data Shape After Imputation: {df_imputed.shape}")

X = df_imputed.iloc[:, :-1]
y = df_imputed.iloc[:, -1]

print(f"📊 Features Shape: {X.shape}, Target Shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- 3️⃣ Baseline Model: Support Vector Machine (SVM) ----------------------

print("🏗️ Training SVM model...")
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
y_pred_svm = svr_model.predict(X_test_scaled)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
print(f"📊 Baseline Model (SVM) MAE: {mae_svm:.4f}")

# ---------------------- 4️⃣ Optimizing Random Forest with Optuna ----------------------

X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_sample, y_sample)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

print("🚀 Running Optuna Optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print(f"🎯 Best Hyperparameters: {best_params}")

rf_model = RandomForestRegressor(**best_params, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"🚀 Optimized Random Forest MAE: {mae_rf:.4f}")

# ---------------------- 5️⃣ Performance Comparison ----------------------

improvement = (mae_svm - mae_rf) / mae_svm * 100
print(f"🔍 Performance Improvement Over SVM: {improvement:.2f}%")

# ---------------------- 6️⃣ MAE Comparison Visualization ----------------------

models = ['SVM (Baseline)', 'Random Forest (Optimized)']
mae_values = [mae_svm, mae_rf]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, mae_values, color=['skyblue', 'lightgreen'])
plt.ylabel('Mean Absolute Error')
plt.title('📊 Model MAE Comparison')
plt.ylim(0, max(mae_values) + 0.01)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.001, f'{height:.4f}', ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---------------------- ✅ Clean Up ----------------------

os.remove(csv_filename)
