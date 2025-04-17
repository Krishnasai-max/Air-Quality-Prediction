# 📈 Air Quality Prediction using Machine Learning & Optuna

This project focuses on building predictive models to estimate air quality using data from the [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+quality). We implement and compare two models:
- A baseline **Support Vector Regression (SVR)**
- An **optimized Random Forest Regressor** tuned using **Optuna**

The objective is to minimize **Mean Absolute Error (MAE)** and showcase the performance gain after hyperparameter tuning.

---

## 🚀 Features

- 📥 Automatic dataset download and extraction from UCI repository
- 🔧 Data cleaning and preprocessing with `pandas` and `SimpleImputer`
- 🔍 Feature scaling with `StandardScaler`
- 🧠 Baseline model using **Support Vector Regression**
- 🌲 Optimized model using **Random Forest** + **Optuna Hyperparameter Tuning**
- 📊 Visualization of model comparison (MAE)
- ✅ Automatic cleanup of intermediate files

---

## 📁 Dataset

**Source:** [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+quality)  
The dataset contains 9358 instances of hourly averaged responses from an array of chemical sensors deployed in an Italian city.

---

