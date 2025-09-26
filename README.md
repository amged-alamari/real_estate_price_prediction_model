# 🏠 Bengaluru House Price Prediction using Machine Learning

This project predicts housing prices in Bengaluru, India using features such as square footage, number of bathrooms, BHK (bedroom-hall-kitchen), and location. It demonstrates a full machine learning pipeline — from data cleaning to model deployment.

---

## 📌 Problem Statement

Accurately predict property prices based on features in the Bengaluru housing dataset using regression models and feature engineering.

---

## 📂 Dataset

- **Source**: [Kaggle - Bengaluru House Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
- **Size**: 13,000+ rows × 9 columns
- **Features**: `location`, `size`, `total_sqft`, `bath`, `price`, etc.

---

## 📊 Workflow Summary

### 🧹 1. Data Cleaning
- Removed missing/null values
- Converted `size` (e.g., "2 BHK") to numerical BHK
- Converted `total_sqft` ranges to average values
- Removed outliers (e.g., unrealistic BHK-to-area ratio)

### 🏗️ 2. Feature Engineering
- Created `price_per_sqft`
- Grouped rare locations into `"other"`
- One-hot encoded `location`

### 📈 3. Modeling
- Used `LinearRegression`, `Lasso`, `DecisionTreeRegressor`
- Tuned hyperparameters with `GridSearchCV`
- Achieved R² score of **0.84**
- 5-fold cross-validation used for robustness

### 🧠 4. Prediction Function
- Custom function: `predict_price(location, sqft, bath, bhk)`
- Easily integrates with Streamlit/Flask

---

## ✅ Final Model Performance

| Metric            | Score   |
|-------------------|---------|
| R² Score          | 0.845   |
| CV (5-fold R² avg)| ~0.82   |

---

## 🧪 Tech Stack

- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- GridSearchCV
- (Optional) Streamlit or Flask

---

## 📁 Repository Structure

