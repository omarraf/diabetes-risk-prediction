# 🩺 Diabetes Risk Prediction Using Logistic Regression

This project applies a logistic regression model—implemented manually using NumPy—to predict an individual's risk of diabetes based on health and lifestyle indicators. The dataset comes from the CDC's 2015 Behavioral Risk Factor Surveillance System (BRFSS) and is publicly available on the UCI Machine Learning Repository.

## 📌 Project Objective

- Predict diabetes risk using selected health indicators
- Implement logistic regression from scratch using Python (NumPy)
- Interpret model weights to generate real-world health insights
- Evaluate the model using standard classification metrics
- Demonstrate the practical application of machine learning in healthcare

## 📊 Dataset

- Source: UCI Machine Learning Repository – CDC BRFSS 2015  
  https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Original Size: ~253,000 survey records
- Features: 21 health-related variables (e.g., BMI, Age, HighBP, Cholesterol)
- Target:
  - **0** = No diabetes (including only during pregnancy)
  - **1** = Prediabetes
  - **2** = Diabetes

This project uses a **balanced subset** with binary labels:
- **0** = No diabetes
- **1** = Diabetes or prediabetes  
Total samples: **9,262**

## ⚙️ Preprocessing

- Controlled downsampling to balance classes (50/50 split)
- Normalization of continuous features using MinMaxScaler
- Feature selection based on correlation analysis and interpretability
- Selected Features:
  - `BMI`, `Age`, `PhysHlth`, `HighBP`, `HighChol`

## 🧠 Model Implementation

- Logistic Regression implemented **from scratch** using NumPy
- Core components:
  - Sigmoid activation
  - Binary cross-entropy loss
  - Gradient descent optimization
- Trained for 1,000 epochs
- No use of scikit-learn for model fitting (per course constraints)

## 🧪 Model Evaluation

| Metric            | Value     |
|-------------------|-----------|
| Accuracy          | 64.3%     |
| Recall (Class 1)  | 70%       |
| F1-Score (Class 1)| 66%       |
| Macro F1-Score    | 64%       |

These results show the model effectively identifies individuals at risk for diabetes, prioritizing **recall** for the positive class to minimize missed diagnoses.

## 🔍 Feature Insights

The model's learned weights revealed the most influential risk factors:

1. **HighChol (Weight = 0.52)** – Strongest predictor
2. **Age (0.41)** – Older individuals more at risk
3. **HighBP (0.07)** – Moderate influence
4. **PhysHlth (0.06)** – Physical unwellness days
5. **BMI (0.02)** – Minimal additional contribution (likely overlaps with other indicators)

## 📁 Project Structure

