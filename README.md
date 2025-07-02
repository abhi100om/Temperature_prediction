# 🌡️ Temperature Prediction Using Humidity

This project demonstrates how to build a **machine learning regression model** to predict **temperature** based on **humidity** using Python and Scikit-learn. It follows a full machine learning pipeline from data preprocessing to prediction and visualization.

---

## 📌 Problem Statement

Given the **humidity percentage**, predict the corresponding **temperature** using historical data. This can be useful in weather forecasting, HVAC control, and environmental studies.

---

## 📊 Dataset

- **File**: `humidity.csv`
- **Features**:
  - `Humidity` (input)
  - `Temperature` (target)

---

## 🔧 Technologies Used

- Python 3.x
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 🧠 Machine Learning Approach

- **Model**: Linear Regression
- **Scaler**: MinMaxScaler (to normalize `Humidity`)
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

---

## 🚀 How to Use

### ✅ Predict Temperature from Humidity

You can use the saved model and scaler to make real-time predictions:

```python
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('temperature_model.pkl')
scaler = joblib.load('scaler.pkl')

# Input humidity in percentage
humidity = float(input("Enter Humidity (%): "))
input_df = pd.DataFrame({'Humidity': [humidity]})
scaled = scaler.transform(input_df)

# Predict temperature
predicted_temp = model.predict(scaled)
print(f"Predicted Temperature: {predicted_temp[0]:.2f} °C")
```

## 📝 License

This project is licensed under the [MIT License](LICENSE).
