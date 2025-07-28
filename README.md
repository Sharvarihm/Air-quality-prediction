# Air Quality Prediction using Machine Learning

This project focuses on analyzing and forecasting air quality using machine learning models. It includes both classification (categorizing air quality) and regression (predicting exact AQI values), integrated into an interactive Streamlit dashboard.

---

## Objective

To build an accurate and interpretable model that predicts air quality based on environmental features and provides category-wise or numerical AQI forecasts, useful for environmental monitoring and public health awareness.

---

## Highlights

- **Data Preprocessing**: Cleaning, handling missing values, and feature transformation.
- **Feature Engineering**: Includes derived metrics like `Pollution_Density`, `Wind_PM_Interaction`, `Temp_Humidity_Interaction`, and others.
- **Classification**: Predict AQI categories — *Very Healthy*, *Healthy*, *Unhealthy*, *Very Unhealthy* — using **XGBoost** with **SMOTE** for class balancing.
- **Regression**: Predict the exact AQI values using **XGBoost Regressor**.
- **Visualization**: Performance metrics and feature importance visualized using **Matplotlib** and **Seaborn**.
- **Forecasting**: Uses the last known data point to predict AQI trends over future days.
- **Interactive Dashboard**: Built with **Streamlit** to allow users to switch between classification and regression modes and enter custom input for predictions.

---

## Dataset

- **Source**: Air quality data collected from multiple environmental sensors.
- **Target Variable**: `airqualitydataset` (numeric) used for both classification and regression.
- **Features Include**:
  - Pollutants: `PM2.5`, `PM10`, `NOx`, `NO2`, `CO`
  - Weather: `Temperature`, `Humidity`, `Pressure`, `WindSpeed`
  - Time-based: `Date`, `Time` converted to `DateTime`

---

## Key Deliverables

- Trained XGBoost models for classification and regression
- Streamlit-based dashboard with two modes:
  - AQI Classification: Category-wise output
  - AQI Regression: Numerical AQI prediction and mapped category
- Forecasting future AQI values based on past trends
- Clean visualizations and performance metrics

---

## Results

- Achieved high classification accuracy and meaningful regression R² scores
- Confusion matrix and feature importance charts provide model interpretability
- Forecasting visualizations for up to 30 future days

---

## Usage & Impact

This project can be extended for:
- Real-time AQI monitoring systems
- Urban planning and pollution control measures
- Alert systems for sensitive populations

---

## Tech Stack

- Python  
- Scikit-learn, XGBoost, Imbalanced-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit

---

