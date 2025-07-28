
# Air Quality Prediction using Machine Learning

This project focuses on analyzing and forecasting air quality using machine learning models. It includes both classification (categorizing air quality) and regression (predicting exact AQI values), integrated into an interactive Streamlit dashboard.

---

## Objective

To build an accurate and interpretable model that predicts air quality based on environmental features and provides category-wise or numerical AQI forecasts, useful for environmental monitoring and public health awareness.

---

## Highlights

- Data cleaning and transformation
- Feature engineering using pollutant interactions and weather conditions
- Classification model using XGBoost with SMOTE
- Regression model using XGBoost Regressor
- Streamlit dashboard for live predictions and forecasting
- Visualizations for feature importance, confusion matrix, and regression performance

---

## Dataset

- **Source**: Custom dataset (`airqualityDataset.csv`).The sample data is available in the dataset section.
- **Features**: PM2.5, PM10, NOx, CO, NO2, Temperature, Humidity, Pressure, WindSpeed, DateTime
- **Target**: `AirQualityIndex` (AQI) - both numeric and categorical formats

---

## Process Flow

<img width="439" height="476" alt="image" src="https://github.com/user-attachments/assets/3de227e2-8957-4046-a792-d0ac660549d1" />


## Outputs

### AQI Classification and Classification Report
<img width="506" height="415" alt="image" src="https://github.com/user-attachments/assets/adebcf70-8672-49c1-a9d2-6c14934b0d6f" />

### Confusion matrix
<img width="583" height="488" alt="image" src="https://github.com/user-attachments/assets/6e39ce88-56ad-4523-8816-4862e5a9e1ce" />

### Feature Importances of XGBoost model
<img width="581" height="317" alt="image" src="https://github.com/user-attachments/assets/adbdd593-8ec0-4b1e-afc3-c0e7961f941a" />

### Forecasting 
<img width="577" height="575" alt="image" src="https://github.com/user-attachments/assets/3829bb8a-3004-4efa-bfdf-0f624e8f01a2" />

###Task AQI Regression with MSE, R2 Score, Sample Predictions
<img width="302" height="480" alt="image" src="https://github.com/user-attachments/assets/43afbb8c-ed68-421a-a918-5a751de3a6f1" />

###Actual vs Predicted scatter plot
<img width="553" height="367" alt="image" src="https://github.com/user-attachments/assets/c7d2d433-a61b-40e9-81b2-25ec8eb3fae5" />

###User input of parameters and predicted AQI
<img width="581" height="431" alt="image" src="https://github.com/user-attachments/assets/0863dce1-7cd8-481f-a277-62adc2b1cf1b" />
<img width="487" height="278" alt="image" src="https://github.com/user-attachments/assets/dcec0da1-09b0-496d-9850-d87944471a37" />


---


Install all libraries  with:
```bash
pip install -r requirements.txt
```

---

## Tech Stack

- Python, Scikit-learn, XGBoost
- Imbalanced-learn for SMOTE
- Streamlit for frontend visualization
- Matplotlib, Seaborn for result plotting

---

