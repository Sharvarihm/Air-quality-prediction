# Air Quality Prediction (Classification & Regression with Forecasting)

This project predicts air quality using machine learning models and visualizes predictions with a Streamlit dashboard. It includes:

* A classification model using XGBoost and SMOTE
* A regression model using XGBoost for AQI value prediction
* A Streamlit dashboard for user interaction and forecasting

---

##  Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn, XGBoost, Imbalanced-learn
* Matplotlib, Seaborn
* Streamlit (for dashboard)

---

##  Repository Structure

```
air-quality-prediction/
├── data/
│   └── AirQualityData.csv         # Sample or reference dataset
├── prediction.py                  # ML pipeline: classification with XGBoost
├── forecast_app.py                # Streamlit dashboard: classification + regression
├── requirements.txt               # Project dependencies
├── results/
│   ├── confusion_matrix.png       # Classification result
│   └── feature_importance.png     # Top features
├── screenshots/
│   └── streamlit_ui.png           # UI screenshot (optional)
└── README.md                      # Project documentation
```

---

##  Features

* Cleaned and engineered air quality dataset
* Feature interaction and transformation
* XGBoost classification (Healthy, Unhealthy, Very Unhealthy)
* Regression for predicting exact AQI values
* Streamlit UI for forecasting future AQI categories
* Visualization of metrics and predictions

---

##  Dataset

* Name: `AirQualityData.csv`
* Fields include: PM2.5, PM10, NOx, CO, NO2, Temperature, Humidity, WindSpeed, Pressure, etc.
* Contains datetime stamps and AirQualityIndex (AQI)

> **Note**: If the full dataset cannot be published, included a sample 

---





## 🚀 How to Run

### To Train and Evaluate Model (offline):

```bash
python prediction.py
```

This script will train the XGBoost classifier, print metrics, and show:

* Confusion matrix
* Feature importances

### To Launch the Streamlit App:

```bash
streamlit run forecast_app.py
```

This opens a browser-based dashboard with two modes:

* **AQI Classification**: Predict category
* **AQI Regression**: Predict AQI value + user input + forecasting

---

## 🧪 Outputs

* `confusion_matrix.png`: Visualization of predicted vs actual categories
* `feature_importance.png`: Ranked feature importances
* Forecasted AQI categories for future days using most recent data

---

## 📸 Screenshots

### Streamlit Dashboard

![Streamlit UI](screenshots/streamlit_ui.png)

---

## 🔒 License

This project is intended for academic and educational use. You may fork and modify with credit. Please do not redistribute proprietary datasets without permission.

---

## 📎 Credits

Created by \[Your Name]. Based on data science, visualization, and interactive ML deployment best practices.
