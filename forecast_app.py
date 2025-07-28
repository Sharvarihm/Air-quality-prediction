import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor

# Load the dataset
df = pd.read_csv("AirQualityData.csv")

# Combine Date and Time columns into a single DateTime column
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.drop(columns=['Date', 'Time'])

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# Feature Engineering
df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']
df['Wind_PM_Interaction'] = df['WindSpeed'] * df['PM2.5']
df['TotalPollution'] = df['PM2.5'] + df['PM10'] + df['CO(GT)'] + df['NOx(GT)'] + df['NO2(GT)']
df['PM2.5_log'] = np.log1p(df['PM2.5'])
df['Temp_Pressure_Index'] = df['Temperature'] * df['Pressure']
df['Pollution_Density'] = (df['PM2.5'] + df['PM10']) / (df['WindSpeed'] + 1)

# AQI Category Function for Regression
def simplify_aqi(aqi):
    if aqi <= 50:
        return "Very Healthy"
    elif aqi <= 100:
        return "Healthy"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

# Streamlit Selection
task = st.radio("Select Task", ["AQI Classification", "AQI Regression"])

if task == "AQI Classification":
    # Classification part (no AQI category calculation here)
    df['AQI_Category'] = df['AirQualityIndex'].apply(simplify_aqi)
    le = LabelEncoder()
    df['AQI_Label'] = le.fit_transform(df['AQI_Category'])

    X = df.drop(columns=['AirQualityIndex', 'AQI_Category', 'AQI_Label', 'DateTime'])
    y = df['AQI_Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    model = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                          eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("🎯 Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    st.write("✅ Accuracy:", accuracy_score(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - AQI Classification")
    st.pyplot(plt)

    importances = model.feature_importances_
    features = X.columns
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
    plt.title("Feature Importances - XGBoost Model")
    st.pyplot(plt)

    forecast_days = st.number_input('Enter number of days to forecast', min_value=1, max_value=30, value=10)
    last_row = df.iloc[-1]
    forecast_data = np.tile(last_row.drop(['AirQualityIndex', 'AQI_Category', 'AQI_Label', 'DateTime']).values, (forecast_days, 1))
    forecast_data_scaled = scaler.transform(forecast_data)
    forecast_predictions = model.predict(forecast_data_scaled)

    forecast_aqi_categories = le.inverse_transform(forecast_predictions)
    forecast_dates = pd.date_range(start=df['DateTime'].iloc[-1], periods=forecast_days + 1, freq='D')[1:]

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted AQI Category': forecast_aqi_categories
    })

    st.write(f"🔮 Forecast for the next {forecast_days} days:")
    st.write(forecast_df)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=forecast_df, x='Date', y='Forecasted AQI Category', marker='o')
    plt.title(f"Forecasted AQI Categories for the Next {forecast_days} Days")
    plt.xticks(rotation=45)
    st.pyplot(plt)

elif task == "AQI Regression":
    X = df.drop(columns=['AirQualityIndex', 'DateTime'])
    y = df['AirQualityIndex']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Adding XGBRegressor for AQI Regression
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("📈 Regression Performance")
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R² Score:", r2_score(y_test, y_pred))

    predicted_category = [simplify_aqi(val) for val in y_pred]
    actual_category = [simplify_aqi(val) for val in y_test]

    results_df = pd.DataFrame({
        "Actual AQI": y_test,
        "Predicted AQI": y_pred,
        "Predicted Category": predicted_category
    }).reset_index(drop=True)

    st.write("🔍 Sample Predictions")
    st.write(results_df.head(10))

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI")
    st.pyplot(plt)

    st.write("✅ Regression Model predicts AQI values and maps them to categories: Very Healthy, Healthy, Unhealthy, Very Unhealthy.")

    # Collect user input for AQI prediction
    st.write("Enter the values for the following features to predict AQI:")

    input_data = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        input_data.append(val)

    if st.button("Predict AQI"):
        prediction = model.predict([input_data])[0]
        predicted_category = simplify_aqi(prediction)
        st.success(f"Predicted AQI: {round(prediction, 2)}")
        st.write(f"The predicted AQI falls under the category: **{predicted_category}**")
