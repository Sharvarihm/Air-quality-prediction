
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
sns.set(style="whitegrid")
df = pd.read_csv("AirQualityData.csv")
print("Initial Shape:", df.shape)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Dropping non-numeric column: {col}")
        df = df.drop(columns=[col])
df = df.dropna(subset=['AirQualityIndex'])
df = df.fillna(df.mean(numeric_only=True))
df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']
df['Wind_PM_Interaction'] = df['WindSpeed'] * df['PM2.5']
df['TotalPollution'] = df['PM2.5'] + df['PM10'] + df['CO(GT)'] + df['NOx(GT)'] + df['NO2(GT)']
df['PM2.5_log'] = np.log1p(df['PM2.5'])
df['Temp_Pressure_Index'] = df['Temperature'] * df['Pressure']
df['Pollution_Density'] = (df['PM2.5'] + df['PM10']) / (df['WindSpeed'] + 1)
def simplify_aqi(aqi):
    if aqi <= 100:
        return "Healthy"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"
df['AQI_Category'] = df['AirQualityIndex'].apply(simplify_aqi)
le = LabelEncoder()
df['AQI_Label'] = le.fit_transform(df['AQI_Category'])
X = df.drop(columns=['AirQualityIndex', 'AQI_Category', 'AQI_Label'])
y = df['AQI_Label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
model = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                      eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - AQI Classification")
plt.show()
importances = model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importances - XGBoost Model")
plt.show()