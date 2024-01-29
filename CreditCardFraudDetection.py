import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the dataset
data = pd.read_csv('D:\PROGRAMMING\Projects\CreditCardFraudDetection/creditcard.csv')
# Extract the features and target variable
X = data.drop('Class', axis=1)
y = data['Class']
# Handle class imbalance using RandomOverSampler
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)
# Preprocess the data by scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Spliting datasets into Training and Testing Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Train a logistic Regression model and Forest model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the Logistic Regression model and Random Forest model
y_pred_lr = lr_model.predict(X_test)
print(classification_report(y_test, y_pred_lr))

y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))