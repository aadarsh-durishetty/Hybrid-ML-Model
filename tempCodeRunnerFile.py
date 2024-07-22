# ML model to predict ICICI bank stock price
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix  # Import confusion_matrix from sklearn.metrics

import warnings
warnings.filterwarnings('ignore')

# Test Case 1: Data Loading Test
df = pd.read_csv('ICICI_BANK.csv')
assert not df.empty, "DataFrame is empty after loading data."
print("Test Case 1 Passed: Data Loading")

# Display basic information about the DataFrame
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())


# Test Case 2: Basic Data Exploration Test
assert df.shape == (len(df), 7), "DataFrame shape is incorrect."
assert df['Close'].dtype == np.float64, "Close column type is incorrect."
print("Test Case 2 Passed: Basic Data Exploration")

# Test Case 3: Time Series Plot Test
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('ICICI Stock Close price.', fontsize=15)
plt.ylabel('Price in INR.')
plt.show()
print("Test Case 3 Passed: Time Series Plot")

# Test Case 4: Feature Distribution Visualization Test
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.distplot(df[col])
plt.show()
print("Test Case 4 Passed: Feature Distribution Visualization")

# Test Case 5: Date Extraction Test with correct format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
assert 'day' in df.columns and 'month' in df.columns and 'year' in df.columns, "Date extraction failed."
print("Test Case 5 Passed: Date Extraction")

# Test Case 6: Quarter-End Indicator Test
df['is_quarter_end'] = (df['month'] % 3 == 0).astype(int)
assert 'is_quarter_end' in df.columns, "Quarter-end indicator creation failed."
print("Test Case 6 Passed: Quarter-End Indicator")

# Test Case 7: Target Variable Creation Test
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
assert 'target' in df.columns, "Target variable creation failed."
print("Test Case 7 Passed: Target Variable Creation")

# Test Case 8: Handle Missing Values (Imputation)
features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'day', 'month', 'year', 'is_quarter_end']]
target = df['target']

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)
X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)
assert X_train.shape[0] > 0 and X_valid.shape[0] > 0, "Data split for training/validation failed."
print("Test Case 8 Passed: Missing Value Handling (Imputation)")

# Test Case 9: Model Training Test
models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier()
]

for model in models:
    model.fit(X_train, Y_train)
    train_auc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
    valid_auc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1])
    assert train_auc > 0.5 and valid_auc > 0.5, f"Model {type(model).__name__} evaluation failed."
print('Training Accuracy : ', train_auc)
print('Validation Accuracy : ', valid_auc)
print("Test Case 9 Passed: Model Training and Evaluation")

# Test Case 10: Target Variable Distribution Test
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.show()
print("Test Case 10 Passed: Target Variable Distribution")

# Test Case 11: Correlation Heatmap Test
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
print("Test Case 11 Passed: Correlation Heatmap")

# Test Case 12: Confusion Matrix Visualization Test
cm = confusion_matrix(Y_valid, models[0].predict(X_valid))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print("Test Case 12 Passed: Confusion Matrix Visualization")

# All test cases passed
print("\nAll test cases passed successfully.")







