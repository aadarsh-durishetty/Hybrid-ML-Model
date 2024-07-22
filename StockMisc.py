



# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn import metrics
# import datetime

# # Data Loading
# df = pd.read_csv('ICICI_BANK.csv')

# # Date Extraction
# df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
# df['day'] = df['Date'].dt.day
# df['month'] = df['Date'].dt.month
# df['year'] = df['Date'].dt.year
# df['is_quarter_end'] = (df['month'] % 3 == 0)  # Boolean (True or False)

# # Target Variable Creation
# df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# # Handling Imputation
# features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'day', 'month', 'year', 'is_quarter_end']]
# target = df['target']

# # Impute missing values with mean
# imputer = SimpleImputer(strategy='mean')
# features_imputed = imputer.fit_transform(features)

# # Scale features
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features_imputed)

# # Split data into training and validation sets
# X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)

# # Model Training
# model = XGBClassifier()
# model.fit(X_train, Y_train)

# # User Input for Date Prediction
# user_date_input = input("Enter a date (DD-MM-YYYY) to predict the ICICI Bank stock price: ")
# input_date = pd.to_datetime(user_date_input, format='%d-%m-%Y')

# # Extract features for the input date
# input_day = input_date.day
# input_month = input_date.month
# input_year = input_date.year
# input_is_quarter_end = (input_month % 3 == 0)  # Boolean (True or False)

# # Create feature array for prediction with valid feature names
# input_features = pd.DataFrame({
#     'Open': [np.nan],
#     'High': [np.nan],
#     'Low': [np.nan],
#     'Close': [np.nan],
#     'Volume': [np.nan],
#     'day': [input_day],
#     'month': [input_month],
#     'year': [input_year],
#     'is_quarter_end': [input_is_quarter_end]
# })

# # Impute missing values and scale features for prediction
# input_features_imputed = imputer.transform(input_features)
# input_features_scaled = scaler.transform(input_features_imputed)

# # Predict using the trained model
# predicted_prob = model.predict_proba(input_features_scaled)[0, 1]  # Probability of class 1 (increase)

# # Determine prediction result and display actionable message
# if predicted_prob >= 0.5:
#     prediction_text = "Stock price is predicted to increase."
#     # Calculate expected percentage change and profit/loss
#     expected_change_percentage = predicted_prob * 100
#     action_message = f"Sell the stock."

#     # Format profit message with ANSI escape sequences for terminal color
#     profit_message = f"\033[92mINR +{expected_change_percentage:.2f}% (Profit)\033[0m"
# elif predicted_prob < 0.5:
#     prediction_text = "Stock price is predicted to decrease."
#     # Calculate expected percentage change and profit/loss
#     expected_change_percentage = (1 - predicted_prob) * 100
#     action_message = f"Buy the stock."

#     # Format loss message with ANSI escape sequences for terminal color
#     profit_message = f"\033[91mINR -{expected_change_percentage:.2f}% (Loss)\033[0m"
# else:
#     prediction_text = "Stock price is predicted to remain unchanged."
#     action_message = "Hold the stock"

#     # For unchanged prediction, profit_message is not applicable

# print(prediction_text)
# print(profit_message)
# print(action_message)
