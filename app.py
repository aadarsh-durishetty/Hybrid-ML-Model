from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('ICICI_BANK.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['is_quarter_end'] = (df['month'] % 3 == 0)
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'day', 'month', 'year', 'is_quarter_end']]
target = df['target']

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)

model = XGBClassifier()
model.fit(X_train, Y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_date_input = request.form['date']
        input_date = pd.to_datetime(user_date_input, format='%d-%m-%Y')

        input_day = input_date.day
        input_month = input_date.month
        input_year = input_date.year
        input_is_quarter_end = (input_month % 3 == 0)

        input_features = pd.DataFrame({
            'Open': [np.nan],
            'High': [np.nan],
            'Low': [np.nan],
            'Close': [np.nan],
            'Volume': [np.nan],
            'day': [input_day],
            'month': [input_month],
            'year': [input_year],
            'is_quarter_end': [input_is_quarter_end]
        })

        input_features_imputed = imputer.transform(input_features)
        input_features_scaled = scaler.transform(input_features_imputed)

        predicted_prob = model.predict_proba(input_features_scaled)[0, 1]

        if predicted_prob >= 0.5:
            prediction_text = "Stock price is predicted to increase."
            expected_change_percentage = predicted_prob * 100
            action_message = "Sell the stock."
            profit_message = f"INR +{expected_change_percentage:.2f}% (Profit)"
        else:
            prediction_text = "Stock price is predicted to decrease."
            expected_change_percentage = (1 - predicted_prob) * 100
            action_message = "Buy the stock."
            profit_message = f"INR -{expected_change_percentage:.2f}% (Loss)"

        return render_template('result.html', prediction=prediction_text, action=action_message, profit=profit_message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

