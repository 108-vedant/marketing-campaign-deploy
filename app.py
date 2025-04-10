from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved pipeline
with open("model_d.pkl", 'rb') as f:
    pipeline = pickle.load(f)

scaler = pipeline['scaler']
power_transformer = pipeline['power_transformer']
columns_order = pipeline['columns_order']
model = pipeline['model']

def preprocess_input(df):
    # Create new features
    
    df['balance_age_ratio'] = df['balance'] / (df['age'] + 1)
    df['pdays_inv'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1/x)
    df['age_duration_interaction'] = df['age'] * df['duration']
    if 'previous' in df.columns:
       df['total_prev_contacts'] = df['pdays_inv'] + df['previous']
    else:
       df['total_prev_contacts'] = df['pdays_inv']

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Define numeric columns
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous','balance_age_ratio',
       'pdays_inv', 'age_duration_interaction', 'total_prev_contacts']
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    # Apply scaling & Transform
    df[numeric_cols] = power_transformer.transform(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Align with training data columns
    for col in columns_order:
        if col not in df.columns:
            df[col] = 0
    df = df[columns_order]
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    input_df = pd.DataFrame([input_data])
    
    # Convert numeric fields
    for col in ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous','day']:
        input_df[col] = pd.to_numeric(input_df[col])

    # Preprocess
    processed_df = preprocess_input(input_df)

    # Predict
    prediction = model.predict(processed_df)[0]

    return render_template('index.html', prediction_text=f'Prediction: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)

