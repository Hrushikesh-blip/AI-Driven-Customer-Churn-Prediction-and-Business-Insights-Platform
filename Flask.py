from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/churn_model.pkl")

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.DataFrame([request.form])
    prediction = model.predict(data)
    return f"Prediction: {prediction[0]}"

if __name__ == "__main__":
    app.run(debug=True)
