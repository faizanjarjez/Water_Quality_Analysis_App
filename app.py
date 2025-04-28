from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('water_quality_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    prediction = model.predict(final_features)[0]
    output = round(prediction, 2)
    
    if output >= 0.5:
        result = "Safe to Drink (Potable)"
    else:
        result = "Not Safe to Drink (Not Potable)"
    
    return render_template('index.html', prediction_text=f'Water Quality Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)