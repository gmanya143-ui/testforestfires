import pickle
from flask import Flask, request, render_template

# Initialize Flask
application= Flask(__name__)
app=application

# Load trained model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/sc.pkl', 'rb'))

# Home page
@app.route("/")
def index():
    return render_template('index.html')

# Prediction page
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoints():
    result = None
    if request.method == 'POST': 

        try:
            # Get form data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes=float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Prepare feature array (exclude target)
            features = [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Region]
            features_2d = [features] 
            # Scale features
            features_scaled = standard_scaler.transform([features])

            # Predict
            prediction = ridge_model.predict(features_scaled)
            result = float(prediction[0])

        except Exception as e:
            result = f"Error: {e}"

    else:
        return render_template('home.html', results=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
