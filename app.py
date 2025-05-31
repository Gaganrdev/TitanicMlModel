from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

try:
    model = pickle.load(open("titanic.pkl", "rb"))
    print("Model loaded successfully.", flush=True)
except Exception as e:
    print("Failed to load model:", e, flush=True)
    raise


@app.route("/")  # by default all requests are get
def home():
    return "Titanic Survival Prediction API"


@app.route("/predict", methods=["POST"])  # user sends some data
def predict():
    data = request.get_json(force=True)

    features = [data["Pclass"], data["Age"], data["Sex_male"]]

    prediction = model.predict([np.array(features)])

    return {"Prediction" : int(prediction[0])}