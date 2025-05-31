from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("titanic.pkl", "rb"))


@app.route("/")  # by default all requests are get
def home():
    return "Titanic Survival Prediction API"


@app.route("/predict", methods=["POST"])  # user sends some data
def predict():
    data = request.get_json(force=True)

    features = [data["Pclass"], data["Age"], data["Sex_male"]]

    prediction = model.predict([np.array(features)])

    return f"Survival Prediction: {int(prediction[0])}"


if __name__ == "__main__":
    app.run(debug=True)