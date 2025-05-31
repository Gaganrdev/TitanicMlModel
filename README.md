# 🧠 Titanic Survival Prediction API (Flask + ML Model Deployment)

This project is a simple Flask-based REST API that predicts Titanic survival outcomes using a trained machine learning model. The model uses three features: Pclass, Sex, and Age.

The entire project runs locally using Python and Flask, and can be tested via Postman. The source code is version-controlled and hosted on **GitHub.

---

## 📦 Tech Stack

- Python 3.10+
- scikit-learn
- Flask (REST API)
- pickle (for model serialization)
- Postman (for API testing)
- GitHub (for code hosting)

---

## 📁 Project Structure


.
├── app.py               # Flask API
├── titanic.pkl          # Trained ML model (pickled)
├── requirements.txt     # Python dependencies
├── titanic.ipynb
└── README.md


---

## 🔍 Model Summary

- Features Used: Pclass, Sex, Age
- Preprocessing:
  - Sex is label-encoded (0 = female, 1 = male)
- Classifier: Logistic Regression
- Training: Conducted in a Jupyter Notebook 
- Model Format: Saved using pickle

---

## 🚀 Getting Started Locally

### 1. Clone the Repository

bash
git clone https://github.com/your-username/titanic-flask-api.git
cd titanic-flask-api


### 2. Set Up the Environment

bash
# Optional: Create virtual environment
python -m venv .venv
source .venv/bin/activate   # For Linux/macOS
# .venv\Scripts\activate  # For Windows

# Install dependencies
pip install -r requirements.txt


### 3. Run the Flask App

bash
python app.py


API will be running at: http://127.0.0.1:5000/predict

---

## 🔄 API Usage

### Endpoint


POST /predict


### Request JSON

json
{
  "Pclass": 3,
  "Age": 22,
  "Sex_male": 1
}


### Response JSON

json
{
  "survived": 0
}


Where 0 = Did not survive, 1 = Survived.

---

## 🧪 Testing the API with Postman

1. Open Postman.
2. Create a new POST request to: http://127.0.0.1:5000/predict
3. Go to Body > raw > select JSON.
4. Paste the request body:

json
{
  "Pclass": 2,
  "Age": 30,
  "Sex_male": 0
}


5. Click Send to get the prediction response.

---


## 📄 License

MIT License. Free to use, modify, and share.

---

## 🙌 Acknowledgements

- Titanic dataset from Github  
- scikit-learn  
- Flask  
- Postman