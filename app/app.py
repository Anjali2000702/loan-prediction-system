
# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # load model
# model = joblib.load("../models/lgb_model.pkl")

# @app.route("/")
# def home():
#     return "Loan Prediction API Running"

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
    
#     features = np.array(data["features"]).reshape(1, -1)
    
#     prediction = model.predict(features)[0]
#     probability = model.predict_proba(features)[0][1]
    
#     return jsonify({
#         "prediction": int(prediction),
#         "probability": float(probability)
#     })

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("../models/lgb_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    
    prediction = model.predict([data])[0]
    probability = model.predict_proba([data])[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run()