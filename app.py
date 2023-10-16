from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and data vectorizer
with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X_client = dv.transform(data)
        probability = model.predict_proba(X_client)[:, 1]
        return jsonify({"probability": probability[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)