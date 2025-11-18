from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import load_model, predict_single

app = Flask(__name__)
CORS(app)
model = load_model('models/ensemble_tabular.joblib')

@app.route('/')
def home():
    return 'Cancer Prediction API running ✅'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error':'no JSON provided'}), 400
    res = predict_single(data, model)
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)