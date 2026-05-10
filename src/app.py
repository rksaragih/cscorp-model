from flask import Flask, request, jsonify
from flask_cors import CORS

from production.predictor import predict_cost
from toxic.predictor import predict_toxic

app = Flask(__name__)
CORS(app)  

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "ML API is running..."

@app.route("/predict-production", methods=["POST"])
def predict_production():
    try:
        data = request.json

        result = predict_cost(data)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

@app.route('/check-toxic', methods=['POST'])
def check_toxic():
    try:
        data = request.json

        message = data.get('message', '')

        result = predict_toxic(message)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)