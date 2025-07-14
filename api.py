from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Charger le modèle (quand tu l'auras)
# model = joblib.load("eligibility_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Simulation du modèle pour l'instant
    # prediction = model.predict([data["features"]])[0]
    prediction = 1 if data["income"] > 100000 else 0

    return jsonify({"score": prediction})

if __name__ == "__main__":
    app.run(debug=True)
