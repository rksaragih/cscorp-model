import joblib
import re


# =========================
# LOAD MODEL
# =========================
model = joblib.load("../../models/toxic/toxic_model.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# =========================
# PREDICT FUNCTION
# =========================
def predict_toxic(message):
    cleaned = clean_text(message)

    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0]

    toxic_score = probability[1]

    return {
        "is_toxic": bool(prediction),
        "toxic_score": float(toxic_score)
    }


# =========================
# TEST
# =========================
if __name__ == "__main__":
    text = input("Enter message: ")

    result = predict_toxic(text)

    print(result)