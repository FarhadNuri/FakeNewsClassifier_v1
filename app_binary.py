from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import pickle
import re
import numpy as np
from scipy.sparse import hstack
import os

app = Flask(__name__)
CORS(app)

tfidf_word = None
tfidf_char = None
model = None



REPO_ID = "nuri63/fake-news_binary-model"


with open(os.path.join(os.path.dirname(__file__), "templates", "index_binary.html"), encoding="utf-8") as _f:
    _HTML = _f.read()

def load_model():
    global tfidf_word, tfidf_char, model

    print("Downloading model files from Hugging Face …")
    model_path    = hf_hub_download(repo_id=REPO_ID, filename="best_model.pkl")
    word_vec_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_word_vectorizer.pkl")
    char_vec_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_char_vectorizer.pkl")

    with open(word_vec_path, "rb") as f:
        tfidf_word = pickle.load(f)
    with open(char_vec_path, "rb") as f:
        tfidf_char = pickle.load(f)
    with open(model_path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, str):
        raise RuntimeError(
            f"best_model.pkl contains the string '{raw}' instead of a fitted "
            f"model object.\n\n"
            f"Fix in Colab – re-run the save cell with:\n"
            f"  best_model = results[best_model_name]['model']\n"
            f"  with open('best_model.pkl', 'wb') as f:\n"
            f"      pickle.dump(best_model, f)   # <-- save model, NOT name\n"
            f"Then re-upload to Hugging Face."
        )

    model = raw
    print(f"Model loaded: {type(model).__name__}")
    print(f"Word vectorizer vocab size: {len(tfidf_word.vocabulary_):,}")
    print(f"Char vectorizer vocab size: {len(tfidf_char.vocabulary_):,}")


def clean_bengali_text(text: str) -> str:
    """Matches clean_bengali_text() from the training notebook exactly."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    text = re.sub(r"\[.*?\]", "", text)

    text = re.sub(r"Source-\s*\w+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"[^\u0980-\u09FF\s\u0964\u0965]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_text(text: str) -> dict:
    cleaned = clean_bengali_text(text)

    if not cleaned:
        raise ValueError(
            "No Bengali characters found after cleaning. "
            "Please enter text in Bengali (Unicode)."
        )

    w = tfidf_word.transform([cleaned])
    c = tfidf_char.transform([cleaned])
    X = hstack([w, c])

    prediction = int(model.predict(X)[0])   
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]   # [P(Fake), P(Authentic)]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]       # positive = Authentic
        prob_authentic = float(sigmoid(score))
        probabilities = np.array([1.0 - prob_authentic, prob_authentic])
    else:
        probabilities = np.zeros(2)
        probabilities[prediction] = 1.0

    label = "Authentic" if prediction == 1 else "Fake"

    all_scores = [
        {"label": "Fake",      "score": float(probabilities[0])},
        {"label": "Authentic", "score": float(probabilities[1])},
    ]
    all_scores.sort(key=lambda x: x["score"], reverse=True)

    return {
        "prediction": label,
        "confidence": float(probabilities[prediction]),
        "all_scores": all_scores,
    }


@app.route("/")
def home():
    try:
        return render_template_string(_HTML)
    except Exception:
        html_path = os.path.join(os.path.dirname(__file__), "templates", "index_binary.html")
        with open(html_path) as f:
            return f.read()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or tfidf_word is None or tfidf_char is None:
            return jsonify({"error": "Model not loaded yet. Please wait and try again."}), 503

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON in request body."}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided."}), 400

        result = predict_text(text)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    load_model()