from flask import Flask, request, jsonify, render_template
import pickle
import os
import json
import numpy as np

app = Flask(__name__)

# Model-Pfad setzen
model_path = os.path.join(os.getcwd(), "question_model.pkl")

# Modell laden, wenn die Datei existiert
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        sentence_model, clf = model  # Entpacke das Modell
else:
    print("Model not found")
    model = None

# Antworten aus der JSON-Datei laden
with open("questions.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    answers = data["answers"]

# Funktion zum Verarbeiten der Frage
def process_question(question):
    if not model:
        return ["Model not loaded"]

    # Erstelle ein Embedding für die Eingabe
    question_embedding = sentence_model.encode([question])

    # Finde das nächste Match
    distances, indices = clf.kneighbors(question_embedding, n_neighbors=1)

    # Falls die Distanz zu hoch ist, Rückgabe: "I don't know this yet."
    if distances[0][0] > 0.5:
        return ["I don't know this yet."]

    return [answers[indices[0][0]]]  # Antwort zurückgeben

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question")
    if user_question:
        answer = process_question(user_question)
        return jsonify({"answer": answer[0]})
    else:
        return jsonify({"error": "No question provided"})

if __name__ == "__main__":
    app.run(debug=True)
