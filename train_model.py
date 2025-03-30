import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier

# Load a pre-trained language model that understands sentence meaning
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the questions and answers from the JSON file
with open("questions.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    questions = data["questions"]
    answers = data["answers"]

print(f"Loaded {len(questions)} questions and answers.")

# Convert questions into sentence embeddings
question_embeddings = sentence_model.encode(questions)

# Train a KNN classifier on the embeddings
clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
clf.fit(question_embeddings, answers)

# Save the trained model
with open("question_model.pkl", "wb") as model_file:
    pickle.dump((sentence_model, clf), model_file)

print("Model has been trained and saved as 'question_model.pkl'.")

# Function to predict answers
def predict_answer(question):
    # Convert the input question to an embedding
    question_embedding = sentence_model.encode([question])
    
    # Get the closest match
    distances, indices = clf.kneighbors(question_embedding, n_neighbors=3)
    
    # If the best match is too far, return a fallback response
    if distances[0][0] > 0.5:  # Adjust threshold as needed
        # Try to improvise by checking the next best matches
        improvise_answer = "I'm not entirely sure, but here's something related:"
        related_answers = [answers[i] for i in indices[0]]
        improvise_answer += " " + ", ".join(related_answers)
        return improvise_answer
    
    return answers[indices[0][0]]

# Example usage
user_question = "angel of mercy cost mana"
print(f"Predicted answer: {predict_answer(user_question)}")
