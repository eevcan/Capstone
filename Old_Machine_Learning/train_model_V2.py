import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from textblob import TextBlob

# Load the questions and answers from the JSON file
with open("questions.json", "r", encoding='utf-8') as file:
    data = json.load(file)
    questions = data['questions']
    answers = data['answers']

# Print out the number of questions loaded
print(f"Loaded {len(questions)} questions and answers.")

# Preprocess the questions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Correct spelling mistakes
    text = str(TextBlob(text).correct())
    return text

# Apply preprocessing to questions
questions = [preprocess_text(q) for q in questions]

# Create a pipeline for text classification
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),  # Converts text to numerical features
    ("clf", KNeighborsClassifier(n_neighbors=3))  # KNN Classifier
])

# Train the model
print("Training the model...")
pipeline.fit(questions, answers)

# Save the trained model to a pickle file
with open("question_model.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)

print("Model has been trained and saved as 'question_model.pkl'.")

# Function to predict answers
def predict_answer(question):
    # Preprocess the input question
    question = preprocess_text(question)
    
    # Predict the answer using the trained model
    prediction = pipeline.predict([question])
    
    # If the answer is too uncertain, return a fallback response
    if prediction == "unknown":
        return "I don't know this yet."
    return prediction[0]
