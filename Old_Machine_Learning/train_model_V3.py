import pickle
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load spaCy model for better text processing
nlp = spacy.load("en_core_web_sm")

# Load the questions and answers from the JSON file
with open("questions.json", "r", encoding='utf-8') as file:
    data = json.load(file)
    questions = data['questions']
    answers = data['answers']

print(f"Loaded {len(questions)} questions and answers.")

# Preprocess the questions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Use spaCy for better sentence reconstruction
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return lemmatized_text

# Apply preprocessing to questions
questions = [preprocess_text(q) for q in questions]

# Create a pipeline for text classification
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),  # TF-IDF with bigrams
    ("clf", KNeighborsClassifier())  # KNN Classifier
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "clf__n_neighbors": [3, 5, 7],
    "clf__weights": ['uniform', 'distance']
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(questions, answers)

# Best parameters from GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}")

# Save the trained model to a pickle file
with open("question_model.pkl", "wb") as model_file:
    pickle.dump(grid_search.best_estimator_, model_file)

print("Model has been trained and saved as 'question_model.pkl'.")

# Function to predict answers
def predict_answer(question):
    # Preprocess the input question
    question = preprocess_text(question)
    
    # Predict the answer using the trained model
    prediction = grid_search.best_estimator_.predict([question])
    
    return prediction[0]

# Evaluation - Classification Report and Confusion Matrix
predictions = grid_search.best_estimator_.predict(questions)

# Classification report
print("Classification Report:")
print(classification_report(answers, predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(answers, predictions)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(answers), yticklabels=set(answers))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
