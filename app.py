from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Get the current working directory and set the model path relative to that
model_path = os.path.join(os.getcwd(), 'question_model.pkl')

# Load the model if the file exists
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    print("Model not found")

# Function to process the question (this should match how your model works)
def process_question(question):
    # Implement the logic to process the question and use the model
    # Example:
    answer = model.predict([question])  # Adjust according to your model's prediction method
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('question')
    if user_question:
        answer = process_question(user_question)
        return jsonify({'answer': answer[0]})  # Adjust according to your model output
    else:
        return jsonify({'error': 'No question provided'})

if __name__ == '__main__':
    app.run(debug=True)
