import pandas as pd
import json

# Load the card data
cards = pd.read_csv("data/cards.csv")

# List to store questions and answers
questions = []
answers = []

# Iterate through rows of the cards DataFrame to create questions
for _, row in cards.iterrows():
    for column in cards.columns:
        value = row[column]
        if pd.notna(value) and len(questions) < 1000:  # Limit to X questions
            question = f"What is the {column} of {row['name']}?"
            questions.append(question)
            answers.append(str(value))

# Save questions and answers to a file for training
questions_data = {'questions': questions, 'answers': answers}
with open("questions.json", "w", encoding='utf-8') as file:
    json.dump(questions_data, file, ensure_ascii=False, indent=4)

print(f"Generated {len(questions)} questions and saved to 'questions.json'.")
