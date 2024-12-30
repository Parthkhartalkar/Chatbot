import random
import json
import torch
import mysql.connector
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

# Setup device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Fetch response from the MySQL database
def fetch_from_database(tag):
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(
            host="enter host name",       # Your MySQL host
            user="enter username",            # Your MySQL username
            password="enter password",     # Your MySQL password
            database="enter database name"      # Your database name
        )
        cursor = conn.cursor()

        # Query to fetch the response for a given tag
        query = "SELECT response FROM responses WHERE tag = %s"
        cursor.execute(query, (tag,))
        result = cursor.fetchone()

        conn.close()
        return result[0] if result else "Sorry, I couldn't find an answer for that."
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return "Database error. Please try again later."

# Generate a response based on the user message
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if "database" in intent and intent["database"]:
                    return fetch_from_database(tag)
                return random.choice(intent['responses'])

    return "I do not understand..."

# Function to determine greeting based on time
def get_time_based_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning! How can I help you today?"
    elif 12 <= current_hour < 18:
        return "Good afternoon! How can I help you today?"
    elif 18 <= current_hour < 22:
        return "Good evening! How can I help you today?"
    else:
        return "Hello! How can I help you today?"

@app.route('/')
def home():
    greeting = get_time_based_greeting()
    return render_template('index.html', greeting=greeting)

@app.route('/get', methods=['GET', 'POST'])
def chat():
    user_message = request.args.get('msg')
    bot_response = get_response(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
