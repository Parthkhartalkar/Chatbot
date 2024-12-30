import random
import json
import mysql.connector
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
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

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# MySQL Database Connection
def get_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host="enter host name",
            user="enter user name",
            password="enetr password",
            database="chatbot"
        )
        print("MySQL connection established successfully.")
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Function to fetch response from MySQL
def get_response_from_mysql(tag):
    try:
        connection = get_mysql_connection()
        if connection is None:
            return None  # If the connection fails, return None

        cursor = connection.cursor()
        query = "SELECT response FROM responses WHERE tag = %s"
        cursor.execute(query, (tag,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return None
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        if connection:
            connection.close()

# Main response function
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

    # First, check if we have a matching response in the intents.json file
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    response = get_response_from_mysql(tag)
    if response:
        return response

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

# Main loop
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    print(f"Bot: {get_time_based_greeting()}")  # Initialize the conversation with a time-based greeting
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(f"Bot: {resp}")
