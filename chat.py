print("Chatbot is starting...")

import random
import json
import torch
import openai
import requests
from model import NeuralNet
from nltk_utils import tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "E_Learn"

def get_response(msg):
    if msg.lower() == "quit":
        print("Exiting chatbot...")
        exit()
    
    words = tokenize(msg)
    response = "Hi, I'm E-Learn. How can I assist you?"

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if msg.lower() == pattern.lower():
                response = random.choice(intent['responses'])
                return response

    for intent in intents["intents"]:
        if any(msg.lower() in pattern.lower() for pattern in intent["patterns"]):
            response = random.choice(intent['responses'])
            return response

    for intent in intents["intents"]:
        for keyword in intent.get("keywords", []):
            if keyword.lower() in msg.lower():
                for pattern in intent["patterns"]:
                    if keyword.lower() in pattern.lower():
                        response = random.choice(intent['responses'])
                        return response

    response = None

    if not response:
        response = search_with_google(msg)

    if not response:
        response = "Sorry, I couldn't find an answer to your question."

    return response


def search_with_google(query):
    params = {
        'key': "YOUR_GOOGLE_API_KEY",
        'cx': "YOUR_SEARCH_ENGINE_ID",
        'q': query
    }

    response = requests.get('https://www.googleapis.com/customsearch/v1', params=params)
    
    if response.status_code != 200:
        print("Google API Error:", response.status_code)
        print(response.text)
        return None

    data = response.json()

    if 'items' in data:
        first_result = data['items'][0]
        title = first_result.get('title')
        link = first_result.get('link')
        snippet = first_result.get('snippet')
        snippet = ' '.join(snippet.split()[:1500])
        return f"{title} - {snippet} ({link})"

    print("No items found in search results.")
    return None

if __name__ == "__main__":
    print("E_Learn: Hello! How can I help you?")
    while True:
        msg = input("You: ")
        response = get_response(msg)
        print(f"{bot_name}: {response}")
