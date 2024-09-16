import random
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import spacy


# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load intents and preprocessing data
intents = json.loads(open('ai_bot\\chatbot\\new_intents.json').read())
words = pickle.load(open('ai_bot\\chatbot\\words.pkl', 'rb'))
classes = pickle.load(open('ai_bot\\chatbot\\classes.pkl', 'rb'))

# Define and load the PyTorch model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Modify path to your trained PyTorch model file
model_path = 'ai_bot\\chatbot\\best_chatbot.pth'
model = ChatbotModel(input_size=len(words), hidden_size=512, output_size=len(classes))
model.load_state_dict(torch.load(model_path))
model.eval()

def clean_up_sentence(sentence):
    doc = nlp(sentence)
    sentence_words = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    bow_tensor = torch.from_numpy(bow).float().unsqueeze(0)
    outputs = model(bow_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]
    return predicted_class

def get_response(predicted_intent, intents_json):
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            return response
    return "Sorry, I don't have the answer for that."

def get_chatbot_response(transcribed_text):
    predicted_intent = predict_class(transcribed_text)
    response = get_response(predicted_intent, intents)
    return response
