import torch
import torch.nn as nn
import spacy
import numpy as np
import json
import pickle

# Load spaCy model and intents
nlp = spacy.load('en_core_web_sm')

with open('ai_bot\\chatbot\\about_jec.json', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('ai_bot\\chatbot\\about_jec_words.pkl', 'rb'))
classes = pickle.load(open('ai_bot\\chatbot\\about_jec_classes.pkl', 'rb'))

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

# Load the pre-trained model
model_path = 'ai_bot\\chatbot\\about_jec_best_chatbot.pth'
model = ChatbotModel(input_size=len(words), hidden_size=512, output_size=len(classes))
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to clean up the input sentence
def clean_up_sentence(sentence):
    doc = nlp(sentence)
    sentence_words = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return sentence_words

# Function to create a bag of words from the sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

# Function to predict the class of the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    bow_tensor = torch.from_numpy(bow).float().unsqueeze(0)
    outputs = model(bow_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]
    print(f"Predicted class: {predicted_class}")  
    return predicted_class

# Function to check if the query is about the CAN organization
def is_query_about_jec(query):
    predicted_class = predict_class(query)
    
    return predicted_class == 'janakpur_engineering_college'

def check_query(query):
    if query is None:
        print("In none section")
        return None
    print("Not none section")
    return is_query_about_jec(query)
