import streamlit as st
import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the intents JSON data
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Prepare data for training
data = []
labels = []
for intent in intents:  # Iterate through the list of intents
    for pattern in intent['patterns']:  # Use 'patterns' instead of 'text'
        data.append(pattern)
        labels.append(intent['tag'])  # Use 'tag' instead of 'intent'

# Vectorize the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
y = labels

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier
clf = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Define the chatbot function
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_text_vectorized)[0]
    
    # Find and return a response based on the tag
    for intent in intents:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, I don't understand."

# Streamlit UI
st.set_page_config(page_title="Chatbot", page_icon="", layout="wide")
# Add some custom CSS for styling the chat UI
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f7f7f7;
        border-radius: 12px;
        border: 1px solid #e1e1e1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 50px;
    }
    .message-box {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 16px;
        line-height: 1.5;
    }
    .user-message {
        background-color: #d1f7c4;
        text-align: right;
        border-left: 3px solid #4CAF50;
    }
    .bot-message {
        background-color: #e3f2fd;
        text-align: left;
        border-left: 3px solid #2196F3;
    }
    .current-message {
        background-color: #e3ffe6;
        text-align: center;
        border: 1px solid #4CAF50;
    }
    .input-box {
        width: 80%;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-top: 10px;
        font-size: 16px;
    }
    .send-button {
        padding: 12px;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        margin-top: 10px;
        font-size: 16px;
    }
    .send-button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title(" Chatbot")
st.write("Ask the chatbot a question and get a response!")

# Chat container to hold the messages
chat_container = st.container()

# User input and send button
user_input = st.text_input("Your message", placeholder="Type something...", key="user_input", label_visibility="collapsed")
if st.button("Send", key="send_button"):
    if user_input:
        response = chatbot(user_input)

        # Display user message
        with chat_container:
            st.markdown(f'<div class="message-box user-message">{user_input}</div>', unsafe_allow_html=True)

        # Display bot response
        with chat_container:
            st.markdown(f'<div class="message-box bot-message">{response}</div>', unsafe_allow_html=True)

        # Highlight current message with a different style (for better distinction)
        with chat_container:
            st.markdown(f'<div class="message-box current-message">Typing...</div>', unsafe_allow_html=True)
