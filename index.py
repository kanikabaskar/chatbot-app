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

# Define the chatbot function (with a single response)
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_text_vectorized)[0]
    
    # Find and return one response based on the tag
    for intent in intents:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, I don't understand."

# Streamlit UI
st.set_page_config(page_title="Chatbot Application", page_icon="🤖", layout="centered")

# Add some custom CSS for styling the chat UI
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f2f2f2;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    .chat-container {
        flex-grow: 1;
        width: 100%;
        overflow-y: auto;  /* Allow only vertical scrolling in the response container */
        padding: 20px;
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
    .input-container {
        padding: 10px;
        background-color: #fff;
        border-top: 1px solid #ddd;
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
    }
    .input-box {
        width: 80%;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ccc;
        font-size: 16px;
    }
    .send-button {
        padding: 12px;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        margin-left: 10px;
    }
    .send-button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🤖 Chatbot Application")
st.write("Ask the chatbot a question and get a response!")

# Chat container to hold the messages
chat_container = st.container()

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Add messages to the chat history
def add_message(text, is_user=True):
    if is_user:
        st.session_state['messages'].append({'message': text, 'is_user': True})
    else:
        st.session_state['messages'].append({'message': text, 'is_user': False})

# Display all previous messages inside the scrollable container
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state['messages']:
        if msg['is_user']:
            st.markdown(f'<div class="message-box user-message">{msg["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-box bot-message">{msg["message"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# User input and send button (fixed at the bottom)
with st.container():
    user_input = st.text_input("Type your message here...", placeholder="Type something...", key="user_input", label_visibility="collapsed", value="")
    send_button = st.button("Send", key="send_button", help="Send message")
    
    if send_button and user_input:
        # Add user message to history
        add_message(user_input, is_user=True)
        
        # Get chatbot response (single response)
        response = chatbot(user_input)
        
        # Add bot response to history
        add_message(response, is_user=False)
        
        # Clear the input field by setting the value to an empty string
        user_input = ""
