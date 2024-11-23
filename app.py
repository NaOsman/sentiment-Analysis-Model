import streamlit as st
import joblib
import pandas as pd

# Load the model and label encoder
model = joblib.load('random_forest_model.pkl')  # Load your trained model
label_encoder = joblib.load('label_encoder.pkl')  # Load the label encoder

# Streamlit UI setup
st.title("Kabul Superstore's Sentiment Analysis")

# Display the questions
st.write("Please answer the following questions:")

# Input questions and options
questions = [
    "How would you rate your overall shopping experience at any superstore?",
    "How likely are you to recommend that superstore to friends and family?",
    "Was this your first visit to any superstore?",
    "Were you able to find all the products you needed?",
    "How satisfied are you with the quality of products at that store?",
    "How would you rate the cleanliness of that store?",
    "How organized was the store layout?",
    "Did you find it easy to locate products and navigate the store?",
    "Was the checkout process smooth and efficient?",
    "How would you rate the helpfulness and friendliness of that staff?",
    "Was there enough staff available to assist you when needed?",
    "Was the staff knowledgeable and able to answer your questions?",
    "How would you rate the pricing of products at that store?",
    "How satisfied are you with the parking facilities or public transport options available at that store?",
    "How would you rate the convenience of that store's location?",
    "How satisfied are you with the promotions, discounts, and loyalty programs available?"
]

# Create a dictionary for the user's responses
responses = {}

# Yes/No questions mapped to radio buttons
yes_no_questions = [2, 3, 7, 8, 10, 11]  # Indices of Yes/No questions (Q3, Q4, Q8, Q9, Q11, Q12)
for idx, question in enumerate(questions):
    if idx in yes_no_questions:
        responses[f"Q{idx+1}"] = st.radio(question, ('Yes', 'No'))
    else:
        responses[f"Q{idx+1}"] = st.slider(question, min_value=1, max_value=5)

# Function to encode Yes/No responses
def encode_yes_no(x):
    return 5 if x == 'Yes' else 1

# Apply encoding for Yes/No questions
for idx in yes_no_questions:
    responses[f"Q{idx+1}"] = encode_yes_no(responses[f"Q{idx+1}"])

# When the submit button is pressed
if st.button("Submit"):
    # Convert responses to DataFrame
    new_data = {key: [value] for key, value in responses.items()}
    new_df = pd.DataFrame(new_data)

    # Predict sentiment
    prediction = model.predict(new_df)
    predicted_label = label_encoder.inverse_transform(prediction)

    # Display result with color based on sentiment
    sentiment = predicted_label[0]
    if sentiment == 'Positive':
        st.markdown(f'<p style="color: green; font-size: 20px;">Predicted Sentiment: {sentiment}</p>', unsafe_allow_html=True)
    elif sentiment == 'Negative':
        st.markdown(f'<p style="color: red; font-size: 20px;">Predicted Sentiment: {sentiment}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color: gray; font-size: 20px;">Predicted Sentiment: {sentiment}</p>', unsafe_allow_html=True)

