import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#load the model
model = load_model('nextword.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)



def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit UI
st.title('Next Word Prediction')
input_text = st.text_input("Enter a sentence:", "To be or not to be")

if st.button('Predict Next Word'):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    if next_word:
        st.write(f"Next word prediction: {next_word}")
    else:
        st.write("Could not predict the next word.")

       

