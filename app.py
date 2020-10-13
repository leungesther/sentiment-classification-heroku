# import libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import regex as re
from nltk.corpus import wordnet, stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
import pickle

st.title("Sentiment Analyzer for NTUC Fairprice Mobile App")
st.subheader("Esther Leung (DSI-16)")
st.write('\n\n')

classifier = pickle.load(open('model.pkl', 'rb'))

review = st.text_input("Enter The Review","Write Here...")
if st.button('Predict Sentiment'):
    review_text = BeautifulSoup(review, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = str(letters_only).lower().split()
    stops = set(stopwords.words('english'))
    stops.update(['app', 'apps', 'application', 'ntuc','fairprice'])
    meaningful_words = [w for w in words if w not in stops]
    lemmatizer = WordNetLemmatizer()
    meaningful_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    text = " ".join(meaningful_words)
    prediction = classifier.predict([text])
    prediction_proba = classifier.predict_proba([text])
    st.subheader('Prediction')
    review_sentiment = np.array(['Positive','Negative'])
    st.write(review_sentiment[prediction])
    st.subheader('Prediction Probability')
    st.write(prediction_proba)

else:
    st.write("Press the above button...")
