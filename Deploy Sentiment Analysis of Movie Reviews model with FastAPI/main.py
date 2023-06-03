# -*- coding: utf-8 -*-
"""
@author: Ammar
"""

# Data manipulation
import pandas as pd 
# from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
import re
import pickle

# text preprocessing modules
import string
from string import punctuation
import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Classification 
from sklearn.linear_model import RidgeClassifier

# Dealing with Models, Vectorizers, Checkpoints saved files
import joblib

# FastAPI, and an ASGI server.
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

# load the sentiment Tfidf Vectorizer
TfidfVec = joblib.load('TfidfVec_Lemma.pkl')
# load the sentiment Ridge Classifier
model = joblib.load('Ridge_Classifier.pkl')
# Remove HTMLs tags.
def remove_htmls(text):
    '''Remove HTMLs tags'''
    return BeautifulSoup(text, "lxml").text

# Remove urls tags.
def remove_urls(text):
    '''Remove urls tags'''
    pattern = r'http[^\s]*'
    return re.sub(pattern, '', text)

# Remove Images.
def remove_images(text):
    '''Remove Images'''
    text = re.sub(r"pic\.twitter\.com/\S+",'', text)
    text = re.sub("\w+(\.png|\.jpg|\.gif|\.jpeg)", " ", text)
    return text

# Remove Mention.
def remove_mention(text):
    '''Remove Mention'''
    pattern = r"@\S+"
    return re.sub(pattern, '', text)

# Function to remove emoji.
def remove_emoji(text):
    '''Function to remove emoji'''
    emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Function for removing emoticons.
def remove_emoticons(text):
    '''Function for removing emoticons'''
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U0001F1F2-\U0001F1F4"  # Macau flag
    u"\U0001F1E6-\U0001F1FF"  # flags
    u"\U0001F600-\U0001F64F"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U0001F1F2"
    u"\U0001F1F4"
    u"\U0001F620"
    u"\u200d"
    u"\u2640-\u2642"
    "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'',text)
    return text

# Remove Mention.
def remove_non_ascii(text):
    '''Remove Mention'''
    return ''.join(i for i in text if i in string.printable)

# Remove Punctuation.
def remove_punctuation(text):
    '''Remove Punctuation'''
    return re.sub('[%s]' % re.escape(string.punctuation), '', text)

# Remove more than one alphabatic characters.
def remove_extra_alphabatic(text):
    '''Remove more than one alphabatic characters'''
    pattern = r'(.{2})\1+'
    return re.sub(pattern, r'\1', text)

# Remove first and end spaces.
def remove_first_end_spaces(string):
    '''Remove first and end spaces'''
    return "".join(string.rstrip().lstrip())

# Remove Numbers.
def remove_numbers(text):
    '''Remove Numbers'''
    text = re.sub(r'[d]+', r'', text)
    text = re.sub(r'[0-9]*','',text)
    text = re.sub(r'([0-9]*\-[0-9]*)*', '', text)
    return text

# Remove single char.
def remove_single_char(text):
    '''
    Removes single characters from string, if present

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Removes words whose length falls below the threshold (by default = 1)

    Args:
      text (str): String to which the functions are to be applied, string

    Returns:
      String with removed words whose length was below the threshold (by default = 1)
    ''' 
    threshold = 2

    words = word_tokenize(text)
    text = ' '.join([word for word in words if len(word) > threshold])
    return text

english_stop_words = set(stopwords.words('english'))
english_stop_words.update(punctuation)

# Remove stop words.
def remove_stop_words(text):
    '''Remove stop words'''
    return ' '.join([word for word in word_tokenize(text) if word not in english_stop_words])

words = set(nltk.corpus.words.words())
def non_eng(text):
  sent = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
  return sent

def process_text_data(text_data):
    '''Applying Remove HTMLs, Remove URLs, Remove Images, Remove mentions, Remove Emoji, Remove Emoticons
    Remove non-ASCII character, Remove Punctuation, Remove Extra Alphabatic characters, Remove first and end spaces,
    Remove Numbers, Convert text words to lower case, Remove single character, Remove non english characters, and Remove stop words functions'''
    text_data = (text_data
                 .apply(lambda word : word.lower())
                 .apply(remove_urls)
                 .apply(remove_htmls)
                 .apply(remove_images)
                 .apply(remove_mention)
                 #.apply(remove_emoji)
                 #.apply(remove_emoticons)
                 .apply(remove_punctuation)
                 .apply(remove_non_ascii)
                 .apply(remove_first_end_spaces)
                 .apply(remove_extra_alphabatic)
                 .apply(remove_single_char)
                 .apply(remove_stop_words)
                 .apply(remove_numbers)
                 .apply(non_eng)
                #.apply(lemmatize)
                 )
    
    return text_data

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    
    proc_test_data = pd.DataFrame(data={'review': [review]})
    
    # clean the review
    cleaned_review = process_text_data(proc_test_data['review'])

    # Convert the data again to dataframe.
    proc_test_data = pd.DataFrame(data={'review': cleaned_review})
    
    # Extract text lemmatization
    lemma = []
    for review in nlp.pipe(proc_test_data['review']):
        if review.has_annotation("DEP"):
            lemma.append([n.lemma_ for n in review])
        else:
            lemma.append(None)
    proc_test_data['review'] = lemma

    # Lemmatized and re-joined.
    proc_test_data['review'] = proc_test_data['review'].apply(' '.join)
    proc_test_data['review'] = proc_test_data['review'].apply(remove_single_char)
    
    # Vectorization
    tfidfdata = TfidfVec.transform(proc_test_data['review'])
    tfidfdata = tfidfdata.reshape(1, -1)
    # perform prediction
    prediction = model.predict(tfidfdata)
    output = int(prediction[0])
    
    probas = model.decision_function(tfidfdata)
    
    output_probability = "{:.2f}".format(float(probas))
    
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}

    return result