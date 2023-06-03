# Sentiment Analysis of Movie Reviews
Using Machine Learning, Data Science and NLP techniques to detect the Reviews Sentiment using significant features given by the most linked features that extraction from that are taken into consideration when evaluating the Sentiment of Movie.

## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Evaluation](#evaluation)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)
* [Sources](#sources)

## Introduction
Understanding and Processing language and deriving criteria, ideas and answers from it is very important in data and AI, we attach importance to Natural Language Processing and we noticed in the past few days the qualitative leap achieved by the use of LLM models.

Hence I present to you my Sentiment Analysis project (sentiment analysis of movie reviews). In this project I put my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning, Data Science, and NLP.\
Hoping to improve it gradually in the coming times.

## Dataset General info
**General info about the dataset:**
* About:

A movie review dataset. NLP tasks Sentiment Analysis.
Note : all the movie review are long sentence(most of them are longer than 200 words.)
Content

two columns used (text : the review of the movie and label : the sentiment label of the movie review)

## Evaluation

1. Features Selection.
2. Compute ROC of the best performance.
3. Cross-Validation.
4. Learning Curve.
5. Classification report.
6. Validation set.
7. Accuracy Score.

## Technologies
* Programming language: Python.
* Libraries: numpy, python-math, collection, more-itertools, DateTime, regex, string, matplotlib, pandas, seaborn, beautifulsoup4, nltk, wordcloud, gensim, scikit-learn, scipy, imblearn, xgboost, swifter, random, copy, pickle, os, tqdm, joblib, torch, tensorflow, tensorflow-hub, transformers, LRFinder, sentencepiece, accelerate. 
* Application: Jupyter Notebook.

## Setup
To run this project setup the following libraries on your local machine using pip on the terminal after installing Python:

!git clone https://github.com/WittmannF/LRFinder.git

'''\
pip install numpy\
pip install python-math\
pip install collection\
pip install more-itertools\
pip install DateTime\
pip install regex\
pip install strings\
pip install matplotlib\
pip install pandas\
pip install seaborn\
pip install beautifulsoup4\
pip install nltk\
pip install wordcloud\
pip install gensim\
pip install scikit-learn\
pip install scipy\
pip install imblearn\
pip install xgboost\
pip install swifter\
pip install random2\
pip install pickle5\
pip install tqdm\
pip install joblib\
pip install torch\
pip install tensorflow\
pip install tensorflow-hub\
pip install keras\
pip install transformers\
pip install sentencepiece\
pip install --upgrade accelerate

'''\
To install these packages with conda run:\
'''\
conda install -c anaconda numpy\
conda install -c conda-forge mpmath\
conda install -c lightsource2-tag collection\
conda install -c conda-forge more-itertools\
conda install -c trentonoliphant datetime\
conda install -c conda-forge re2\
conda install -c conda-forge r-stringi\
conda install -c conda-forge matplotlib\
conda install -c anaconda pandas\
conda install -c anaconda seaborn\
conda install -c anaconda bs4\
conda install -c anaconda nltk\
conda install -c conda-forge wordcloud\
conda install -c anaconda gensim\
conda install -c anaconda scikit-learn\
conda install -c anaconda scipy\
conda install -c conda-forge imbalanced-learn\
conda install -c anaconda py-xgboost\
conda install -c conda-forge swifter\
conda install -c conda-forge mkl_random\
conda install -c conda-forge pickle5\
conda install -c conda-forge tqdm\
conda install -c anaconda joblib\
conda install -c pytorch pytorch\
conda install -c anaconda tensorflow\
conda install -c conda-forge keras\
conda install -c conda-forge transformers\
conda install -c conda-forge sentencepiece\
conda install -c conda-forge accelerate

'''

## Features
* I present to you my project solving the problem of sentiment analysis of movie reviews using a lot of effective algorithm and techniques with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it in the best possible ways and the current capabilities using Data Science, Machine Learning and NLP.

### To Do:
**Briefly about the process of the project work, here are (some) insights that I took care of it:**

* Planning.
* Exploring data.
* Exploratory data analysis (EDA).
* Data Cleaning and PreProcessing.
* Deep EDA (Find Sents, Tokens, Lemma, POS, NER, N-Grams).
* Vectorizing (Count Vectorizer, TF-IDF Vectorizer).
* Modelling (ML - Decision Tree, BernoulliNB, Complement NB, Random Forest, SVC, XGBoost, SGD, Voting, Multinomial NB, AdaBoost, Gradient Boosting, Bagging, Logistic Regression, Stacking, Ridge Classifiers).
* Evaluate and making analysis of ML models (ROC, Classification report, Cross-validation, Learning curve, Validation set, and Features Selection).
* Modelling (DL - GRU, LSTM, BiLSTM with pretrained embedding layer - GloVE).
* Evaluate DL models on (Test set, Validation set using accuracy score).
* Modelling (Transformers - Roberta-Base with TensorFlow, bert-base-uncased fine tuning using Pytorch).
* Parameters Choosing (Halving Grid Search CV).
* Deployment the model using FastAPI.

## Run Example

To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage.

2. Load the dataset.

3. Select which cell you would like to run and show its output.

4. Run Selection/Line in Python Terminal command (Shift+Enter).

In case of deployment it using FastAPI, you can follow this steps:

1. make sure you install the latest version (with pip): pip install fastapi
2. You will also need an ASGI server for production such as uvicorn: pip install uvicorn
3. Run the API
 3.1 The following command will help us to run the FastAPI app we have created: uvicorn main:app --reload

4. FastAPI provides an Automatic Interactive API documentation page. To access it navigate to http://127.0.0.1:8000/docs in your browser and then you will see the documentation page created automatically by FastAPI.

5. To make a prediction first click the “predict-review” route and then click on the button “Try it out”, it allows you to fill the review parameter and directly interact with the API.

## Sources
This data was taken from Kaggle:\
(https://www.kaggle.com/competitions/shai-training-2023-a-level-2/?fbclid=IwAR0HoK9niAS7LOYw7Lx0XKbUyG4j2tPCeQOBj6QMQNicLKxgK51lf2q-VnI)
