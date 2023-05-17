# -*- coding: utf-8 -*-

import unicodedata
import string
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud, STOPWORDS
from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
import plotly
from sklearn.model_selection import train_test_split
from textblob import Word
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize.toktok import ToktokTokenizer
import spacy
#from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import preprocessor as p
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle

# pip install tweet-preprocessor

# 1. Import Libraries

# 2. Loading Dataset

original_questions = pd.read_csv('questions.csv')
# make a copy so that the original dataset is not affected
questions = original_questions.copy()

# questions.count()
# Dropping Empty and Duplicate Data

questions = questions.dropna()
questions = questions.drop_duplicates()
questions = questions.rename(columns={'Y': 'target'})
questions = questions.drop(columns='Id')

# 3. Data Pre-processing


def preprocess_Title(row):
    Title = row['Title']
    Title = p.clean(Title)
    return Title


questions['Title'] = questions.apply(preprocess_Title, axis=1)

# questions.head()

# questions = questions.dropna()
# questions = questions.drop_duplicates()
# questions.count()


def stopword_removal(row):
    Title = row['Title']
    Title = remove_stopwords(Title)
    return Title


questions['Title'] = questions.apply(stopword_removal, axis=1)

questions['Title'] = questions['Title'].str.lower().str.replace(
    '[^\w\s]', ' ').str.replace('\s\s+', ' ')

# 4. Tokenization

nltk.download('punkt')

questions['Title'] = questions['Title'].apply(word_tokenize)

# 5. Stemming and Lemmatization

stemmer = PorterStemmer()


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text])


questions['Title'] = questions['Title'].apply(lambda text: stem_words(text))

# 6. Lexicon Based Sentiment Calculation


questions['polarity'] = questions['Title'].apply(
    lambda x: TextBlob(x).sentiment.polarity)
questions['subjectivity'] = questions['Title'].apply(
    lambda x: TextBlob(x).sentiment.subjectivity)


def getAnalysis(score):
    if score < 0:
        return '-1'
    elif score == 0:
        return '0'
    else:
        return '1'


questions['Analysis'] = questions['polarity'].apply(getAnalysis)

# sns.set_style(style= 'darkgrid')
# sns.relplot(x='polarity', y='subjectivity',
#             sizes=(40, 400), alpha=.5, palette="mako",
#             height=6, aspect = 3,data=questions)

# Word Cloud

# from PIL import Image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# import matplotlib.pyplot as plt

# text = " ".join(review for review in questions.Title)
# print ("There are {} words in the combination of all review.".format(len(text)))

# wordcl = WordCloud(background_color="white").generate(text)
# plt.imshow(wordcl, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# questions.head()

# questions.tail()

# 7. Word Embedding (Bag of Words)

bow = CountVectorizer(min_df=2, max_features=100000)
bow.fit(questions['Title'])
questions_processed = bow.transform(questions['Title']).toarray()

# 8. Splitting of Datasets

y = questions.drop(
    labels=['polarity', 'subjectivity', 'Id', 'Title', 'Y'], axis=1)
X = questions.drop(labels=['Analysis', 'Id', 'Title', 'Y'], axis=1)

# y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_test

# y_test.info()

# 9. Model Prediction


# Calling the Class
NBC = BernoulliNB()

# Fitting the data to the classifier
NBC.fit(X_train, y_train.values.ravel())

# Predict on test data
y_pred = NBC.predict(X_test)

# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test,y_pred))

# import seaborn as sns
# import matplotlib.pyplot as plt
# ax= plt.subplot()
# annot=True to annotate cells
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax = ax,cmap='Blues',fmt='');
# labels, title and ticks
# ax.set_xlabel('Predicted labels');
# ax.set_ylabel('True labels');
# ax.set_title('Confusion Matrix (NB)');
# ax.xaxis.set_ticklabels(['Positive', 'Neutral', 'Negative']); ax.yaxis.set_ticklabels(['Positive','Neutral', 'Negative']);

# make pickle file
# pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(NBC, open("model.pkl", "wb"))
