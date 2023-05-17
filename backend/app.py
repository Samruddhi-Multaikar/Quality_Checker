import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import nltk
import preprocessor as p
from textblob import TextBlob
from gensim.parsing.preprocessing import remove_stopwords
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stem_words(text):       
    return " ".join([stemmer.stem(word) for word in text])

# preprocessing the incoming data
def preprocess(question):
  question = p.clean(question)
  question = remove_stopwords(question)
  question = question.lower().replace('[^\w\s]',' ').replace('\s\s+', ' ')
  question = word_tokenize(question)
  question = stem_words(question)
  polarity = TextBlob(question).sentiment.polarity
  subjectivity = TextBlob(question).sentiment.subjectivity
  return [polarity, subjectivity]

# load the classifier
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

# Create flask app
app = Flask(__name__)

# define home page using route method
@app.route("/")
def Home():
    return render_template("index.html")
    

# define predict method
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        search = request.form.get('search')
        query = preprocess(search)
        arr = np.array([query])
        df = pd.DataFrame(arr, columns = ['polarity','subjectivity'])
        my_prediction = model.predict(df)
        print(my_prediction)
        if my_prediction[0] == '0':
            prediction = "HQ"
        elif my_prediction[0] == '-1':
            prediction = "LQ_CLOSE"
        else:
            prediction = "LQ_EDIT"

        return render_template("index.html", prediction_text="The quality of the question is {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
