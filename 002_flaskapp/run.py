from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import pandas as pd
import re
import string
import nltk


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)



# data cleaning
def clean_text(text):
    text = text.lower()                                  # lower-case all characters
    text =  re.sub(r'@\S+', '',text)                     # remove twitter handles
    text =  re.sub(r'http\S+', '',text)                  # remove urls
    text =  re.sub(r'pic.\S+', '',text) 
    text =  re.sub(r"[^a-zA-Z+']", ' ',text)             # only keeps characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ')      # keep words with length>1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.tokenize.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')   # remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i)>2])
    text= re.sub("\s[\s]+", " ",text).strip()            # remove repeated/leading/trailing spaces
    return text


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    
    genre_label = {0: "action",
                   1: "comedy",
                   2: "documentary",
                   3: "drama",
                   4: "short"}    

    if request.method == "POST":
        title = request.form["title"]
        year = request.form["year"]
        summary = request.form["summary"]
        print(summary) 
        # preprocessing
        summary_clean = clean_text(summary)
        X = pd.Series(summary_clean)
        pred_num = model.predict(X)
        pred=genre_label[pred_num[0]]
        print(pred)
        
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
