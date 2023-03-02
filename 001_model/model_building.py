import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('imdb.csv')

#split the data for training and test
X = df['summary']
y = df['genre']

# Split the data into training and testing sets, stratifying by y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_df = 0.25,
                             min_df = 15,
                             ngram_range = (1,3))),
    ('clf', MultinomialNB(fit_prior=True, class_prior=None, alpha=0.01)),
])

model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
