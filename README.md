# imdbgenre

This is a simple practice to build a web app with Flask. 

imdb.csv: 

  1. imdb datasets from kaggle:  https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb
  2. combined the genres into five categories and utilized LabelEncoder to label as: 0: "action", 1: "comedy", 2: "documentary", 3: "drama", 4: "short"
  3. add "year" and "title" columns by extracting from the movie column
  4. clean the "title" and "summary" columns by removing the stopwords, lowercase, etc. 
  5. clean the "year" column by removing the tuples with wrong and unreasonable values. 

model
  1. Tfidf + Naive Bayes
