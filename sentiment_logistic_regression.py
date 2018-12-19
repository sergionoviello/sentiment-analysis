import numpy as np
import pandas as pd
import re
import pickle

import nltk.data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

class SentimentAnalysisLogReg:
  def __init__(self):
    self.appos = {
      "aren't" : "are not",
      "can't" : "cannot",
      "couldn't" : "could not",
      "didn't" : "did not",
      "doesn't" : "does not",
      "don't" : "do not",
      "hadn't" : "had not",
      "hasn't" : "has not",
      "haven't" : "have not",
      "he'd" : "he would",
      "he'll" : "he will",
      "he's" : "he is",
      "i'd" : "I would",
      "i'd" : "I had",
      "i'll" : "I will",
      "i'm" : "I am",
      "isn't" : "is not",
      "it's" : "it is",
      "it'll":"it will",
      "i've" : "I have",
      "let's" : "let us",
      "mightn't" : "might not",
      "mustn't" : "must not",
      "shan't" : "shall not",
      "she'd" : "she would",
      "she'll" : "she will",
      "she's" : "she is",
      "shouldn't" : "should not",
      "that's" : "that is",
      "there's" : "there is",
      "they'd" : "they would",
      "they'll" : "they will",
      "they're" : "they are",
      "they've" : "they have",
      "we'd" : "we would",
      "we're" : "we are",
      "weren't" : "were not",
      "we've" : "we have",
      "what'll" : "what will",
      "what're" : "what are",
      "what's" : "what is",
      "what've" : "what have",
      "where's" : "where is",
      "who'd" : "who would",
      "who'll" : "who will",
      "who're" : "who are",
      "who's" : "who is",
      "who've" : "who have",
      "won't" : "will not",
      "wouldn't" : "would not",
      "you'd" : "you would",
      "you'll" : "you will",
      "you're" : "you are",
      "you've" : "you have",
      "'re": " are",
      "wasn't": "was not",
      "we'll":" will",
      "didn't": "did not",
      "loooooooovvvvvvee": "love"
    }

    self.df = self.read_data(filename='data/training.1600000.processed.noemoticon.csv', limit=3000)
    self.dftest = self.read_data(filename='data/testdata.manual.2009.06.14.csv', limit=1000)
    self.vect = None

  def build_train_model(self):
    self.clean_text(self.df)
    X = self.bag_of_words()
    y = self.df['label']
    X_train, _, Y_train, _ = train_test_split(X,y, test_size = 0.3, random_state = 42)
    self.model = self.build_model(X_train, Y_train)
    self.evaluate_train()

  def make_predictions_on_test_model(self):
    self.clean_text(self.dftest)
    X_test = self.vect.fit(self.dftest['clean_text']).transform(self.dftest['clean_text'])
    Y_test = self.dftest['label']
    preds = self.predict(X_test, Y_test)
    self.evaluate_test(X_test, Y_test)
    return preds

  def pre_process(self, text):
    stops = set(stopwords.words("english"))
    text = text.lower() # lower case
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) # remove links urls

    # convert arent' to are not
    words = text.split()
    reformed = [self.appos[word] if word in self.appos else word for word in words]
    reformed = " ".join(reformed)

    #remove punctuation
    tokens = nltk.word_tokenize(reformed)
    words = [word for word in tokens if word.isalpha()]
    #remove stop words
    words = [w for w in words if not w in stops]
    #stemming
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]

    return ' '.join(words)

  def read_data(self, filename,  limit=None):
    columns=['label', 'id', 'created_at', 'query', 'user', 'text']
    df = pd.read_csv(filename, header=None, names=columns, encoding = "ISO-8859-1")
    if limit != None:
      pos = df[df.label == 0]
      neg = df[df.label == 4]
      pos = pos.head(limit)
      neg = neg.head(limit)
      return pd.concat([pos, neg], ignore_index=True)

    return df

  def clean_text(self, df):
    df['clean_text'] = df['text'].apply(self.pre_process)
    df['label'] = df['label'].replace(4,1)

  def bag_of_words(self, min_df=3, ngrams=(1, 2)):
    X = self.df['clean_text']
    self.vect = CountVectorizer(min_df=min_df, ngram_range=ngrams, analyzer='word')
    bow_transformer = self.vect.fit(X)
    X = bow_transformer.transform(X)
    return X

  def build_model(self, X_train, Y_train):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
    grid.fit(X_train, Y_train)

    return grid

  def evaluate_train(self):
    print("Score val set: {:.2f}".format(self.model.best_score_))
    # print("Best parameters: ", self.model.best_params_)
    # print("Best estimator: ", self.model.best_estimator_)

  def evaluate_test(self, X_test, Y_test):
    lr = self.model.best_estimator_
    print("Score test set: {:.2f}".format(lr.score(X_test, Y_test)))

  def predict(self, X_test, Y_test):
    lr = self.model.best_estimator_
    lr.fit(X_test, Y_test)
    preds = lr.predict(X_test)
    return preds

analyzer = SentimentAnalysisLogReg()
analyzer.build_train_model()
preds = analyzer.make_predictions_on_test_model()