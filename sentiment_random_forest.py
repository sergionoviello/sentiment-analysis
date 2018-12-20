import numpy as np
import pandas as pd
import re
import pickle

import nltk.data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from text_preprocessor import TextPreprocessor

class SentimentAnalysisLogReg:
  def __init__(self):
    self.text_preprocessor = TextPreprocessor()
    self.df = self.read_data(filename='data/training.1600000.processed.noemoticon.csv', limit=25000)
    self.dftest = self.read_brand_test_data()
    # self.dftest = self.read_data(filename='data/testdata.manual.2009.06.14.csv', limit=22000)
    self.vect = None

  def is_not_ascii(self, string):
    return string is not None and any([ord(s) >= 128 for s in string])

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

  def read_brand_test_data(self):
    df = pd.read_csv('data/minnesota_test2.csv')
    df = df[~df['Sentiment'].apply(self.is_not_ascii)]
    df = df.rename(columns={'Snippet': 'text', 'Sentiment': 'label'})
    df = df[df.label != 'neutral']
    df['label'] = df['label'].apply(lambda x: 0 if x == 'negative' else 4)
    return df

  def clean_text(self, df):
    df['clean_text'] = df['text'].apply(self.text_preprocessor.pre_process)
    df['label'] = df['label'].replace(4,1)

  def bag_of_words(self, min_df=1, ngrams=(1, 1)):
    X = self.df['clean_text']
    self.vect = CountVectorizer(min_df=min_df, ngram_range=ngrams, analyzer='word')
    bow_transformer = self.vect.fit(X)
    X = bow_transformer.transform(X)
    return X

  def build_model(self, X_train, Y_train):
    # param_grid = {
    #     'n_estimators': [200, 500, 1000],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth' : [4,5,6,7,8],
    #     'criterion' :['gini', 'entropy']
    # }
    best_param_grid = {
        'n_estimators': [1000],
        'max_features': ['log2'],
        'max_depth' : [8],
        'criterion' :['entropy']
    }
    rfc = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rfc, best_param_grid, cv=5)
    grid.fit(X_train, Y_train)

    return grid

  def evaluate_train(self):
    print("Score val set: {:.2f}".format(self.model.best_score_))
    print("Best parameters: ", self.model.best_params_)
    # print("Best estimator: ", self.model.best_estimator_)

  def evaluate_test(self, X_test, Y_test):
    lr = self.model.best_estimator_
    print("Score test set: {:.2f}".format(lr.score(X_test, Y_test)))

  def predict(self, X_test, Y_test):
    lr = self.model.best_estimator_
    lr.fit(X_test, Y_test)
    preds = lr.predict(X_test)
    return preds

nltk.download('punkt')
nltk.download('stopwords')

analyzer = SentimentAnalysisLogReg()
analyzer.build_train_model()
preds = analyzer.make_predictions_on_test_model()