import numpy as np
import pandas as pd
import re
import pickle
import csv
import sys
from tqdm import tqdm

import nltk.data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report

from text_preprocessor import TextPreprocessor

VECTORIZER_FILE = 'vect.pkl'
MODEL_FILE = 'sentiment_logistic_regression.pkl'

class SentimentAnalysisLogReg: 
  def __init__(self, train_filename='data/train_pre_processed.csv', test_filename='data/minnesota_test.csv'):
    self.df = pd.read_csv(train_filename, encoding = "ISO-8859-1")
    self.df = self.df[pd.notnull(self.df['clean_text'])]
    
    self.dftest = self.read_brand_test_data(test_filename)
    # self.dftest = self.read_data(filename='data/testdata.manual.2009.06.14.csv', limit=22000)
    self.vect = None

  def load_model(self):
    with open("saved_models/{}".format(MODEL_FILE), 'rb') as fid:
      self.model = pickle.load(fid)
    with open("saved_models/{}".format(VECTORIZER_FILE), 'rb') as f2:
      self.vect = pickle.load(f2)
    return self.model 

  def train_model(self):
    print('bag of words...')
    X = self.bag_of_words()
    y = self.df['label']
    X_train, _, Y_train, _ = train_test_split(X,y, test_size = 0.3, random_state = 42)

    print('training model...')
    self.model = self.build_model(X_train, Y_train)

    with open("saved_models/{}".format(MODEL_FILE), 'wb') as fid:
      pickle.dump(self.model, fid)

    print('evaluate model...')
    self.evaluate_train()
    return self.model

  def predict(self, model):
    tqdm.pandas()
    print('preprocessing test data...')
    tp = TextPreprocessor() 
    self.dftest['clean_text'] = self.dftest['text'].progress_apply(tp.pre_process_text)
    self.dftest['label'] = self.dftest['label'].replace(4,1)

    print('bag of words test data...')
    X_test = self.vect.transform(self.dftest['clean_text'])
    Y_test = self.dftest['label']

    print('predict...')
    preds = self.evaluate_test(X_test, Y_test, model)
    return preds

  def is_not_ascii(self, string):
    return string is not None and any([ord(s) >= 128 for s in string])

  def read_brand_test_data(self, test_filename):
    df = pd.read_csv(test_filename)
    df = df[~df['Sentiment'].apply(self.is_not_ascii)]
    df = df.rename(columns={'Snippet': 'text', 'Sentiment': 'label'})
    df = df[df.label != 'neutral']
    df['label'] = df['label'].apply(lambda x: 0 if x == 'negative' else 4)

    return df

  def bag_of_words(self, min_df=1, ngrams=(1, 1)):
    X = self.df['clean_text']

    self.vect = CountVectorizer(min_df=min_df, ngram_range=ngrams, analyzer='word')
    bow_transformer = self.vect.fit(X)
    X = bow_transformer.transform(X)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X)

    with open("saved_models/{}".format(VECTORIZER_FILE), 'wb') as fid:
      pickle.dump(self.vect, fid)

    return X_train_tfidf

  def build_model(self, X_train, Y_train):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
    grid.fit(X_train, Y_train)

    return grid

  def evaluate_train(self):
    print("Score val set: {:.2f}".format(self.model.best_score_))
    # print("Best parameters: ", self.model.best_params_)
    # print("Best estimator: ", self.model.best_estimator_)

  def evaluate_test(self, X_test, Y_test, model):

    preds = model.predict(X_test)
    #print(confusion_matrix(Y_test, preds))
    #print('\n')
    print(classification_report(Y_test, preds))

    return preds

nltk.download('punkt')
nltk.download('stopwords')

'''
  --------------------------------------------------
  MAIN
  --------------------------------------------------
'''

if len(sys.argv) == 1:
  print("task name is required. USAGE: python3 sentiment_logistic_regression.py <task>")
elif sys.argv[1] == 'train':
  analyzer = SentimentAnalysisLogReg(train_filename='data/train_pre_processed.csv', test_filename='data/minnesota_test.csv')
  analyzer.train_model()
elif sys.argv[1] == 'test':
  analyzer = SentimentAnalysisLogReg(train_filename='data/train_pre_processed.csv', test_filename='data/minnesota_test.csv')
  model = analyzer.load_model()
  preds = analyzer.predict(model)