import numpy as np
import pandas as pd
import pickle

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import activations, initializers, regularizers, constraints
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

VECTORIZER_FILE = 'vect_cnn.pkl'
MODEL_FILE = 'sentiment_cnn.hdf5'
CHECKPOINT_FILE = 'sentiment_cnn_weights.hdf5'
MAX_FEATURES = 2000
EPOCHS = 3
BATCH_SIZE= 32

EMBED_DIMS = 32

class SentimentAnalysisCnn:
  def __init__(self, train_filename='data/full_preprocessed.csv', test_filename='data/minnesota_test.csv'):
    self.df = pd.read_csv(train_filename, encoding = "ISO-8859-1")
    self.df = self.df[pd.notnull(self.df['clean_text'])]

    # self.dftest = self.read_brand_test_data(test_filename)
    self.vect = None
    self.max_fatures = MAX_FEATURES

  def train(self):
    filepath="saved_models/{}".format(CHECKPOINT_FILE)

    X, max_len = self.tokenize()
    y = pd.get_dummies(self.df['label']).values

    X_train, X_val, Y_train, Y_val = train_test_split(X,y, test_size = 0.3, random_state = 42)

    model = self.build_model(max_len)

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val), callbacks = [checkpoint])

    model.save("saved_models/{}".format(MODEL_FILE))

    score,acc = model.evaluate(X_val, Y_val, verbose = 2, batch_size = BATCH_SIZE)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

  def build_model(self, max_len):
    model = Sequential()
    model.add(Embedding(self.max_fatures, EMBED_DIMS, input_length=max_len))

    model.add(Conv1D(filters=32, kernel_size=3, padding='valid', strides=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(2,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

  def tokenize(self):
    self.df['max_len'] = self.df['clean_text'].apply(lambda x: len(x))
    max_len = self.df['max_len'].max()

    tokenizer = Tokenizer(num_words=self.max_fatures)
    tokenizer.fit_on_texts(self.df['clean_text'].values)
    sequences = tokenizer.texts_to_sequences(self.df['clean_text'].values)
    x_train_seq = pad_sequences(sequences, maxlen=max_len)
    print('Found %d unique words.' % len(tokenizer.word_index))

    with open("saved_models/{}".format(VECTORIZER_FILE), 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
      print ('tokenizer saved')

    return x_train_seq, max_len

analyzer = SentimentAnalysisCnn()
analyzer.train()