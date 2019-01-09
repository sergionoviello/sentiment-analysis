import numpy as np
import pandas as pd
import pickle

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import activations, initializers, regularizers, constraints
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

VECTORIZER_FILE = 'vect_cnn_pretrained.pkl'
MODEL_FILE = 'sentiment_cnn_pretrained.hdf5'
CHECKPOINT_FILE = 'sentiment_cnn_weights_pretrained.hdf5'
GLOVE = 'glove.twitter.27B.100d.txt'

EPOCHS = 10
BATCH_SIZE= 1024

NB_WORDS = 30000# 222697
MAX_LEN = 50
EMBED_DIMS = 100

class SentimentAnalysisCnn:
  def __init__(self, train_filename='data/full_no_stem_preprocessed.csv', test_filename='data/minnesota_test.csv'):
    columns=['label', 'id', 'created_at', 'query', 'user', 'text', 'clean_text']
    self.df = pd.read_csv(train_filename, header=None, names=columns, encoding = "ISO-8859-1")
    self.df = self.df[pd.notnull(self.df['clean_text'])]

    # self.dftest = self.read_brand_test_data(test_filename)
    self.vect = None
    self.vocab_size = NB_WORDS

  def train(self):
    filepath="saved_models/{}".format(CHECKPOINT_FILE)

    self.df['max_len'] = self.df['clean_text'].apply(lambda x: len(x))
    print('max sentence length', self.df['max_len'].max())
    print('avg sentence el', self.df['max_len'].mean())
    max_len = MAX_LEN

    X = self.word_embeddings(self.df['clean_text'].values, max_len)
    y = np_utils.to_categorical(self.df['label'].values)

    X_train, X_val, Y_train, Y_val = train_test_split(X,y, test_size = 0.3, random_state = 42)

    self.embed_dict = self.create_word_embeddings_dict()
    self.embed_matrix = self.create_word_embeddings_matrix(self.embed_dict)

    model = self.build_model(max_len)

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val), callbacks = [checkpoint])

    model.save("saved_models/{}".format(MODEL_FILE))

    score,acc = model.evaluate(X_val, Y_val, verbose = 2, batch_size = BATCH_SIZE)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

  def build_model(self, max_len):
    model = Sequential()
    model.add(Embedding(self.vocab_size, EMBED_DIMS, input_length=max_len, weights=[self.embed_matrix], trainable=False))

    #model.add(Conv1D(filters=100, kernel_size=2, padding='valid', strides=1, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
#    model.add(Dropout(0.2))

    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    #model.add(Flatten())
    model.add(Dense(2,activation='softmax'))

    # model.layers[0].set_weights([self.embed_matrix])
    # model.layers[0].trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

  def create_word_embeddings_matrix(self, embed_dict):
    emb_matrix = np.zeros((self.vocab_size, EMBED_DIMS))
    for w, i in self.tokenizer.word_index.items():
        if i < self.vocab_size:
            vect = embed_dict.get(w)
            if vect is not None:
              emb_matrix[i] = vect
        else:
            break
    return emb_matrix

  def create_word_embeddings_dict(self):
    filename = "data/{}".format(GLOVE)
    emb_dict = {}
    glove = open(filename, 'r', encoding = "utf-8")
    for line in glove:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        emb_dict[word] = vector
    glove.close()
    return emb_dict

  def word_embeddings(self, texts, max_len):
    # self.tokenizer = Tokenizer(num_words=max_fatures)
    self.tokenizer = Tokenizer()
    self.tokenizer.fit_on_texts(texts)

    sequences = self.tokenizer.texts_to_sequences(texts)
    self.vocab_size = len(self.tokenizer.word_index) + 1
    print('Found %d unique words.' % len(self.tokenizer.word_index))

    x_train = pad_sequences(sequences, maxlen=max_len)
    return x_train

  # def tokenize(self):
  #   #self.df['max_len'] = self.df['clean_text'].apply(lambda x: len(x))
  #   #max_len = self.df['max_len'].max()
  #   max_len = MAX_LEN

  #   tokenizer = Tokenizer(num_words=self.max_fatures)
  #   tokenizer.fit_on_texts(self.df['clean_text'].values)
  #   sequences = tokenizer.texts_to_sequences(self.df['clean_text'].values)
  #   x_train_seq = pad_sequences(sequences, maxlen=max_len)
  #   print('Found %d unique words.' % len(tokenizer.word_index))

  #   with open(VECTORIZER_FILE, 'wb') as handle:
  #     pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
  #     print ('tokenizer saved')

  #   return x_train_seq, max_len

analyzer = SentimentAnalysisCnn()
analyzer.train()