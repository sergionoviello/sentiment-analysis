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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


EPOCHS = 3
BATCH_SIZE= 32

NB_WORDS = 10000
MAX_LEN = 24
EMBED_DIMS = 8

class SentimentAnalysisCnn:
  def __init__(self):
    self.vect = None

    # df = pd.read_csv('data/train_pre_processed.csv')
    # df = df.head(2)
    # print(df['clean_text'].values[0])

  def train(self):
    texts = ['deep learning is very deep', 'this is deep', 'the sun is bright']

    X = self.word_embeddings(texts)
    y = [1, 0, 1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=37)

    # Keras provides a convenient way to convert each word into a multi-dimensional vector.
    # This can be done with the Embedding layer. It will compute the word embeddings (or use pre-trained embeddings) and look up each word in a dictionary to find its vector representation.
    # Here we will train word embeddings with 8 dimensions.

    model = Sequential()
    model.add(Embedding(NB_WORDS, EMBED_DIMS, input_length=MAX_LEN))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

  def train_pre_trained_model(self):
    texts = ['deep learning is very deep', 'this is deep', 'the sun is bright']

    X = self.word_embeddings(texts)
    y = [1, 0, 1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=37)

    embed_dict = create_word_embeddings_dict()
    # With the GloVe embeddings loaded in a dictionary we can look up the embedding
    # for each word in the corpus of the airline tweets.
    # These will be stored in a matrix with a shape of NB_WORDS and GLOVE_DIM.
    # If a word is not found in the GloVe dictionary, the word embedding values for the word are zero.
    embed_matrix = create_word_embeddings_matrix(embed_dict)

    glove_model = models.Sequential()
    glove_model.add(layers.Embedding(NB_WORDS, EMBED_DIMS, input_length=MAX_LEN))
    glove_model.add(layers.Flatten())
    glove_model.add(layers.Dense(3, activation='softmax'))

    # In the Embedding layer (which is layer 0 here) we set the weights for the words to those found in
    # the GloVe word embeddings. By setting trainable to False we make sure that the GloVe word embeddings
    # cannot be changed. After that we run the model.
    glove_model.layers[0].set_weights([embed_matrix])
    glove_model.layers[0].trainable = False

    glove_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = glove_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

  def create_word_embeddings_matrix(self, embed_dict):
    emb_matrix = np.zeros((NB_WORDS, EMBED_DIMS))
    for w, i in tk.word_index.items():
        if i < NB_WORDS:
            vect = embed_dict.get(w)
            if vect is not None:
            emb_matrix[i] = vect
        else:
            break
    return emb_matrix

  def create_word_embeddings_dict(self):
    filename = "data/{}".format(GLOVE)
    emb_dict = {}
    glove = open(filename)
    for line in glove:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        emb_dict[word] = vector
    glove.close()
    return emb_dict

  def bag_of_words(self, texts):
    vect = CountVectorizer(min_df=1, ngram_range=(1,1), analyzer='word')
    X = vect.fit_transform(texts)

    print(X.todense())


  def bag_of_words_tfidf(self, texts):
    vect = TfidfVectorizer(min_df=1, ngram_range=(1,1), analyzer='word')
    X = vect.fit_transform(texts)
    print(vect.get_feature_names())
    print(X.todense())


  def word_embeddings(self, texts):
    max_fatures = NB_WORDS

    tokenizer = Tokenizer(num_words=max_fatures)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    x_train = pad_sequences(sequences, maxlen=MAX_LEN)
    return x_train

analyzer = SentimentAnalysisCnn()
analyzer.train()