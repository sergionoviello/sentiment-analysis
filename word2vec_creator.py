
import pandas as pd
from gensim.models import word2vec

OUTPUT_FILE = 'sentiment140.w2v'
INPUT_FILE = 'full_no_stem_preprocessed.csv'

# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10


class Word2VecCreator:
  def __init__(self):
    columns=['label', 'id', 'created_at', 'query', 'user', 'text', 'clean_text']
    self.df = pd.read_csv("data/{}".format(INPUT_FILE), header=None, names=columns, encoding = "ISO-8859-1")
    self.df = self.df[pd.notnull(self.df['clean_text'])]

    self.documents = [_text.split() for _text in self.df.clean_text]


  def train(self):
    w2v_model = word2vec.Word2Vec(size=W2V_SIZE,
                                  window=W2V_WINDOW,
                                  min_count=W2V_MIN_COUNT,
                                  workers=8)
    w2v_model.build_vocab(self.documents)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("Vocab size", vocab_size)

    w2v_model.train(self.documents, total_examples=len(self.documents), epochs=W2V_EPOCH)

    w2v_model.save("saved_models/{}".format(OUTPUT_FILE))


w2v = Word2VecCreator()
w2v.train()
