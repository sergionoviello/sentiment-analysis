import nltk.data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import csv
import re
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
# from helpers import read_data

class TextPreprocessor:
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
    "wtf": "angry",
    "xxx": "kiss",
    "xx": "kiss",
    "xxxx": "kiss",
    "yay": "yes",
    "yaaaay": "yes",
    "aaaaand": "and"
    }


  def pre_process_docs(self, train_file, output_file, with_stemming=True):
    #columns=['label', 'id', 'created_at', 'query', 'user', 'text']
    #df = pd.read_csv("data/{}".format(train_file), header=None, names=columns, encoding = "ISO-8859-1")
    df = pd.read_csv("data/{}".format(train_file), encoding = "ISO-8859-1")
    df = df.rename(columns={'Snippet': 'text', 'Sentiment': 'label'})
    di = { 'positive': 2, 'neutral': 1, 'negative': 0 }
    #df['label'] = df['label'].apply(lambda x: 0 if x == 'negative' else 4)
    df["label"].replace(di, inplace=True)

    print('preprocessing train data...')
    tqdm.pandas()
    if with_stemming:
      df['clean_text'] = df['text'].progress_apply(self.pre_process_text)
    else:
      df['clean_text'] = df['text'].progress_apply(self.pre_process_text_no_stemming)

    # df['label'] = df['label'].replace(4,1)
    df.to_csv("data/{}".format(output_file), header=False)


  def pre_process_text(self, text):
    stops = set(stopwords.words("english"))
    text = text.lower() # lower case
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) # remove links urls
    # convert brb in be right back
    text = self.translator(text)

    # convert arent' to are not
    words = text.split()
    reformed = [self.appos[word] if word in self.appos else word for word in words]
    reformed = " ".join(reformed)

    # this seems to be slow and it doesn't improve accuracy
    # spelling correction
    # reformed = str(TextBlob(reformed).correct())

    #remove punctuation
    tokens = nltk.word_tokenize(reformed)
    words = [word for word in tokens if word.isalpha()]
    #remove stop words
    words = [w for w in words if not w in stops]
    #stemming
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]

    return ' '.join(words)

  def pre_process_text_no_stemming(self, text):
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

    return ' '.join(words)

  def translator(self, user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "data/slang.txt"
        # File Access mode [Read Mode]
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return ' '.join(user_string)


'''
  --------------------------------------------------
  MAIN
  --------------------------------------------------
'''

# TRAIN_FILE = 'training.1600000.processed.noemoticon.csv'
TRAIN_FILE = 'tweets_sergio.csv'

if len(sys.argv) < 2:
  print("task name is required. USAGE: python3 text_preprocessor.py <task>")
elif sys.argv[1] == 'preprocess':
  nltk.download('punkt')
  nltk.download('stopwords')

  text_processor = TextPreprocessor()
  # text_processor.pre_process_docs(TRAIN_FILE, 'full_preprocessed.csv')
  text_processor.pre_process_docs(TRAIN_FILE, 'full_preprocessed_sergio.csv', False)