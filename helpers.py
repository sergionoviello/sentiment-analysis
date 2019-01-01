import pandas as pd

def read_data(filename,  limit=None):
  columns=['label', 'id', 'created_at', 'query', 'user', 'text']
  df = pd.read_csv(filename, header=None, names=columns, encoding = "ISO-8859-1")
  if limit != None:
    pos = df[df.label == 0]
    neg = df[df.label == 4]
    pos = pos.head(limit)
    neg = neg.head(limit)
    return pd.concat([pos, neg], ignore_index=True)

  return df