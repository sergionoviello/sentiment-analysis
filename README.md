# Sentiment Analysis

## Dataset
sentiment140: http://help.sentiment140.com/for-students (1600000 tweets)

## Installation
```
docker build -t sentiment-analysis .
```

## Run
```
docker run --rm -v $(pwd):/app -v $(pwd)/data:/app/data -it --network syfl-net sentiment-analysis
```

## Sentiment analysis with logistic regressions
```
pre process train dataframe : python3 text_preprocessor.py

python3 sentiment_logistic_regression.py train
python3 sentiment_logistic_regression.py test
```