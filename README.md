# Sentiment Analysis

## Dataset
sentiment140: http://help.sentiment140.com/for-students (1600000 tweets)

## Installation
```
docker build -t sentiment-keras .
```

## Run
```
docker run --rm -v $(pwd):/app -v $(pwd)/data:/app/data -it --network dev-pg sentiment-keras
```

## Sentiment analysis with logistic regressions
```
python3 sentiment_logistic_regression.py
```