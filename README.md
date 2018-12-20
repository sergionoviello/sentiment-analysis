# Sentiment Analysis

## Dataset
sentiment140: http://help.sentiment140.com/for-students (1600000 tweets)

## Installation
```
docker build -t sentiment-analysis .
```

## Run
```
docker run --rm -v $(pwd):/app -v $(pwd)/data:/app/data -it --network dev-pg sentiment-analysis
```

## Sentiment analysis with logistic regressions
```
python3 sentiment_logistic_regression.py
```