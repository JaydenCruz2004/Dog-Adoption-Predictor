## Dog Adoption Project
## Belmont University
## Jayden Cruz
## DSC-4900-01: Data Science Project/Portfolio (Spring 2026)

# features.py
# All feature engineering functions live here
# Called by main.py to prepare the data before training

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from models.config import SPEED_TO_SCORE, DOG_TYPE

nltk.download("vader_lexicon", quiet=True)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(train_path, test_path):
    """
    Load train and test CSVs and filter to dogs only.
    Returns two dataframes: train and test.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = train[train["Type"] == DOG_TYPE].copy()
    test = test[test["Type"] == DOG_TYPE].copy()

    print("Training dogs:", len(train))
    print("Test dogs:", len(test))

    return train, test


# ==============================================================================
# ADOPTION SCORE
# ==============================================================================

def add_adoption_score(df):
    """
    Convert the AdoptionSpeed column (0-4) to a continuous 0-1 score.
    Higher score means the dog was adopted faster.
    Mapping is defined in config.py.
    """
    df["AdoptionScore"] = df["AdoptionSpeed"].map(SPEED_TO_SCORE)
    return df


# ==============================================================================
# SENTIMENT FEATURES
# ==============================================================================

def add_sentiment_features(df):
    """
    Run VADER sentiment analysis on the Description column.
    Adds four new columns to the dataframe:
        sentiment_neg      — how negative the text is (0 to 1)
        sentiment_neu      — how neutral the text is  (0 to 1)
        sentiment_pos      — how positive the text is (0 to 1)
        sentiment_compound — overall tone (-1 very negative to +1 very positive)
    """
    sia = SentimentIntensityAnalyzer()

    # Fill missing descriptions so VADER does not crash on NaN
    descriptions = df["Description"].fillna("")
    scores = descriptions.apply(lambda text: sia.polarity_scores(text))

    df["sentiment_neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment_neu"] = scores.apply(lambda s: s["neu"])
    df["sentiment_pos"] = scores.apply(lambda s: s["pos"])
    df["sentiment_compound"] = scores.apply(lambda s: s["compound"])

    return df
