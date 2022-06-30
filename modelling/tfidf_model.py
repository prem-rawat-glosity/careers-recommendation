import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(training_data: pd.Series, configs: dict) -> tuple:
    params = configs["TFIDF"]["PARAMS"]
    model = TfidfVectorizer(**params)
    model.fit(training_data)
    return model, params
