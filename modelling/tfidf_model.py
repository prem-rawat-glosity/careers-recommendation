import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(training_data: pd.Series, configs: dict) -> tuple:
    params = configs["TFIDF"]["PARAMS"]
    model = TfidfVectorizer(**params)
    model.fit(training_data)
    vocab_count = model.get_feature_names_out()
    params["TFIDF"]["VOCAB_SIZE"] = vocab_count
    return model, params
