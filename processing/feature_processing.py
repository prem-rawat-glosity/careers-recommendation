import pandas as pd
import spacy
import string


def preprocessing(text: str, model):
    document = model(text)
    removed_punct = [t for t in document if t.text not in string.punctuation]
    removed_stop_words = [t for t in removed_punct if not t.is_stop]
    lemmatized = " ".join([t.lemma_ for t in removed_stop_words])
    return lemmatized.lower()


def generate_training_data(features: pd.Series, preprocessing_model) -> pd.Series:
    nlp = spacy.load(preprocessing_model)
    features = features.apply(preprocessing, args=(nlp,))
    return features
