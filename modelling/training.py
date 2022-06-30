from etl.load_data import load_careers
from processing.feature_engineering import feature_engineer
from processing.feature_processing import generate_training_data
from modelling.d2v_model import build_d2v
from modelling.tfidf_model import build_tfidf


def build_model(service_api, configs, d2v=False):
    careers_dataset, career_cols = load_careers(service_api=service_api,
                                                configs=configs)

    engineered_feature = feature_engineer(careers_dataset,
                                          cols=career_cols)

    training_data = generate_training_data(engineered_feature,
                                           preprocessing_model=configs["SPACY_MODEL"])

    if d2v:
        return build_d2v(training_data, configs)
    else:
        return build_tfidf(training_data, configs)
