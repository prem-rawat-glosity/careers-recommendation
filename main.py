from modelling.training import build_model
from etl.utils import read_configs
from joblib import dump
from config import settings


SERVICE_API = f"{settings.API_BASE_URL}/api/{settings.API_VERSION}/{settings.SERVICE_TYPE}"


def generate_d2v_model():
    configs = read_configs(file="config.yml")
    # service_api = get_service_api(configs)
    model, params, matrices = build_model(service_api=SERVICE_API,
                                          configs=configs,
                                          d2v=True)

    dump(model, filename="d2v", compress=0)

    return params, matrices


def generate_tfidf_model():
    configs = read_configs(file="config.yml")
    # service_api = get_service_api(configs)
    model, params = build_model(service_api=SERVICE_API,
                                configs=configs,)

    dump(model, filename="tfidf", compress=0)
    return params


if __name__ == '__main__':
    print(generate_d2v_model())
    print(generate_tfidf_model())
