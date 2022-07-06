from flask import Flask, jsonify
from modelling.training import build_model
from careers_embedding import updateCareerEmbedding
import requests
from utils import read_configs, get_service_api
from config import settings
from joblib import dump

app = Flask(__name__)

SERVICE_API = f"{settings.api_base_url}/api/{settings.api_version}/{settings.service_type}"


@app.route("/careers/d2v/train/", methods=['GET'])
def generate_d2v_model():
    configs = read_configs(file="./config.yml")
    # service_api = get_service_api(configs)
    model, params, matrices = build_model(service_api=SERVICE_API,
                                          configs=configs,
                                          d2v=True)

    dump(model, filename="d2v", compress=0)

    model_file = {"model": open("d2v", "rb")}

    response = requests.post(f"{SERVICE_API}/{configs['D2V']['SERVER_UPLOAD_API']}",
                             files=model_file)

    model_file["model"].close()

    if response["status"] == 200:
        updation_parameters = {"key": response["key"],
                               "version": configs["D2V"]["VERSION"],
                               "matrices": {"params": params, "score": matrices}
                               }

        test_response = requests.post(f"{SERVICE_API}/{configs['D2V']['DB_UPLOAD_API']}",
                                      json=updation_parameters).json()
        return test_response
    else:
        return None


@app.route("/careers/tfidf/train/", methods=['GET'])
def generate_tfidf_model():
    configs = read_configs(file="./config.yml")
    # service_api = get_service_api(configs)
    model, params = build_model(service_api=SERVICE_API,
                                configs=configs, )

    dump(model, filename="tfidf", compress=0)

    model_file = {"model": open("tfidf", "rb")}

    response = requests.post(f"{SERVICE_API}/{configs['D2V']['SERVER_UPLOAD_API']}",
                             files=model_file).json()
    model_file["model"].close()

    if response["status"] == 200:
        updation_parameters = {"key": response['key'],
                               "version": configs['TFIDF']['VERSION'],
                               "matrices": {'params': params}
                               }

        test_response = requests.post(f"{SERVICE_API}/{configs['TFIDF']['DB_UPLOAD_API']}",
                                      json=updation_parameters).json()
        return test_response
    else:
        return None


@app.route("/careers/update_embeddings", methods=['GET'])
def update_career_embedding():
    configs = read_configs(file="./config.yml")
    # service_api = get_service_api(configs)
    updateCareerEmbedding(service_api=SERVICE_API,
                          configs=configs)
    return jsonify({"message": "Successfully Update All Career Embeddings"})


@app.route("/")
def welcome():
    message = """
    Welcome to the Career Recommendation APIs<br><br>
    
    <b>"/d2v/train/":</b> This API train the D2V model<br>
    <b>"/tfidf/train/":</b> This API train the TFIDF model<br>
    """
    return message


if __name__ == '__main__':
    app.run(debug=True)
