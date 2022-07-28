import time

from flask import Flask, jsonify, request
from modelling.training import build_model
from careers_embedding import updateCareerEmbedding
from candidate_embedding import updateUserEmbedding
from recommend import get_topN_careers
import requests
from etl.utils import read_configs, print_time_taken
from config import settings
from joblib import dump

app = Flask(__name__)

SERVICE_API = f"{settings.api_base_url}/api/{settings.api_version}/{settings.service_type}"


@app.route("/ai/careers/d2v/train/", methods=['GET'])
def generate_d2v_model():
    start = time.time()
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

    if response.ok:
        updation_parameters = {"key": response.json()["key"],
                               "version": configs["D2V"]["VERSION"],
                               "matrices": {"params": params, "score": matrices}
                               }

        test_response = requests.post(f"{SERVICE_API}/{configs['D2V']['DB_UPLOAD_API']}",
                                      json=updation_parameters).json()
        end = time.time()
        print_time_taken(start, end, api_name="tfidf_model_building")
        return test_response, 200
    else:
        return None


@app.route("/ai/careers/tfidf/train/", methods=['GET'])
def generate_tfidf_model():
    start = time.time()
    configs = read_configs(file="./config.yml")
    # service_api = get_service_api(configs)
    model, params = build_model(service_api=SERVICE_API,
                                configs=configs, )

    dump(model, filename="tfidf", compress=0)

    model_file = {"model": open("tfidf", "rb")}

    response = requests.post(f"{SERVICE_API}/{configs['D2V']['SERVER_UPLOAD_API']}",
                             files=model_file).json()
    model_file["model"].close()

    if response.ok:
        updation_parameters = {"key": response.json()["key"],
                               "version": configs['TFIDF']['VERSION'],
                               "matrices": {'params': params}
                               }

        test_response = requests.post(f"{SERVICE_API}/{configs['TFIDF']['DB_UPLOAD_API']}",
                                      json=updation_parameters).json()
        end = time.time()
        print_time_taken(start, end, api_name="tfidf_model_building")
        return test_response, 200
    else:
        return None


@app.route("/ai/careers/update_embeddings", methods=['GET'])
def update_career_embedding():
    start = time.time()
    configs = read_configs(file="./config.yml")
    # service_api = get_service_api(configs)
    updateCareerEmbedding(service_api=SERVICE_API,
                          configs=configs)
    end = time.time()
    print_time_taken(start, end, api_name="career_embedding")
    return jsonify({"message": "Successfully Update All Career Embeddings"}), 200


@app.route("/ai/careers/user_embedding/", methods=['GET', 'POST'])
def update_candidate_embedding():
    start = time.time()
    cid = request.values.get('candidateID')  # initialize

    configs = read_configs(file="./config.yml")
    # service_api = get_service_api(configs)
    response = updateUserEmbedding(cid=cid,
                                   service_api=SERVICE_API,
                                   configs=configs).json()
    end = time.time()
    print_time_taken(start, end, api_name="user_embedding")
    return jsonify(response)


@app.route("/ai/careers/recommend/", methods=['GET', 'POST'])
def generate_N_recommendation():
    start = time.time()
    cid = request.values.get("candidateID")
    if cid is None:
        return {"messsage": "{} is not given by client".format("candidateID")}
    try:
        freq = request.values.get("N", default=-1, type=int)
    except ValueError:
        freq = -1

    response = get_topN_careers(cid=cid,
                                service_api=SERVICE_API,
                                n=freq)
    end = time.time()
    print_time_taken(start, end, api_name="n_recommendation")
    return jsonify(response)


@app.route("/ai/careers/")
def career_api():
    message = """Welcome to the Career Recommendation APIs"""
    d2v_train_api = "{server}/ai/careers/d2v/train/"
    tfidf_train_api = "{server}/ai/careers/tfidf/train/"
    career_embedding_api = "{server}/ai/careers/update_embeddings/"
    candidate_embedding_api = "{server}/ai/careers/user_embedding/"
    n_recommendation_api = "{server}/ai/careers/recommend/"
    return jsonify({"api_message": message,
                    "endpoints": {"d2v_training_endpoint": d2v_train_api,
                                  "tfidf_training_endpoint": tfidf_train_api,
                                  "career_embedding_endpoint": career_embedding_api,
                                  "user_embedding_endpoint": candidate_embedding_api,
                                  "n_recommendation": n_recommendation_api
                                  }
                    })


@app.route("/")
def welcome():
    message = """
    <b>Welcome to the AI APIs</b></br></br>
    
    <b>GO TO {server}/ai/careers/</b>
    """
    return message


if __name__ == '__main__':
    app.run(debug=True)
