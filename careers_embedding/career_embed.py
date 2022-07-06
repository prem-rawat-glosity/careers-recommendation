import requests
import os
import joblib
import json
from config import settings
from etl.load_data import load_careers
from processing.feature_engineering import feature_engineer
from processing.feature_processing import generate_training_data


def download_model(model_url: str, destination_folder: str):
    filename = model_url.split('/')[-1].replace(" ", "_")
    model_file_path = os.path.join(destination_folder, filename)

    # create folder if it does not exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    else:
        # return path of the model file if it
        # already exist
        if os.path.exists(model_file_path):
            return os.path.abspath(model_file_path)

    response = requests.get(model_url, stream=True)

    if response.ok:
        with open(model_file_path, 'wb') as fp:
            for chunk in response.iter_content(chunk_size=1024 * 8):
                if chunk:
                    fp.write(chunk)
                    fp.flush()
                    os.fsync(fp.fileno())
        return os.path.abspath(model_file_path)
    else:  # HTTP status code 4XX/5XX
        raise Exception("Download failed: status code {}\n{}".format(response.status_code,
                                                                     response.text))


def get_model_object(api_url, d2v=True):
    if d2v:
        endpoint = "{}/getD2VModel".format(api_url)
        model_info = requests.get(endpoint).json()["d2vModelData"][0]
        # model download url
        model_download_link = f"{settings.models_base_url}/d2v_models/{model_info['key']}"
        model_filepath = download_model(model_url=model_download_link, destination_folder="models")
        model_obj = joblib.load(model_filepath)
    else:
        endpoint = "{}/getTFIDFModel".format(api_url)
        model_info = requests.get(endpoint).json()["tfidfModelData"][0]
        # model download url
        model_download_link = f"{settings.models_base_url}/d2v_models/{model_info['key']}"
        model_filepath = download_model(model_url=model_download_link, destination_folder="models")
        model_obj = joblib.load(model_filepath)
    return model_obj


def updateEmbeddingIntoDB(x, saving_url):
    print("{}: ".format(str(x["career_name"])), end="")
    payload = json.dumps({"career_name": str(x["career_name"]),
                          "tfidf_embedding": list(map(lambda p: float(p), x["tfidf_vectors"])),
                          "d2v_embedding": list(map(lambda p: float(p), x["d2v_vectors"]))
                          })
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", saving_url, headers=headers, data=payload)

    if response.ok:
        print("Embedding for this {}".format(response.json()["message"]))
    else:
        raise Exception("Inappropriate Response Code: {}".format(response.status_code))


def updateCareerEmbedding(service_api, configs):
    # initialize d2v model
    d2v_obj = get_model_object(api_url=service_api, d2v=True)

    # initialize tfidf model
    tfidf_obj = get_model_object(api_url=service_api, d2v=False)

    # get careers with corresponding hashtags
    print("Started Preprocessing of Careers Data...")
    careers_dataset, career_cols = load_careers(service_api=service_api, configs=configs)
    engineered_feature = feature_engineer(careers_dataset, cols=career_cols)
    careers = generate_training_data(engineered_feature, preprocessing_model=configs["SPACY_MODEL"])
    print("Successfully Processed.")

    print("Started Getting D2V Embeddings...")
    careers_dataset['d2v_vectors'] = careers.apply(lambda x: d2v_obj.model.infer_vector(x.split(" ")))
    print("D2V Embedding Successfully Generated for all careers.")

    print("Started Getting TFIDF Embeddings...")
    careers_dataset['tfidf_vectors'] = careers.apply(lambda x: tfidf_obj.transform([x]).toarray()[0])
    print("TFIDF Embedding Successfully Generated for all careers.")

    # combine career_name with vectors
    careers_with_embedding = careers_dataset.to_dict('records')

    # DB saving api url
    print("Saving Embedding into DB...")
    careers_embedding_saving_api = "{}/saveCareers".format(service_api)
    for career in careers_with_embedding:
        updateEmbeddingIntoDB(career, saving_url=careers_embedding_saving_api)
    print("All Embedding Successfully Saved into DB")
