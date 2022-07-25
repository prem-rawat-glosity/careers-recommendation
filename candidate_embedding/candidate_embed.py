import requests
import json
import pandas as pd
from processing.feature_engineering import feature_engineer
from processing.feature_processing import generate_training_data
from careers_embedding.career_embed import get_model_object


def updateUserEmbedding(cid, service_api, configs):
    # initialize d2v model
    d2v_obj = get_model_object(api_url=service_api, d2v=True)

    # initialize tfidf model
    tfidf_obj = get_model_object(api_url=service_api, d2v=False)

    # get careers with corresponding hashtags
    print("Loading user info...")
    user_info_url = "{}/{}?candidate_id={}".format(service_api, "getCandidateDetailsById", str(cid))
    user = requests.request(method="GET", url=user_info_url).json()

    if user['status'] == 200:
        user_data = pd.DataFrame.from_dict({"aspirations": [user["candidate"]["aspirations"]],
                                            "hashtags": [user["candidate"]["hashtags"]]})
        print("Safely loaded candidate info.")
        # print(user_data.head())
        engineered_feature = feature_engineer(user_data, cols=["aspirations", "hashtags"])
        # print(engineered_feature)
        user_data = generate_training_data(engineered_feature, preprocessing_model=configs["SPACY_MODEL"])
        print(user_data.head())
        print("Successfully Processed.")

        print("Started Getting D2V Embeddings...")
        d2v_vectors = user_data.apply(lambda x: d2v_obj.model.infer_vector(x.split(" ")))
        # print(d2v_vectors.loc[0])
        print("D2V Embedding Successfully Generated.")

        print("Started Getting TFIDF Embeddings...")
        tfidf_vectors = user_data.apply(lambda x: tfidf_obj.transform([x]).toarray()[0])
        # print(tfidf_vectors.loc[0])
        print("TFIDF Embedding Successfully Generated.")

        # DB saving api url
        print("Saving Embedding into DB...")
        user_embedding_saving_api = "{}/updateCandidateEmbeddings".format(service_api)

        payload = json.dumps({"candidate_id": str(cid),
                              "tfidf_embedding": list(map(lambda p: float(p), tfidf_vectors.loc[0])),
                              "d2v_embedding": list(map(lambda p: float(p), d2v_vectors.loc[0]))
                              })

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", user_embedding_saving_api,
                                    headers=headers, data=payload)
        print("Successfully Saved Embedding of user into DB.")

        return response
    else:
        return user
