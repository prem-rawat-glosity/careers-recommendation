import requests
import numpy as np
import scipy.spatial as sp
from operator import itemgetter
from config import settings
import json


def calculateSimilarityScores(vector1: np.matrix, vector2: np.matrix) -> np.matrix:
    similarity_score = 1 - sp.distance.cdist(vector1, vector2, 'cosine')
    return similarity_score


def get_harmonic_mean(d2vscore, tfidfscore):
    if tfidfscore <= 0:
        return 0.0
    elif d2vscore <= 0:
        return np.mean([0.0, tfidfscore])
    else:
        return np.mean([d2vscore, tfidfscore])


def getSortedScoreIndices(scores: np.array):
    return np.argsort(scores)[::-1]


def choose_n(n, careersData: list):
    if n < 0:
        n = len(careersData)
    elif n > len(careersData):
        n = len(careersData)
    return n


def get_topN_careers(cid, service_api, n: int = -1):
    print("Started to generate recommendation for candidate id {}".format(str(cid)))
    user_info_url = "{}/{}?candidate_id={}".format(service_api, "getCandidateDetailsById",
                                                   str(cid))
    response = requests.request("GET", url=user_info_url).json()
    # print("get user info: {}".format(response))

    if str(response['status']) == '200':
        user_info = response['candidate']
        if len(user_info['d2v_embedding']) > 0 and len(user_info['tfidf_embedding']) > 0:
            user_d2v_mat = np.matrix(user_info['d2v_embedding'])
            user_tfidf_mat = np.matrix(user_info['tfidf_embedding'])
        else:
            print("Note: Not found user embeddings!")
            print("Generating and Updating user embeddings...")
            url = f"http://{settings.FLASK_RUN_HOST}:{settings.FLASK_RUN_PORT}/ai/careers/user_embedding/"
            url += f"?candidateID={str(cid)}"
            # print(url)
            updated_info = requests.request(method='GET', url=url).json()
            if updated_info['status'] == 200:
                user_d2v_mat = np.matrix(updated_info['candidateData']['d2v_embedding'])
                user_tfidf_mat = np.matrix(updated_info['candidateData']['tfidf_embedding'])
                print("Successfully generated and updated the user embeddings.")
            else:
                return {"status": '412', "message": 'Not Found Embeddings'}
    else:
        return response

    """######### Print User Basic Info ################
    print("User Information:")
    print("Name: {}".format(user_info['name']))
    print("Aspirations: {}".format(user_info['aspirations']))
    print("HashTags: {}".format(user_info['hashtags']))
    ################################################"""

    careers_info_url = "{}/{}".format(service_api, "getCareers")
    response = requests.request("GET", url=careers_info_url)

    if response.ok:
        careers_info = response.json()
        careers_docvectors = np.matrix(list(map(lambda x: x['d2v_embedding'], careers_info['careersData'])))
        careers_tfidfvectors = np.matrix(list(map(lambda x: x['tfidf_embedding'], careers_info['careersData'])))
        d2v_scores = calculateSimilarityScores(user_d2v_mat, careers_docvectors).flatten()

        tfidf_scores = calculateSimilarityScores(user_tfidf_mat, careers_tfidfvectors).flatten()

        combined_scores = np.array(list(map(get_harmonic_mean, d2v_scores, tfidf_scores)))

        sorted_indices = getSortedScoreIndices(combined_scores)

        n = choose_n(n, careersData=careers_info['careersData'])

        top_n_careers = list(itemgetter(*sorted_indices.tolist()[:n])(careers_info['careersData']))

        top_n_careers = list(map(lambda x: {x['career_name']: x['career_tags']}, top_n_careers))

        """print("Top {} recommended careers:".format(n))
        for i, career in enumerate(top_n_careers):
            print("{}. {}".format(i + 1, list(career.keys())[0]))"""

        update_recs_url = "{}/{}".format(service_api, "updateCandidateRecommendations")
        payload = json.dumps({"candidate_id": cid,
                              "recommended_careers": top_n_careers
                              })
        # print(payload)
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", update_recs_url, headers=headers, data=payload).json()
        print("Successfully got and updated N recommendations")
    return response['candidateData']['recommended_careers']
