import requests
import pandas as pd


def unique_fields(careers_info: list[dict]):
    if isinstance(careers_info, list):
        return set(key for keys in careers_info for key in keys)
    else:
        raise Exception("Input Argument 'career_info' should be List Type.")


def load_careers(service_api: str, configs: dict):
    if "CAREERS_DATA" in configs.keys():
        if "GET_API" in configs["CAREERS_DATA"]:
            get_data_api = service_api + "/" + configs["CAREERS_DATA"]["GET_API"]
            data_fields = configs["CAREERS_DATA"]["FIELDS"]
        else:
            raise Exception("GET_API key is not present in config file within CAREERS_DATA")
    else:
        raise Exception("CAREERS_DATA key is not found in config file")

    response = requests.get(get_data_api)
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception("Request is NOT Succeeded!")

    if len(data["careersData"]) >= 1 and set(data_fields).issubset(unique_fields(data["careersData"])):
        return pd.DataFrame(data["careersData"], columns=data_fields), data_fields
    else:
        raise Exception("Not Matching Expected Fields in Response i.e. {}.".format(set(data_fields)))
