import os
import yaml
import pprint


def read_configs(file) -> dict:
    filename = os.path.join(os.getcwd(), file)
    with open(filename, "r") as stream:
        configs = yaml.safe_load(stream)
        return configs


def get_service_api(configs: dict):
    api = ""
    if "ROOT_API" in configs.keys():
        api += configs["ROOT_API"]
    else:
        raise Exception("Not Found ROOT_API key in the config file!")

    if "API_VERSION" in configs.keys():
        api += "/" + configs["API_VERSION"]

    if "SERVICE_TYPE" in configs.keys():
        api += "/" + configs["SERVICE_TYPE"]
    return api
