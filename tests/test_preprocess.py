import pandas as pd
import spacy
import pytest
from etl.utils import read_configs
from processing.feature_engineering import feature_engineer
from processing.feature_processing import (preprocessing,
                                           generate_training_data)


class TestProcessing:
    @pytest.fixture(scope="class")
    def columns(self):
        return ["career_name", "career_tools", "career_skills"]

    @pytest.fixture(scope="class")
    def testing_data(self):
        test_case = {
            "career_name": ["Graphic Designer", "Big Data Engineer", "Marketing Analyst"],
            "career_tools": ["Adobe Illustrator, Adobe Photoshop, Adobe Creative Suite",
                             "Apache STORM, Amazon Web Services (AWS), OpenShift, Cassandra",
                             "Microsoft Excel, Tableau, Data Studio, SPSS"
                             ],
            "career_skills": ["Typography, Logo Creation, Choosing Fonts, Color Theory, Modifying Designs",
                              "Big Data Concepts, Logical Thinking, Fundamentals of Algorithms",
                              "Statistics, RFM Analysis, SWOT Analysis, Marketing Segmentation"
                              ],
            "added_time": ["2018-07-27", "2021-11-10", "2022-05-28"]
        }
        return pd.DataFrame.from_dict(test_case)

    @pytest.fixture(scope="class")
    def engineered_data(self):
        engi_data = {
            "features": ["Graphic Designer Adobe Illustrator, Adobe Photoshop, Adobe Creative Suite "
                         "Typography, Logo Creation, Choosing Fonts, Color Theory, Modifying Designs",
                         "Big Data Engineer Apache STORM, Amazon Web Services (AWS), OpenShift, "
                         "Cassandra Big Data Concepts, Logical Thinking, Fundamentals of Algorithms",
                         "Marketing Analyst Microsoft Excel, Tableau, Data Studio, SPSS Statistics, "
                         "RFM Analysis, SWOT Analysis, Marketing Segmentation"]
        }
        return pd.DataFrame(engi_data)['features']

    @pytest.fixture(scope="class")
    def preprocessed_text(self):
        features = ["graphic designer adobe illustrator adobe photoshop adobe creative suite "
                    "typography logo creation choose fonts color theory modifying design",
                    "big data engineer apache storm amazon web services aws openshift "
                    "cassandra big data concepts logical thinking fundamental algorithms",
                    "marketing analyst microsoft excel tableau data studio spss statistics "
                    "rfm analysis swot analysis marketing segmentation"]
        return pd.Series(features, name='features')

    @pytest.fixture(scope="class")
    def configs(self):
        return read_configs(file="tests/test_config.yml")

    @pytest.fixture(scope="class")
    def nlp_model(self, configs):
        return spacy.load(configs["SPACY_MODEL"])

    def test_merge_features(self, testing_data, engineered_data, columns):
        result = feature_engineer(data=testing_data, cols=columns)
        assert result.loc[0] == engineered_data.loc[0]
        assert result.loc[1] == engineered_data.loc[1]
        assert result.loc[2] == engineered_data.loc[2]

    def test_preprocessed_text(self, engineered_data, preprocessed_text, nlp_model):
        result = preprocessing(text=engineered_data.loc[0], model=nlp_model)
        assert result == preprocessed_text.loc[0]

        result = preprocessing(text=engineered_data.loc[1], model=nlp_model)
        assert result == preprocessed_text.loc[1]

        result = preprocessing(text=engineered_data.loc[2], model=nlp_model)
        assert result == preprocessed_text.loc[2]

    def test_generate_training_data(self, engineered_data, preprocessed_text, configs):
        result = generate_training_data(features=engineered_data,
                                        preprocessing_model=configs["SPACY_MODEL"])
        assert result.loc[0] == preprocessed_text.loc[0]
        assert result.loc[1] == preprocessed_text.loc[1]
        assert result.loc[2] == preprocessed_text.loc[2]
