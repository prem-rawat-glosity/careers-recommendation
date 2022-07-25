import os
import pandas as pd
from etl.load_data import unique_fields, load_careers
from etl.utils import read_configs
from config import settings
import pytest


class TestETLModule:
    unique_fields_test_case_1 = ([{"career_name": "Python Developer", "career_tags": "Python, Flask, REST API"},
                                  {"career_name": "Java Developer", "career_tags": "Java, SpringBoot"},
                                  {"career_name": "Business Analyst", "career_tags": "Business Managemenet, Statistics"}
                                  ], {"career_name", "career_tags"})

    unique_fields_test_case_2 = ([{"career_name": "Python Developer"},
                                  {"career_name": "Java Developer", "career_tags": "Java, SpringBoot"},
                                  {"career_name": "Business Analyst"}
                                  ], {"career_name", "career_tags"})
    unique_fields_test_case_3 = ([{"career_name": "Python Developer", "career_tags": "Python, Flask, REST API"},
                                  {"career_name": "Java Developer", "career_tags": "Java, SpringBoot"},
                                  {"career_name": "Business Analyst", "career_embedding": [0.21, 0.031, 0.315]}
                                  ], {"career_name", "career_tags", "career_embedding"})

    @pytest.mark.parametrize("input_info_dict, expected_cols", [unique_fields_test_case_1,
                                                                unique_fields_test_case_2,
                                                                unique_fields_test_case_3])
    def test_unique_fields(self, input_info_dict, expected_cols):
        unique_cols = unique_fields(input_info_dict)
        assert unique_cols == expected_cols

    def test_load_careers(self):
        data, columns = load_careers(
            service_api=f"{settings.api_base_url}/api/{settings.api_version}/{settings.service_type}",
            configs=read_configs(os.path.join(os.getcwd(), "tests/test_config.yml")))
        assert type(data) == pd.DataFrame
        assert set(data.columns).issubset(set(columns))
