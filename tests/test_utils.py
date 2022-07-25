from etl.utils import read_configs

expected_content = {'CAREERS_DATA': {'FIELDS': ['career_name', 'career_tags'],
                                     'GET_API': 'getCareers'},
                    'D2V': {'DB_UPLOAD_API': 'updateD2VModel',
                            'PARAMS': {'alpha': 0.025,
                                       'dbow_words': 1,
                                       'dm': 1,
                                       'epochs': 100,
                                       'min_alpha': 0.001,
                                       'shrink_windows': True,
                                       'vector_size': 100,
                                       'window': 5},
                            'SERVER_UPLOAD_API': 'updateD2VModelToServer',
                            'VERSION': 'V.1.0'},
                    'SPACY_MODEL': 'en_core_web_sm',
                    'TFIDF': {'DB_UPLOAD_API': 'updateTFIDFModel',
                              'PARAMS': {'encoding': 'utf-8',
                                         'lowercase': True,
                                         'ngram_range': [1, 1],
                                         'stop_words': 'english'},
                              'SERVER_UPLOAD_API': 'updateTFIDFModelToServer',
                              'VERSION': 'V.1.0'}}


def test_read_configs_instance():
    result = read_configs(file="tests/test_config.yml")
    assert isinstance(result, dict)


def test_read_configs_content():
    result = read_configs(file="tests/test_config.yml")
    assert result == expected_content
