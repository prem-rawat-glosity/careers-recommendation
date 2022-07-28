import pytest
import os
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from config import settings
from etl.utils import read_configs
from modelling.d2v_model import Doc2VecModel, build_d2v
from modelling.tfidf_model import build_tfidf
from modelling.training import build_model

TOL = 0.1


class TestDoc2VecModel:
    @pytest.fixture(scope="class")
    def d2v_params(self):
        return {'D2V': {'PARAMS': {'alpha': 0.025, 'dbow_words': 1, 'dm': 1, 'epochs': 5,
                                   'min_alpha': 0.001, 'shrink_windows': True, 'min_count': 1,
                                   'vector_size': 5, 'window': 1
                                   }
                        }
                }

    @pytest.fixture(scope="class")
    def d2v(self, d2v_params):
        return Doc2VecModel(**d2v_params['D2V']['PARAMS'])

    @pytest.fixture(scope="module")
    def training_data(self):
        data = pd.Series(["graphic designer adobe illustrator adobe photoshop adobe",
                          "big data concepts logical thinking",
                          "analyst microsoft excel tableau",
                          "machine learning deep learning data science",
                          "react js javascript swift android developer"
                          ]
                         )
        return data

    @pytest.fixture(scope="class")
    def examples(self):
        raw_data = ["graphic designer adobe illustrator adobe photoshop adobe",
                    "big data concepts logical thinking"]
        return [simple_preprocess(raw_data[0]), simple_preprocess(raw_data[1])]

    @pytest.fixture(scope="class")
    def tagged_examples(self, examples):
        return [TaggedDocument(words=examples[0], tags=[0]), TaggedDocument(words=examples[1], tags=[1])]

    def test_d2v_init(self, d2v):
        assert d2v.size == 5
        assert d2v.dm == 1
        assert d2v.window == 1
        assert d2v.epochs == 5
        assert isinstance(d2v.model, Doc2Vec)
        assert d2v.tagged_documents is None
        assert d2v.incrementer == -1

    def test_tag_example(self, d2v, examples, tagged_examples):
        assert d2v.tag_example(examples[0]) == tagged_examples[0]
        assert d2v.tag_example(examples[1]) == tagged_examples[1]
        d2v.incrementer = -1

    def test_fit(self, training_data, d2v, d2v_params):
        d2v.fit(training_data)
        parameters = d2v_params['D2V']['PARAMS']
        assert d2v.incrementer == training_data.shape[0] - 1
        assert isinstance(d2v.tagged_documents, pd.Series)
        assert isinstance(d2v.model, Doc2Vec)
        assert len(d2v.model.wv.key_to_index) > 0
        assert d2v.model.epochs == parameters['epochs']
        assert d2v.model.corpus_count == training_data.shape[0]
        vector1 = d2v.model.infer_vector(doc_words=training_data.loc[0].split(' '))
        assert vector1.shape[0] == parameters['vector_size']
        vector2 = d2v.model.infer_vector(doc_words=training_data.loc[0].split(' '))
        assert vector2.shape[0] == parameters['vector_size']
        vector1 = np.reshape(vector1, newshape=(vector1.shape[0], 1))
        vector2 = np.reshape(vector2, newshape=(1, vector2.shape[0]))
        sim_value = d2v.model.dv.cosine_similarities(vector1, vector2)
        assert sim_value[0][0] == pytest.approx(0.96, TOL)

    def test_score(self, d2v):
        result = d2v.score()
        assert isinstance(result, float)
        assert result == pytest.approx(0.5, 1.0)

    def test_build_d2v(self, training_data, d2v_params):
        model, params, score = build_d2v(training_data=training_data,
                                         configs=d2v_params)
        assert isinstance(model, Doc2VecModel)
        assert isinstance(model.model, Doc2Vec)
        assert model.incrementer == 4
        assert params == d2v_params['D2V']['PARAMS']
        assert isinstance(score, float)


class TestTFIDFModel:
    @pytest.fixture(scope="class")
    def tfidf_params(self):
        return {'TFIDF': {'PARAMS': {'encoding': 'utf-8', 'lowercase': True,
                                     'ngram_range': [1, 1], 'stop_words': 'english'
                                     }
                          }
                }

    @pytest.fixture(scope="class")
    def training_data(self):
        data = pd.Series(["graphic designer adobe illustrator adobe photoshop adobe",
                          "big data concepts logical thinking",
                          "analyst microsoft excel tableau",
                          "machine learning deep learning data science",
                          "react js javascript swift android developer"
                          ]
                         )
        return data

    def test_build_tfidf(self, tfidf_params, training_data):
        model, params = build_tfidf(training_data=training_data,
                                    configs=tfidf_params)
        assert isinstance(model, TfidfVectorizer)
        assert isinstance(params, dict)
        assert isinstance(params['TFIDF'], dict)
        assert isinstance(params['TFIDF']['PARAMS'], dict)
        assert isinstance(params['TFIDF']['VOCAB_SIZE'], int)
        assert params['TFIDF']['PARAMS']['ngram_range'] == [1, 1]
        assert params['TFIDF']['PARAMS']['encoding'] == 'utf-8'
        assert params['TFIDF']['PARAMS']['stop_words'] == 'english'
        assert params['TFIDF']['VOCAB_SIZE'] == 24


class TestTrainingModel:
    @pytest.fixture(scope="class")
    def get_service_api(self):
        return f"{settings.api_base_url}/api/{settings.api_version}/{settings.service_type}"

    @pytest.fixture(scope="class")
    def get_config(self):
        return read_configs(file=os.path.join(os.getcwd(), "config.yml"))

    @pytest.mark.parametrize("is_d2v, output_len, exp_model", [(True, 3, Doc2VecModel),
                                                               (False, 2, TfidfVectorizer)]
                             )
    def test_build_model(self, is_d2v, output_len, exp_model, get_service_api, get_config):
        outcome = build_model(service_api=get_service_api, configs=get_config, d2v=is_d2v)
        assert len(outcome) == output_len
        assert isinstance(outcome[0], exp_model)
        assert isinstance(outcome[1], dict)
