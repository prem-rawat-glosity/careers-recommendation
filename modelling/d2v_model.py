import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin


class Doc2VecModel(BaseEstimator, TransformerMixin):
    def __init__(self, dm=1, vector_size=1, window=1, epochs=5, **args):
        self.size = vector_size
        self.window = window
        self.dm = dm
        self.epochs = epochs
        self.model = Doc2Vec(vector_size=self.size, window=self.window,
                             dm=self.dm, epochs=self.epochs, **args)
        self.tagged_documents = None
        self.incrementer = -1  # for indexing each document

    def tag_example(self, raw_document):
        self.incrementer += 1
        return TaggedDocument(words=raw_document, tags=[self.incrementer])

    def fit(self, raw_documents):
        processed_documents = raw_documents.apply(simple_preprocess, )

        # Generate Tagged Examples
        self.tagged_documents = processed_documents.apply(self.tag_example, )

        # Build vocabulary
        self.model.build_vocab(corpus_iterable=self.tagged_documents)

        if len(self.model.wv.key_to_index) == 0:
            raise Exception("Either number of min_count is very less or no sample is present is input data.")

        # Train model
        self.model.train(corpus_iterable=self.tagged_documents,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)

    def score(self):
        ranks = []
        for tagged_document in self.tagged_documents:
            try:
                inferred_vector = self.model.infer_vector(self.tagged_documents[tagged_document.tags[0]].words)
            except IndexError:
                continue
            sims = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))
            rank = [docid for docid, sim in sims].index(tagged_document.tags[0])
            ranks.append(rank)
        counter_0 = ranks.count(0)
        return counter_0 / self.tagged_documents.shape[0]


def build_d2v(training_data: pd.Series, configs: dict):
    params = configs["D2V"]["PARAMS"]
    d2v = Doc2VecModel(**params)
    d2v.fit(training_data)
    accuracy_score = d2v.score()
    return d2v, params, accuracy_score


confi = {'D2V': {'PARAMS': {'alpha': 0.025, 'dbow_words': 1, 'dm': 1, 'epochs': 100,
                            'min_alpha': 0.001, 'min_count': 1,
                            'vector_size': 5, 'window': 1
                            }
                 }
         }

train_series = pd.Series(["graphic designer adobe illustrator adobe photoshop adobe",
                          "big data concepts logical thinking",
                          "analyst microsoft excel tableau",
                          "machine learning deep learning data science",
                          "react js javascript swift android developer"
                          ],
                         name='features'
                         )

model, par, score = build_d2v(training_data=train_series, configs=confi)

print(type(model))
print(par)
print(score)
