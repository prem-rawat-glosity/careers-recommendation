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
        self.tagged_documents = list()

    def tag_examples(self, raw_documents):
        self.tagged_documents.clear()
        for index, tokens in raw_documents.iteritems():
            self.tagged_documents.append(TaggedDocument(words=tokens,
                                                        tags=[index]))

    def fit(self, raw_documents):
        processed_documents = raw_documents.apply(simple_preprocess,)
        # Generate Tagged Examples
        self.tag_examples(processed_documents)

        # Build vocabulary
        self.model.build_vocab(self.tagged_documents)

        # Train model
        self.model.train(self.tagged_documents,
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
        return counter_0/len(self.tagged_documents)


def build_d2v(training_data: pd.Series, configs: dict):
    params = configs["D2V"]["PARAMS"]
    d2v = Doc2VecModel(**params)
    d2v.fit(training_data)
    accuracy_score = d2v.score()
    return d2v, params, accuracy_score

"""

param_grid = {'window': [5],
              'dm': [1],
              'size': [50, 100, 200],
              'epochs': [100, 200, 500]
              }

# pipe_log = Pipeline([('doc2vec', Doc2VecModel())])
d2v_model = Doc2VecModel(size=100, dm=1, window=5, epochs=100,
                         shrink_windows=True, alpha=0.025, min_alpha=0.001,
                         dbow_words=1)

dataset = pd.read_excel("data.xlsx")
data = feature_engineer(dataset, cols=CAREER_COLUMNS)
training_data = generate_training_data(data)
print(training_data)
d2v_model.fit(training_data)
print(d2v_model.score())
"""