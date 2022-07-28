import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(training_data: pd.Series, configs: dict) -> tuple:
    params = configs["TFIDF"]["PARAMS"]
    model = TfidfVectorizer(**params)
    model.fit(training_data)
    vocab_count = len(model.get_feature_names_out())
    configs["TFIDF"]["VOCAB_SIZE"] = vocab_count
    return model, configs


"""confi = {'TFIDF': {'PARAMS': {'encoding': 'utf-8', 'lowercase': True,
                              'ngram_range': [1, 1], 'stop_words': 'english'
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

mod, par = build_tfidf(training_data=train_series, configs=confi)

print(type(mod))
print(par)
"""