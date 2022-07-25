import pandas as pd


def merge_features(x: pd.Series, cols: str, is_careers: bool, ) -> str:
    if is_careers:
        if x.isnull()[1]:
            return str(x[cols[2]])
        elif x.isnull()[2]:
            return str(x[cols[1]])
        else:
            return ", ".join([x[cols[1]], x[cols[2]]])
    else:
        pass


def feature_engineer(data: pd.DataFrame, cols: list) -> pd.Series:
    data = data.dropna(how="all", subset=cols)
    data = data.reset_index(0, drop=True)
    data = data[cols].astype(str)
    data["features"] = data[cols].agg(" ".join, axis=1)
    return data["features"]

"""import numpy as np
print(feature_engineer(data=pd.DataFrame.from_dict({"X": ["y", "z", "K p"],
                                                    "Y": [1, 2, 12],
                                                    "Z": ["a", "b", np.nan]
                                                    }),
                       cols=['X', 'Y']))"""
