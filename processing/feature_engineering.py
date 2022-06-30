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
    data = data.dropna(how="all")
    data = data.reset_index(0, drop=True)
    data["features"] = data[cols].agg(", ".join, axis=1)
    return data["features"]
