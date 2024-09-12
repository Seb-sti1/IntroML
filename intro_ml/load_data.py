import os

import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_data(print_info=False):
    """
    :return: features, targets as a pandas dataframe
    """
    wine = fetch_ucirepo(id=109)

    X = wine.data.features
    y = wine.data.targets

    if print_info:
        print(wine.metadata)
        print(wine.variables)
    return X, y


def get_data():
    """
    :return: the data as a pandas dataframe
    """
    if os.path.exists("wine.csv"):
        return pd.read_csv("wine.csv")
    else:
        X, y = fetch_data()

        data = pd.concat([y, X], axis=1)
        data.to_csv("wine.csv", index=False)
        return data


if __name__ == '__main__':
    fetch_data(True)
