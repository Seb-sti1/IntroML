import os

import pandas as pd
from ucimlrepo import fetch_ucirepo

column_correction = {"Alcalinity_of_ash": "Alkalinity of ash",
                     "Total_phenols": "Total phenols",
                     "Flavanoids": "Flavonoids",
                     "Nonflavanoid_phenols": "Non flavonoid phenols",
                     "Color_intensity": "Color intensity",
                     "0D280_0D315_of_diluted_wines": "0D280 0D315 of diluted wines"}


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
        df = pd.read_csv("wine.csv")
        df = df.rename(columns=column_correction)
        df.to_csv("wine.csv", index=False)
        return df
    else:
        X, y = fetch_data()
        data = pd.concat([y, X], axis=1)
        data = data.rename(columns=column_correction)
        data.to_csv("wine.csv", index=False)
        return data


if __name__ == '__main__':
    fetch_data(True)
