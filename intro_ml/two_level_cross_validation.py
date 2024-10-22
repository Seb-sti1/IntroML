from typing import Tuple

import numpy as np
from sklearn.model_selection import KFold


class Model:
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplemented("The Model.train function needs to be implemented")

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        raise NotImplemented("The Model.test function needs to be implemented")


def two_level_cross_val(X: np.ndarray, y: np.ndarray, K1: int, K2: int, models: list[Model]) \
        -> Tuple[float, np.ndarray]:
    """
    This function preforms a two level cross validation.

    This is an implementation of Algorithm 5 (See lecture 6 slide 52)
    :param X: the data
    :param y: the result (to predict)
    :param K1: the number of outer folds
    :param K2: the number of inner folds
    :param models: the models to compare
    :return: e_circumflex_gen (float),
             result (np.ndarray) of shape K1 x 2. It corresponds to a column for a method in the Table 2 of the project2.pdf.
               - first column is the index of the optimal model for the given outer fold
               - second column is the e_test_i : the test error of the optimal model
    """

    e_circumflex_gen = 0
    result = np.zeros((K1, 2))

    outer_fold = KFold(K1, shuffle=True)
    for i, (d_par, d_test) in enumerate(outer_fold.split(X, y)):
        X_par = X[d_par]
        y_par = y[d_par]
        X_test = X[d_test]
        y_test = y[d_test]

        e_circumflex_gen_s = np.zeros((len(models)))

        inner_fold = KFold(K2, shuffle=True)
        for d_train, d_val in inner_fold.split(X_par, y_par):
            X_train = X_par[d_train]
            y_train = y_par[d_train]
            X_val = X_par[d_val]
            y_val = y_par[d_val]

            for s, m in enumerate(models):
                m.train(X_train, y_train)
                e_circumflex_gen_s[s] += len(d_val) / len(d_par) * m.test(X_val, y_val)

        s_star = np.argmin(e_circumflex_gen_s)
        m_star = models[s_star]
        m_star.train(X_par, y_par)
        e_test_i = m_star.test(X_test, y_test)

        e_circumflex_gen += len(d_test) / len(X) * e_test_i
        result[i, 0] = s_star
        result[i, 1] = e_test_i

    return e_circumflex_gen, result
