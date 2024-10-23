from typing import Tuple

import numpy as np
from sklearn.model_selection import KFold


class Model:
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplemented("The Model.train function needs to be implemented")

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        raise NotImplemented("The Model.test function needs to be implemented")


def two_level_cross_val(X: np.ndarray, y: np.ndarray, K1: int, K2: int, models: list[list[Model]]) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    This function preforms a two level cross validation.

    For models with one element, this is an implementation of Algorithm 5 (See lecture 6 slide 52)
    :param X: the data
    :param y: the result (to predict)
    :param K1: the number of outer folds
    :param K2: the number of inner folds
    :param models: A list of "list of models to compare".
                    For instance, the first list would be a list of Multinomial Regression (with different params)
                    and the second list would be a list of KNN (with different k)
    :return: e_circumflex_gen_list (np.ndarray) for each element of models e_circumflex_gen,
             result (np.ndarray) of shape K1 x (2*len(models)). It corresponds to a Table 1/Table 2 of the project2.pdf.
             For each list of models to compare (ie each element of models):
               - first column is the index of the optimal model for the given outer fold
               - second column is the e_test_i : the test error of the optimal model
    """

    e_circumflex_gen_list = np.zeros((len(models)))
    result = np.zeros((K1, 2 * len(models)))

    outer_fold = KFold(K1, shuffle=True)
    for i, (d_par, d_test) in enumerate(outer_fold.split(X, y)):
        X_par = X[d_par]
        y_par = y[d_par]
        X_test = X[d_test]
        y_test = y[d_test]

        # for each list of models to compare, prepare computation the e_circumflex_gen_s
        e_circumflex_gen_s_list = [np.zeros((len(models_to_compare))) for models_to_compare in models]

        inner_fold = KFold(K2, shuffle=True)
        for d_train, d_val in inner_fold.split(X_par, y_par):
            X_train = X_par[d_train]
            y_train = y_par[d_train]
            X_val = X_par[d_val]
            y_val = y_par[d_val]

            # for each list of models to compare, compute the e_circumflex_gen_s
            for k, models_to_compare in enumerate(models):
                for s, m in enumerate(models_to_compare):
                    m.train(X_train, y_train)
                    e_circumflex_gen_s_list[k][s] += len(d_val) / len(d_par) * m.test(X_val, y_val)

        # for each list of models to compare, find the best one
        s_star_list = [np.argmin(e_circumflex_gen_s_list[k]) for k in range(len(models))]
        m_star_list = [models[k][s_star] for k, s_star in enumerate(s_star_list)]
        for m_star in m_star_list:
            m_star.train(X_par, y_par)
        e_test_i_list = [m_star.test(X_test, y_test) for m_star in m_star_list]

        for k, e_test_i in enumerate(e_test_i_list):
            e_circumflex_gen_list[k] += len(d_test) / len(X) * e_test_i
            result[i, 2 * k] = s_star_list[k]
            result[i, 2 * k + 1] = e_test_i

    return e_circumflex_gen_list, result
