import typing as tp

import numpy as np
from scipy.stats import t
from sklearn.model_selection import KFold

from intro_ml.two_level_cross_validation import Model


def setup_ii(X: np.ndarray, y: np.ndarray,
             model_a: Model, model_b: Model,
             K: int = 5, J: int = 200,
             alpha=0.95) -> tp.Tuple[float, float, float, float]:
    """
    Method described in box 11.4.1

    :param X: data
    :param y: the ouput
    :param model_a: the model A
    :param model_b: the model B
    :param K: the number of fold, the pdf indicates to keep K low such that the test set is greater than 30 items
    :param J:
    :param alpha: the confidence for the CI
    :return: r_circumflex, z_l, z_u, p
    """
    r_list = np.zeros(J)

    for j in range(0, J, K):
        fold = KFold(K, shuffle=True)
        for k, (d_train, d_test) in enumerate(fold.split(X, y)):
            X_train = X[d_train]
            y_train = y[d_train]
            X_test = X[d_test]
            y_test = y[d_test]

            # train the models
            model_a.train(X_train, y_train)
            model_b.train(X_train, y_train)

            # predict
            error_a = model_a.test(X_test, y_test)
            error_b = model_b.test(X_test, y_test)

            r_list[j + k] = error_a - error_b

    r_circumflex = 1 / J * r_list.sum()
    s_circumflex_squared = (r_list - r_circumflex).T @ (r_list - r_circumflex) / (J - 1)
    rho = 1 / K
    sigma_tilde = np.sqrt((1 / J + rho / (1 - rho)) * s_circumflex_squared)

    z_l = t.ppf(alpha / 2, J - 1, loc=r_circumflex, scale=sigma_tilde)
    z_u = t.ppf(1 - alpha / 2, J - 1, loc=r_circumflex, scale=sigma_tilde)

    sigma_circumflex = sigma_tilde / np.sqrt((1 / J + 1 / (K - 1)))
    t_circumflex = r_circumflex / sigma_circumflex / np.sqrt((1 / J + rho / (1 - rho)))

    p = 2 * t.cdf(- abs(t_circumflex), J - 1, loc=0, scale=1)

    return r_circumflex, z_l, z_u, p


def result_to_latex(r_circumflex: float, z_l: float, z_u: float, p: float):
    print("$\hat{r} = %s$ with a CI [$%s$, $%s$]"
          " and a $p=%s$" % ("{:.3e}".format(r_circumflex).replace("e", r" \times 10^{") + "}",
                             "{:.3e}".format(z_l).replace("e", r" \times 10^{") + "}",
                             "{:.3e}".format(z_u).replace("e", r" \times 10^{") + "}",
                             "{:.6e}".format(p).replace("e", r" \times 10^{") + "}"))
