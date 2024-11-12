import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from intro_ml.compare_models import setup_ii, result_to_latex
from intro_ml.load_data import get_data
from intro_ml.two_level_cross_validation import two_level_cross_val, Model, result_to_latex_table


def classification_error(y_predicted, y_test):
    return len(np.argwhere(y_predicted != y_test)) / len(y_test)


def standardise(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


class BaselineRegression(Model):
    """
    This implements the baseline model for the regression part

    See project2.pdf page 4:
    > The baseline will be a model which compute the mean of the training data, and predict
    > everything in the test-data as the mean (corresponding to the optimal prediction by a linear
    > regression model with a bias term and no features).
    """

    def __init__(self):
        pass

    def name(self) -> str:
        return "Baseline"

    def param_name(self) -> str:
        return ""

    def param_value(self) -> str:
        return ""

    def train(self, _: np.ndarray, y_train: np.ndarray) -> None:
        self.predicted_mean = np.mean(y_train)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return mean_squared_error(self.predicted_mean * np.ones((len(X_test))), y_test)


class RidgeRegression(Model):
    def __init__(self, l):
        """
        :param l: the regularization parameter
        """
        self.l = l
        self.regularised_linear_regression = Ridge(alpha=l)

    def name(self) -> str:
        return "Ridge regression"

    def param_name(self) -> str:
        return "\\lambda"

    def param_value(self) -> str:
        return str(self.l)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.regularised_linear_regression.fit(X_train, y_train)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return mean_squared_error(self.regularised_linear_regression.predict(X_test), y_test)


class ANN(Model):
    def __init__(self, l):
        """
        :param l: the regularization parameter
        """
        self.l = l
        self.ann = MLPRegressor(hidden_layer_sizes=(l,), max_iter=1000, batch_size=15)

    def name(self) -> str:
        return "Artificial Neural Network"

    def param_name(self) -> str:
        return "\\lambda"

    def param_value(self) -> str:
        return str(self.l)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.ann.fit(X_train, y_train)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return mean_squared_error(self.ann.predict(X_test), y_test)


if __name__ == '__main__':
    wine_data = get_data()

    wine_data_np = wine_data.to_numpy()
    # winedata normalised
    winedata_normalised = standardise(wine_data_np)

    # Separate features (X) and target (y)
    target_column = wine_data.columns.get_loc("Color intensity")
    y = winedata_normalised[:, target_column]
    # Use all other columns for X
    X = np.delete(winedata_normalised, target_column, axis=1)

    # Add offset attribute
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    # definition of lambdas for regularization?
    lambdas = np.logspace(-5, 9, 20)  # Range from 0.0001 to 100

    hidden_layer_sizes = range(1, 20)  # Range from 1 to 20

    models_rlr = [RidgeRegression(l) for l in lambdas]
    models_ann = [ANN(h) for h in hidden_layer_sizes]
    models_baseline = [BaselineRegression()]
    models = [models_rlr, models_ann, models_baseline]
    e_circumflex_gen_list, result = two_level_cross_val(X, y, 10, 10, models)

    result_to_latex_table("Comparison of linear Ridge regression, ANN and baseline regression", models,
                          e_circumflex_gen_list, result)

    a = 0.95
    best_lambda = 2.5
    best_h = 3
    result_to_latex(*setup_ii(X, y, ANN(best_h), RidgeRegression(best_lambda), alpha=a, J=50))
    result_to_latex(*setup_ii(X, y, BaselineRegression(), RidgeRegression(best_lambda), alpha=a, J=50))
    result_to_latex(*setup_ii(X, y, BaselineRegression(), ANN(best_h), alpha=a, J=50))
