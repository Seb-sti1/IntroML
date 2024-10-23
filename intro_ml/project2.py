import numpy as np
from sklearn.linear_model import LogisticRegression

from intro_ml.load_data import get_data
from intro_ml.two_level_cross_validation import two_level_cross_val, Model, result_to_latex_table


def classification_error(y_predicted, y_test):
    return len(np.argwhere(y_predicted != y_test)) / len(y_test)


class Baseline(Model):
    """
    This implements the baseline model for the classification part

    See project2.pdf page 6:
    > The baseline will be a model which compute the largest class on the training
    > data, and predict everything in the test-data as belonging to that class (corresponding
    > to the optimal prediction by a logistic regression model with a bias
    > term and no features).
    """

    def __init__(self):
        self.predicted_class = None

    def name(self) -> str:
        return "Baseline"

    def param_name(self) -> str:
        return ""

    def param_value(self) -> str:
        return ""

    def train(self, _: np.ndarray, y_train: np.ndarray) -> None:
        possible_classes = np.unique(y_train)
        count = [len(np.argwhere(y_train == possible_class)) for possible_class in possible_classes]
        self.predicted_class = possible_classes[np.argmax(count)]

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return classification_error(self.predicted_class * np.ones((len(X_test))),
                                    y_test)


class MultinomialRegression(Model):
    def __init__(self, l):
        """
        :param l: the regularization parameter
        """
        self.l = l
        self.lr = LogisticRegression(C=1 / l, tol=0.1)

    def name(self) -> str:
        return "Multinomial Regression"

    def param_name(self) -> str:
        return "\\lambda"

    def param_value(self) -> str:
        return str(self.l)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.lr.fit(X_train, y_train)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return classification_error(self.lr.predict(X_test), y_test)


if __name__ == '__main__':
    wine_data = get_data()
    wine_data_np = wine_data.to_numpy()
    X, y = wine_data_np[:, 1:], wine_data_np[:, 0]

    models_mr = [MultinomialRegression(l) for l in [1, 2, 3, 4, 5]]
    models_bl = [Baseline()]
    models = [models_mr, models_bl]
    e_circumflex_gen_list, result = two_level_cross_val(X, y, 5, 5, models)

    result_to_latex_table("Comparison of Multinomial Regression and Baseline",
                          models, e_circumflex_gen_list, result)
