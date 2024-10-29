import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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
        # lbfgs solver is compatible with multinomial multiclass
        # multi_class is deprecated and should be left at the default value
        # https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit
        self.lr = LogisticRegression(solver='lbfgs', C=1 / l)

    def name(self) -> str:
        return "Multinomial regression"

    def param_name(self) -> str:
        return "\\lambda"

    def param_value(self) -> str:
        return str(self.l)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.lr.fit(X_train, y_train)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return classification_error(self.lr.predict(X_test), y_test)


class KNN(Model):
    def __init__(self, k):
        """
        :param k: the controlling parameter
        """
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def name(self) -> str:
        return "$k$-nearest neighbor"

    def param_name(self) -> str:
        return "k"

    def param_value(self) -> str:
        return str(self.k)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.knn.fit(X_train, y_train)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return classification_error(self.knn.predict(X_test), y_test)


if __name__ == '__main__':
    wine_data = get_data()

    wine_data_np = wine_data.to_numpy()
    X, y = wine_data_np[:, 1:], wine_data_np[:, 0]
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    models_mr = [MultinomialRegression(10 ** l) for l in range(-6, 6)]
    models_knn = [KNN(k) for k in range(1, 16)]
    models_bl = [Baseline()]
    models = [models_mr, models_knn, models_bl]
    e_circumflex_gen_list, result = two_level_cross_val(X, y, 10, 10, models)

    result_to_latex_table("Comparison of multinomial regression, $k$-nearest neighbor and baseline",
                          models, e_circumflex_gen_list, result)
