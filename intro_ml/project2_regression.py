import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from dtuimldmtools import rlr_validate

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


from intro_ml.load_data import get_data
from intro_ml.two_level_cross_validation import two_level_cross_val, Model, result_to_latex_table
from intro_ml.plot import plot_val_error_v_lambdas, plot_generalization_train_val_error_v_lambdas


def classification_error(y_predicted, y_test):
    return len(np.argwhere(y_predicted != y_test)) / len(y_test)

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# class linearRegression(Model):
#
#     def __init__(self, l):
#
#     def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
#         self.lr.fit(X_train, y_train)
#
#     def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
#         return classification_error(self.lr.predict(X_test), y_test)
#
#






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
        def __init__(self, l):
            """
            :param l (lambda): the regularization parameter
            """
            self.l = l

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


class ClassificationTree(Model):
    def __init__(self, d):
        self.d = d
        self.cf = DecisionTreeClassifier(criterion="gini", min_samples_split=d)

    def name(self) -> str:
        return "Classification tree"

    def param_name(self) -> str:
        return "d"  # TODO change this

    def param_value(self) -> str:
        return self.d

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.cf.fit(X_train, y_train)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return classification_error(self.cf.predict(X_test), y_test)


if __name__ == '__main__':
    wine_data = get_data()

    wine_data_np = wine_data.to_numpy()
    #winedata normalised
    winedata_normalised = normalize(wine_data_np)

    # Separate features (X) and target (y)
    target_column = wine_data.columns.get_loc("Color intensity")
    y = winedata_normalised[:, target_column]
    # Use all other columns for X
    X = np.delete(winedata_normalised, target_column, axis=1)

    # Add offset attribute
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    #definition of lambdas for regularization?
    lambdas = np.logspace(-7, 7, 14)  # Range from 0.0001 to 100

    # K-Fold CV and Generalization Error Estimation
    K = 10  # Number of folds
    kf = KFold(n_splits=K, shuffle=True, random_state=2)


    # Placeholders to store errors for each λ
    train_errors = []
    val_errors = []

    # Iterate over each λ value
    for lam in lambdas:
        fold_train_errors = []
        fold_val_errors = []

        # Perform K-fold cross-validation
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Apply Ridge Regression with the current λ as the regularization parameter
            model = Ridge(alpha=lam)
            model.fit(X_train, y_train)

            # Predict on the validation and training sets and compute MSE
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Calculate errors for the current fold
            fold_train_errors.append(mean_squared_error(y_train, y_train_pred))
            fold_val_errors.append(mean_squared_error(y_val, y_val_pred))

        # Average the MSE over all folds for the current λ
        avg_train_error = np.mean(fold_train_errors)
        avg_val_error = np.mean(fold_val_errors)

        # Store the errors
        train_errors.append(avg_train_error)
        val_errors.append(avg_val_error)

    # Plot the generalization error as a function of λ
    plot_val_error_v_lambdas(lambdas, val_errors)

    plot_generalization_train_val_error_v_lambdas(lambdas, val_errors, train_errors)