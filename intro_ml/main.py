import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import svd

from intro_ml.load_data import get_data
from intro_ml.plot import boxplot, histplot, corr_matrix


def pca():
    wine_data_no_class = wine_data.drop("class", axis=1)
    wine_data_np = wine_data_no_class.to_numpy()
    Y = wine_data_np - np.ones((wine_data_np.shape[0], 1)) * wine_data_np.mean(axis=0)

    # PCA by computing SVD of Y
    U, S, V = svd(Y, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()

    threshold = 0.9

    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, "x-")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
    plt.plot([1, len(rho)], [threshold, threshold], "k--")
    plt.title("Variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    wine_data = get_data()

    # scatter_by_class(wine_data, "Alcohol", "Hue")
    corr_matrix(wine_data)
    boxplot(wine_data)
    histplot(wine_data)
    pca()
