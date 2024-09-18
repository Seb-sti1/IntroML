import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import svd

from intro_ml.load_data import get_data
from intro_ml.plot import histplot, corr_matrix, boxplot


def pca():
    wine_data_no_class = wine_data.drop("class", axis=1)
    wine_data_s = wine_data_no_class / wine_data_no_class.max()
    wine_data_m = wine_data_s - wine_data_s.mean()
    Y = wine_data_m.to_numpy()

    # PCA by computing SVD of Y
    U, S, V = svd(Y, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()

    threshold = 0.9

    # Plot variance explained
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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
