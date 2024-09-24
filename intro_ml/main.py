import numpy as np
import pandas as pd
from scipy.linalg import svd

from intro_ml.load_data import get_data
from intro_ml.plot import plot_pca_variance, histplot, boxplot, corr_matrix, plot_Vh, pairplot

if __name__ == '__main__':
    wine_data = get_data()

    corr_matrix(wine_data)
    boxplot(wine_data)
    histplot(wine_data)

    # PCA computation
    wine_data_no_class = wine_data.drop("class", axis=1)
    wine_data_std = wine_data_no_class.std(axis=0)
    wine_data_s = wine_data_no_class / wine_data_std
    wine_data_m = wine_data_s - wine_data_s.mean()
    Y = wine_data_m.to_numpy()
    U, S, Vh = svd(Y, full_matrices=False)

    plot_pca_variance(S)
    plot_Vh(Vh, wine_data_no_class.columns)

    # project the data
    Y_t = np.matmul(Y, Vh)
    wine_data_t = pd.DataFrame(np.concatenate([wine_data["class"].to_numpy()[:, np.newaxis], Y_t], axis=1),
                               columns=["class"] + [f"PCA {i + 1}" for i in range(Vh.shape[0])])

    N = 5  # the number of PCA in the plot
    pairplot(wine_data_t[["class"] + [f"PCA {i + 1}" for i in range(N)]])
