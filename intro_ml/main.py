import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from scipy.linalg import svd

from intro_ml.load_data import get_data
from intro_ml.plot import scatter_by_class, plot_pca_variance, histplot, boxplot, corr_matrix

if __name__ == '__main__':
    wine_data = get_data()

    # scatter_by_class(wine_data, "Alcohol", "Hue")
    corr_matrix(wine_data)
    boxplot(wine_data)
    histplot(wine_data)

    wine_data_no_class = wine_data.drop("class", axis=1)
    wine_data_std = wine_data_no_class.std(axis=0)
    wine_data_s = wine_data_no_class / wine_data_std
    wine_data_m = wine_data_s - wine_data_s.mean()
    Y = wine_data_m.to_numpy()
    U, S, Vh = svd(Y, full_matrices=False)

    plot_pca_variance(S)

    fig = plt.figure(figsize=(10, 8))
    sb.heatmap(Vh, yticklabels=wine_data_no_class.columns,
               xticklabels=[f"PCA {i + 1}" for i in range(Vh.shape[0])],
               cmap='Blues', annot=True, fmt='.2f')
    fig.tight_layout()
    plt.show()

    Y_t = np.matmul(Y, Vh)
    wine_data_t = pd.DataFrame(np.concatenate([wine_data["class"].to_numpy()[:, np.newaxis], Y_t], axis=1),
                               columns=["class"] + [f"PCA {i + 1}" for i in range(Vh.shape[0])])
    scatter_by_class(wine_data_t, "PCA 1", "PCA 2")
    scatter_by_class(wine_data_t, "PCA 1", "PCA 3")
    scatter_by_class(wine_data_t, "PCA 1", "PCA 4")
    scatter_by_class(wine_data_t, "PCA 1", "PCA 5")
    scatter_by_class(wine_data_t, "PCA 2", "PCA 3")
    scatter_by_class(wine_data_t, "PCA 2", "PCA 4")
    scatter_by_class(wine_data_t, "PCA 2", "PCA 5")
    scatter_by_class(wine_data_t, "PCA 3", "PCA 4")
    scatter_by_class(wine_data_t, "PCA 3", "PCA 5")
    scatter_by_class(wine_data_t, "PCA 4", "PCA 5")
