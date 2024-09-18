from scipy.linalg import svd

from intro_ml.load_data import get_data
from intro_ml.plot import histplot, corr_matrix, boxplot, plot_pca_variance

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


