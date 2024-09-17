import numpy as np
from matplotlib import pyplot as plt


def scatter_by_class(wine_data, x_axis_column, y_axis_column, group_column="class"):
    groups = wine_data.groupby(group_column)

    fig, ax = plt.subplots()
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group[x_axis_column],
                group[y_axis_column],
                marker='o', linestyle='', ms=12, label=name)
    plt.xlabel(x_axis_column)
    plt.ylabel(y_axis_column)
    ax.legend()

    plt.show()


def corr_matrix(wine_data):
    # TODO add number on the plot
    corr = wine_data.corr()
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111)
    caxes = axes.matshow(corr, )
    figure.colorbar(caxes)
    plt.xticks(rotation=90)
    axes.set_xticks(np.arange(len(wine_data.columns)))
    axes.set_yticks(np.arange(len(wine_data.columns)))
    axes.set_xticklabels(list(wine_data.columns))
    axes.set_yticklabels(list(wine_data.columns))
    figure.tight_layout()
    plt.show()


def boxplot(wine_data):
    normalized = wine_data / wine_data.max(axis=0)
    normalized = normalized.drop("class", axis=1)

    fig = plt.figure(figsize=(10, 8))
    axes = fig.add_subplot(111)
    normalized.plot.box(ax=axes)
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.show()


def histplot(wine_data):
    # TODO add lines
    wine_data_no_class = wine_data.drop("class", axis=1)
    wine_data_no_class.hist(bins=30, figsize=(10, 10), layout=(5, 3))
    plt.tight_layout()
    plt.show()
