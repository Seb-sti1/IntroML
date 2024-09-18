import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


def scatter_by_class(wine_data, x_axis_column, y_axis_column, group_column="class"):
    groups = wine_data.groupby(group_column)

    colors = ['green', 'pink', 'blue']

    fig, ax = plt.subplots()
    ax.margins(0.05)
    for i, (name, group) in enumerate(groups):
        ax.plot(group[x_axis_column],
                group[y_axis_column],
                color=colors[i],
                marker='o', linestyle='', ms=12, label=name)
    plt.xlabel(x_axis_column)
    plt.ylabel(y_axis_column)
    ax.legend()

    plt.show()


def corr_matrix(wine_data):
    data = wine_data.drop("class", axis=1)
    corr = data.corr()
    fig = plt.figure(figsize=(10, 8))
    sb.heatmap(corr, cmap='Blues', annot=True, fmt='.1f')
    fig.tight_layout()
    plt.show()


def boxplot(wine_data):
    normalized = wine_data / wine_data.max(axis=0)
    normalized = normalized.drop("class", axis=1)

    fig = plt.figure(figsize=(10, 8))
    axes = fig.add_subplot(111)
    normalized.plot.box(ax=axes)
    plt.xticks(rotation=90)
    plt.ylabel("Normalized value between 0 and 1")
    fig.tight_layout()
    plt.show()


def histplot(wine_data):
    data = wine_data.drop("class", axis=1)
    fig = plt.figure(figsize=(8, 10))
    axes = fig.subplots(nrows=7, ncols=2).flatten()

    for ax, col in zip(axes[:len(data.columns)], data.columns):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        sb.histplot(data[col], bins=25, ax=ax, kde=True, edgecolor=None)

    for ax in axes[len(data.columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_pca_variance(S):
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
    # plt.title("Variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()
    plt.show()
