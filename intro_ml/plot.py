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
    ax = fig.add_subplot(111)
    normalized.plot.box(ax=ax)
    plt.xticks(rotation=90)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)  # Set y-axis label size
    plt.ylabel("Normalized value between 0 and 1")
    fig.tight_layout()
    plt.show()


def histplot(wine_data):
    normalized = wine_data / wine_data.max(axis=0)
    normalized = normalized.drop("class", axis=1)
    fig = plt.figure(figsize=(14, 8))
    axes = fig.subplots(nrows=4, ncols=4).flatten()

    for ax, col in zip(axes[:len(normalized.columns)], normalized.columns):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        sb.histplot(normalized[col], bins=25, ax=ax, kde=True, edgecolor=None)
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # Set x-axis label size
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)  # Set y-axis label size

    for ax in axes[len(normalized.columns):]:
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


def plot_Vh(Vh, columns):
    fig = plt.figure(figsize=(12, 8))
    g = sb.heatmap(Vh, yticklabels=columns,
                   xticklabels=[f"PCA {i + 1}" for i in range(Vh.shape[0])],
                   cmap='PRGn', annot=True, fmt='.2f',
                   vmin=-.85, vmax=.85)
    ax = g.axes
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.show()


def pairplot(data):
    g = sb.pairplot(data,
                    hue="class",
                    palette="Set1",
                    diag_kind="hist",
                    corner=True)
    for ax in g.axes.flat:
        if ax:
            ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # Set x-axis label size
            ax.set_ylabel(ax.get_ylabel(), fontsize=14)  # Set y-axis label size
    plt.show()


def bar_pca(N, attributeNames, V):
    pcs = range(N)
    legendStrs = ["PC" + str(e + 1) for e in pcs]
    bw = 0.2
    r = np.arange(1, len(attributeNames) + 1)
    fig = plt.figure(figsize=(10, 8))
    for i in pcs:
        plt.bar(r + i * bw, V[:, i], width=bw)
    plt.xticks(r + bw, attributeNames)
    plt.xlabel("Attributes")
    plt.xticks(rotation=90)
    plt.ylabel("Component coefficients")
    plt.legend(legendStrs)
    plt.grid()
    fig.tight_layout()
    plt.show()
