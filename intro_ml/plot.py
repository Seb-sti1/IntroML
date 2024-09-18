import seaborn as sb
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
