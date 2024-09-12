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

