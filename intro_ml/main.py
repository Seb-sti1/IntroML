from intro_ml.load_data import get_data
from intro_ml.plot import scatter_by_class

if __name__ == '__main__':
    wine_data = get_data()
    print(wine_data)

    scatter_by_class(wine_data, "Alcohol", "Hue")
