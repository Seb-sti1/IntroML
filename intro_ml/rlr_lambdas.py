import numpy as np


from intro_ml.project2_regression import standardise
from intro_ml.load_data import get_data
from intro_ml.plot import plot_val_error_v_lambdas, plot_generalization_train_val_error_v_lambdas, plot_weights_bar_chart

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


wine_data = get_data()

wine_data_np = wine_data.to_numpy()
#winedata normalised
winedata_normalised = standardise(wine_data_np)

# Separate features (X) and target (y)
target_column = wine_data.columns.get_loc("Color intensity")
y = winedata_normalised[:, target_column]
# Use all other columns for X
X = np.delete(winedata_normalised, target_column, axis=1)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

#definition of lambdas for regularization?
# lambdas = np.logspace(-1, 1, 16)  # Range from 0.0001 to 100
lambdas = np.logspace(-5, 9, 40)
# lambdas[0] = 1
# lambdas.sort()


# K-Fold CV and Generalization Error Estimation
K = 10  # Number of folds
kf = KFold(n_splits=K, shuffle=True, random_state=2)


# Placeholders to store errors for each λ
train_errors = []
val_errors = []

# Iterate over each λ value
for lam in lambdas:
    fold_train_errors = []
    fold_val_errors = []

    # Perform K-fold cross-validation
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Apply Ridge Regression with the current λ as the regularization parameter
        model = Ridge(alpha=lam)
        model.fit(X_train, y_train)

        # Predict on the validation and training sets and compute MSE
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Calculate errors for the current fold
        fold_train_errors.append(mean_squared_error(y_train, y_train_pred))
        fold_val_errors.append(mean_squared_error(y_val, y_val_pred))

    # Average the MSE over all folds for the current λ
    avg_train_error = np.mean(fold_train_errors)
    avg_val_error = np.mean(fold_val_errors)

    # Store the errors
    train_errors.append(avg_train_error)
    val_errors.append(avg_val_error)


optimal_lambda_index = np.argmin(val_errors)
optimal_lambda = lambdas[optimal_lambda_index]
print(f"Optimal lambda: {optimal_lambda}")

optimal_model = Ridge(alpha=1.1659)
optimal_model.fit(X, y)
coef_weights = optimal_model.coef_
bias_term = optimal_model.intercept_

coef_weights = [bias_term] + list(coef_weights)
attribute_names = ["Bias"] + list(wine_data.columns)



#Plot the generalization error as a function of λ + the weights bar chart
plot_val_error_v_lambdas(lambdas, val_errors)

plot_generalization_train_val_error_v_lambdas(lambdas, val_errors, train_errors)

plot_weights_bar_chart(coef_weights, attribute_names, f"Weights of the optimal model (\u03BB=1.1659) in Ridge Regression")