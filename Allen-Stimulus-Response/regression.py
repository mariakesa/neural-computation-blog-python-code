from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

def ridge_regression(dat_dct):

    y_train, y_test, X_train, X_test= dat_dct['y_train'], dat_dct['y_test'], dat_dct['X_train'], dat_dct['X_test']

    regr=Ridge(10)

    # Fit the model with scaled training features and target variable
    regr.fit(X_train, y_train.T)

    # Make predictions on scaled test features
    predictions = regr.predict(X_test)

    # Calculate R-squared score
    r2 = r2_score(y_test.T, predictions)

    print("R-squared score:", r2)