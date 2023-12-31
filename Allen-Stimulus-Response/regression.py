from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import numpy as np
import json

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

    #null_coefficients=null_space(regr.coef_)
    #print('null shp', null_coefficients.shape, X_test.shape, y_test.shape)
    #projection_onto_nullspace_train=null_coefficients.T@X_train.T
    #projection_onto_nullspace_test=null_coefficients.T@X_test.T
    #r2_null=r2_score(y_test.T, projection_onto_nullspace_test)
    #print(projection_onto_nullspace_test.shape)
    #ridge2=Ridge(100)
    #ridge2.fit(projection_onto_nullspace_train.T, y_train.T)

    #pred_nn=ridge2.coef_@projection_onto_nullspace_test
    #print('My prds', pred_nn.shape)

    scores=[]
    for i in range(0,y_test.shape[0]):
        #scores.append(r2_score(y_test.T[:,i], predictions[:,i]))
        #print(r2_score(y_test.T[:,i], predictions[:,i]))
        scores.append(r2_score(y_test.T[:,i], predictions[:,i]))
        print(r2_score(y_test.T[:,i], predictions[:,i]))
        plt.plot(y_test.T[:,i],label='test')
        plt.plot(predictions[:,i],label='pred')
        plt.title('Single trial, Variance explained: ' + str(r2_score(y_test.T[:,i], predictions[:,i])))
        plt.legend()
        plt.show()
    #print(len(scores))
    #scores=np.array(scores)
    #plt.hist(scores[scores>0.1])
    #plt.show()
    return scores

def make_visualizations(cell_ids, dat_dct):
    y_train, y_test, X_train, X_test= dat_dct['y_train'], dat_dct['y_test'], dat_dct['X_train'], dat_dct['X_test']

    regr=Ridge(10)

    # Fit the model with scaled training features and target variable
    regr.fit(X_train, y_train.T)

    # Make predictions on scaled test features
    predictions = regr.predict(X_test)

    scores={}
    for i in range(0,y_test.shape[0]):
        #scores.append(r2_score(y_test.T[:,i], predictions[:,i]))
        scores[i]=r2_score(y_test.T[:,i], predictions[:,i])
    #plt.hist(scores)
    #plt.show()

    # Save scores to a JSON file with an indent level of 2
    with open('scores.json', 'w') as json_file:
        json.dump(scores, json_file, indent=2)




    