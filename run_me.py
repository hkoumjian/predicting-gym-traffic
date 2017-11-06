# numpy to load the data
import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer

# Regression Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Hyperparemeter selection models
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Feature selection Model
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold

# Graphing tool
import matplotlib.pyplot as plt

# Load the Berkeley data
berkeley_data = np.load("berkeley.npy")

# Shuffle the data to prevent overfitting
np.random.shuffle(berkeley_data)

# Get the train and test data from the berkeley dataset and split it up 70/30
berkeley_train = berkeley_data[:int(berkeley_data.shape[0]*0.7)]
berkeley_test = berkeley_data[int(berkeley_data.shape[0]*0.7):]

# Split the berkeley up into training and testing data
train_x = berkeley_train[:, 0:berkeley_data.shape[1]-2]
train_y = berkeley_train[:, -1]


test_x = berkeley_test[:, 0:berkeley_test.shape[1]-2]
test_y = berkeley_test[:, -1]

# Umass data will be used for testing
umass_data = np.load("umass.npy")

umass_x = umass_data[:, 0:umass_data.shape[1]-2]
umass_y = umass_data[:, -1]

ridge = Ridge()
rf = RandomForestRegressor()
svr = SVR()

# Function for predicting a regressor on a set of test data
def pred(regressor, test):
    predictions = regressor.predict(test)
    return predictions

# Function for calculating the Root Mean Squared Error
def rmse(truth,predictions):
    return math.sqrt(mse(truth,predictions))

def pipeline(fs, hp, tx, ty):
    print "Training Feature Selected Model"
    # Train the feature selection model
    fs.fit(tx,ty)
    new_x = fs.fit_transform(tx,ty)
    print "Training Hyperparameter selector on new data"
    # Train the hyperparameter selection model
    hp.fit(new_x,ty)
    # Return the best model
    return hp.best_estimator_

# Parameters to give the GridSearchCV object

# Parameters for Ridge Regression
params_ridge = {
    'alpha': np.arange(0,1,0.001)
}
# Parameters for RandomForestRegressor
params_rf = {
    'n_estimators': range(1,101),
    'min_samples_split': [2,5,10,15,30]
}
# Parameters for SVR
params_svr = {
    'C': [1,10,25,50,100]
}
# Parameters to give the RFE Feature selector
rfe_params = {
    'step': 1,
    'cv': 3
}

# Create the scoring function (RMSE) to give to the Feature selector and hyperparameter selector
score_func = make_scorer(rmse,greater_is_better=False)

# Create the feature selection implementations
rfe = RFECV(estimator=ridge, scoring = score_func, step=rfe_params["step"], cv=rfe_params["cv"], verbose = 10, n_jobs=-1)
vt = VarianceThreshold(threshold = (0.999)*(0.001))

# Create the hyperparameter optimizer implementations
gs = GridSearchCV(estimator=ridge, param_grid=params_ridge, scoring=score_func, verbose = 10)
rs = RandomizedSearchCV(estimator=ridge, param_distributions=params_ridge, n_iter=100, scoring=score_func, n_jobs=-1, verbose=10)

# Run the pipeline to try to find the optimal features and hyperparameters
# Must specify in the code which model you want to run and which hyperparameter selection function and which feature selection function you want to run.
# Only runs one at a time.
optimal = pipeline(vt,gs,train_x,train_y)
optimal.fit(train_x,train_y)
predictions = optimal.predict(test_x)
print rmse(test_y,predictions)

ridge = Ridge()
rf = RandomForestRegressor()
svr = SVR()

# Run the regression models on the default hyperparameters
ridge.fit(train_x,train_y)
ridge_predictions =ridge.predict(test_x)
print rmse(test_y,ridge_predictions)

# Function for graphing hyperparameters against their RMSE's
def graph(reg, param_range, trainx, trainy, testx, testy):
    score_function = make_scorer(rmse, greater_is_better=False)
    x_vals = []
    y_vals = []

    params = {
        'n_estimators': param_range
    }

    gs = GridSearchCV(estimator=reg, param_grid=params, scoring=score_function)
    gs.fit(trainx,trainy)
    y_vals = np.abs(gs.cv_results_['mean_test_score'])
    x_vals = param_range

    print y_vals[0]

    plt.xlabel("N-Estimators")
    plt.ylabel("RMSE")
    plt.title("RMSE at Different N-Estimators")
    plt.plot(x_vals,y_vals,'-')
    plt.show()

graph(rf,range(1,101),train_x,train_y,test_x,test_y)
