
#sudo docker build -t "afl_xgboot_tuner" .
#docker run -v /path/to/where/your/data/needs/to/save/on/host:/usr/src/app/data afl_xgboot_tuner

import numpy as np 
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.base import clone
from sklearn.utils import shuffle
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from afl_utils import *
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# Get the data and validation obkect
data_for_predict, val,_ = get_train_data()

# Get Level and features
X = normalize(data_for_predict[:,1:],axis = 0)
Y = data_for_predict[:,0].astype(int)

# Get cv index
cv = val.get_iterable()

#Determine the grid
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'learning_rate': [0.05,0.1,0.3], #so called `eta` value
              'max_depth': [3,6],
              'min_child_weight': [0,1,10],
              'colsample_bytree': [0.7,1],
              'n_estimators': [10]} #number of trees, change it to 1000 for better results}



# Generate an xgboost classifier
xgb_clf = XGBClassifier()

#Calculate the grid
grid = GridSearchCV(xgb_clf, 
       parameters,
			cv=cv, 
			scoring="accuracy",
			verbose = 2,
			n_jobs = -1, # -1 for complete parallelism
			return_train_score = True)

# Fit the grid
grid.fit(X,Y)

# Get the results
result = pd.DataFrame(grid.cv_results_)

# Print to the data file
result.to_csv('data/xg_boost_tuning_result.csv')

# Print the best result
best_parameters, score = grid.best_params_ , grid.best_score_ 

# Print the best accuracy
print('Accuracy:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

