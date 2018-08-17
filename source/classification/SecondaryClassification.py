import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
import time
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#Import classification function


# Read feature files
X_train_s_fs = pd.read_csv('Data/X_train_s_fs_chopped.csv')
X_test_s_fs = pd.read_csv('Data/X_test_s_fs_chopped.csv')
X_train_d_fs = pd.read_csv('Data/X_train_d_fs_chopped.csv')
X_test_d_fs = pd.read_csv('Data/X_test_d_fs_chopped.csv')
X_train_t_fs = pd.read_csv('Data/X_train_t_fs_chopped.csv')
X_test_t_fs = pd.read_csv('Data/X_test_t_fs_chopped.csv')
y_train_s = np.load('Data/y_train_s.npy')



#Train Base Classifiers on the training data, make predictions on the test data
costSensitive = False
rbf_probs_s, rbf_params_s = run_clf('rbf', X_train_s_fs, y_train_s, X_test_s_fs, costSensitive)
lin_probs_s, lin_params_s = run_clf('lin', X_train_s_fs, y_train_s, X_test_s_fs, costSensitive)
rf_probs_s, rf_params_s = run_clf('rf', X_train_s_fs, y_train_s, X_test_s_fs, costSensitive)
log_probs_s, log_params_s = run_clf('log', X_train_s_fs, y_train_s, X_test_s_fs, costSensitive)

rbf_probs_d, rbf_params_d = run_clf('rbf', X_train_d_fs, y_train_s, X_test_d_fs, costSensitive)
lin_probs_d, lin_params_d = run_clf('lin', X_train_d_fs, y_train_s, X_test_d_fs, costSensitive)
rf_probs_d, rf_params_d = run_clf('rf', X_train_d_fs, y_train_s, X_test_d_fs, costSensitive)
log_probs_d, log_params_d = run_clf('log', X_train_d_fs, y_train_s, X_test_d_fs, costSensitive)

rbf_probs_t, rbf_params_t = run_clf('rbf', X_train_t_fs, y_train_s, X_test_t_fs, costSensitive)
lin_probs_t, lin_params_t = run_clf('lin', X_train_t_fs, y_train_s, X_test_t_fs, costSensitive)
rf_probs_t, rf_params_t = run_clf('rf', X_train_t_fs, y_train_s, X_test_t_fs, costSensitive)
log_probs_t, log_params_t = run_clf('log', X_train_t_fs, y_train_s, X_test_t_fs, costSensitive)

#Store predicted probabilities
test_probs = np.zeros((417, 12))

for index in range(0,417):
    test_probs[index,0] = rbf_probs_s[index]
    test_probs[index,1] = lin_probs_s[index]
    test_probs[index,2] = rf_probs_s[index]
    test_probs[index,3] = log_probs_s[index]
    test_probs[index,4] = rbf_probs_d[index]
    test_probs[index,5] = lin_probs_d[index]
    test_probs[index,6] = rf_probs_d[index]
    test_probs[index,7] = log_probs_d[index]
    test_probs[index,8] = rbf_probs_t[index]
    test_probs[index,9] = lin_probs_t[index]
    test_probs[index,10] = rf_probs_t[index]
    test_probs[index,11] = log_probs_t[index]

# Results for base classifiers
print ('AUCs for static(lin, rbf, rf), dynamic(lin, rbf, rf), temporal(lin, rbf, rf)')
print (roc_auc_score(y_test_s, lin_probs_s))
print (roc_auc_score(y_test_s, rbf_probs_s))
print (roc_auc_score(y_test_s, rf_probs_s))
print (roc_auc_score(y_test_s, log_probs_s))

print (roc_auc_score(y_test_s, lin_probs_d))
print (roc_auc_score(y_test_s, rbf_probs_d))
print (roc_auc_score(y_test_s, rf_probs_d))
print (roc_auc_score(y_test_s, log_probs_d))

print (roc_auc_score(y_test_s, lin_probs_t))
print (roc_auc_score(y_test_s, rbf_probs_t))
print (roc_auc_score(y_test_s, rf_probs_t))
print (roc_auc_score(y_test_s, log_probs_t))



# Also make predictions for the training set
X_train_s_fs = X_train_s_fs.as_matrix()
X_train_d_fs = X_train_d_fs.as_matrix()
X_train_t_fs = X_train_t_fs.as_matrix()

num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = True)
training_probs = np.zeros((972, 12))

for train, test in kf.split(np.arange(X_train_s_fs.shape[0])):
    CV_X_train_s_fs, CV_X_test_s_fs, CV_y_train_s, CV_y_test_s = X_train_s_fs[train,:], X_train_s_fs[test,:], y_train_s[train], y_train_s[test]
    CV_X_train_d_fs, CV_X_test_d_fs, CV_y_train_d, CV_y_test_d = X_train_d_fs[train,:], X_train_d_fs[test,:], y_train_s[train], y_train_s[test]
    CV_X_train_t_fs, CV_X_test_t_fs, CV_y_train_t, CV_y_test_t = X_train_t_fs[train,:], X_train_t_fs[test,:], y_train_s[train], y_train_s[test]
        
    rbf_probs_s, rbf_params_s = run_clf('rbf', CV_X_train_s_fs, CV_y_train_s, CV_X_test_s_fs, costSensitive)
    lin_probs_s, lin_params_s = run_clf('lin', CV_X_train_s_fs, CV_y_train_s, CV_X_test_s_fs, costSensitive)
    rf_probs_s, rf_params_s = run_clf('rf', CV_X_train_s_fs, CV_y_train_s, CV_X_test_s_fs, costSensitive)
    log_probs_s, log_params_s = run_clf('log', CV_X_train_s_fs, CV_y_train_s, CV_X_test_s_fs, costSensitive)

    rbf_probs_d, rbf_params_d = run_clf('rbf', CV_X_train_d_fs, CV_y_train_d, CV_X_test_d_fs, costSensitive)
    lin_probs_d, lin_params_d = run_clf('lin', CV_X_train_d_fs, CV_y_train_d, CV_X_test_d_fs, costSensitive)
    rf_probs_d, rf_params_d = run_clf('rf', CV_X_train_d_fs, CV_y_train_d, CV_X_test_d_fs, costSensitive)
    log_probs_d, log_params_d = run_clf('log', CV_X_train_d_fs, CV_y_train_d, CV_X_test_d_fs, costSensitive)

    rbf_probs_t, rbf_params_t = run_clf('rbf', CV_X_train_t_fs, CV_y_train_t, CV_X_test_t_fs, costSensitive)
    lin_probs_t, lin_params_t = run_clf('lin', CV_X_train_t_fs, CV_y_train_t, CV_X_test_t_fs, costSensitive)
    rf_probs_t, rf_params_t = run_clf('rf', CV_X_train_t_fs, CV_y_train_t, CV_X_test_t_fs, costSensitive)
    log_probs_t, log_params_t = run_clf('log', CV_X_train_t_fs, CV_y_train_t, CV_X_test_t_fs, costSensitive)
    
    for index,value in enumerate(test):
        training_probs[value,0] = rbf_probs_s[index]
        training_probs[value,1] = lin_probs_s[index]
        training_probs[value,2] = rf_probs_s[index]
        training_probs[value,3] = log_probs_s[index]
        training_probs[value,4] = rbf_probs_d[index]
        training_probs[value,5] = lin_probs_d[index]
        training_probs[value,6] = rf_probs_d[index]
        training_probs[value,7] = log_probs_d[index]
        training_probs[value,8] = rbf_probs_t[index]
        training_probs[value,9] = lin_probs_t[index]
        training_probs[value,10] = rf_probs_t[index]
        training_probs[value,11] = log_probs_t[index]


# Generate X_meta and y_meta
Xw_train = training_probs
Xw_test = test_probs
yw_train = y_train_s 
yw_test = y_test_s 


# Training and testing of Meta Classifiers
rbf_probs_s, rbf_params_s = run_clf('rbf', Xw_train, yw_train, Xw_test, costSensitive)
lin_probs_s, lin_params_s = run_clf('lin', Xw_train, yw_train, Xw_test, costSensitive)
rf_probs_s, rf_params_s = run_clf('rf', Xw_train, yw_train, Xw_test, costSensitive)
log_probs_s, log_params_s = run_clf('log', Xw_train, yw_train, Xw_test, costSensitive)

print (roc_auc_score(yw_test, lin_probs_s))
print (roc_auc_score(yw_test, rbf_probs_s))
print (roc_auc_score(yw_test, rf_probs_s))
print (roc_auc_score(yw_test, log_probs_s))


