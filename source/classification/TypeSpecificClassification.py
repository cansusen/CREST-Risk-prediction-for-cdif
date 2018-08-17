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