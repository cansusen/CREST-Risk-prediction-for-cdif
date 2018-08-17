def run_clf(clf_name, X_train, y_train, X_test, costSensitive): # input should only be which classifier we want and data
    '''This function takes in training/testing data and a classifier name request, performs a cross-validated
    hyperparameter grid search and returns a prediction vector for testing data. It will be easy to add cost
    sensitivity to SVM's and any new classifiers at a later time.'''
    if clf_name == 'rbf':
        if costSensitive:
            param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [10,1,0.1,0.01,0.001, 0.0001, 0.00001]}
            ]
            clf_gs = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
            clf_gs.fit(X_train, y_train)
            clf_rbf = SVC(class_weight={0:clf_gs.best_params_.get('C'),1:1.5*clf_gs.best_params_.get('C')},
                          gamma=clf_gs.best_params_.get('gamma'), kernel = 'rbf', probability=True)
            clf_rbf.fit(X_train, y_train)
            probs_clf = clf_rbf.predict_proba(X_test)[:, 1]
        else:
            param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [10,1,0.1,0.01,0.001, 0.0001, 0.00001]}
            ]
            clf_gs = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
            clf_gs.fit(X_train, y_train)
            clf_rbf = SVC(C=clf_gs.best_params_.get('C'), gamma=clf_gs.best_params_.get('gamma'), kernel = 'rbf', probability=True) #class_weight='balanced'
            clf_rbf.fit(X_train, y_train)
            probs_clf = clf_rbf.predict_proba(X_test)[:, 1]
        params = tuple((clf_gs.best_params_.get('C'), clf_gs.best_params_.get('gamma')))
    if clf_name == 'lin':
        if costSensitive:
            param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
            ]
            clf_gs = GridSearchCV(LinearSVC(), param_grid, cv=5)
            clf_gs.fit(X_train, y_train)
            clf_lin = LinearSVC(class_weight={0:clf_gs.best_params_.get('C'),1:1.5*clf_gs.best_params_.get('C')})
            clf_lin.fit(X_train, y_train)
            probs_clf = clf_lin.decision_function(X_test)
        else:
            param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
            ]
            clf_gs = GridSearchCV(LinearSVC(), param_grid, cv=5)
            clf_gs.fit(X_train, y_train)
            clf_lin = LinearSVC(C=clf_gs.best_params_.get('C'))
            clf_lin.fit(X_train, y_train)
            probs_clf = clf_lin.decision_function(X_test)
        params = clf_gs.best_params_.get('C')
    if clf_name == 'poly':
        if costSensitive:
            param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [10,1,0.1,0.01,0.001, 0.0001, 0.00001]},
            ]
            clf_gs = GridSearchCV(SVC(kernel='poly'), param_grid, cv=5)
            clf_gs.fit(X_train, y_train)
            clf_poly = SVC(class_weight={0:clf_gs.best_params_.get('C'),1:1.5*clf_gs.best_params_.get('C')},
                           gamma=clf_gs.best_params_.get('gamma'), kernel = 'poly', probability=True)
            clf_poly.fit(X_train, y_train)
            probs_clf = clf_poly.predict_proba(X_test)[:, 1]
        else:
            param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [10,1,0.1,0.01,0.001, 0.0001, 0.00001]},
            ]
            clf_gs = GridSearchCV(SVC(kernel='poly'), param_grid, cv=5)
            clf_gs.fit(X_train, y_train)
            clf_poly = SVC(C=clf_gs.best_params_.get('C'), gamma=clf_gs.best_params_.get('gamma'), kernel = 'poly', probability=True)
            clf_poly.fit(X_train, y_train)
            probs_clf = clf_poly.predict_proba(X_test)[:, 1]
        params = tuple((clf_gs.best_params_.get('C'), clf_gs.best_params_.get('gamma')))
    if clf_name == 'rf':
        rfc = RandomForestClassifier(n_jobs = -1, max_features = 'sqrt', n_estimators=32, oob_score = True)
        param_grid = {
            'n_estimators': [50, 100, 200, 500, 1000],
            'max_features': ['sqrt', 'log2']
        }
        clf_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 5)
        clf_rfc.fit(X_train, y_train)
        clf_rf_1 = RandomForestClassifier(n_jobs=-1, n_estimators=clf_rfc.best_params_.get('n_estimators'), max_features=clf_rfc.best_params_.get('max_features'), oob_score = True)
        clf_rf_1.fit(X_train, y_train)
        probs_clf = clf_rfc.predict_proba(X_test)[:, 1]
        params = tuple((clf_rfc.best_params_.get('n_estimators'), clf_rfc.best_params_.get('max_features')))
    if clf_name == 'log':
        if costSensitive:
            param_grid = [
                {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            ]
            clf_gs = GridSearchCV(LogisticRegression(), param_grid, cv = 5)
            clf_gs.fit(X_train, y_train)
            clf_log = LogisticRegression(class_weight={0:clf_gs.best_params_.get('C'),1:1.5*clf_gs.best_params_.get('C')})
            clf_log.fit(X_train, y_train)
            probs_clf = clf_log.predict_proba(X_test)[: ,1]
        else:
            param_grid = [
                {'C': [0.001, 0.1, 1, 10, 1000], 'penalty':['l1','l2'] }
            ]
            clf_gs = GridSearchCV(LogisticRegression(), param_grid, cv = 5)
            clf_gs.fit(X_train, y_train)
            clf_log = LogisticRegression(C = clf_gs.best_params_.get('C'), penalty =clf_gs.best_params_.get('penalty'), tol=0.01)
            clf_log.fit(X_train, y_train)
            probs_clf = clf_log.predict_proba(X_test)[: ,1]
        params = tuple((clf_gs.best_params_.get('C'), clf_gs.best_params_.get('penalty')))
    if clf_name == 'nn':
        param_grid = [
            {'alpha': [1e-1,1e-3,1e-5,1e-7], 'hidden_layer_sizes':[(5, 2),(4, 2),(3, 2)] }
        ]
        MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        
        clf_gs = GridSearchCV(MLPClassifier(solver='lbfgs'), param_grid, cv = 5)
        clf_gs.fit(X_train, y_train)
        clf_log = MLPClassifier(solver='lbfgs', alpha = clf_gs.best_params_.get('alpha'), hidden_layer_sizes =clf_gs.best_params_.get('hidden_layer_sizes'),random_state=1)
        clf_log.fit(X_train, y_train)
        probs_clf = clf_log.predict_proba(X_test)[: ,1]
        params = tuple((clf_gs.best_params_.get('alpha'), clf_gs.best_params_.get('hidden_layer_sizes')))

    return(probs_clf, params) # returns a set of predicted probabilities/response values from the chosen classifier



