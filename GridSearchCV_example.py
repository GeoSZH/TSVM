# This is an example for using GridSearchCV
# Note: We need to set X2 because GridSearchCV.fit only accept two parameters,
# but we can enter X2 as **fit_params.

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# Cross validation
kfold = StratifiedKFold(n_splits=10, random_state = 2022, shuffle=True)

# set the parameter grid
tsvm_param_grid = {'kernel': ['linear'],
                   'C': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                   'Cl': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]}

gsTSVM = GridSearchCV(tsvm, param_grid=tsvm_param_grid, scoring="accuracy", cv=kfold, n_jobs=-1, verbose=1)

gsTSVM.fit(np.array(X_train), np.array(y_train), X2=np.array(X_test_pred))

TSVM_best = gsTSVM.best_estimator_

print('The best parameters of model are:', gsTSVM.best_params_)
print('The best score：', gsTSVM.best_score_)