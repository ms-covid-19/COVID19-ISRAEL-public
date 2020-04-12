from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.individual_level_models.utils import *
from src.individual_level_models.defs import X_COLS, Y_COL, MAIN_DATASET

# Split the dataset in two equal parts
data_main = globals()['load_processed_{}'.format(MAIN_DATASET)]()
# X_COLS.remove('symptom_headache')
X = data_main[X_COLS]
y = data_main[Y_COL].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'learning_rate': [0.001, 0.002, 0.005, 0.01], 'n_estimators': [500, 1000, 2000],
                     'max_depth': [4, 5], 'subsample': [0.7, 0.8], 'colsample_bytree': [0.8],
                     'min_child_weight': [10, 25, 50]}]

print("# Tuning hyper-parameters for roc_auc\n")

clf = GridSearchCV(XGBClassifier(), tuned_parameters, scoring='roc_auc')
clf.fit(X_train, y_train)

print("Best parameters set found on development set:\n")
print(clf.best_params_)
print("Grid scores on development set:\n")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print("Detailed classification report:\n")
print("The model is trained on the full development set.\n")
print("The scores are computed on the full evaluation set.\n")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
