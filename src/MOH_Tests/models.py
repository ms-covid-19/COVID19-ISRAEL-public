import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle

from src.MOH_Tests.defs import *
from src.MOH_Tests.clf_eval_utils import quickplot_eval_3, quickplot_eval_4


def load_data():
    return pd.read_csv(MOH_T_PROCESSED_FILE, index_col=0)


def get_imputed_data(df, x_cols):
    imp = SimpleImputer(strategy="most_frequent")
    df[x_cols] = imp.fit_transform(df[x_cols])
    return df


def create_model(model, X_train, y_train, save=False, **kwargs):
    clf = model(**kwargs)
    clf = clf.fit(X_train, y_train)
    if save:
        pickle.dump(clf, open(os.path.join(MODELS_DIR, 'model_{}.sav'.format(str(clf).split('(')[0])), 'wb'))
    return clf


if __name__ == '__main__':
    data = load_data()
    print(data.isna().sum())
    imputed_data = get_imputed_data(data, X_cols)
    X_train, X_test, y_train, y_test = train_test_split(data[X_cols], data[Y_col], test_size=0.3)
    # X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(imputed_data[X_cols], imputed_data[Y_col],
    #                                                                     test_size=0.3)

    # Decision Tree
    clf = create_model(DecisionTreeClassifier, X_train, y_train, save=SAVE_MODELS, min_samples_leaf=100)
    print('\nDecisionTreeClassifier Feature Importance')
    for rank in (-clf.feature_importances_).argsort():
        print('{} - {:.3}'.format(X_train.columns[rank], clf.feature_importances_[rank]))
    y_pred = clf.predict(X_test)
    quickplot_eval_3(y_test, y_pred, 'DecisionTree')
    quickplot_eval_4(y_test, y_pred, 'DecisionTree')


    # RF
    clf = create_model(RandomForestClassifier, X_train, y_train, save=SAVE_MODELS, min_samples_leaf=100)
    print('\nRandomForestClassifier Feature Importance')
    for rank in (-clf.feature_importances_).argsort():
        print('{} - {:.3}'.format(X_train.columns[rank], clf.feature_importances_[rank]))
    y_pred = clf.predict(X_test)
    quickplot_eval_3(y_test, y_pred, 'RF')
    quickplot_eval_4(y_test, y_pred, 'RF')

    # Boosted trees
    clf = create_model(GradientBoostingClassifier, X_train, y_train, save=SAVE_MODELS, min_samples_leaf=100)
    print('\nRandomForestClassifier Feature Importance')
    for rank in (-clf.feature_importances_).argsort():
        print('{} - {:.3}'.format(X_train.columns[rank], clf.feature_importances_[rank]))
    y_pred = clf.predict(X_test)
    quickplot_eval_3(y_test, y_pred, 'Boostedtrees')
    quickplot_eval_4(y_test, y_pred, 'Boostedtrees')

    # XGBoost
    clf = create_model(XGBClassifier, X_train, y_train, save=SAVE_MODELS)
    print('\nXGBClassifier Feature Importance')
    for rank in (-clf.feature_importances_).argsort():
        print('{} - {:.3}'.format(X_train.columns[rank], clf.feature_importances_[rank]))
    y_pred = clf.predict(X_test)
    quickplot_eval_3(y_test, y_pred, 'XGB')
    quickplot_eval_4(y_test, y_pred, 'XGB')

    # LinearRegression
    clf = create_model(LinearRegression, X_train, y_train, save=SAVE_MODELS)
    print('\nLinearRegression Coefficients')
    for rank in (-clf.coef_.ravel()).argsort():
        print('{} - {:.3}'.format(X_train.columns[rank], clf.coef_.ravel()[rank]))
    y_pred = clf.predict(X_test)
    quickplot_eval_3(y_test, y_pred, 'LinearRegression')
    quickplot_eval_4(y_test, y_pred, 'LinearRegression')

    # LogisticRegression
    clf = create_model(LogisticRegression, X_train, y_train, save=SAVE_MODELS)
    print('\nLogisticRegression Coefficients')
    for rank in (-clf.coef_[0]).argsort():
        print('{} - {:.3}'.format(X_train.columns[rank], clf.coef_[0][rank]))
    y_pred = clf.predict(X_test)
    quickplot_eval_3(y_test, y_pred, 'LogisticRegression')
    quickplot_eval_4(y_test, y_pred, 'LogisticRegression')
