from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import pickle

from src.individual_level_models.utils import *
from src.individual_level_models.clf_eval_utils import quickplot_eval_2, feature_importance_plot
from src.individual_level_models.shap_wrappers import plot_shap_summary, plot_shap_summary_ages


def create_model_from_dataset(model, X, y, model_name, dataset_name=MAIN_DATASET, save=True, **kwargs):
    clf = model(**kwargs)
    y_pred = cross_val_predict(clf, X, y, cv=3, method='predict_proba')[:, 1]
    clf.fit(X, y)
    if save:
        pickle.dump(clf, open(os.path.join(MODELS_DIR, 'model_{}_dataset_{}.sav'.format(model_name, dataset_name)), 'wb'))
    return clf, y_pred


if __name__ == '__main__':
    # load and create X and y
    data = globals()['load_processed_{}'.format(MAIN_DATASET)]()
    X = data[X_COLS]
    y = data[Y_COL].values.ravel()
    imputed_data = get_imputed_data(data)
    X_imp = imputed_data[X_COLS]

    # XGBoost
    fit_model, y_pred = create_model_from_dataset(XGBClassifier, X, y, 'XGBClassifier')
    quickplot_eval_2(y, y_pred, 'XGBClassifier')
    plot_shap_summary(fit_model, X, 'XGBClassifier')
    # plot_shap_summary_ages(fit_model, X[X.age_over_60 == 0], X[X.age_over_60 == 1])

    # LogisticRegression
    fit_model, y_pred = create_model_from_dataset(LogisticRegression, X_imp, y, 'LinearRegression')
    quickplot_eval_2(y, y_pred, 'LinearRegression')
    feature_importance_plot(fit_model, X_imp, 'LinearRegression')

    # LogisticRegression with interactions
    # X_imp_interact = add_interactions(X_imp, X_COLS, age_col='age_over_60', gender_col='gender')
    # fit_model, y_pred = create_model_from_dataset(LogisticRegression, X_imp_interact, y, 'LinearRegressionInteract')
    # quickplot_eval_2(y, y_pred, 'LinearRegressionInteract')
    # feature_importance_plot(fit_model, X_imp_interact, 'LinearRegressionInteract')
