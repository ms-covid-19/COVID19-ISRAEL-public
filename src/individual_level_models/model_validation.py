import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from src.individual_level_models.utils import *
from src.individual_level_models.clf_eval_utils import quickplot_eval_3
from src.individual_level_models.main_model import create_model_from_dataset


def predict(model_func, train_data_name, validation_data_name, **kwargs):
    model_name = str(model_func).split("'")[1].replace('.', '_')

    train_data = globals()['load_processed_{}'.format(train_data_name)]()
    validation_data = globals()['load_processed_{}'.format(validation_data_name)]()

    if model_func in NEED_IMPUTATION_MODELS:
        train_data.loc[:, X_COLS] = get_imputed_data(train_data, X_COLS)
        validation_data.loc[:, X_COLS] = get_imputed_data(validation_data, X_COLS)

    x_train, y_train = train_data[X_COLS], train_data[Y_COL].values.ravel()
    x_val, y_val = validation_data[X_COLS], validation_data[Y_COL]

    fit_model, _ = create_model_from_dataset(model_func, x_train, y_train, model_name, save=False)
    y_pred = fit_model.predict(x_val)
    quickplot_eval_3(y_val, y_pred, 'main_{}_val_{}_{}'.format(train_data_name, validation_data_name, model_name))


if __name__ == '__main__':
    BEST_PARAMS_XGB_MAIN_SET = globals()['BEST_PARAMS_XGB_{}'.format(MAIN_DATASET)]
    for validation_set in VALIDATION_SETS:
        predict(XGBClassifier, MAIN_DATASET, validation_set, **BEST_PARAMS_XGB_MAIN_SET)
        predict(LogisticRegression, MAIN_DATASET, validation_set)




