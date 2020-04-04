import os
import pandas as pd
import numpy as np

from src.utils.aggregations import questionnaire_summary, aggregate_regions, filter_regions, adjacent_regions_mean
from src.utils.lamas_features import create_city_features, create_neighborhood_features
from src.train.constants import DATE_COL, TIME_COL, PRED_COL, GT_COL
from src.train.train_config import train_date, label_date, test_date, lower_cut_date, agg_col, y_col_name, data_file, \
    MINIMUM_PER_REGION, model_features_list, kfold, model, save_map, upper_cut_date, neighborhood_type, city_type
from src.utils.CONSTANTS import CITY_ID_COL, NEIGHBORHOOD_ID_COL, LAMAS_ID_COL
from config import LAMAS_DATA, LAMAS_NEIGHBERHOOD_DATA, LAMAS_CITIES_DATA
from src.utils.map_utils import create_map
from config import LAMAS_DATA
from src.utils.map_utils import create_map
from config import UNIFIED_FORMS_FILE

from sklearn.metrics import r2_score
from copy import deepcopy


def add_date_column(all_data_df):
    # todo update with current time column
    all_data_df[DATE_COL] = all_data_df[TIME_COL].apply(lambda x: x.split('T')[0])
    return all_data_df


def add_lamas_features(aggregated_df):
    agg_columns = aggregated_df.columns.to_list()

    if agg_col == CITY_ID_COL:
        if not os.path.exists(LAMAS_CITIES_DATA):
            create_city_features(LAMAS_DATA)
        lamsas_features_df = pd.read_csv(os.path.join(LAMAS_CITIES_DATA), index_col=0)
        aggregated_df = aggregated_df.merge(lamsas_features_df, left_on=agg_col,
                                            right_on=LAMAS_ID_COL).drop([LAMAS_ID_COL], axis=1)
        agg_columns = aggregated_df.columns.to_list()
    elif agg_col == NEIGHBORHOOD_ID_COL:
        if not os.path.exists(LAMAS_NEIGHBERHOOD_DATA):
            create_neighborhood_features(LAMAS_DATA)
        lamsas_features_df = pd.read_csv(os.path.join(LAMAS_NEIGHBERHOOD_DATA), index_col=0)
        aggregated_df = aggregated_df.merge(lamsas_features_df, left_on=agg_col,
                                            right_on=LAMAS_ID_COL).drop([LAMAS_ID_COL], axis=1)
        agg_columns = aggregated_df.columns.to_list()

    return agg_columns, aggregated_df


def create_accumulative_df(all_data_df, mode='upto', n_days=3):
    # possible modes: 'each', 'upto', ('range' with n_days)

    # todo use variables from function header and constants as default
    aggregation_dates = all_data_df[DATE_COL].unique()
    min_date = np.min(all_data_df[DATE_COL].unique())

    slicer = pd.IndexSlice

    tmp_aggregated_df = aggregate_regions(all_data_df[all_data_df[DATE_COL] <= min_date], region_id_col=agg_col,
                                          filter_region_func=filter_regions, minimum_questionnaires=MINIMUM_PER_REGION,
                                          aggregation_func=questionnaire_summary)

    agg_columns, tmp_aggregated_df = add_lamas_features(tmp_aggregated_df)
    agg_columns = tmp_aggregated_df.columns.to_list()

    index_names = pd.MultiIndex.from_product([sorted(all_data_df[agg_col].unique()),
                                              sorted(aggregation_dates)], names=[agg_col, DATE_COL])
    full_df = pd.DataFrame(index=index_names, columns=agg_columns)

    for id_ld, last_date in enumerate(sorted(aggregation_dates)):
        if mode == 'single':
            dates_included_list = [last_date]
        elif mode == 'upto':
            dates_included_list = [item for item in sorted(aggregation_dates) if item <= last_date]
        elif mode == 'range':
            dates_included_list = [item for item in sorted(aggregation_dates) if item <= last_date]
            if len(dates_included_list) > n_days:
                dates_included_list = dates_included_list[-1 * n_days:]
        else:
            raise ValueError

        tmp_data_df = all_data_df[all_data_df[DATE_COL].isin(dates_included_list)]
        tmp_aggregated_df = aggregate_regions(tmp_data_df, region_id_col=agg_col, filter_region_func=filter_regions,
                                              minimum_questionnaires=MINIMUM_PER_REGION,
                                              aggregation_func=questionnaire_summary)

        agg_columns, tmp_aggregated_df = add_lamas_features(tmp_aggregated_df)
        agg_columns = tmp_aggregated_df.columns.to_list()

        full_df.loc[slicer[tmp_aggregated_df[agg_col], last_date], agg_columns] = tmp_aggregated_df.values

    return full_df


def main(model_features_list=None, imputation_model=False, agg_col=agg_col):
    slicer = pd.IndexSlice

    all_data_df = pd.read_csv(UNIFIED_FORMS_FILE)

    # rows validity checks # todo move to amit's pipeline
    all_data_df = all_data_df[(all_data_df['age'].astype(float) > 0) & (all_data_df['age'].astype(float) < 100)]
    all_data_df['gender'] = (all_data_df['gender'] == 'M').astype(
        int)  # todo change column names to constants from Amit
    all_data_df = all_data_df[all_data_df[agg_col].notnull()]
    # todo change based on aggregated df too and get from amit's pipeline

    # add time step column - currently by date
    all_data_df = add_date_column(all_data_df)

    # remove dates from the past if needed
    if lower_cut_date is not None:
        all_data_df = all_data_df[all_data_df[DATE_COL] >= lower_cut_date]
    if upper_cut_date is not None:
        all_data_df = all_data_df[all_data_df[DATE_COL] < upper_cut_date]

    full_df = create_accumulative_df(all_data_df)

    if not model_features_list:
        model_features_list = ['age', 'gender'] + [item for item in full_df.columns.to_list()  # todo  'body_temp'
                                                   if ('symptom' in item) or ('condition' in item) or
                                                   ('smoking' in item) or ('isolation' in item) or (
                                                           'patient' in item) or ('lms' in item)]

    if model_features_list and not set(model_features_list).issubset(set(full_df.columns.tolist())):
        raise Exception("all_features not in passed dataframe columns")

    X_df = full_df.loc[slicer[:, train_date], :].dropna(axis=0)
    y_df = full_df.loc[slicer[X_df[agg_col], label_date], :].dropna(axis=0)
    X_df = X_df.loc[slicer[y_df[agg_col], train_date], :]

    if imputation_model:
        X_df = adjacent_regions_mean(X_df, model_features_list, agg_col)
        y_df = y_df.loc[slicer[X_df[agg_col], label_date], :].dropna(axis=0)

    y_pred = []
    y_true = []

    X_tmp = X_df[model_features_list].values
    y_tmp = y_df[[y_col_name]].values

    for train_index, test_index in kfold.split(X_tmp):
        current_model = deepcopy(model)

        X_train_cv, X_val_cv, y_train_cv, y_val_cv = X_tmp[train_index], X_tmp[test_index], \
                                                     y_tmp[train_index], y_tmp[test_index]

        current_model.fit(X_train_cv, y_train_cv)

        current_score = current_model.score(X_val_cv, y_val_cv)
        print(current_score)
        y_pred.extend(current_model.predict(X_val_cv))
        y_true.extend(y_val_cv)

    current_model = deepcopy(model)

    current_model.fit(X_tmp, y_tmp)

    current_r2 = r2_score(y_true, y_pred)
    print(current_r2)

    test_df = full_df.loc[slicer[X_df[agg_col], test_date], :].dropna(axis=0)

    future_day = current_model.predict(test_df[model_features_list].values)
    pred_df = pd.DataFrame(np.squeeze(future_day), index=list(test_df[agg_col].values),
                           columns=[PRED_COL])
    pred_df.index.name = agg_col

    test_vec = full_df.loc[slicer[test_df[agg_col], test_date], [y_col_name]].dropna(axis=0).values
    test_vec = np.squeeze(test_vec)
    gt_df = pd.DataFrame(test_vec, index=list(test_df[agg_col].values), columns=[GT_COL])
    gt_df.index.name = agg_col

    if save_map:
        if agg_col == CITY_ID_COL:
            create_map(pred_df.reset_index(), PRED_COL, 'city_pred', city_level=True)
            create_map(gt_df.reset_index(), GT_COL, 'city_true', city_level=True)
        elif agg_col == NEIGHBORHOOD_ID_COL:
            create_map(pred_df.reset_index(), PRED_COL, 'neighborhood_pred', city_level=False)
            create_map(gt_df.reset_index(), GT_COL, 'neighborhood_true', city_level=False)

    return current_model


if __name__ == '__main__':
    current_model = main(model_features_list, imputation_model=True, agg_col=agg_col)
