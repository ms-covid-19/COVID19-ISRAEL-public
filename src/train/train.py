import pandas as pd
import numpy as np
import os

from datetime import datetime, timedelta
from src.utils.aggregations import adjacent_regions_mean
from src.train.constants import DATE_COL, PRED_COL, GT_COL
from src.train.train_config import kfold, model, save_map, upper_cut_date, lower_cut_date, agg_col, y_col_name,\
    x_train_date, y_train_date, x_test_date, y_test_date, x_agg_mode, y_agg_mode, model_features_list, \
    imputation_model, test, AGGREGATED_DIR
from src.train.train_utils import create_accumulative_df, get_all_data_df, clip_to_range
from src.utils.CONSTANTS import CITY_ID_COL, NEIGHBORHOOD_ID_COL
from src.utils.map_utils import create_map
from config import UNIFIED_FORMS_FILE, PROCESSED_DATA_DIR
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from copy import deepcopy


def get_cache_name(arg,lower_cut=lower_cut_date,
                   upper_cut=upper_cut_date, agg_col=agg_col,
                   y_col_name=y_col_name):

    if arg['mode'] == 'range':
        suffix = (agg_col + '_mode_' + arg['mode'] + '_n_days_' + str(arg['n_days']) + '_min_res_' + str(
        arg['min_per_region']) + '_lcd_' +
              lower_cut[-2:] + '_ucd_' + upper_cut[-2:])

    else:
        suffix = (agg_col + '_mode_' + arg['mode'] +
          '_min_res_' + str(arg['min_per_region']) + '_lcd_'
          + lower_cut[-2:] + '_ucd_' + upper_cut[-2:])

    suffix = suffix +'_' + y_col_name + '_' + '.csv'

    return  suffix

def get_model_mode(arg):
    if 'model_mode' in arg.keys():
        model_mode = arg['model_mode']
    else:
        model_mode = 'single_date'
    return model_mode


def upload_data(x_train_date, y_train_date, x_test_date, y_test_date, imputation_model, data_path, arg_x, arg_y,
                model_features_list=None, feature_function_x =create_accumulative_df,
                feature_function_y=create_accumulative_df, lower_cut=lower_cut_date, upper_cut=upper_cut_date,
                col_date=DATE_COL, agg_col=agg_col, y_col_name=y_col_name, test=False, cache=False):
    # todo add cache on/off
    full_df_x =None
    full_df_y =None

    model_mode = get_model_mode(arg_x)

    cache_name_x = get_cache_name(arg_x,lower_cut=lower_cut,
                   upper_cut=upper_cut, agg_col=agg_col,
                   y_col_name=y_col_name)
    cache_x_path = os.path.join(AGGREGATED_DIR, cache_name_x)
    if os.path.exists(cache_x_path):
        full_df_x = pd.read_csv(cache_x_path, index_col=[0,1], skipinitialspace=True)

    cache_name_y = get_cache_name(arg_y,lower_cut=lower_cut,
                   upper_cut=upper_cut, agg_col=agg_col,
                   y_col_name=y_col_name)
    cache_y_path = os.path.join(AGGREGATED_DIR, cache_name_y)
    if os.path.exists(cache_y_path):
        full_df_y = pd.read_csv(cache_y_path, index_col=[0,1], skipinitialspace=True)

    if (full_df_x is None) or (full_df_y is None):
        # get data
        all_data_df = get_all_data_df(data_path)
        # remove rows if needed
        all_data_df = clip_to_range(all_data_df, lower_cut=lower_cut, upper_cut=upper_cut, col=col_date)
        all_data_df = all_data_df[all_data_df[agg_col].notnull()]

    if  full_df_x is None:
        # aggregate data (allows different aggregation methods for each)
        full_df_x = feature_function_x(all_data_df, **arg_x)
        full_df_x.to_csv(cache_x_path)
    if full_df_y is None:
        full_df_y = feature_function_y(all_data_df, **arg_y)
        full_df_y.to_csv(cache_y_path)

    if not model_features_list:
        model_features_list = [item for item in full_df_x.columns.to_list()]
        print('Model features list is set to: ', model_features_list, '\n', len(model_features_list))

    if model_features_list and not set(model_features_list).issubset(set(full_df_x.columns.tolist())):
        print([item for item in model_features_list if item not in full_df_x.columns.tolist()])
        raise Exception("all_features not in passed dataframe columns")
    if y_col_name not in full_df_y.columns.tolist():
        print([item for item in model_features_list if item not in full_df_y.columns.tolist()])
        raise Exception("y_col_name not in passed dataframe columns")

    X_df, y_df = get_xy(full_df_x, full_df_y, x_train_date, y_train_date, model_features_list,  agg_col=agg_col,
                        imputation_model=imputation_model, y_col_name=y_col_name, mode=model_mode, arg_x=arg_x,
                        arg_y=arg_y, lower_cut =lower_cut)
    df_dict = {'X_df': X_df, 'y_df': y_df, 'pred_df': pd.DataFrame(), 'gt_df': pd.DataFrame()}

    if test:
        pred_df, gt_df = get_xy(full_df_x, full_df_y, x_test_date, y_test_date, model_features_list, agg_col=agg_col,
                                imputation_model=imputation_model, y_col_name=y_col_name, mode='single_date')
        df_dict['pred_df']=pred_df
        df_dict['gt_df'] = gt_df

    for key in df_dict.keys():
        print(f"shape of {key} is {df_dict[key].shape}")

    return df_dict, model_features_list, full_df_x, full_df_y


def get_xy(full_df_x, full_df_y, x_date, y_date, model_features_list, agg_col, imputation_model=False,
           y_col_name=y_col_name, mode='single_date', arg_x=None, arg_y=None, lower_cut=lower_cut_date):
    slicer = pd.IndexSlice

    if mode == 'single_date':
        valid_x = [item[0] for item in
                   list(full_df_x.loc[slicer[:, x_date], :].dropna(axis=0, subset=model_features_list).index.values)]
        valid_y = [item[0] for item in
                   list(full_df_y.loc[slicer[:, y_date], :].dropna(axis=0, subset=[y_col_name]).index.values)]

        agg_col_in_x_and_y = [item for item in valid_x if item in valid_y]

        X_df = full_df_x.loc[slicer[agg_col_in_x_and_y, x_date], :]
        y_df = full_df_y.loc[slicer[agg_col_in_x_and_y, y_date], :]
    elif mode == 'max_date':
        x_start = (datetime.strptime(lower_cut, '%Y-%m-%d') +
                                timedelta(days=arg_x['n_days'])).strftime("%Y-%m-%d")
        prediction_days = (datetime.strptime(y_date, '%Y-%m-%d') - datetime.strptime(x_date, '%Y-%m-%d')).days
        y_start = (datetime.strptime(x_start, '%Y-%m-%d') + timedelta(days=prediction_days)).strftime("%Y-%m-%d")
        X_df = full_df_x.loc[slicer[:, x_start:x_date], :]
        y_df = full_df_y.loc[slicer[:, y_start:y_date], :]

        valid_x = [item for item in list(X_df.dropna(axis=0, subset=model_features_list).index.values)]
        valid_y = [item for item in list(y_df.dropna(axis=0, subset=[y_col_name]).index.values)]
        agg_col_in_x, agg_col_in_y = [], []
        for item in valid_x:
            matching_y_date = (datetime.strptime(item[1], '%Y-%m-%d') +
                               timedelta(days=prediction_days)).strftime("%Y-%m-%d")
            if (item[0], matching_y_date) in valid_y:
                agg_col_in_x.append(item)
                agg_col_in_y.append((item[0], matching_y_date))

        X_df = X_df.loc[slicer[agg_col_in_x], :]
        y_df = y_df.loc[slicer[agg_col_in_y], :]

    if imputation_model:
        X_df = adjacent_regions_mean(X_df, model_features_list, agg_col)
        y_df = y_df.loc[slicer[X_df[agg_col], y_date], :]

    return X_df, y_df



def train(df_dict, agg_col, model_features_list, model, y_col_name, gt_col=GT_COL, pred_col=PRED_COL, save_map=False,
                test=False):


    X_df = df_dict['X_df']
    y_df = df_dict['y_df']

    # Specify current model features
    X_tmp = X_df[model_features_list].values
    y_tmp = y_df[[y_col_name]].values
    # todo if model features list is none we still get zip code etc
    y_true, y_pred = [], []

    # Cross validate for results evaluation
    for train_index, test_index in kfold.split(X_tmp):
        current_model = deepcopy(model)

        X_train_cv, X_val_cv, y_train_cv, y_val_cv = X_tmp[train_index], X_tmp[test_index], \
                                                     y_tmp[train_index], y_tmp[test_index]

        current_model.fit(X_train_cv, y_train_cv)

        current_score = current_model.score(X_val_cv, y_val_cv)

        print('Fold: ', current_score, spearmanr(np.squeeze(y_val_cv), np.squeeze(current_model.predict(X_val_cv))))

        y_pred.extend(current_model.predict(X_val_cv))
        y_true.extend(y_val_cv)

    current_model = deepcopy(model)

    current_model.fit(X_tmp, y_tmp)
    current_r2 = r2_score(y_true, y_pred)
    mse_train =  mean_squared_error(y_true, y_pred)
    df_dict['CV_r2'] = current_r2
    df_dict['train_mse'] =  mse_train
    print('Total CV: ', current_r2, spearmanr(y_true, y_pred),mse_train )

    # Evaluate on test
    if test:
        pred_df = df_dict['pred_df']
        gt_df = df_dict['gt_df']

        if pred_df.shape[0] == 0:
            Exception('cannot perform test no mutual cities in these date')
        # todo check if doesnt return nan's
        pred_df.index.names = [agg_col, DATE_COL] #todo check
        cities_ids = pred_df.index.get_level_values(agg_col)
        future_day = current_model.predict(pred_df.loc[:, model_features_list].values)
        pred_df = pd.DataFrame(np.squeeze(future_day), index=cities_ids, columns=[pred_col])

        test_vec = np.squeeze(gt_df.loc[:, y_col_name].values)
        gt_df = pd.DataFrame(test_vec, index=cities_ids, columns=[gt_col])
        gt_df.index.name = agg_col

        current_r2 = r2_score(test_vec, np.squeeze(future_day))
        spearman = spearmanr(test_vec, np.squeeze(future_day))
        mse_test =  mean_squared_error(test_vec, np.squeeze(future_day))


        print('Test: ', current_r2, spearman, mse_test)

        df_dict['pred_df'] = pred_df
        df_dict['gt_df'] = gt_df
        df_dict['test_r2'] = current_r2
        df_dict['spearman'] = spearman
        df_dict['test_mse'] =   mse_test

        if save_map:
            if agg_col == CITY_ID_COL:
                create_map(pred_df.reset_index(), pred_col, 'city_pred', city_level=True)
                create_map(gt_df.reset_index(), gt_col, 'city_true', city_level=True)
            elif agg_col == NEIGHBORHOOD_ID_COL:
                create_map(pred_df.reset_index(), pred_col, 'neighborhood_pred', city_level=False)
                create_map(gt_df.reset_index(), gt_col, 'neighborhood_true', city_level=False)

    return current_model, df_dict


def main(x_train_date, y_train_date, x_test_date, y_test_date, data_path, arg_x, arg_y,
         model_features_list, feature_function_x,feature_function_y, model, lower_cut=lower_cut_date,
         upper_cut=upper_cut_date, col_date=DATE_COL, agg_col=agg_col, y_col_name=y_col_name, imputation_model=False,
         gt_col=GT_COL,pred_col=PRED_COL, save_map=False, test=False, mode='single_date'):

    df_dict, model_features_list, full_df_x, full_df_y = \
        upload_data(x_train_date, y_train_date, x_test_date, y_test_date, imputation_model, data_path, arg_x, arg_y,
        model_features_list, feature_function_x, feature_function_y,lower_cut, upper_cut, col_date, agg_col, y_col_name,
                    test=test)

    current_model, df_dict = train(df_dict=df_dict, agg_col=agg_col, model_features_list=model_features_list,
                                   model=model, y_col_name=y_col_name, gt_col=gt_col, pred_col=pred_col,
                                   save_map=save_map, test=test)

    return current_model, df_dict, full_df_x, full_df_y, model_features_list


if __name__ == '__main__':
    current_model, df_dict, full_df_x, full_df_y, model_features_list = \
        main(x_train_date=x_train_date, y_train_date=y_train_date, x_test_date=x_test_date, y_test_date=y_test_date,
             data_path=UNIFIED_FORMS_FILE, arg_x=x_agg_mode, arg_y=y_agg_mode, model_features_list=model_features_list,
             feature_function_x=create_accumulative_df, feature_function_y=create_accumulative_df, model=model,
             lower_cut=lower_cut_date, upper_cut=upper_cut_date, col_date=DATE_COL, agg_col=agg_col,
             y_col_name=y_col_name, imputation_model=imputation_model, gt_col=GT_COL, pred_col=PRED_COL,
             save_map=save_map, test=test)
