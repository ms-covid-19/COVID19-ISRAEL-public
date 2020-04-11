import os
import numpy as np
import pandas as pd
import datetime
import shap
import sys
slicer = pd.IndexSlice
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

from config import OUT_DIR
from src.train.train import upload_data, train, create_accumulative_df
from src.train.experiments_list import *



MODELS = MODELS
N_iterations = N_iterations
N_splits = N_splits
results_dir = os.path.join(OUT_DIR,'prediction_results')

columns = ['agg_col', 'Y','features_list','model', 'CV_r2','test_r2', 'spearman', 'x_train_date',
                                'y_train_date','x_test_date', 'y_test_date',
                               'n_days_x', 'n_days_y','number_of_regions_train','number_of_regions_test',
                               'number_of_features',
                               'alpha', 'colsample_bytree', 'learning_rate', 'lower_cut_date', 'max_depth', 'min_child_weight',
                               'min_per_region_x', 'min_per_region_y', 'mode_x','mode_y',
                               'n_estimators',  'subsample', 'test_mse',
                                'train_mse', 'upper_cut_date' ]

results_df = pd.DataFrame(columns =columns)

dict_list = [dict_14]

start_time_of_experiment = str(datetime.datetime.utcnow())

only_zeros_columns = ['symptom_well', 'symptom_not_well', 'symptom_fatigue', 'symptom_infirmity',
                      'condition_asthma', 'ow_sr_symptom_well', 'ow_sr_symptom_not_well', 'ow_sr_symptom_fatigue',
                      'ow_sr_symptom_infirmity', 'ow_sr_condition_asthma', 'ow_sr_patient_location_recovered']
noisy_columns = ['body_temp', 'symptom_fever']
mostly_zero_columns = patient_loc_cols

features_to_drop = only_zeros_columns + noisy_columns + mostly_zero_columns


calc_shap = False
shap_df = pd.DataFrame()

for dates_setup in dict_list:
    for outcome in dates_setup['outcome']:

        y_col_name = outcome
        x_agg_mode = dates_setup['x_agg_mode']
        y_agg_mode = dates_setup['y_agg_mode']
        lower_cut_date = dates_setup['lower_cut_date']
        upper_cut_date = dates_setup['upper_cut_date']
        x_train_date = dates_setup['x_train_date']
        y_train_date = dates_setup['y_train_date']
        x_test_date = dates_setup['x_test_date']
        y_test_date = dates_setup['y_test_date']
        agg_col = dates_setup['agg_col']
        features_combinations = dates_setup['features_combinations']


        df_dict, model_features_list, full_df_x, full_df_y = \
            upload_data(x_train_date=x_train_date, y_train_date=y_train_date, x_test_date=x_test_date,
                        y_test_date=y_test_date, imputation_model=imputation_model, data_path=data_path,
                        arg_x=x_agg_mode, arg_y=y_agg_mode,
                        model_features_list=None,  # upload all features first
                        feature_function_x=create_accumulative_df, feature_function_y=create_accumulative_df,
                        lower_cut=lower_cut_date, upper_cut=upper_cut_date, col_date=DATE_COL, agg_col=agg_col,
                        y_col_name=y_col_name, test=test)

        _df_dict = df_dict.copy()

        for id_f, features_list in enumerate(features_combinations.keys()):

            model_features_list = features_combinations[features_list]

            for model_type in MODELS:

                if outcome not in patients_cols:
                    model_features_list = [feat for feat in model_features_list if feat not in patients_cols]
                if len(model_features_list) == 0:
                    print('No features left. Skipping experiment.')
                    continue

                model_features_list = [item for item in model_features_list if item not in features_to_drop]

                model_dict = dict.fromkeys(columns)
                model_dict['features_list'] = features_list
                model_dict['model'] = model_type[2]
                model_dict['Y'] = y_col_name
                model_dict['agg_col'] = agg_col
                model_dict['mode_x'] = x_agg_mode['mode']
                if x_agg_mode['mode'] == 'range':
                    model_dict['n_days_x'] = x_agg_mode['n_days']
                model_dict['min_per_region_y'] = y_agg_mode['min_per_region']
                model_dict['mode_y'] = y_agg_mode['mode']
                if y_agg_mode['mode'] == 'range':
                    model_dict['n_days_y'] = y_agg_mode['n_days']
                model_dict['min_per_region_y'] = y_agg_mode['min_per_region']
                model_dict['lower_cut_date'] = lower_cut_date
                model_dict['upper_cut_date'] = upper_cut_date
                model_dict['x_train_date'] = x_train_date
                model_dict['y_train_date'] = y_train_date
                model_dict['x_test_date'] = x_test_date
                model_dict['y_test_date'] = y_test_date

                if model_type[2] == 'Ridge':
                    model_dict['alpha'] = ridge_alpha

                model = deepcopy(model_type[0])
                df_dict = _df_dict.copy()

                X_df = df_dict['X_df']
                y_df = df_dict['y_df']

                model_dict['number_of_regions_train'] = X_df.shape[0]
                try:
                    model_dict['number_of_regions_test'] = df_dict[3].shape[0]
                except:
                    pass
                model_dict['number_of_features'] = len(model_features_list)

                if model_type[1] != {}:

                    X_tmp = df_dict['X_df'][model_features_list].values
                    y_tmp = df_dict['y_df'][[y_col_name]].values

                    rand_src = RandomizedSearchCV(model, model_type[1], n_iter=N_iterations, cv=N_splits, iid=True)
                    rand_src.fit(X_tmp, y_tmp)


                    print("Best: %f using %s" % (rand_src.best_score_, rand_src.best_params_))
                else:
                    params_dict = model_type[0].get_params()

                print(model_type[2])
                print(model_features_list)

                model = deepcopy(model_type[0])

                if model_type[1] != {}:
                    # print(rand_src.best_params_)
                    model.set_params(**rand_src.best_params_)
                    params_dict = rand_src.best_params_

                if model_type[2] == 'XGB':
                    model_dict['learning_rate'] = params_dict['learning_rate']
                    model_dict['n_estimators'] = params_dict['n_estimators']
                    model_dict['min_child_weight'] = params_dict['min_child_weight']
                    model_dict['subsample'] = params_dict['subsample']
                    model_dict['colsample_bytree'] = params_dict['colsample_bytree']
                    model_dict['max_depth'] = params_dict['max_depth']

                print(f"scores are for {outcome} outcome")

                current_model, df_dict = train(df_dict=df_dict, agg_col=agg_col,
                                               model_features_list=model_features_list, model=model,
                                               y_col_name=y_col_name, gt_col=GT_COL, pred_col=PRED_COL,
                                               save_map=save_map, test=test)

                X_df = df_dict['X_df']

                if (model_type[2] == 'XGB') and calc_shap:
                    explainer = shap.TreeExplainer(current_model)
                    shap_values = explainer.shap_values(X_df[model_features_list].values)

                    fig, ax = plt.subplots(1, 1)
                    shap.summary_plot(shap_values, features=X_df[model_features_list].values,
                                      feature_names=model_features_list, show=False, title=[y_col_name])
                    plt.savefig(os.path.join(OUT_DIR, 'figs', 'o-' + y_col_name + '-f-' + features_list + '.png'),
                                bbox_inches='tight')

                    fig, ax = plt.subplots(1, 1)
                    shap.summary_plot(shap_values, features=X_df[model_features_list].values,
                                      feature_names=model_features_list,
                                      plot_type='bar', show=False, title=[y_col_name])
                    plt.savefig(os.path.join(OUT_DIR, 'figs', 'bar-o-' + y_col_name + '-f-' + features_list + '.png'),
                                bbox_inches='tight')

                model_dict['CV_r2'] = df_dict['CV_r2']
                model_dict['test_r2'] = df_dict['test_r2']
                model_dict['test_mse'] = df_dict['test_mse']
                model_dict['train_mse'] = df_dict['train_mse']
                model_dict['spearman'] = df_dict['spearman']

                model_parameters = pd.DataFrame.from_dict(model_dict, orient='index').T

                results_df = pd.concat([results_df, model_parameters])
                results_df.to_csv(os.path.join(results_dir, '_03_03_20_patients_results_'+start_time_of_experiment+'.csv'))
