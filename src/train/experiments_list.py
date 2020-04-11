import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
from src.train.train_config import *
from src.utils.CONSTANTS import PATIENT_COL

###### Models ######


ridge_alpha = 0.1
MODELS = [  # (object, search params, name)
    #(Ridge(alpha=ridge_alpha), {}, 'Ridge'),
    #(xgb.XGBRegressor(objective="reg:squarederror", max_depth=2, n_estimators=800, learning_rate=0.01), {}, 'XGB'),
    (xgb.XGBRegressor(objective="reg:squarederror"), {
        'learning_rate': np.arange(0.001, 0.005, 0.001),
        'n_estimators': np.arange(1000, 2500, 300),
        'min_child_weight': np.arange(3, 5, 1),
        'subsample': np.arange(0.5, 0.8, 0.05),
        'colsample_bytree': np.arange(0.4, 0.7, 0.05),
        'max_depth': np.arange(2, 5, 1)
    }, 'XGB'),

]

N_iterations = 15
N_splits = 3

###### features #######

## all baseline features
baseline_features = lamas_cols + city_places + patients_cols

## all baseline features with symptoms
baseline_and_symptoms = baseline_features + gender_col + age_col + symptoms_cols

## all baseline features with all questionere
baseline_and_questionere = baseline_and_symptoms + conditions_cols + smoking_cols \
                           + isolation_cols + patient_loc_cols + SR_cols

all_features = gender_col + age_col +symptoms_cols + conditions_cols \
               +smoking_cols + isolation_cols + patient_loc_cols + \
               lamas_cols+ SR_cols + age_groups_cols + only_with_sr_avgs \
               +groups_totals + city_places+ patients_cols

SR_demo_and_lamas = gender_col + age_col + lamas_cols+ SR_cols + city_places + patients_cols

baseline_no_patients = lamas_cols + city_places
baseline_no_patients_and_SR = baseline_no_patients + gender_col + age_col + SR_cols
baseline_and_symptoms = baseline_features + gender_col + age_col + symptoms_cols
#baseline_and_questionere = baseline_and_symptoms + conditions_cols + smoking_cols \
#                           + isolation_cols

features_combinations_v0 = {'baseline_features':baseline_features, 'baseline_and_symptoms': baseline_and_symptoms,
                         'baseline_and_questionere': baseline_and_questionere,
                         'all_features': all_features,
                         'SR_demo_and_lamas':SR_demo_and_lamas}

features_combinations_v2 = {'baseline_no_patients':baseline_no_patients,
                            'baseline_no_patients_and_SR':baseline_no_patients_and_SR}

lamas_and_city_places = city_places + lamas_cols

questionere = gender_col + age_col + symptoms_cols + conditions_cols \
               + smoking_cols + isolation_cols + patient_loc_cols + SR_cols

lamas_city_places_and_questionere = lamas_and_city_places + questionere
confirmed_cases_list = ['confirmed_cases', 'norm_confirmed_cases']

features_combinations_v3 = {'confirmed_cases': confirmed_cases_list,
                            'questionere': questionere,
                            'lamas_and_city_places': lamas_and_city_places,
                            'lamas_city_places_and_questionere': lamas_city_places_and_questionere,
                            }

questionere_enriched = questionere + age_groups_cols  + groups_totals
features_combinations_v4 = {'SRt':['SRt']}

features_combinations_v5 = {'questionere':questionere}
features_combinations_v6 = {'questionere_enriched':questionere_enriched}


######## Experiments ###########

dict_1 = {"lower_cut_date":'2020-03-24',"upper_cut_date":'2020-04-01',
            "x_train_date":'2020-03-27',"y_train_date" :'2020-03-29',
              "x_test_date":'2020-03-29',"y_test_date":'2020-03-31',
         'x_agg_mode':{'mode': 'range', 'n_days': 4, 'min_per_region': 50},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':['SRs','SRt', PATIENT_COL],
         'features_combinations':features_combinations_v2,
          'agg_col':agg_col}

dict_2 = {"lower_cut_date":'2020-03-22',"upper_cut_date":'2020-04-01',
            "x_train_date":'2020-03-26',"y_train_date" :'2020-03-28',
              "x_test_date":'2020-03-28',"y_test_date":'2020-03-31',
         'x_agg_mode':{'mode': 'range', 'n_days': 4, 'min_per_region': 50},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':['SRs','SRt',PATIENT_COL],
         'features_combinations':features_combinations_v2,
          'agg_col':agg_col}

dict_3 = {"lower_cut_date":'2020-03-21',"upper_cut_date":'2020-04-05',
            "x_train_date":'2020-03-26',"y_train_date" :'2020-03-30',
              "x_test_date":'2020-03-30',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 4, 'min_per_region': 50},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':[PATIENT_COL],
         'features_combinations':features_combinations_v0,
          'agg_col':agg_col}

dict_4 = {"lower_cut_date":'2020-03-17',"upper_cut_date":'2020-04-05',
            "x_train_date":'2020-03-20',"y_train_date" :'2020-03-27',
              "x_test_date":'2020-03-27',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 4, 'min_per_region': 50},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':[PATIENT_COL],
         'features_combinations':features_combinations_v0,
          'agg_col':agg_col}

dict_5 = {"lower_cut_date":'2020-03-21',"upper_cut_date":'2020-04-05',
            "x_train_date":'2020-03-26',"y_train_date" :'2020-03-30',
              "x_test_date":'2020-03-30',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 50},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':[PATIENT_COL],
         'features_combinations':features_combinations_v0,
          'agg_col':agg_col}

dict_6 = {"lower_cut_date":'2020-03-17',"upper_cut_date":'2020-04-05',
            "x_train_date":'2020-03-20',"y_train_date" :'2020-03-27',
              "x_test_date":'2020-03-27',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 50},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':[PATIENT_COL],
         'features_combinations':features_combinations_v0,
          'agg_col':agg_col}

dict_7 = {"lower_cut_date":'2020-03-24',"upper_cut_date":'2020-04-01',
            "x_train_date":'2020-03-27',"y_train_date" :'2020-03-29',
              "x_test_date":'2020-03-29',"y_test_date":'2020-03-31',
         'x_agg_mode':{'mode': 'range', 'n_days': 4, 'min_per_region': 30},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':['SRs','SRt'],
         'features_combinations':features_combinations_v0,
          'agg_col':NEIGHBORHOOD_ID_COL}

dict_8 = {"lower_cut_date":'2020-03-22',"upper_cut_date":'2020-04-01',
            "x_train_date":'2020-03-26',"y_train_date" :'2020-03-28',
              "x_test_date":'2020-03-28',"y_test_date":'2020-03-31',
         'x_agg_mode':{'mode': 'range', 'n_days': 4, 'min_per_region': 30},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':['SRs','SRt'],
         'features_combinations':features_combinations_v0,
          'agg_col':NEIGHBORHOOD_ID_COL}

dict_9 = {"lower_cut_date":'2020-03-20',"upper_cut_date":'2020-04-04',
            "x_train_date":'2020-03-27',"y_train_date" :'2020-03-29',
              "x_test_date":'2020-03-29',"y_test_date":'2020-03-31',
         'x_agg_mode':{'mode': 'range', 'n_days': 4, 'min_per_region': 50},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':['norm_confirmed_cases', PATIENT_COL],
         'features_combinations':features_combinations_v3,
          'agg_col':agg_col}

dict_10 = {"lower_cut_date":'2020-03-18',"upper_cut_date":'2020-04-04',
            "x_train_date":'2020-03-28',"y_train_date" :'2020-04-03',
              "x_test_date":'2020-03-28',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 50},
         'y_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 15},
         'outcome':['norm_confirmed_cases', PATIENT_COL],
         'features_combinations':features_combinations_v3,
          'agg_col':agg_col,
           }

dict_11 = {
    "lower_cut_date":'2020-03-25',"upper_cut_date":'2020-04-01',
    "x_train_date":'2020-03-27',"y_train_date" :'2020-03-29',
    "x_test_date":'2020-03-29',"y_test_date":'2020-03-31',
    'x_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 50},
    'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
    'outcome': ['SRt', 'SRs'],
    'features_combinations': features_combinations_v3,
    'agg_col':agg_col,
           }

dict_12 = {"lower_cut_date":'2020-03-20',"upper_cut_date":'2020-04-04',
            "x_train_date":'2020-03-28',"y_train_date" :'2020-03-31',
              "x_test_date":'2020-03-28',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 20, 'model_mode': 'max_date'},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':['SRs'],
         'features_combinations':features_combinations_v4,
          'agg_col':agg_col,
           }

dict_13 = {"lower_cut_date":'2020-03-14',"upper_cut_date":'2020-04-04',
            "x_train_date":'2020-03-28',"y_train_date" :'2020-04-03',
              "x_test_date":'2020-03-28',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 20},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':['norm_confirmed_cases'],
         'features_combinations':features_combinations_v4,
          'agg_col':agg_col,
           }

training_params = { 'nfold' :3, "num_round":10, "N_iterations" : 5}

dict_14 = {"lower_cut_date":'2020-03-14',"upper_cut_date":'2020-04-04',
            "x_train_date":'2020-03-22',"y_train_date" :'2020-03-28',
              "x_test_date":'2020-03-28',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 20, 'model_mode': 'max_date'},
         'y_agg_mode':{'mode': 'range', 'n_days': 2, 'min_per_region': 15},
         'outcome':['norm_confirmed_cases'],
         'features_combinations':features_combinations_v5,
          'agg_col':agg_col,
        'training_params':training_params
           }

dict_15 = {"lower_cut_date":'2020-03-14',"upper_cut_date":'2020-04-04',
            "x_train_date":'2020-03-24',"y_train_date" :'2020-03-28',
              "x_test_date":'2020-03-30',"y_test_date":'2020-04-03',
         'x_agg_mode':{'mode': 'range', 'n_days': 4, 'min_per_region': 20, 'model_mode': 'max_date'},
         'y_agg_mode':{'mode': 'range', 'n_days': 3, 'min_per_region': 15},
         'outcome':['norm_confirmed_cases'],
         'features_combinations':features_combinations_v6,
          'agg_col':agg_col,
        'training_params':training_params
           }