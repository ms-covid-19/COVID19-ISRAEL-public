import os
from sklearn.linear_model import LogisticRegression

from src.utils.unify_forms import *
from src.maoz_analysis.defs import MAOZ_PROCESSED_FILE
from src.M_analysis.defs import M_FILE_PROCESSED
from src.MOH_Tests.defs import MOH_T_PROCESSED_FILE
from config import UNIFIED_FORMS_FILE, OUT_DIR

# main model - MOH_TESTS
MAIN_DATASET = 'MOH_TESTS'
VALIDATION_SETS = ['MAOZ']
X_COLS = ['gender', 'age_over_60', enum_to_column(Symptom.COUGH), enum_to_column(Symptom.FEVER),
          enum_to_column(Symptom.SORE_THROAT), enum_to_column(Symptom.SHORTNESS_OF_BREATH),
          enum_to_column(Symptom.HEADACHE)]

# main model - FORMS_MOH_Tests
# MAIN_DATASET = 'FORMS_MOH_Tests'
# BEST_PARAMS_XGB_FORMS_MOH_Tests = {}

# X_COLS = ['gender', 'age_over_60', enum_to_column(Symptom.COUGH), enum_to_column(Symptom.FEVER),
#           enum_to_column(Symptom.SORE_THROAT), enum_to_column(Symptom.SHORTNESS_OF_BREATH),
#           enum_to_column(Symptom.HEADACHE)]

# VALIDATION_SETS = ['MAOZ']
# X_COLS = ['gender', 'age_over_60', enum_to_column(Symptom.COUGH), enum_to_column(Symptom.FEVER),
#           enum_to_column(Symptom.SORE_THROAT), enum_to_column(Symptom.SHORTNESS_OF_BREATH)]

# VALIDATION_SETS = ['MACCABI']
# X_COLS = [enum_to_column(Symptom.COUGH), enum_to_column(Symptom.FEVER), enum_to_column(Symptom.SORE_THROAT),
#           enum_to_column(Symptom.SHORTNESS_OF_BREATH), enum_to_column(Symptom.HEADACHE)]

# VALIDATION_SETS = ['MACCABI', 'MAOZ']
# X_COLS = [enum_to_column(Symptom.COUGH), enum_to_column(Symptom.FEVER), enum_to_column(Symptom.SORE_THROAT),
#           enum_to_column(Symptom.SHORTNESS_OF_BREATH)]

# model parameters
BEST_PARAMS_XGB_MOH_TESTS = {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 4, 'min_child_weight': 10,
                             'n_estimators': 1000, 'subsample': 0.8}

BEST_PARAMS_XGB_NO_HEADACHE_MOH_TESTS = {'colsample_bytree': 0.8, 'learning_rate': 0.001, 'max_depth': 4,
                                         'min_child_weight': 10, 'n_estimators': 1000, 'subsample': 0.7}

# main model - constants
MODELS_DIR = os.path.join(OUT_DIR, 'individual_level_models')
NEED_IMPUTATION_MODELS = [LogisticRegression]
SAVE_MODELS = True
Y_COL = 'label'

# validation models
PROCESSED_FILE_MAOZ = MAOZ_PROCESSED_FILE
PROCESSED_FILE_MACCABI = M_FILE_PROCESSED
PROCESSED_FILE_FORMS = UNIFIED_FORMS_FILE
PROCESSED_FILE_MOH_TESTS = MOH_T_PROCESSED_FILE

# moh load data
SYMPTOMATIC_PERC_OLD = 0.7
SYMPTOMATIC_PERC_YOUNG = 0.4

# symptoms columns
SYMPTOMS_MOH_TESTS = [enum_to_column(Symptom.COUGH), enum_to_column(Symptom.FEVER), enum_to_column(Symptom.SORE_THROAT),
                      enum_to_column(Symptom.SHORTNESS_OF_BREATH), enum_to_column(Symptom.HEADACHE)]

# individual defining columns
INDIVIDUAL_COLS_FORMS = \
    ['lat',
     'lng',
     'CITY_ID',

     'age',
     'gender',

     'condition_any',
     'condition_asthma',
     'condition_cancer',
     'condition_diabetes',
     'condition_hypertention',
     'condition_ischemic_heart_disease',
     'condition_kidney_disease',
     'condition_lung_disease',

     'smoking_currently',
     'smoking_never',
     'smoking_past',
     'smoking_past_less_than_five_years_ago',
     'smoking_past_more_than_five_years_ago',
]