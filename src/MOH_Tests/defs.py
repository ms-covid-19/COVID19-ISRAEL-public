import os
import numpy as np

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUT_DIR
from src.utils.unify_forms import *

# Preprocess
MOH_T_FOLDER_RAW = os.path.join(RAW_DATA_DIR, 'MOH_Tests')
MOH_T_FOLDER_PROCESSED = os.path.join(PROCESSED_DATA_DIR, 'MOH_Tests')
MOH_T_RAW_FILE = os.path.join(MOH_T_FOLDER_RAW, 'corona_tested_individuals_ver.xlsx')
MOH_T_PROCESSED_FILE = os.path.join(MOH_T_FOLDER_PROCESSED, 'corona_tested_individuals.csv')
MOH_T_OUT = os.path.join(OUT_DIR, 'MOH_Tests')

COLS_DICT = {
    'cough': enum_to_column(Symptom.COUGH),
    'fever': enum_to_column(Symptom.FEVER),
    'sore_throat': enum_to_column(Symptom.SORE_THROAT),
    'shortness_of_breath': enum_to_column(Symptom.SHORTNESS_OF_BREATH),
    'head_ache': enum_to_column(Symptom.HEADACHE),
    'age_60_and_above': 'age_over_60'
}

GENDER_DICT = {'נקבה': 0, 'זכר': 1}
AGE_DICT = {'No': 0, 'Yes': 1}
CORONA_RES_DICT = {'אחר': np.nan, 'שלילי': 0, 'חיובי': 1}

# models
SAVE_MODELS = True
MODELS_DIR = os.path.join(MOH_T_OUT, 'models')

X_cols = [enum_to_column(Symptom.COUGH), enum_to_column(Symptom.FEVER), enum_to_column(Symptom.SORE_THROAT),
          enum_to_column(Symptom.SHORTNESS_OF_BREATH), enum_to_column(Symptom.HEADACHE), 'age_over_60', 'gender']
Y_col = ['corona_result']
