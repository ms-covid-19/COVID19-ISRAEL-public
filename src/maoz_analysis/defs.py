import os

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUT_DIR

# Preprocess
MAOZ_FOLDER_RAW = os.path.join(RAW_DATA_DIR, 'maoz')
MAOZ_FOLDER_PROCESSED = os.path.join(PROCESSED_DATA_DIR, 'maoz')
MAOZ_PROCESSED_FILE = os.path.join(MAOZ_FOLDER_PROCESSED, 'maoz_all_processed.csv')
MAOZ_PROCESSED_FILE_EXCEL = os.path.join(MAOZ_FOLDER_PROCESSED, 'maoz_all_processed.xlsx')
MAOZ_OUT = os.path.join(OUT_DIR, 'maoz')

COLUMNS_DICT = {
    'sex': 'gender',
    'gil': 'age',
    'var4': 'isolation_diagnosed',
    'var5': 'isolation_isolated',
    'var6': 'symptom_well',
    'var10': 'symptom_sore_throat',
    'var11': 'symptom_cough',
    'var12': 'symptom_shortness_of_breath',
    'var13': 'symptom_smell_or_taste_loss',
    'var14': 'body_temp_measured',
    'var15': 'condition_any',
    'var16': 'patient_location_home'
}

CITY_DICT_HB_EN = {'בני ברק': 'Bene-Brak', 'בית שמש': 'Beit-Shemesh', 'אום אל פאחם': 'Um El-Fahem', 'סחנין': 'Sakhnin',
                   'עמנואל': 'Immanu\'el', 'חצור הגלילית': 'Hatzor HaGlilit', 'אלעד': 'El\'ad', 'צפת': 'Zefat',
                   'רכסים': 'Rekhasim', 'דבוריה': 'Daburiyya', 'שפרעם': 'Shfar\'am'}

CITY_DICT_EN_HB = dict((v, k) for k, v in CITY_DICT_HB_EN.items())

BINARY_COLS = ['isolation_diagnosed', 'isolation_isolated', 'symptom_well', 'symptom_sore_throat', 'symptom_cough',
               'symptom_shortness_of_breath', 'condition_any', 'symptom_smell_or_taste_loss',
               'patient_location_home', 'symptom_fever']

# TABLE1
TABLE_COLS = ['gender', 'age', 'isolation_diagnosed', 'isolation_isolated', 'symptom_well', 'symptom_sore_throat',
              'symptom_cough', 'symptom_shortness_of_breath', 'symptom_loss_of_taste_and_smell', 'condition_any',
              'symptom_fever', 'one_or_more_symptoms', 'symptom_ratio_weighted']

TABLE_LABELS = ['Male', 'Age', 'Diagnosed', 'Isolation', 'Feeling Well', 'Sore throat', 'Cough', 'Shortness of breath',
                'Loss of taste & smell', 'Prior medical conditions', 'Fever (above 38)', 'Symptoms >= 1',
                'Symptoms Ratio']

TABEL_COLS_TYPE = ['count_perc', 'continuous'] + ['count_perc'] * 10 + ['continuous']

AGE_GROUPS_DICT = {1: '< 20', 2: '21-30', 3: '31-40', 4: '41-50', 5: '51-60', 6: '61-70', 7: '> 70'}
AGE_GROUPS_TRANSFORMER = {1: 10, 2: 25, 3: 35, 4: 45, 5: 55, 6: 65, 7: 75}
