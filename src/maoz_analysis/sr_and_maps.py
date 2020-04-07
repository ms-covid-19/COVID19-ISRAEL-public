import os
import pandas as pd
import numpy as np

from config import RAW_DATA_DIR
from src.utils.map_utils import create_map
from src.utils.add_location import add_location
from src.utils.CONSTANTS import NEIGHBORHOOD_ID_COL, SYMPTOM_RATIO_WEIGHTED
from src.maoz_analysis.calc_sr_maoz import calc_symtom_ratio
from src.maoz_analysis.defs import COLUMNS_DICT, MAOZ_PROCESSED_FILE, MAOZ_FOLDER_RAW, MAOZ_PROCESSED_FILE_EXCEL, \
    BINARY_COLS, CITY_DICT_EN_HB, MAOZ_OUT, AGE_GROUPS_DICT


def process_data(df):
    processed_df = df.copy()
    processed_df.rename(columns=COLUMNS_DICT, inplace=True)
    # remove rows with no answers
    processed_df.dropna(subset=list(COLUMNS_DICT.values()), how='all', inplace=True)
    # remove rows with age out of categories
    processed_df = processed_df[processed_df.age.isin(AGE_GROUPS_DICT.keys())]

    # gender
    woman_idx = processed_df['gender'] == 2
    processed_df.loc[woman_idx, 'gender'] = 0

    # adult, 1 if over 18 0 o.w
    processed_df['adult'] = 0
    adult_idx = processed_df['age'] > 1
    processed_df.loc[adult_idx, 'adult'] = 1

    # fever
    processed_df['symptom_fever'] = 0
    fever_idx = processed_df['body_temp_measured'] == 3
    processed_df.loc[fever_idx, 'symptom_fever'] = 1

    # binary columns: 1 = yes, 2 = no
    for col in BINARY_COLS:
        no_idx = processed_df[col] == 2
        processed_df.loc[no_idx, col] = 0

    # one or more symptoms
    processed_df['sum_symptom'] = processed_df[['symptom_cough', 'symptom_sore_throat', 'symptom_shortness_of_breath',
                                                'symptom_smell_or_taste_loss', 'symptom_fever']].sum(axis=1)
    processed_df['one_or_more_symptoms'] = 0
    sypmtos_idx = processed_df['sum_symptom'] >= 1
    processed_df.loc[sypmtos_idx, 'one_or_more_symptoms'] = 1

    return processed_df


if __name__ == '__main__':
    maoz_files = []
    for f in os.listdir(MAOZ_FOLDER_RAW):
        if f.endswith('.csv'):
            new_file = pd.read_csv(os.path.join(MAOZ_FOLDER_RAW, f))
        elif f.endswith('.xlsx'):
            new_file = pd.read_excel(os.path.join(MAOZ_FOLDER_RAW, f))
        else:
            continue
        if 'lat' not in new_file.columns:
            new_file.rename(columns={'Unnamed: 1': 'street'}, inplace=True)
            new_file['street'] = new_file['street'].fillna(value='')
            new_file['city'] = CITY_DICT_EN_HB[f.split('.')[0]]
            new_file['lat'] = np.nan
            new_file['lng'] = np.nan
            new_file['source'] = np.nan
            new_file = add_location(new_file, RAW_DATA_DIR)
            new_file.drop(['source'], axis=1)
        maoz_files.append(new_file)

    raw_all_data = pd.concat(maoz_files, sort=False)

    all_data = process_data(raw_all_data)
    all_data = calc_symtom_ratio(all_data)
    all_data.to_csv(MAOZ_PROCESSED_FILE)
    all_data.to_excel(MAOZ_PROCESSED_FILE_EXCEL)

    neighborhoods = all_data.groupby([NEIGHBORHOOD_ID_COL]).mean()
    BINARY_COLS.extend([SYMPTOM_RATIO_WEIGHTED])
    for col in ['one_or_more_symptoms', SYMPTOM_RATIO_WEIGHTED]:
        try:
            create_map(neighborhoods.dropna(subset=[col]), col, 'maoz_neighborhoods_{}'.format(col),
                       out_dir=MAOZ_OUT, maoz=True)
        except Exception as e:
            print('Columns: {}, Exception: {}'.format(col, str(e)))









