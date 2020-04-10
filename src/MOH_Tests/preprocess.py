import pandas as pd

from src.MOH_Tests.defs import *


def process(df):
    df.rename(columns=COLS_DICT, inplace=True)
    df['corona_result'] = df['corona_result'].replace(CORONA_RES_DICT)
    df['gender'] = df['gender'].replace(GENDER_DICT)
    df['age_over_60'] = df['age_over_60'].replace(AGE_DICT)

    df.dropna(subset=['corona_result'], inplace=True)
    return df


if __name__ == '__main__':
    data_raw = pd.read_excel(MOH_T_RAW_FILE)
    data_processed = process(data_raw)
    data_processed.to_csv(MOH_T_PROCESSED_FILE)
