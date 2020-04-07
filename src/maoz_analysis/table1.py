import os
import pandas as pd

from src.Table1.table1_global import make_table1
from src.maoz_analysis.defs import MAOZ_PROCESSED_FILE, AGE_GROUPS_TRANSFORMER, CITY_DICT_HB_EN, TABLE_COLS, TABLE_LABELS, \
    TABEL_COLS_TYPE, AGE_GROUPS_DICT, MAOZ_OUT


if __name__ == "__main__":
    all_data = pd.read_csv(MAOZ_PROCESSED_FILE, index_col=0)
    all_data.rename(columns={'age': 'age_category'}, inplace=True)
    all_data['age'] = all_data['age_category'].replace(AGE_GROUPS_TRANSFORMER)
    all_data.fillna(0, inplace=True)

    # Table1 by cities
    TX_labels = CITY_DICT_HB_EN
    TX_col = 'city'
    table1 = make_table1(all_data, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels, decimals=2).dropna(axis=1)
    table1.to_csv(os.path.join(MAOZ_OUT, 'Table1_cities.csv'))

    # Table1 per city vs others - adults and all
    for min_age, max_age in zip([20, 0], [100, 100]):
        for curr_city in CITY_DICT_HB_EN.keys():
            ages_data = all_data[(all_data.age >= min_age) & (all_data.age <= max_age)].copy()
            ages_data['curr_city'] = 0
            city_idx = ages_data.city == curr_city
            ages_data.loc[city_idx, 'curr_city'] = 1
            TX_col = 'curr_city'
            TX_labels = {1: CITY_DICT_HB_EN[curr_city], 0: 'Other cities'}
            table1 = make_table1(ages_data, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels, decimals=2).dropna(
                axis=1)
            table1.to_csv(os.path.join(MAOZ_OUT, 'Table1_cities_{}-{}_{}.csv'.format(min_age, max_age,
                                                                                    CITY_DICT_HB_EN[curr_city])))

    # Table1 per city with age groups
    TX_col = 'age_category'
    TX_labels = AGE_GROUPS_DICT
    for curr_city in CITY_DICT_HB_EN.keys():
        city_data = all_data[all_data.city == curr_city].copy()
        table1 = make_table1(city_data, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels,
                             decimals=2).dropna(
            axis=1)
        table1.to_csv(os.path.join(MAOZ_OUT, 'Table1_{}_age_groups.csv'.format(CITY_DICT_HB_EN[curr_city])))

    # Table1 by cities - adults only
    adults_data = all_data[all_data.adult == 1]
    TX_labels = CITY_DICT_HB_EN
    TX_col = 'city'
    table1 = make_table1(adults_data, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels, decimals=2).dropna(axis=1)
    table1.to_csv(os.path.join(MAOZ_OUT, 'Table1_cities_only_adults.csv'))

    # Table1 by adults-children
    TX_labels = {'1': 'Adult', '0': 'Child'}
    TX_col = 'adult'
    table1 = make_table1(all_data, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels, decimals=2).dropna(axis=1)
    table1.columns = ['All patients', 'Children (up to 20)', 'Adults']
    table1.to_csv(os.path.join(MAOZ_OUT, 'Table1_adults_vs_children.csv'))



