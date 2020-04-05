import os
import pandas as pd

from src.Table1.table1_global import make_table1
from src.Table1.defs import *
from config import OUT_DIR, UNIFIED_FORMS_FILE
from src.utils.CONSTANTS import AGE_THRESHOLD
from src.utils.unify_forms import Symptom, Condition, Smoking, Isolation, enum_to_column

gender_dictionary = {'F': 0, 'M': 1}

if __name__ == "__main__":
    df = pd.read_csv(UNIFIED_FORMS_FILE)

    # Modify some columns
    df['Adult'] = (df.age > AGE_THRESHOLD).astype(int)
    df['gender'] = df['gender'].replace(gender_dictionary)

    # Table1 by isolation status
    TX_labels = {'1': 'Quarantine', '0': 'Not quarantied'}
    TX_col = enum_to_column(Isolation.ISOLATED)
    table1_isolcation = make_table1(df, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels, decimals=2).dropna(axis=1)
    table1_isolcation.columns = ['All patients', 'Not in isolation', 'In isolation']

    # Table1 by adults-children
    TX_labels = {'1': 'Adult', '0': 'Child'}
    TX_col = 'Adult'
    table1_adults = make_table1(df, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels, decimals=2).dropna(axis=1)
    table1_adults.columns = ['All patients', 'Children (up to 18)', 'Adults']

    # Merge tables and save
    table1 = table1_isolcation.merge(table1_adults.drop(['All patients'], axis=1)[['Adults', 'Children (up to 18)']],
                                     left_index=True, right_index=True)
    table1.to_csv(os.path.join(OUT_DIR, 'Table1.csv'))
