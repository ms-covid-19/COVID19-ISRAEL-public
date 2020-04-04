import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.table1_global import make_table1
from src.utils.CONSTANTS import *
from config import OUT_DIR, GOOGLE_FORM_PROCESSED_DIR

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(GOOGLE_FORM_PROCESSED_DIR, 'COVID_19-All_with_location.csv'),
                     index_col=0, low_memory=False)

    # Add some columns
    df[HIGH_FEVER] = (df[BODY_TEMP] > FEVER_TEMP).astype(int)
    df[ISOLATION] = ((df[ISOLATION_CONTACT] == 1) | (df[ISOLATION_TRAVEL] == 1)).astype(int)
    df['Adult'] = (df.Age > AGE_THRESHOLD).astype(int)

    # Table1 by isolation status
    TX_labels = {'1': 'Quarantine', '0': 'Not quarantied'}
    TX_col = ISOLATION
    table1 = make_table1(df, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels, decimals=2).dropna(axis=1)
    table1.columns = ['All patients', 'Not in home isolation', 'In home isolation']
    table1.to_csv(os.path.join(OUT_DIR, 'Table1_Isolation.csv'))

    # Table1 by adults-children
    TX_labels = {'1': 'Adult', '0': 'Child'}
    TX_col = 'Adult'
    table1 = make_table1(df, TABLE_COLS, TABLE_LABELS, TABEL_COLS_TYPE, TX_col, TX_labels, decimals=2).dropna(axis=1)
    table1.columns = ['All patients', 'Children (up to 18)', 'Adults']
    table1.to_csv(os.path.join(OUT_DIR, 'Table1_Adults.csv'))
