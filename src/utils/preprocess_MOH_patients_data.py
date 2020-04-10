import os
import numpy as np
import pandas as pd
from glob import glob
from datetime import date

from src.utils.add_location import add_location

from config import PATIENTS_RAW_DIR, PATIENTS_PROCESSED_DIR, RAW_DATA_DIR, LAMAS_DATA

existing_dates = [date.fromisoformat(os.path.basename(f).split('.')[0]) for f in
                  glob(os.path.join(PATIENTS_RAW_DIR, '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].xlsx'))]

patients_MOH_df = pd.DataFrame(columns=['city'])

for d in existing_dates:
    d_df = pd.read_excel(os.path.join(PATIENTS_RAW_DIR, str(d) + '.xlsx'))\
        .dropna().rename(columns={'patients': d})
    d_df['city'] = d_df['city'].str.strip()
    if 'ירושלים' not in d_df['city'].tolist():
        d_df['city'] = [city[::-1] for city in d_df['city']]

    patients_MOH_df = patients_MOH_df.set_index('city').join(d_df.set_index('city'), how='outer').reset_index()

patients_MOH_df['street'] = ''
patients_MOH_df['lat'] = np.nan
patients_MOH_df['lng'] = np.nan
patients_MOH_df['source'] = np.nan

patients_MOH_df = add_location(patients_MOH_df, RAW_DATA_DIR)
# in case two cities got the same CITY_ID take the one with the most patients
patients_MOH_df.groupby('CITY_ID').filter(lambda x: x.shape[0] > 1).sort_values('CITY_ID')\
    .to_excel(os.path.join(PATIENTS_PROCESSED_DIR, 'p_problems.xlsx'))  # report the duplicates
patients_MOH_df = patients_MOH_df.sort_values(existing_dates[::-1], ascending=False).drop_duplicates(['CITY_ID'])

cities_mtd = pd.read_csv(os.path.join(LAMAS_DATA, 'cities_meta.csv'))\
    .rename(columns={'OBJECTID_1': 'CITY_ID'})
patients_MOH_df = patients_MOH_df.set_index('CITY_ID').join(cities_mtd.set_index('CITY_ID'), how='left').reset_index()

os.makedirs(PATIENTS_PROCESSED_DIR, exist_ok=True)
patients_MOH_df.to_csv(os.path.join(PATIENTS_PROCESSED_DIR, 'MOH_confirmed_patients.csv'))
