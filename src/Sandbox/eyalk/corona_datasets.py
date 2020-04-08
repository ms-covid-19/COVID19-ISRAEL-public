import os
import json
import glob
import pandas as pd
from pandas import DataFrame
from datetime import datetime


from config import OUT_DIR, LAMAS_DATA, UNIFIED_FORMS_FILE, HAMAGEN_DATA, PATIENTS_PROCESSED_DIR
from src.utils.CONSTANTS import LAMAS_ID_COL, CITY_ID_COL, NEIGHBORHOOD_ID_COL, SYMPTOM_RATIO


REF_DATETIME = datetime.strptime('2020-03-01', '%Y-%m-%d')


def load_unified_forms() -> DataFrame:

    data = pd.read_csv(UNIFIED_FORMS_FILE, index_col=0, low_memory=False)

    data['datetime'] = data.timestamp.map(
        lambda r: datetime.strptime(r, '%Y-%m-%dT%H:%M:%S'))
    data['date_int'] = (data.datetime - REF_DATETIME).days
    data['date_num'] = (data.datetime - REF_DATETIME).total_seconds() / (24 * 3600)

    return data


def load_hamagen_data() -> DataFrame:

    files_list = glob.glob(os.path.join(HAMAGEN_DATA, 'Points_*.json'))
    all_files = []
    for filename in files_list:
        with open(os.path.join(files_list), 'rb') as json_data:
            points = json.load(json_data)
        points_df = pd.json_normalize(points['features'])
        # normalize date columns and set thm as index
        points_df['fromTime'] = pd.to_datetime(points_df['properties.fromTime'] // 1000, unit='s')
        points_df['toTime'] = pd.to_datetime(points_df['properties.toTime'] // 1000, unit='s')
        points_df['from_date_int'] = (points_df.fromTime - REF_DATETIME).days
        points_df['from_date_num'] = (points_df.fromTime - REF_DATETIME).total_seconds() / (24 * 3600)
        points_df['duration_hours'] = (points_df.toTime.total_seconds()
                                       - points_df.fromTime.total_seconds()) / 3600
        #points_df = points_df.set_index(pd.DatetimeIndex(points_df['fromTime']))
        all_files.append(points_df)

    data = pd.concat(all_files, axis=0, sort=False).drop_duplicates(ignore_index=True)

    return data


def load_confirmed_by_day_and_city() -> DataFrame:

    data = pd.read_csv(os.path.join(PATIENTS_PROCESSED_DIR, 'confirmed_patients_by_day_and_city.csv'))
    # TODO: What post processing is needed?

    return data
