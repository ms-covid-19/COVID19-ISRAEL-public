import os
import json
import glob
from typing import Optional, List
import pandas as pd
import geopandas as gpd

from geopandas import GeoDataFrame
from pandas import DataFrame
from datetime import datetime


from config import OUT_DIR, LAMAS_DATA, UNIFIED_FORMS_FILE, HAMAGEN_DATA, PATIENTS_PROCESSED_DIR, \
    GENERAL_CACHE_DIR
from src.train.constants import DATE_COL, TIME_COL
from src.utils.CONSTANTS import LAMAS_ID_COL, CITY_ID_COL, NEIGHBORHOOD_ID_COL, SYMPTOM_RATIO


REF_DATETIME = datetime.strptime('2020-03-01', '%Y-%m-%d')


def load_unified_forms(agg_col: Optional[str] = None) -> DataFrame:
    """

    Args:
        agg_col: If not None, we keep only rows with agg_col.notnull()

    Returns:

    """
    data = pd.read_csv(UNIFIED_FORMS_FILE, index_col=0, low_memory=False)

    data[DATE_COL] = data[TIME_COL].apply(lambda x: x.split('T')[0])

    data['datetime'] = data.timestamp.map(
        lambda r: datetime.strptime(r, '%Y-%m-%dT%H:%M:%S'))
    data['date_int'] = data.datetime.map(lambda r: (r - REF_DATETIME).days)
    data['date_num'] = data.datetime.map(lambda r: (r - REF_DATETIME).total_seconds() / (24 * 3600))

    data = data[(data.age.astype(float) > 0) &
                (data.age.astype(float) < 100)]
    data.gender = (data.gender == 'M').astype(int)

    if agg_col is not None:
        data = data[data[agg_col].notnull()]

    return data


def load_hamagen_data() -> DataFrame:

    files_list = glob.glob(os.path.join(HAMAGEN_DATA, 'Points_*.json'))
    all_files = []
    for filename in files_list:
        with open(os.path.join(filename), 'rb') as json_data:
            points = json.load(json_data)
        points_df = pd.json_normalize(points['features'])
        # normalize date columns and set thm as index
        points_df['fromTime'] = pd.to_datetime(points_df['properties.fromTime'] // 1000, unit='s')
        points_df['toTime'] = pd.to_datetime(points_df['properties.toTime'] // 1000, unit='s')
        points_df['from_date_int'] = points_df.fromTime.map(lambda r: (r - REF_DATETIME).days)
        points_df['from_date_num'] = points_df.fromTime.map(
            lambda r: (r - REF_DATETIME).total_seconds() / (24 * 3600))
        points_df['duration_hours'] = points_df.apply(
            lambda r: (r.toTime - r.fromTime).total_seconds() / 3600, axis=1)
        points_df.rename(columns={'properties.POINT_X': 'lat',
                                  'properties.POINT_Y': 'lng',
                                  'properties.Place': 'Place',
                                  'properties.OBJECTID': 'OBJECTID'
                                  }, inplace=True)
        points_df.drop(columns=['geometry.coordinates',  # We already have the data
                                ] ,inplace=True)
        #points_df = points_df.set_index(pd.DatetimeIndex(points_df['fromTime']))
        all_files.append(points_df)

    data = pd.concat(all_files, axis=0, sort=False)
    data.drop_duplicates(  # TODO: Verify that this is the correct way
        ['Place', 'OBJECTID', 'lat', 'lng', 'fromTime', 'toTime'],
        ignore_index=True, inplace=True)

    return data


def load_confirmed_by_day_and_city() -> DataFrame:

    data = pd.read_csv(os.path.join(PATIENTS_PROCESSED_DIR, 'confirmed_patients_by_day_and_city.csv'))
    # TODO: What post processing is needed?

    return data


def load_confirmed_patients_by_cities_mar_two_dates(
        drop_nan_city_flag: bool = True, city_filter: Optional[List] = None) -> DataFrame:
    data  = pd.read_csv(os.path.join(
        PATIENTS_PROCESSED_DIR, 'confirmed_patients_by_cities.csv'))
    if drop_nan_city_flag:
        data = data[~data.City_En.isna()]
    if city_filter is not None:
        data = data[data.City_En.isin(city_filter)]
    data.set_index('City_En', inplace=True)

    return data


def load_lamas_data(cache_file_name_prefix: Optional[str] = GENERAL_CACHE_DIR + r'\cities_lms_cache',
                    reset_cache_flag: bool = False) -> GeoDataFrame:
    """

    Args:
        cache_file_name_prefix: If not None, use it for caching the results (should be full path here)
        reset_cache_flag: If True, don't read from cache (but write if cache_file_name is not None)

    Returns: DataFrame with lamas data per city

    """
    cache_file_name = cache_file_name_prefix + '.zip'
    if not reset_cache_flag and cache_file_name is  not None and os.path.exists(cache_file_name):
        return pd.read_pickle(cache_file_name)

    cities_gpd = gpd.read_file(os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp'), encoding='utf-8')
    # create demographic features only for cities we have population number for
    cities_gpd = cities_gpd[cities_gpd.Pop_Total > 0]
    cities_gpd['City_En'] = cities_gpd.SHEM_YIS_1
    # TODO: Here do more computations such as density

    if cache_file_name is not None:
        cities_gpd.to_pickle(cache_file_name)

    return cities_gpd
