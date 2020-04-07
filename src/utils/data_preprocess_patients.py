import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
from datetime import timedelta
from pytz import UTC
import os
from config import LAMAS_DATA, PATIENTS_PROCESSED_DIR, PATIENTS_RAW_DIR
from src.utils.CONSTANTS import *
from src.utils.district_helper import *

loc_desc_col = 'location_description'
date_col = 'date'
lat_col = 'lat'
lng_col = 'lng'
COLS_DICT = {'תאריך':date_col, 'x':lng_col,'y':lat_col,'מקום':loc_desc_col, 'description hebrew':loc_desc_col}
start_time_col = 'start time'
end_time_col = 'end time'


def preprocess_old_file(df):
    df = pd.concat([df, df.position.str.split(',', expand=True)], axis=1).rename(columns={0: 'lat', 1: 'lng'})
    return df


def preprocess_new_file(df, date_col, times_col, start_time_col, end_time_col):
    df = pd.concat([df, df[times_col].str.split('-', expand=True)], axis=1)
    # df[date_col] = pd.to_datetime(df[date_col])
    df.rename(columns={0:start_time_col, 1:end_time_col}, inplace=True)
    df[start_time_col] = np.where(pd.to_datetime(df[start_time_col].str.strip(), errors='coerce').isnull(), '00:00',
                                  df[start_time_col].str.strip())
    df[end_time_col] = np.where(pd.to_datetime(df[end_time_col].str.strip(), errors='coerce').isnull(), '23:59',
                                df[end_time_col].str.strip())
    return df


def process_patient_data(patients_data, date_col, start_time_col, end_time_col, lat_col=lat_col, lng_col=lng_col,
                         loc_desc_col=loc_desc_col):
    gdf_neighbor = gpd.read_file(os.path.join(LAMAS_DATA, 'neighbor_polygons.shp'), encoding='utf-8')
    gdf_cities = gpd.read_file(os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp'), encoding='utf-8')
    patients_data[date_col] = pd.to_datetime(patients_data[date_col], format='%d/%m/%Y')
    patients_data[lat_col] = patients_data[lat_col].astype(float)
    patients_data[lng_col] = patients_data[lng_col].astype(float)
    patients_data = patients_data[[date_col, lat_col, lng_col, start_time_col, end_time_col, loc_desc_col]]
    patients_data[TIME_STAY_HR] = pd.to_timedelta(pd.to_datetime(patients_data[end_time_col]) -
                                                  pd.to_datetime(patients_data[start_time_col]))
    patients_data[TIME_STAY_HR] = np.where(patients_data[TIME_STAY_HR] < timedelta(0),
                                           patients_data[TIME_STAY_HR] + timedelta(1),
                                           patients_data[TIME_STAY_HR])
    patients_data[TIME_STAY_FLT] = patients_data[TIME_STAY_HR].apply(lambda x: x.total_seconds() / (60 * 60))


    # Add neighborhood ID to match the Lamas neighborhoods
    points_geometry = [Point(p) for p in (list(zip(patients_data.lng, patients_data.lat)))]
    patients_data_gdf = gpd.GeoDataFrame(patients_data, geometry=points_geometry)
    patients_data_gdf.crs = {'init': 'epsg:4326'}
    patients_data_gdf = gpd.sjoin(gdf_neighbor, patients_data_gdf)

    patients_data = patients_data_gdf[[date_col, lat_col, lng_col, LAMAS_ID_COL, start_time_col, end_time_col,
                                       TIME_STAY_HR, TIME_STAY_FLT, 'DistrictCo', 'Shem_Yis_1', loc_desc_col]]

    # adds district id to unrecognized cities and adds district name
    patients_data = find_unknown_district(patients_data)
    patients_data = add_district_name(patients_data)

    patients_data.rename(columns={'DistrictCo': 'District_Number', 'Shem_Yis_1': 'City_En',
                                  LAMAS_ID_COL: NEIGHBORHOOD_ID_COL}, inplace=True)

    # Add city ID to match the Lamas cities
    points_geometry = [Point(p) for p in (list(zip(patients_data.lng, patients_data.lat)))]
    patients_data_gdf = gpd.GeoDataFrame(patients_data, geometry=points_geometry)
    patients_data_gdf.crs = {'init': 'epsg:4326'}
    patients_data_gdf = gpd.sjoin(gdf_cities, patients_data_gdf)

    patients_data = patients_data_gdf[patients_data.columns.tolist() + [LAMAS_ID_COL]].drop(['geometry'], axis=1)
    patients_data.rename(columns={LAMAS_ID_COL: CITY_ID_COL}, inplace=True)

    return patients_data


def merge_files(old, new):
    old.lat = old.lat.astype(str).str[0:10].astype(float)
    old.lng = old.lng.astype(str).str[0:10].astype(float)
    new.lat = new.lat.astype(str).str[0:10].astype(float)
    new.lng = new.lng.astype(str).str[0:10].astype(float)
    united = new.append(old)
    united['start time'] = united[['start time', 'end time']].min(axis=1)
    united['end time'] = united[['start time', 'end time']].max(axis=1)
    united.groupby(['date', 'lat', 'lng', 'start time', 'end time']).first().reset_index()\
        .to_csv(os.path.join(PATIENTS_PROCESSED_DIR, 'confirmed_patients_with_polygons.csv'))
    return


if __name__ == "__main__":
    new_patient_data = pd.read_csv(os.path.join(PATIENTS_RAW_DIR, 'מיקומי חשיפה לקורונה.csv'))
    new_patient_data.rename(columns=COLS_DICT, inplace=True)

    times_col = 'שעות שהייה'

    new_patient_data = preprocess_new_file(new_patient_data, date_col, times_col, start_time_col, end_time_col)
    new = process_patient_data(new_patient_data, date_col, start_time_col, end_time_col)
    old = pd.read_csv(os.path.join(PATIENTS_PROCESSED_DIR, 'confirmed_patients_with_polygons.csv'))
    merge_files(old, new)
