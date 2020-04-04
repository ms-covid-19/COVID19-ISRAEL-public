import numpy as np
import os
import geocoder
import concurrent.futures
import tqdm
import geopandas as gpd
from shapely.geometry import Point
import googlemaps

from config import UTILITY_DATA, LAMAS_DATA
from src.utils.CONSTANTS import LAMAS_ID_COL, NEIGHBORHOOD_ID_COL, CITY_ID_COL
from src.utils.district_helper import *


def get_location_osm(session, city, street):
        g = geocoder.osm(city + ', ' + street, session=session, url='https://geocode.datacity.org.il/')
        if g.ok and g.lat and g.lng:
            return g.lat, g.lng
        else:
            return np.nan, np.nan


def get_location_google(google_maps, city, street):
    try:
        return google_maps.geocode(city + ', ' + street)[0]['geometry']['location']
    except IndexError:
        return np.nan


def geocode_locations(location_list):
    results = {}
    num_workers = 10
    with open(os.path.join(UTILITY_DATA, 'google_api_key.txt')) as file:
        KEY = file.readline()
    google_maps = googlemaps.Client(key=KEY)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers,) as executor:
        futures_dict = {}

        for city, street in location_list:
            future = executor.submit(fn=get_location_google, google_maps=google_maps, city=city, street=street,)
            futures_dict[future] = (city, street)

        for index, future in tqdm.tqdm(enumerate(concurrent.futures.as_completed(fs=futures_dict,),),
                                       total=len(location_list), ):
            city, street = futures_dict[future]
            try:
                results[(city, street)] = future.result()
            except Exception as exception:
                print(city, street, exception)
                results[(city, street)] = np.nan
        result_df = pd.DataFrame.from_dict(results).T
        result_df.index.names = ['city', 'street']
        result_df.reset_index(inplace=True)
        result_df.rename(columns={0: 'lat', 1: 'lng'}, inplace=True)
        # result_df.to_csv('new_unique_locations.csv')
    return result_df


def add_location(df, known_locations_dir, city_col='city', street_col='street'):
    """
    Add geolocation to dataframe.
    :param df: DataFrame, must have columns city_col and street_col
    :param known_locations_dir: str, directory where the file "known_locations.csv" is saved
    :param city_col: str, name of the column which contains the city
    :param street_col: str, name of the column which contains the street
    :return: df_with_location_info: DataFrame, same df which was given with additional columns: 'lat', 'lng', 'CITY_ID',
    'NEIGHBOR_ID', 'District_Number' and 'City_En'
    """

    # Bot data comes with location
    if df['source'][0] == 'bot':
        columns_list = [col for col in df.columns.tolist() if col not in ['lat', 'lng']]
        df_with_location = df
    # For google_forms we need to get locations
    else:
        df.drop(['lat', 'lng'], axis=1, inplace=True)
        columns_list = df.columns.tolist()
        known_locations = pd.read_csv(os.path.join(known_locations_dir, 'known_locations.csv'), index_col=0)
        known_unique_locations = set(zip(known_locations[city_col], known_locations[street_col]))
        all_unique_locations = list(set(list(zip(df[city_col], df[street_col]))))
        missing_locations = [location for location in all_unique_locations if location not in known_unique_locations]
        missing_unique_location = list(set(missing_locations))
        if len(missing_unique_location) > 0:
            print('Geocoding {} new locations'.format(len(missing_unique_location)))
            new_location_df = geocode_locations(missing_unique_location)
            all_locations = pd.concat([known_locations, new_location_df], axis=0,
                                      sort=False).drop_duplicates(subset=[city_col, street_col])
        else:
            all_locations = known_locations
        all_locations.to_csv(os.path.join(known_locations_dir, 'known_locations.csv'))

        df_with_location = df.merge(all_locations, on=[city_col, street_col])

    df_with_location.dropna(subset=['lat', 'lng'], inplace=True)

    # Add City ID to match the Lamas cities
    points_geometry = [Point(p) for p in (list(zip(df_with_location.lng, df_with_location.lat)))]
    df_with_location_gpd = gpd.GeoDataFrame(df_with_location, geometry=points_geometry)
    df_with_location_gpd.crs = {'init': 'epsg:4326'}
    gdf = gpd.read_file(os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp'), encoding='utf-8')
    df_with_cities = gpd.sjoin(gdf[[LAMAS_ID_COL, 'geometry']], df_with_location_gpd)
    df_with_cities.rename(columns={LAMAS_ID_COL: CITY_ID_COL}, inplace=True)
    df_with_location = df_with_cities[df_with_location.columns.tolist() + [CITY_ID_COL]].drop(['geometry'], axis=1)

    # Add neighborhood ID to match the Lamas neighborhoods
    points_geometry = [Point(p) for p in (list(zip(df_with_location.lng, df_with_location.lat)))]
    df_with_location = gpd.GeoDataFrame(df_with_location, geometry=points_geometry)
    df_with_location.crs = {'init': 'epsg:4326'}
    gdf_nei = gpd.read_file(os.path.join(LAMAS_DATA, 'neighbor_polygons.shp'), encoding='utf-8')
    df_with_location_info = gpd.sjoin(gdf_nei, df_with_location)
    df_with_location_info = df_with_location_info[columns_list + [CITY_ID_COL, LAMAS_ID_COL, 'DistrictCo',
                                                      'Shem_Yis_1', 'lat', 'lng']]
    # adds district id to unrecognized cities and adds district name
    df_with_location_info = find_unknown_district(df_with_location_info)
    df_with_location_info = add_district_name(df_with_location_info)
    df_with_location_info.rename(columns={'Shem_Yis_1': 'City_En', 'DistrictCo': 'District_Number',
                                          LAMAS_ID_COL: NEIGHBORHOOD_ID_COL}, inplace=True)
    df_with_location_info.reset_index(inplace=True, drop=True)

    return df_with_location_info
