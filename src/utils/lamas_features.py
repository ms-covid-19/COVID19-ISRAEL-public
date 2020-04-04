import os
import geopandas as gpd
import pyproj
from shapely.ops import transform

from config import LAMAS_DATA
from src.utils.CONSTANTS import LAMAS_ID_COL
from src.train.constants import *


def create_city_features(out_path):
    cities_gpd = gpd.read_file(os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp'), encoding='utf-8')
    # create demographic features only for cities we have population number for
    cities_gpd = cities_gpd[cities_gpd.Pop_Total > 0]

    # change crs to UTM to calculate area in squared km
    project = lambda x, y: pyproj.transform(pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='epsg:32636'), x, y)
    cities_gpd['geometry'] = cities_gpd.geometry.apply(lambda g: transform(project, g))
    cities_gpd['area_km^2'] = cities_gpd.geometry.area / 1000000

    population_columns = ['Male_Total', 'Female_Tot', 'age_0_14', 'age_15_19', 'age_20_29', 'age_30_64', 'age_65_up']
    new_population_columns = [LMS_MALE_PERC, LMS_FEMALE_PERC, LMS_0_14_PERC, LMS_15_19_PERC, LMS_20_29_PERC,
                              LMS_30_64_PERC, LMS_65_up_PERC]
    col_dict = {old: new for old, new in list(zip(population_columns + ['Pop_Total'],
                                                  new_population_columns + [LMS_POP_DENSITY]))}

    cities_gpd[population_columns] = cities_gpd[population_columns].div(cities_gpd.Pop_Total, axis=0)
    cities_gpd['Pop_Total'] = cities_gpd['Pop_Total'] / cities_gpd['area_km^2']
    cities_gpd.rename(columns=col_dict, inplace=True)

    cities_gpd[new_population_columns + [LAMAS_ID_COL]].to_csv(os.path.join(out_path, 'cities_lms_features.csv'))


def create_neighborhood_features(out_path):
    nei_gpd = gpd.read_file(os.path.join(LAMAS_DATA, 'neighbor_polygons.shp'), encoding='utf-8')
    # create demographic features only for neighborhoods we have population number for
    nei_gpd = nei_gpd[nei_gpd['pop_thou'] > 0]

    # change crs to UTM to calculate area in squared km
    project = lambda x, y: pyproj.transform(pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='epsg:32636'), x, y)
    nei_gpd['geometry'] = nei_gpd.geometry.apply(lambda g: transform(project, g))
    nei_gpd['area_km^2'] = nei_gpd.geometry.area / 1000000

    nei_gpd.rename(columns={'pop_thou': 'Pop_total_thousands', 'men_thou': 'Men_total_thousands',
                            'women_thou': 'Woman_total_thousands'}, inplace=True)
    population_columns_thousands = ['Pop_total_thousands', 'Men_total_thousands', 'Woman_total_thousands']
    population_columns_perc = ['age0_17_pc', 'age18_64_p', 'age65_pcnt']
    new_population_columns = [LMS_POP_DENSITY, LMS_MALE_PERC, LMS_FEMALE_PERC, LMS_0_17_PERC, LMS_18_64_PERC,
                              LMS_65_up_PERC]
    col_dict = {old: new for old, new in list(zip(population_columns_thousands + population_columns_perc,
                                                  new_population_columns))}
    nei_gpd[['Men_total_thousands', 'Woman_total_thousands']] = \
        nei_gpd[['Men_total_thousands', 'Woman_total_thousands']].div(nei_gpd['Pop_total_thousands'], axis=0)
    nei_gpd['Pop_total_thousands'] = nei_gpd['Pop_total_thousands']*1000 / nei_gpd['area_km^2']
    nei_gpd[population_columns_perc] = nei_gpd[population_columns_perc] / 100
    nei_gpd.rename(columns=col_dict, inplace=True)

    # Two Jewish neighborhoods have religion code 6 instead of 1
    nei_gpd.religion = nei_gpd.religion.replace(6, 1)

    nei_gpd[new_population_columns +
            [LAMAS_ID_COL, LMS_RELIGION_CODE]].to_csv(os.path.join(out_path, 'neighborhoods_lms_features.csv'))


if __name__ == "__main__":
    create_city_features(LAMAS_DATA)
    create_neighborhood_features(LAMAS_DATA)
