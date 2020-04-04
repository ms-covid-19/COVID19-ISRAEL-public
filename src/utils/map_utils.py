import os
import pandas as pd
import folium
import geopandas as gpd
import branca.colormap as cm

from config import OUT_DIR, LAMAS_DATA, PROCESSED_DATA_DIR
from src.utils.CONSTANTS import LAMAS_ID_COL, CITY_ID_COL, NEIGHBORHOOD_ID_COL, COLORS


def create_map(data, color_col, map_name, colors=None, city_level=False, scale=True, english=False):
    """
    Function to create an html map from given data.
    :param data: pandas DataFrame, must contain column given at color name, and an id column - either 'NEIGHBOR_ID'
    or 'CITY_ID'
    :param color_col: str, name of the column to color the map by
    :param map_name: str, name of the map that will be saved to OUT_DIR
    :param colors: list, list of colors that will be used to build the map. If None the colors used are:
    green_to_purple = ['#bae4b3', '#74c476', '#6baed6', '#c51b8a', '#7a0177']
    :param city_level: bool, whether the map is in city or neighborhood level. Default is False. T
    NOTE the id column in data must match the level.
    :param scale: bool, whether to scale the color_col column before drawing the map. Scaling will be done by dividing
    by the max value and multiplying by 100. Default is True.
    :param english: bool, preferred language of the background map. Defulat is False meaning the map will be in Hebrew/
    :return: Html map will be saved to the OUT_DIR
    """

    # scale and categorize column and create color map
    data_map = data.copy()
    if scale:
        data_map[color_col] = (data_map[color_col] / (data_map[color_col].max())) * 100
    color_col_cat = '{}_cat'.format(color_col)
    data_map[color_col_cat] = pd.qcut(data_map[color_col].values, 10, duplicates='drop', labels=[1, 2, 3, 4, 5, 6,
                                                                                                 7, 8, 9, 10])

    if colors:
        color_map = cm.LinearColormap(colors, vmin=data_map[color_col_cat].min(), vmax=data_map[color_col_cat].max())
    else:
        color_map = cm.LinearColormap(COLORS,
                                      vmin=data_map[color_col_cat].min(), vmax=data_map[color_col_cat].max())

    # merge with Lamas data to get polygons
    if city_level:
        lamas_data = gpd.read_file(os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp'), encoding='utf-8')
        data_map = lamas_data.merge(data_map, right_on=CITY_ID_COL, left_on=LAMAS_ID_COL)
    else:
        lamas_data = gpd.read_file(os.path.join(LAMAS_DATA, 'neighbor_polygons.shp'),  encoding='utf-8')
        data_map = lamas_data.merge(data_map, right_on=NEIGHBORHOOD_ID_COL, left_on=LAMAS_ID_COL)

    data_map['geometry'] = data_map['geometry'].apply(lambda x: x.buffer(0))

    if english:
        output_map = folium.Map(location=[32, 34], zoom_start=8,
                                tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
                                attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>')
    else:
        output_map = folium.Map(location=[32, 34], zoom_start=8)

    folium.GeoJson(data_map,
                   style_function=lambda feature: {
                        'fillColor': color_map(feature['properties'][color_col_cat]),
                        'color': 'black',
                        'weight': 0.25,
                        'lineOpacity': 0.2,
                        'fillOpacity': 0.55
                   },).add_to(output_map)

    color_map.add_to(output_map)
    output_map.save(os.path.join(OUT_DIR, '{}.html'.format(map_name)))


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'GT_city_id_2020-03-24.csv'))
    data.rename(columns={'city_id': 'CITY_ID'}, inplace=True)
    create_map(data, 'Next day - questionnaires', 'city_true', city_level=True)
