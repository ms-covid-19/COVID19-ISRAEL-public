import pandas as pd
import os
import folium
import geopandas as gpd
import branca.colormap as cm

from config import OUT_DIR, LAMAS_DATA, UNIFIED_FORMS_FILE
from src.utils.CONSTANTS import LAMAS_ID_COL, CITY_ID_COL, NEIGHBORHOOD_ID_COL, SYMPTOM_RATIO


def create_map_for_figures(gpd, color_column, map_name, color_map, english=False):
    if english:
        mapa = folium.Map(location=[32, 34], zoom_start=8,
                          tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
                          attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>')
    else:
        mapa = folium.Map(location=[32, 34], zoom_start=8)

    folium.GeoJson(gpd,
                   style_function=lambda feature: {
                       'fillColor': color_map(feature['properties'][color_column]),
                       'color': 'black',
                       'weight': 0.25,
                       'lineOpacity': 0.2,
                       'fillOpacity': 0.55
                   },
                   tooltip=folium.GeoJsonTooltip(fields=[color_column, '# Responds'],
                                                 aliases=['Symptoms category', '# Responds'])).add_to(mapa)

    color_map.add_to(mapa)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    mapa.save(os.path.join(OUT_DIR, '{}.html'.format(map_name)))


if __name__ == "__main__":
    # City map
    data = pd.read_csv(UNIFIED_FORMS_FILE, index_col=0,
                       low_memory=False)
    data_g = data.groupby([CITY_ID_COL])[SYMPTOM_RATIO].agg(['mean', 'count'])
    data_g.rename(columns={'mean': 'symptoms_mean', 'count': '# Responds'}, inplace=True)
    data_g = data_g[data_g['# Responds'] > 30]

    gdf_cities = gpd.read_file(os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp'), encoding='utf-8')
    gdf_cities = gdf_cities.merge(data_g, left_on=LAMAS_ID_COL, right_on=CITY_ID_COL)
    gdf_cities['geometry'] = gdf_cities['geometry'].apply(lambda x: x.buffer(0))
    gdf_cities['symptoms_norm'] = (gdf_cities['symptoms_mean'] / (gdf_cities['symptoms_mean'].max())) * 100
    gdf_cities['symptoms_category'] = pd.qcut(gdf_cities['symptoms_norm'].values, 5, duplicates='drop',
                                              labels=[1, 2, 3, 4, 5])

    purples = ['#feebe2', '#fbb4b9', '#f768a1', '#c51b8a', '#7a0177']
    # green_to_red = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']

    color_map = cm.LinearColormap(purples, vmin=1, vmax=5)
    create_map_for_figures(gdf_cities, 'symptoms_category', 'average_symptoms_cities_map', color_map)

    # Neighborhood map
    data_nei_g = data.groupby([NEIGHBORHOOD_ID_COL])[SYMPTOM_RATIO].agg(['mean', 'count'])
    data_nei_g.rename(columns={'mean': 'symptoms_mean', 'count': '# Responds'}, inplace=True)
    data_nei_g = data_nei_g[data_nei_g['# Responds'] > 10]

    gdf_nei = gpd.read_file(os.path.join(LAMAS_DATA, 'neighbor_polygons.shp'), encoding='utf-8')
    gdf_nei = gdf_nei.merge(data_nei_g, left_on=LAMAS_ID_COL, right_on=NEIGHBORHOOD_ID_COL)
    gdf_nei['geometry'] = gdf_nei['geometry'].apply(lambda x: x.buffer(0))
    gdf_nei['symptoms_norm'] = (gdf_nei['symptoms_mean'] / (gdf_nei['symptoms_mean'].max())) * 100
    gdf_nei['symptoms_category'] = pd.qcut(gdf_nei['symptoms_norm'].values, 5, duplicates='drop',
                                           labels=[1, 2, 3, 4, 5])

    create_map_for_figures(gdf_nei, 'symptoms_category', 'average_symptoms_neighborhoos_map', color_map)
