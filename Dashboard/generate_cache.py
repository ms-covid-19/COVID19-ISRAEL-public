import json
import logging
import os

import branca.colormap as cm
import geopandas as gpd
import pandas as pd

from Dashboard.data import COLORS, features_to_perc, NDAYS_SMOOTHING, MIN_OBSERVATIONS_CITY, \
    MIN_OBSERVATIONS_NEIGHBORHOOD
from config import DASH_CACHE_DIR, LAMAS_DATA, PROCESSED_DATA_DIR
from src.utils.aggregations import mean_if_enough_observations
from src.utils.processed_data_class import ProcessedData

log_ = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

COLOR_MAP = lambda vmin, vmax: cm.LinearColormap(COLORS, vmin=vmin, vmax=vmax)


class DashCacheGenerator(object):
    def __init__(self):
        # data = pd.read_csv(os.path.join(NEW_PROCESSED_DATA_DIR, 'bot_and_questionnaire_2403.csv'), index_col=0,
        #                    low_memory=False)
        # data = pd.read_csv(os.path.join(NEW_PROCESSED_DATA_DIR, 'COVID_19-All_with_location.csv'), index_col=0,
        #                    low_memory=False)
        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'forms', 'all_forms.csv'), index_col=0,
                           low_memory=False)

        # tmp = data.copy()
        # tmp = tmp.query("City_En=='HOLON'")
        # tmp['timestamp'] = pd.to_datetime(tmp['timestamp'])
        # min_date = pd.to_datetime('2020-03-29 00:00:00')
        # tmp = tmp.query('timestamp >= @min_date')
        # print(len(tmp))
        # tmp_neicount = tmp.groupby('NEIGHBOR_ID').count()
        # print(tmp_neicount.query('symptom_ratio >= 50')[['City_En', 'symptom_ratio']])

        data = self.process_data(data)
        self.cache_city(data)
        self.cache_neighborhood(data)

    @staticmethod
    def cache_hasadna(df):
        df = ProcessedData.add_city_name(df).set_index(['city_heb', 'city_eng'], append=True)\
            .reorder_levels(['city_id', 'city_heb', 'city_eng', 'neighborhood_id', 'date', 'time'])
        symptom_ratio_wei_norm = df['symptom_ratio_weighted'] / df['symptom_ratio_weighted'].max() * 100
        symptom_ratio_wei_norm_agg = symptom_ratio_wei_norm.groupby(['city_id', 'city_heb', 'city_eng']).agg(['mean', 'std', 'count'])
        symptom_ratio_wei_norm_agg.to_csv(os.path.join(PROCESSED_DATA_DIR, 'symptom_ratio_wei_norm_agg_all_days.csv'))
        symptom_ratio_wei_norm_agg.to_json(os.path.join(PROCESSED_DATA_DIR, 'symptom_ratio_wei_norm_agg_all_days.json'))

        ld = symptom_ratio_wei_norm.index.get_level_values('date').max()
        symptom_ratio_wei_norm_agg_ld = symptom_ratio_wei_norm.to_frame()\
            .query("date == @ld")['symptom_ratio_weighted']\
            .groupby(['city_id', 'city_heb', 'city_eng'])\
            .agg(['mean', 'std', 'count'])
        symptom_ratio_wei_norm_agg_ld.to_csv(os.path.join(PROCESSED_DATA_DIR, 'symptom_ratio_wei_norm_agg_last_day.csv'))
        symptom_ratio_wei_norm_agg_ld.to_json(os.path.join(PROCESSED_DATA_DIR, 'symptom_ratio_wei_norm_agg_last_day.json'))
        pass

    def process_data(self, data):
        # data = ProcessedData.convert_new_processed_bot_and_questionnaire(data)
        data = ProcessedData.convert_processed_united(data)
        data[features_to_perc] = data[features_to_perc] * 100
        self.cache_hasadna(data)
        return data

    @staticmethod
    def day_roller(gdf, ndays=NDAYS_SMOOTHING):
        # append the dataframe with itself shifted 1, 2, ..., ndays forward
        max_day = gdf['date'].max()
        add_dfs = []
        for i in range(1, ndays):
            tmp = gdf.copy()
            tmp['date'] = tmp['date'] + i
            add_dfs.append(tmp)
        undf = gdf.append(add_dfs).query('date <= @max_day')
        # calculate mean for columns with enough observations
        return undf

    def aggregate_polygon(self, gdf, level):
        undf = self.day_roller(gdf)
        if level == 'city_id':
            return undf.groupby('date', as_index=False, group_keys=False).apply(lambda x: mean_if_enough_observations(x, min_observations=MIN_OBSERVATIONS_CITY))
        else:
            return undf.groupby('date', as_index=False, group_keys=False).apply(
                lambda x: mean_if_enough_observations(x, min_observations=MIN_OBSERVATIONS_NEIGHBORHOOD))

    def count_polygon(self, gdf):
        undf = self.day_roller(gdf).set_index('date', append=True)
        return undf.groupby('date').count()

    def aggregate_data(self, data, level):
        log_.info('Start processing level {}'.format(level))
        if level == 'city_id':
            data = data.reset_index('neighborhood_id', drop=True)
        else:
            data = data.reset_index('city_id', drop=True)
        log_.info('Counting...')
        gdf = data.select_dtypes('number').sort_index().reset_index('date').groupby(level)
        data_count = gdf.apply(self.count_polygon)

        # tmp = data_count[['smoking']]
        # sel_date = (pd.to_datetime('2020-03-20 00:00:00') - ANCHOR_DATE).days
        # tmp = tmp.query('date == @sel_date').sort_values('smoking', ascending=False)

        log_.info('Counting finished!')
        log_.info('Aggregating...')
        gdf = data.select_dtypes('number').sort_index().reset_index().groupby(level, as_index=False, group_keys=False)
        data_agg = gdf.apply(self.aggregate_polygon, level=level)
        log_.info('Aggregating finished!')
        data_agg = data_agg.dropna(how='all').drop(columns=['time']).set_index([level, 'date'])
        return data_count, data_agg

    def convert_cache_lamas(self, tag, valid_ids):
        if tag == 'city':
            lamas_fn = os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp')
        else:
            lamas_fn = os.path.join(LAMAS_DATA, 'neighbor_polygons.shp')
        gpdf = gpd.read_file(lamas_fn, encoding='utf-8')[['OBJECTID_1', 'geometry']]
        gpdf = gpdf.rename(columns={'OBJECTID_1': 'id'})
        # write only meaningful polygons (intersect with agg_data)
        gpdf = gpdf.query("id in @valid_ids").reset_index(drop=True)
        gpdf['geometry'] = gpdf['geometry'].apply(lambda x: x.buffer(0))
        centers = gpdf.set_index('id')
        centers = ProcessedData.add_city_name(centers)
        #
        tmp = centers.query("city_eng=='REHOVOT'")

        centers['geometry'] = centers['geometry'].centroid
        centers['lat'] = centers['geometry'].y
        centers['lon'] = centers['geometry'].x
        centers.reset_index().drop(columns=['geometry']).to_feather(os.path.join(DASH_CACHE_DIR, tag + '_centers.feather'))
        gpdf[['id', 'geometry']] \
            .to_file(os.path.join(DASH_CACHE_DIR, tag + '_polygons.json'), driver="GeoJSON")
        self.property_to_id_cache_geolayers(folder=DASH_CACHE_DIR, tag=tag)

    def write_to_cache(self, df, tag):
        fn = os.path.join(DASH_CACHE_DIR, tag + '.feather')
        if tag == 'city':
            df = ProcessedData.add_city_name(df, eng=True, heb=True)
        df = df.reset_index()
        df[tag + '_id'] = df[tag + '_id'].astype(int)
        df['date'] = ProcessedData.date_and_time_to_timeser(date=df['date'], time=None)
        df.to_feather(fn)

    @staticmethod
    def edit_data(data):
        for col in ['symptom_ratio', 'symptom_ratio_weighted']:
            data[col] = (data[col] / data[col].max()) * 100
        return data

    @staticmethod
    def get_color_series(ser):
        f = COLOR_MAP(vmin=ser.min(), vmax=ser.max())
        return ser.dropna().apply(f).reindex(ser.index)

    def concat_agg_count(self, count_df, agg_df):
        color_df = agg_df.groupby('date').apply(lambda gdf: gdf.apply(self.get_color_series))
        df = agg_df.join(count_df, rsuffix='_count').join(color_df, rsuffix='_color')
        return df

    def cache_city(self, data):
        count_data, agg_data = self.aggregate_data(data, level='city_id')
        agg_data = self.edit_data(agg_data)
        df = self.concat_agg_count(count_data, agg_data)
        self.write_to_cache(df, tag='city')
        self.convert_cache_lamas(tag='city', valid_ids=df.index.get_level_values('city_id').unique())

    def cache_neighborhood(self, data):
        count_data, agg_data = self.aggregate_data(data, level='neighborhood_id')
        agg_data = self.edit_data(agg_data)
        df = self.concat_agg_count(count_data, agg_data)
        self.write_to_cache(df, tag='neighborhood')
        self.convert_cache_lamas(tag='neighborhood', valid_ids=df.index.get_level_values('neighborhood_id').unique())

    @staticmethod
    def property_to_id_cache_geolayers(folder, tag):
        fn = os.path.join(folder, tag + '_polygons.json')
        geolayers_fn = os.path.join(folder, tag + '_geolayers.json')
        with open(fn) as in_file:
            polygons = json.load(in_file)
        # log_.info("Removing MultiPolygons...")
        # polygons['features'] = [polygons['features'][i] for i in range(len(polygons['features'])) if
        #                         polygons['features'][i]['geometry']['type'] == 'Polygon']
        log_.info("{}: moving property to id".format(fn))
        geo_layers = {}
        for i in range(len(polygons['features'])):
            polygons['features'][i]['id'] = polygons['features'][i]['properties']['id']
            polygons['features'][i]['properties'] = {}
            indiv = {'type': 'FeatureCollection', 'features': [polygons['features'][i]]}
            geo_layers[polygons['features'][i]['id']] = dict(
                sourcetype="geojson",
                source=indiv,  # base_url + str(year) + "/" + bin + ".geojson",
                type="fill",
                color='red',  # cm[bin],
                opacity=0.5,
                fill=dict(outlinecolor="#afafaf"),
            )
            # with open(os.path.join(folder, 'polygons', tag, str(i) + '.json'), 'w') as out_file:
            #     json.dump(geo_layers[i], out_file)
        with open(fn, 'w') as out_file:
            json.dump(polygons, out_file)
        with open(geolayers_fn, 'w') as out_file:
            json.dump(geo_layers, out_file)


if __name__ == '__main__':
    log_.info('Start...')
    dcg = DashCacheGenerator()
