import json
import logging
import os

import branca.colormap as cm
import geopandas as gpd
import pandas as pd
import numpy as np

from Dashboard.data import COLORS, features_to_perc, NDAYS_SMOOTHING, MIN_OBSERVATIONS_CITY, \
    MIN_OBSERVATIONS_NEIGHBORHOOD
from config import DASH_CACHE_DIR, LAMAS_DATA, PROCESSED_DATA_DIR
from src.utils.processed_data_class import ProcessedData, ANCHOR_DATE

log_ = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

COLOR_MAP = lambda vmin, vmax: cm.LinearColormap(COLORS, vmin=vmin, vmax=vmax)
MAIN_FEATURE = 'symptom_ratio_weighted'


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
        symptom_ratio_wei_norm = df[MAIN_FEATURE]

        def cache_date(gdf, folder, fn=None):
            gdf = gdf.reset_index()
            if fn is None:
                fn = (ANCHOR_DATE + pd.Timedelta(days=gdf['date'].iloc[0])).strftime(format="%Y-%m-%d")
            gdf.rename(columns={'mean': 'prediction'}).to_csv(os.path.join(PROCESSED_DATA_DIR, 'hasadna', folder, fn + '.csv'), index=False)

        def cache_gb(gpb, folder, daily_threshold=50):
            symptom_ratio_wei_norm_agg = gpb.agg(['mean', 'count'])
            symptom_ratio_wei_norm_agg['confidence'] = 1 - np.sqrt((4 * gpb.var(ddof=0)))
            symptom_ratio_wei_norm_agg.loc[symptom_ratio_wei_norm_agg['count'] == 1, 'confidence'] = 0
            # symptom_ratio_wei_norm_agg = symptom_ratio_wei_norm_agg[symptom_ratio_wei_norm_agg >= daily_threshold]#.reindex(symptom_ratio_wei_norm_agg.index)
            symptom_ratio_wei_norm_agg.groupby('date').apply(lambda ddf: cache_date(ddf, folder=folder))
            ld = symptom_ratio_wei_norm_agg.index.get_level_values('date').max()
            cache_date(symptom_ratio_wei_norm_agg.query("date == @ld"), folder=folder, fn='latest')

        cache_gb(symptom_ratio_wei_norm.groupby(['date', 'city_id', 'city_heb', 'city_eng']), folder='city', daily_threshold=50)
        cache_gb(symptom_ratio_wei_norm.groupby(['date', 'neighborhood_id']), folder='neighborhood', daily_threshold=10)


        pass

    def process_data(self, data):
        # data = ProcessedData.convert_new_processed_bot_and_questionnaire(data)
        data = ProcessedData.convert_processed_united(data)
        self.cache_hasadna(data)
        data[features_to_perc] = data[features_to_perc] * 100
        return data

    @staticmethod
    def day_roller(gdf, max_day, ndays=NDAYS_SMOOTHING):
        # append the dataframe with itself shifted 1, 2, ..., ndays forward
        add_dfs = []
        for i in range(1, ndays):
            tmp = gdf.copy()
            tmp['date'] = tmp['date'] + i
            add_dfs.append(tmp)
        undf = gdf.append(add_dfs).query('date <= @max_day')
        # calculate mean for columns with enough observations
        return undf

    # def aggregate_polygon(self, gdf, level):
    #     undf = self.day_roller(gdf)
    #     min_obs = MIN_OBSERVATIONS_CITY if level == 'city_id' else MIN_OBSERVATIONS_NEIGHBORHOOD
    #     gpb = undf.groupby('date', as_index=False, group_keys=False)
    #     undf = gpb.apply(lambda x: mean_if_enough_observations(x, min_observations=min_obs))
    #     return undf

    # def count_polygon(self, gdf):
    #     undf = self.day_roller(gdf).set_index('date', append=True)
    #     return undf.groupby('date').count()

    @staticmethod
    def filter_by_min_obs(df, min_obs):
        return df[df[(df.columns[0][0], 'count')] >= min_obs]

    def count_filter(self, agg_data, level):
        log_.info('Filtering by min count...')
        if level == 'city_id':
            min_obs = MIN_OBSERVATIONS_CITY
        elif level == 'neighborhood_id':
            min_obs = MIN_OBSERVATIONS_NEIGHBORHOOD
        else:
            raise ValueError('level value is not allowed')
        agg_data = agg_data.groupby(level=0, axis=1).apply(lambda g: self.filter_by_min_obs(g, min_obs=min_obs))
        assert agg_data.columns.nlevels == 3
        agg_data.columns = agg_data.columns.droplevel(0)
        return agg_data

    def aggregate_data(self, data, level):
        log_.info('Start processing level {}'.format(level))
        log_.info('Rolling...')
        if 'Unnamed: 0.1' in data.columns:
            data = data.drop(columns=['Unnamed: 0.1'])
        max_day = data.index.get_level_values('date').max()
        rolled = data.select_dtypes('number').reset_index('date')\
            .groupby(level, as_index=False, group_keys=False)\
            .apply(lambda gdf: self.day_roller(gdf, max_day=max_day))
        log_.info('Aggregating...')
        gpb = rolled.set_index('date', append=True).groupby([level, 'date'])
        agg_data = gpb.agg(['mean', 'count'])
        log_.info('Adding weighted symptoms ratio std...')
        agg_data = agg_data.join(gpb[MAIN_FEATURE].agg('std').rename((MAIN_FEATURE, 'std')))
        return agg_data, gpb

    def flatten_add_colors(self, agg_data, gpb):
        agg_data.columns = ['_'.join(col).replace('_mean', '') for col in agg_data.columns.values]
        colors = agg_data.filter(regex="^(?!.*_count).*$").groupby('date').apply(lambda gdf: gdf.apply(self.get_color_series))  # columns not containing colors
        agg_data = agg_data.join(colors, rsuffix='_color')
        log_.info('Aggregating finished!')
        return agg_data

    def geodata_hasadna(self, df, gpdf, centers, tag):
        df = df[[(MAIN_FEATURE, 'mean'), (MAIN_FEATURE, 'count'), (MAIN_FEATURE, 'std')]]
        df.columns = df.columns.droplevel(0)
        latest = (pd.Timestamp.today().normalize() - ANCHOR_DATE).days - 1  # last day
        df = df.query("date == @latest")
        assert df.index.names == [tag + '_id', 'date']
        df.index.names = ['id', 'date']
        df = gpdf.join(df).join(centers).reset_index()
        df['count'] = df['count'].fillna(0)
        if tag == 'city':
            cond0 = df['count'] < 100
        elif tag == 'neighborhood':
            cond0 = df['count'] < 10
        else:
            raise ValueError("tag is not recognized!")
        cond05 = ~cond0 & (df['count'] / df['population'] < 0.01)
        df['confidence'] = 1
        df.loc[cond0, 'confidence'] = 0
        df.loc[cond05, 'confidence'] = 0.5
        # quant_trans = lambda ser: ((ser - ser.quantile(0.05)) / ser.quantile(0.95)).clip(0, 1)
        # df['factor_relnum'] = df['count'] / df['population']
        # df['factor_variance'] = 1 / np.exp(-(df['std']) / 100)
        # df['confidence'] = quant_trans(df['factor_relnum'] * df['factor_variance'])
        # df['confidence'] = df['confidence'].fillna(0)
        # df['mean'] = df['mean'] / 100
        df = df.rename(columns={'confidence': 'latest_confidence',
                                'mean': 'latest_ratio',
                                'count': 'latest_reports',
                                })
        valid_cols = ['id', 'city_eng', 'latest_ratio', 'latest_confidence', 'latest_reports', 'population', 'center', 'geometry']
        df = df.filter(valid_cols)
        # df[['latest_ratio', 'latest_confidence']] = (df[['latest_ratio', 'latest_confidence']] * 100).round()
        log_.info('geopandas shape {}'.format(df.shape))
        # gpd.GeoDataFrame(df).to_file(os.path.join(DASH_CACHE_DIR, tag + '_hasadna.json'), driver="GeoJSON")
        # with open(os.path.join(DASH_CACHE_DIR, tag + '_hasadna2.json'), 'w') as f:
        #     f.write(gpd.GeoDataFrame(df).to_json())
       #  df.query('city_eng == "JERUSALEM"')[['id', 'city_eng', 'latest_ratio', 'latest_confidence', 'latest_reports',
       # 'geometry']].to_file(os.path.join(DASH_CACHE_DIR, tag + '_test.geojson'), driver="GeoJSON")
        df[['id', 'city_eng', 'population', 'latest_ratio', 'latest_confidence', 'latest_reports',
       'geometry']].to_file(os.path.join(DASH_CACHE_DIR, tag + '_hasadna.geojson'), driver="GeoJSON")
       #  gpd.GeoDataFrame(df[['id', 'city_eng', 'latest_ratio', 'latest_confidence', 'latest_reports',
       # 'center']]).to_file(os.path.join(DASH_CACHE_DIR, tag + '_centers_hasadna.json'), driver="GeoJSON")

    def convert_cache_lamas(self, df, gpdf, centers, tag):
        # write only meaningful polygons (intersect with agg_data)
        valid_ids = df.index.get_level_values(tag + '_id').unique()
        gpdf = gpdf.query("id in @valid_ids")
        gpdf.to_file(os.path.join(DASH_CACHE_DIR, tag + '_polygons.json'), driver="GeoJSON")
        centers = centers.query("id in @valid_ids")
        centers['lat'] = centers['geometry'].y
        centers['lon'] = centers['geometry'].x
        centers.reset_index().drop(columns=['geometry']).to_feather(os.path.join(DASH_CACHE_DIR, tag + '_centers.feather'))
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
    def get_color_series(ser):
        f = COLOR_MAP(vmin=ser.min(), vmax=ser.max())
        color_ser = ser.dropna().apply(f).str[:7]
        return color_ser.reindex(ser.index)

    # def concat_agg_count(self, count_df, agg_df):
    #     color_df = agg_df.groupby('date').apply(lambda gdf: gdf.apply(self.get_color_series))
    #     df = agg_df.join(count_df, rsuffix='_count').join(color_df, rsuffix='_color')
    #     return df

    @staticmethod
    def read_lamas_city():
        lamas_fn = os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp')
        gpdf = gpd.read_file(lamas_fn, encoding='utf-8')[['OBJECTID_1', 'geometry', 'Pop_Total']]
        gpdf = gpdf.rename(columns={'OBJECTID_1': 'id', 'Pop_Total': 'population'})
        return gpdf

    @staticmethod
    def read_lamas_neighborhood():
        lamas_fn = os.path.join(LAMAS_DATA, 'neighbor_polygons.shp')
        gpdf = gpd.read_file(lamas_fn, encoding='utf-8')[['OBJECTID_1', 'geometry', 'pop_thou']]
        gpdf['population'] = gpdf['pop_thou'] * 1000
        gpdf = gpdf.drop(columns=['pop_thou']).rename(columns={'OBJECTID_1': 'id'})
        return gpdf

    @staticmethod
    def process_lamas(gpdf):
        gpdf = gpdf.query('population > 0')
        gpdf['geometry'] = gpdf['geometry'].apply(lambda x: x.buffer(0))
        centers = gpdf.set_index('id')[['geometry']]
        centers = ProcessedData.add_city_name(centers)
        centers['geometry'] = centers['geometry'].centroid
        return gpdf, centers

    def routine(self, data, tag):
        level = tag + '_id'
        if tag == 'city':
            gpdf = self.read_lamas_city()
        else:
            gpdf = self.read_lamas_neighborhood()
        gpdf, centers = self.process_lamas(gpdf)
        agg_data, gpb = self.aggregate_data(data, level=level)
        self.geodata_hasadna(df=agg_data, gpdf=gpdf.set_index('id'), centers=centers.rename(columns={'geometry': 'center'}), tag=tag)
        agg_data = self.count_filter(agg_data=agg_data, level=level)
        agg_data = self.flatten_add_colors(agg_data, gpb=gpb)
        self.write_to_cache(agg_data, tag=tag)
        self.convert_cache_lamas(df=agg_data, gpdf=gpdf, centers=centers, tag=tag)

    def cache_city(self, data):
        self.routine(data, tag='city')

    def cache_neighborhood(self, data):
        self.routine(data, tag='neighborhood')

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
