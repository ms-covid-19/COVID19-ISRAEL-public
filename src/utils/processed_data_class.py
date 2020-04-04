import os

import pandas as pd

from config import NEW_PROCESSED_DATA_DIR

ANCHOR_DATE = pd.to_datetime('2020-03-14')
INDEX_LEVELS = ['city_id', 'neighborhood_id', 'date', 'time']


class ProcessedData(object):
    """
    Processed data class:
    1 row per 1 questionnaire/bot response
    index levels: <city_id> city id (int), <neighborhood_id> neighborhood id (int),
                  <date> days since anchor data (int), <time> seconds since midnight
    columns: all features
    """

    @staticmethod
    def timeser_to_date_and_time(timeser):
        """
        Splits pandas timeseries into date and time
        :param timeser: pandas timeseries
        :return: date (days since anchor data), time (seconds since midnight)
        """
        since_anchor = timeser - ANCHOR_DATE
        return since_anchor.dt.days, since_anchor.dt.seconds

    @staticmethod
    def date_and_time_to_timeser(date, time):
        """
        Unites date series and time series into pandas timeseries
        :param date (days since anchor data)
        :param time (seconds since midnight)
        :return: timeser: pandas timeseries
        """
        if time is None:
            return ANCHOR_DATE + pd.to_timedelta(date * 86400, unit='s')
        else:
            return ANCHOR_DATE + pd.to_timedelta(date * 86400 + time, unit='s')


    @staticmethod
    def set_index(df, old_index_to_columns=False, sort_index_flag=True):
        """
        Sets columns from INDEX_LEVELS to index
        :param df: df to reindex
        :param old_index_to_columns: True: to reset the old index to columns, False: drop the old index
        :param sort_index_flag: whether to sort index
        :return: reindexed df
        """
        df = df.reset_index(drop=~old_index_to_columns)
        if len(df.columns.intersection(INDEX_LEVELS)) < len(INDEX_LEVELS):
            raise ValueError('Not all new index levels {} are present in columns. If some the index levels were in the old index, make sure old_index_to_columns is set to True'.format(INDEX_LEVELS))
        for i in INDEX_LEVELS:
            if not pd.api.types.is_numeric_dtype(df[i]):
                raise TypeError('Column {} should be numeric!'.format(i))
            elif df[i].dtype != 'int64':
                df[i] = df[i].astype('int64')
        df = df.set_index(INDEX_LEVELS)
        if sort_index_flag:
            df = df.sort_index()
        return df

    @staticmethod
    def convert_new_processed_bot_and_questionnaire(df):
        df = df.rename(columns={'neighbor_id': 'neighborhood_id'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # len(df[(df['timestamp'] >= pd.to_datetime('2020-03-18 00:00:00')) & (df['timestamp'] <= pd.to_datetime('2020-03-20 00:00:00'))]) == 23660
        df['date'], df['time'] = ProcessedData.timeser_to_date_and_time(df['timestamp'])
        df = df.drop(columns=['timestamp', 'city_en', 'district_number', 'city_hb'])
        return ProcessedData.set_index(df, old_index_to_columns=False)

    @staticmethod
    def convert_processed_united(df):
        df = df.rename(columns={'NEIGHBOR_ID': 'neighborhood_id', 'CITY_ID': 'city_id'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'], df['time'] = ProcessedData.timeser_to_date_and_time(df['timestamp'])
        df['gender'] = df['gender'].replace({'M': 1, 'F': 0})

        # tmp = df.query("City_En == 'AFULA'").groupby(['date']).mean().drop(columns=['time']).reset_index()
        # tmp['date'] = ProcessedData.date_and_time_to_timeser(date=tmp['date'], time=None)
        # tmp.to_csv('c:/users/dkolobok/cache/Afula.csv')

        df = df.drop(columns=['timestamp', 'source', 'city', 'City_En',
                              'lat', 'lng', 'District_Number', 'district_en', 'street', 'zip_code'])
        return ProcessedData.set_index(df, old_index_to_columns=False)

    @staticmethod
    def convert_new_processed_all_with_location(df):
        """
        :param df: dataframe loaded from COVID_19-ALL_with_location.csv
        :return: new df
        """
        df = df.rename(columns={'CITY_ID': 'city_id', 'NEIGHBOR_ID': 'neighborhood_id'})
        df['date'], df['time'] = ProcessedData.timeser_to_date_and_time(pd.to_datetime(df['Timestamp']))
        df = df.drop(columns=['Timestamp', 'index', 'City_En', 'City'])
        return ProcessedData.set_index(df, old_index_to_columns=False)

    @staticmethod
    def add_city_name(df, heb=True, eng=True):
        """
        Adds column(s) with city names
        :param df: dataframe
        :param heb: whether to include Hebrew name
        :param eng: whether to include English name
        :return: dataframe with city names
        """
        cities_mtd = pd.read_csv(os.path.join(NEW_PROCESSED_DATA_DIR, 'cities_meta.csv'))
        cities_mtd = cities_mtd.rename(columns={'OBJECTID_1': 'city_id', 'SHEM_YISHU': 'city_heb', 'SHEM_YIS_1': 'city_eng'})
        sel_columns = []
        if heb:
            sel_columns.extend(['city_heb'])
        if eng:
            sel_columns.extend(['city_eng'])
        cities_mtd = cities_mtd.set_index('city_id')[sel_columns]
        return df.join(cities_mtd)


if __name__ == '__main__':
    olddf = pd.read_csv(os.path.join(NEW_PROCESSED_DATA_DIR, 'COVID_19-ALL_with_location.csv'))
    newdf = ProcessedData.convert_new_processed_all_with_location(olddf)
    newdf_with_city_names = ProcessedData.add_city_name(newdf)
    pass