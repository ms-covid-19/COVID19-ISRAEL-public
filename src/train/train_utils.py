import os
import pandas as pd
import numpy as np

from src.utils.aggregations import questionnaire_summary, aggregate_regions, filter_regions
from src.utils.lamas_features import create_city_features, create_neighborhood_features
from src.train.constants import DATE_COL, TIME_COL
from src.train.train_config import agg_col, MINIMUM_PER_REGION, y_col_name
from src.utils.CONSTANTS import CITY_ID_COL, NEIGHBORHOOD_ID_COL, LAMAS_ID_COL, PATIENT_COL
from config import LAMAS_NEIGHBERHOOD_DATA, LAMAS_CITIES_DATA
from config import LAMAS_DATA
from config import UNIFIED_FORMS_FILE, PATIENTS_CITY_DATE_FILE, PATIENTS_CITY_DATE_FILE_NON_ZERO, PATIENTS_CITY_DATE_FILE_SAAR


def add_date_column(all_data_df, time_col_name=TIME_COL, date_col_name=DATE_COL, separator='T'):
    all_data_df[date_col_name] = all_data_df[time_col_name].apply(lambda x: x.split(separator)[0])
    return all_data_df


def add_lamas_features(aggregated_df):
    agg_columns = aggregated_df.columns.to_list()

    if agg_col == CITY_ID_COL:
        if not os.path.exists(LAMAS_CITIES_DATA):
            create_city_features(LAMAS_DATA)
        lamas_features_df = pd.read_csv(os.path.join(LAMAS_CITIES_DATA), index_col=0)
        aggregated_df = aggregated_df.merge(lamas_features_df, left_on=agg_col,
                                            right_on=LAMAS_ID_COL).drop([LAMAS_ID_COL], axis=1)
        agg_columns = aggregated_df.columns.to_list()
        if os.path.exists(os.path.join(LAMAS_DATA, 'dem_df.csv')):
            dem_features_df = pd.read_csv(os.path.join(LAMAS_DATA, 'dem_df.csv'), index_col=0)
            aggregated_df = aggregated_df.merge(dem_features_df, left_on=agg_col, right_on=CITY_ID_COL)
            agg_columns = aggregated_df.columns.to_list()
        else:
            print(os.path.join(LAMAS_DATA, 'dem_df.csv') + ' does not exist. Features are ignored.')

        if os.path.exists(os.path.join(LAMAS_DATA, 'total_pop_by_city.csv')):
            total_pop_df = pd.read_csv(os.path.join(LAMAS_DATA, 'total_pop_by_city.csv'), index_col=0)
            aggregated_df = aggregated_df.merge(total_pop_df, left_on=agg_col, right_on=CITY_ID_COL)
            agg_columns = aggregated_df.columns.to_list()
        else:
            print(os.path.join(LAMAS_DATA, 'total_pop_by_city.csv') + ' does not exist. Features are ignored.')

    elif agg_col == NEIGHBORHOOD_ID_COL:
        if not os.path.exists(LAMAS_NEIGHBERHOOD_DATA):
            create_neighborhood_features(LAMAS_DATA)
        lamas_features_df = pd.read_csv(os.path.join(LAMAS_NEIGHBERHOOD_DATA), index_col=0)
        aggregated_df = aggregated_df.merge(lamas_features_df, left_on=agg_col,
                                            right_on=LAMAS_ID_COL).drop([LAMAS_ID_COL], axis=1)
        agg_columns = aggregated_df.columns.to_list()

    return agg_columns, aggregated_df


def create_accumulative_df(all_data_df, min_per_region=MINIMUM_PER_REGION, mode='upto', n_days=3,
                           aggregation_dates=None, **kwargs):
    # possible modes: 'single', 'upto', ('range' with n_days)

    # todo use variables from function header and constants as default
    if aggregation_dates is None:
        aggregation_dates = all_data_df[DATE_COL].unique()

    min_date = np.min(aggregation_dates)

    slicer = pd.IndexSlice

    tmp_aggregated_df = aggregate_regions(all_data_df[all_data_df[DATE_COL] <= min_date], region_id_col=agg_col,
                                          filter_region_func=filter_regions, minimum_questionnaires=min_per_region,
                                          aggregation_func=questionnaire_summary)

    agg_columns, tmp_aggregated_df = add_lamas_features(tmp_aggregated_df)
    agg_columns = tmp_aggregated_df.columns.to_list()

    index_names = pd.MultiIndex.from_product([sorted(all_data_df[agg_col].unique()),
                                              sorted(aggregation_dates)], names=[agg_col, DATE_COL])
    full_df = pd.DataFrame(index=index_names, columns=agg_columns)

    for id_ld, last_date in enumerate(sorted(aggregation_dates)):
        if mode == 'single':
            dates_included_list = [last_date]
        elif mode == 'upto':
            dates_included_list = [item for item in sorted(aggregation_dates) if item <= last_date]
        elif mode == 'range':
            dates_included_list = [item for item in sorted(aggregation_dates) if item <= last_date]
            if len(dates_included_list) > n_days:
                dates_included_list = dates_included_list[-1*n_days:]
        else:
            raise ValueError

        tmp_data_df = all_data_df[all_data_df[DATE_COL].isin(dates_included_list)]
        tmp_aggregated_df = aggregate_regions(tmp_data_df, region_id_col=agg_col, filter_region_func=filter_regions,
                                              minimum_questionnaires=min_per_region,
                                              aggregation_func=questionnaire_summary)

        if len(tmp_aggregated_df) == 0 and agg_col == CITY_ID_COL:
            # add confirmed_cases
            tmp_aggregated_df = aggregate_regions(tmp_data_df, region_id_col=agg_col, filter_region_func=None,
                                                  aggregation_func=questionnaire_summary)
            tmp_aggregated_df = tmp_aggregated_df[[agg_col,'confirmed_cases']]

        agg_columns, tmp_aggregated_df = add_lamas_features(tmp_aggregated_df)
        agg_columns = tmp_aggregated_df.columns.to_list()

        full_df.loc[slicer[tmp_aggregated_df[agg_col], last_date], agg_columns] = tmp_aggregated_df.values

    if agg_col == CITY_ID_COL:
        full_df['norm_confirmed_cases'] = (full_df['confirmed_cases'].divide(full_df['Pop_Total']).dropna()) * 1000
        full_df['norm_responses'] = (full_df['N_aggregated'].divide(full_df['Pop_Total']).dropna()) * 1000

    return full_df


def get_all_data_df(all_data_df=UNIFIED_FORMS_FILE):
    if isinstance(all_data_df, str):
        all_data_df = pd.read_csv(all_data_df)
    elif isinstance(all_data_df, pd.DataFrame):
        pass
    else:
        raise ValueError

    if ('gender' in all_data_df.columns.to_list() ) and ('M' in all_data_df['gender'].unique()):
        all_data_df['gender'] = (all_data_df['gender'] == 'M').astype(int) # todo change column names to constants from Amit
    if TIME_COL in all_data_df.columns.to_list():
        # add time step column - currently by date
        all_data_df = add_date_column(all_data_df, date_col_name=DATE_COL)


    if agg_col == CITY_ID_COL:
        patients_df = pd.read_csv(PATIENTS_CITY_DATE_FILE)
        #patients_df = pd.read_csv(PATIENTS_CITY_DATE_FILE_SAAR)
        all_data_df = pd.merge(all_data_df, patients_df, how='outer', on=[agg_col,DATE_COL])

    return all_data_df


def clip_to_range(data_df, lower_cut=None, upper_cut=None, col=DATE_COL):

    if lower_cut is not None:
        data_df = data_df[data_df[col] >= lower_cut]
    if upper_cut is not None:
        data_df = data_df[data_df[col] < upper_cut]

    return data_df
