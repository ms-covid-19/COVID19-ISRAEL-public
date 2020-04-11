import pandas as pd
import numpy as np
from src.utils.CONSTANTS import CITY_ID_COL, NEIGHBORHOOD_ID_COL
from src.train.train_config import city_type, neighborhood_type
from src.Analyses.region_distances import RegionDistances
from config import PROCESSED_DATA_DIR
slicer = pd.IndexSlice
import os



def mean_if_enough_observations(gdf, min_observations):
    """
    :param gdf: multiple row gdf
    :param min_observations: if column count < min_observations, return NaN for that column
    :return: mean for each column
    """
    fdf = gdf.loc[:, gdf.count() >= min_observations]
    fdf = fdf.reindex(columns=gdf.columns)
    return fdf.mean()


def questionnaire_summary(region_questionnaires_df, region_id_col, **kwargs):
    """
    Gets a region questionnaires dataframe and implements the aggregation logic

    Parameters
    ----------
    region_questionnaires_df - region questionnaires answers
    region_id_col - region identifier
    kwargs - keywords required by the implemented logic

    Returns
    -------
    region_dict - dictionary of aggregated values for this region.
    """

    number_of_questionnaires = len(region_questionnaires_df)

    if 'feature_cols' in kwargs.keys():
        feature_cols = kwargs['feature_cols']
    else:
        feature_cols = region_questionnaires_df.select_dtypes(include=['int', 'float']).columns
        EXCLUDED_COLS = ['Unnamed: 0', 'Unnamed: 0.1', 'zip_code', 'lat', 'lng']
        feature_cols = [item for item in feature_cols if item not in EXCLUDED_COLS]

    region_questionnaires_df = region_questionnaires_df.reset_index()
    region_dict = dict(region_questionnaires_df[feature_cols].sum(axis=0, skipna=True))

    if 'symptom_ratio_weighted' in region_questionnaires_df.columns.to_list():
        only_with_symptoms_df = region_questionnaires_df[region_questionnaires_df['symptom_ratio_weighted'] > 0]
        n_with_sr = len(only_with_symptoms_df)
        only_with_symptoms_dict = dict(only_with_symptoms_df[feature_cols].sum(axis=0, skipna=True))

    age_groups_dfs = []
    age_groups_len = []
    age_groups_definition = [(0, 18), (18, 40), (40, 60), (60, 120)]
    for age_group in age_groups_definition:
        age_group_df = region_questionnaires_df[(region_questionnaires_df['age'] >= age_group[0]) &
                                                (region_questionnaires_df['age'] < age_group[1])]
        age_groups_dfs.append(age_group_df)
        age_groups_len.append(len(age_group_df))

    for col in feature_cols:
        region_dict[col] = region_dict[col] / number_of_questionnaires
        if n_with_sr > 0:
            #todo remove columns like city_id, neighborhood_id and district number
            region_dict['ow_sr_' + col] = only_with_symptoms_dict[col] / n_with_sr
        else:
            region_dict['ow_sr_' + col] = 0

        split_to_age_groups_cols = ['symptom_fever', 'symptom_cough', 'symptom_diarrhea', 'symptom_nausea_vomiting',
                                    'symptom_runny_nose', 'symptom_cough_dry', 'symptom_cough_moist']

        if col in split_to_age_groups_cols:
            for id_ag, age_group_df in enumerate(age_groups_dfs):
                new_col_name = 'ag_{}_{}_'.format(age_groups_definition[id_ag][0], age_groups_definition[id_ag][1]) + col
                if age_groups_len[id_ag] > 0:
                    region_dict[new_col_name] = age_group_df[col].sum() / age_groups_len[id_ag]
                else:
                    region_dict[new_col_name] = 0

    region_dict.update({region_id_col: np.unique(region_questionnaires_df[region_id_col])[0]})
    region_dict.update({'N_aggregated': number_of_questionnaires})
    region_dict.update({'N_with_sr': n_with_sr})
    region_dict.update({'With_sr_perc': n_with_sr / number_of_questionnaires})

    for id_ag, n_in_ag in enumerate(age_groups_len):
        new_col_name = 'N_ag_{}_{}'.format(age_groups_definition[id_ag][0], age_groups_definition[id_ag][1])
        region_dict.update({new_col_name: n_in_ag})

    return region_dict


def filter_regions(region_questionnaires_df, **kwargs):
    """
    Gets a region questionnaires dataframe and implements per-region filter logic

    Parameters
    ----------
    region_questionnaires_df - region questionnaires dataframe
    kwargs - keywords required by the implemented logic

    Returns
    -------
    True if the region does not meet the criteria and should be filtered
    False if the region meets the criteria and should not be filtered
    """

    if 'minimum_questionnaires' in kwargs.keys():
        if len(region_questionnaires_df) < kwargs['minimum_questionnaires']:
            return True

    return False


def aggregate_regions(questionnaires_df, region_id_col, filter_region_func=None,
                      aggregation_func=None, **kwargs):
    """
    Gets data of questionnaires answers, and applies filter_region_func and aggregation_func per region_id_col

    Parameters
    ----------
    questionnaires_df - dataframe with questionnaires answers
    region_id_col - col name to filter and aggregate by
    filter_region_func - per-region filter logic
    aggregation_func - per-region filter logic
    kwargs - kwargs which are also passed to filter function and aggregation funcion

    Returns
    -------
    questionnaires_df - filtered and aggregated data frame, one row per region
    """

    if filter_region_func is not None:
        # apply filter func on each part - per region id col and gets filtered or not true\false per region:
        filtered_regions = questionnaires_df.groupby(region_id_col).apply(filter_region_func, **kwargs)
        filtered_regions_list = filtered_regions[filtered_regions.values].index.values
        # updated questionnaires_df to only take the unfiltered regions:
        questionnaires_df = questionnaires_df[~questionnaires_df[region_id_col].isin(filtered_regions_list)]

    if aggregation_func is not None:
        # apply aggregation func on each part - per region id col and gets aggregated dictionary per region:
        aggregated_records = questionnaires_df.groupby(region_id_col).apply(aggregation_func,
                                                                            region_id_col=region_id_col, **kwargs)
        # updated to questionnaires_df by the dictionaries - one line per region id col value with aggregated values
        # per column:
        questionnaires_df = pd.DataFrame.from_records(aggregated_records.values)

    return questionnaires_df

def adjacent_regions_mean(X_df, model_features_list, agg_col):
    """
    Parameters
    ----------
    X_df: multi-indexed pandas dataframe with (CITY_ID,date) as index or some other (region,time) indices
    model_features_list: list of columns from X_df to be used as predictors
    agg_col: string, name of the region index of X_df which the aggregation of the full dataframe was done on.
    region_type: string, 'city' or 'nieghbor' to be passed to RegionDistances class to create intersection matrix

    Returns: pandas dataframe with same features as X_df where for each region the features are the mean features of
    the adjacent regions according to the distance matrix from RegionDistances.
    -------

    """
    # get region_type
    if agg_col == CITY_ID_COL:
        region_type = city_type
    elif agg_col == NEIGHBORHOOD_ID_COL:
        region_type = neighborhood_type
    else:
        Exception('{} is not a region granularity included in RegionDistances class'.format(agg_col))


    X_df = X_df.copy()
    input_dates = X_df.index.get_level_values('date').unique().tolist()
    regions = X_df.index.get_level_values(agg_col).unique().tolist()
    print('loading intersection matrix..')
    if not set(model_features_list).issubset(set(X_df.columns.tolist())):
        raise Exception("all_features not in passed dataframe columns")
    # get intersection matrix
    distance_mat_path = os.path.join(PROCESSED_DATA_DIR, 'intersect_{}.csv'.format(region_type))
    if os.path.exists(distance_mat_path):
        intersect_region = pd.read_csv(distance_mat_path)
    else:
        region_dist_class = RegionDistances(region_type=region_type)
        intersect_region = region_dist_class.intersect

    # change matrix diagonal to false
    intersect_region = intersect_region.set_index(agg_col)
    np_itersect = intersect_region.to_numpy()
    np.fill_diagonal(np_itersect, False)
    intersect_region = pd.DataFrame(np_itersect, index=intersect_region.index, columns=intersect_region.columns)
    intersect_region.reset_index(inplace=True)
    print('changing region features..')

    have_adjacent_regions = []
    for region in regions:
        # get list of all regions numbers adjacent to region
        adjacent_regions = intersect_region[intersect_region[str(region)] == True][agg_col].values
        # filter regions not in X_df from list
        adjacent_regions = list(set(adjacent_regions).intersection(set(regions)))
        if len(adjacent_regions) > 0:
            # create df of only adjacent regions to region
            all_adjacent = X_df.loc[slicer[adjacent_regions, :], model_features_list]
            have_adjacent_regions.append(region)
        # series of the mean of  all columns for a date
            for date in input_dates:
                mean_adjacent = all_adjacent[all_adjacent.index.get_level_values('date') == date].mean(axis=0)
                # change row
                X_df.loc[slicer[region, date], model_features_list] = mean_adjacent
    X_df = X_df.loc[slicer[have_adjacent_regions, :], :]

    return X_df

