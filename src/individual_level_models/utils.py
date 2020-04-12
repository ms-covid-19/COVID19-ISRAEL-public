import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer

from src.individual_level_models.defs import *


def load_processed_MOH_TESTS(age_gender_dropna=True, sample_non_symptomatic=True):
    df = pd.read_csv(PROCESSED_FILE_MOH_TESTS, index_col=0, low_memory=False)
    if age_gender_dropna:
        df = df.dropna(subset=['age_over_60', 'gender'])

    df_symptomatic = df[df[SYMPTOMS_MOH_TESTS].sum(axis=1) > 0]
    if sample_non_symptomatic:
        symptomatic_old_count = df_symptomatic[df_symptomatic.age_over_60 == 1].shape[0]
        symptomatic_young_count = df_symptomatic[df_symptomatic.age_over_60 == 0].shape[0]

        asymptomatic_old_count = int((symptomatic_old_count/SYMPTOMATIC_PERC_OLD) - symptomatic_old_count)
        asymptomatic_young_count = int((symptomatic_young_count/SYMPTOMATIC_PERC_YOUNG) - symptomatic_young_count)

        df_asymptomatic = df[df[SYMPTOMS_MOH_TESTS].sum(axis=1) == 0]
        df_asymptomatic_old = df_asymptomatic[df_asymptomatic.age_over_60 == 1].sample(asymptomatic_old_count)
        df_asymptomatic_young = df_asymptomatic[df_asymptomatic.age_over_60 == 0].sample(asymptomatic_young_count)
        df = pd.concat([df_symptomatic, df_asymptomatic_old, df_asymptomatic_young])

    df.rename(columns={'corona_result': Y_COL}, inplace=True)
    df.dropna(subset=[Y_COL], inplace=True)
    return df


def load_processed_MAOZ(age_gender_dropna=True, **kwargs):
    df = pd.read_csv(PROCESSED_FILE_MAOZ, index_col=0, low_memory=False)
    if age_gender_dropna:
        df = df.dropna(subset=['age', 'gender'])
    df['age_over_60'] = df['age'].isin([6, 7]).replace({True: 1, False: 0})
    df.rename(columns={enum_to_column(Isolation.DIAGNOSED): Y_COL}, inplace=True)
    df.dropna(subset=[Y_COL], inplace=True)
    return df


def load_processed_MACCABI(age_gender_dropna=True, **kwargs):
    df = pd.read_csv(PROCESSED_FILE_MACCABI, index_col=0, low_memory=False)
    df.rename(columns={enum_to_column(Isolation.DIAGNOSED): Y_COL}, inplace=True)
    # TODO - deal with age once we get it
    # if age_gender_dropna:
    #     df = df.dropna(subset=['age', 'gender'])
    #
    df.dropna(subset=[Y_COL], inplace=True)
    return df


def load_processed_FORMS(age_gender_dropna=True, single_entry_per_person=True, **kwargs):
    df = pd.read_csv(PROCESSED_FILE_FORMS, index_col=0, low_memory=False)
    if age_gender_dropna:
        df = df.dropna(subset=['age', 'gender'])
    if single_entry_per_person:  # TODO: change when diagnosed column is fixed
        df = shuffle(df[df[enum_to_column(Isolation.DIAGNOSED)] != 1]).drop_duplicates(INDIVIDUAL_COLS_FORMS)
        # df = df.groupby(INDIVIDUAL_COLS_FORMS).apply(lambda x: \
        #      x.sample(n=1) if x[enum_to_column(Isolation.DIAGNOSED)].sum() == 0
        #      else x[x[enum_to_column(Isolation.DIAGNOSED)].sum() == 1].sort_values('timestamp').iloc[0])
    df.rename(columns={enum_to_column(Isolation.DIAGNOSED): Y_COL}, inplace=True)
    df['age_over_60'] = (df['age'] > 60).replace({True: 1, False: 0})
    df['gender'] = df['gender'].replace({'F': 0, 'M': 1})
    df.dropna(subset=[Y_COL], inplace=True)
    return df


def load_processed_FORMS_MOH_Tests(**kwargs):
    forms = load_processed_FORMS(**kwargs)
    moh = load_processed_MOH_TESTS(sample_non_symptomatic=False, **kwargs)
    forms = forms[forms[Y_COL] == 0]
    moh = moh[moh[Y_COL] == 1]
    overlapped_cols = forms.columns.intersection(moh.columns)
    df = pd.concat([forms[overlapped_cols], moh[overlapped_cols]])
    return df


def add_interactions(df, col_list, age_col='age', gender_col='gender'):
    df_with_interactions = df.copy()
    for col in col_list:
        if col not in ['age', 'gender']:
            df_with_interactions['{}*age'.format(col)] = df_with_interactions[col] * df_with_interactions[age_col]
            df_with_interactions['{}*sex'.format(col)] = df_with_interactions[col] * df_with_interactions[gender_col]
    return df_with_interactions


def get_imputed_data(df, x_cols=X_COLS):
    imp = SimpleImputer(strategy="most_frequent")
    df.loc[:, x_cols] = imp.fit_transform(df.loc[:, x_cols])
    return df

