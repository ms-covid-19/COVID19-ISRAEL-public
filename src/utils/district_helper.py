import pandas as pd

district_ids_names_dict = {
    0: 'Unknown',
    1: 'Jerusalem',
    2: 'North',
    3: 'Haifa',
    4: 'Center',
    5: 'Tel-Aviv',
    6: 'South',
    7: 'West Bank'
}

# this is based on manual google maps and wikipedia searches
unknown_cities_district_ids_dict = {
    None: 0,
    'AL SAYYID': 6,
    'AVENAT': 7,
    "DERIG'AT": 6,
    'EN HAZEVA': 6,
    'HAR AMASA': 6,
    'MAKCHUL': 6,
    'MOLADA*': 6,
    'NURIT': 2,
    'PELEKH': 6,
    'SHITTIM': 6,
    'UMM BATIN': 6,
    "YA'EL": 2,
    # these are known but missing in the patients file
    'ELAT': 6,
    'QIRYAT EQRON': 4,
    'NES ZIYYONA': 4,
    'SEDEROT': 6,
    'BET SHEMESH': 1,
    'NAZERAT ILLIT': 2,
    'ROSH HAAYIN': 4,
}


def find_unknown_district(df, city_col='Shem_Yis_1', district_col='DistrictCo'):
    known_df = df[df[district_col] != 0]
    unknown_df = df[df[district_col] == 0]

    problems = set(unknown_df[city_col]) - set(known_df[city_col]) - set(unknown_cities_district_ids_dict.keys())
    if len(problems) > 0:
        print('I do not know these cities district, please add them to unknown_cities_district_ids_dict\n{}'.format(problems))

    known_cities_district_ids_dict = known_df.drop_duplicates([city_col, district_col]).set_index(city_col)[district_col].to_dict()

    cities_district_ids_dict = known_cities_district_ids_dict.copy()
    cities_district_ids_dict.update(unknown_cities_district_ids_dict)

    new_col = pd.DataFrame(known_df[district_col])\
        .rename(columns={district_col: 'new_'+district_col})
    known_df = pd.concat([known_df, new_col], axis=1)
    new_col = pd.DataFrame(unknown_df[city_col].map(cities_district_ids_dict).fillna(0))\
        .rename(columns={city_col: 'new_'+district_col})
    unknown_df = pd.concat([unknown_df, new_col], axis=1)

    return pd.concat([known_df, unknown_df]).drop([district_col], axis=1).rename(
        columns={'new_' + district_col: district_col})


def add_district_name(df, district_id_col='DistrictCo', district_name_col='district_en'):
    df.loc[:, district_name_col] = df[district_id_col].map(district_ids_names_dict)

    return df
