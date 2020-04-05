"""Sanity checks for preprocessed data. Drops bad data rows and alerts when
something is wrong."""

from pandas import DataFrame


# ----- Main function ----------------------------------------------------------

def sanitize_data(df: DataFrame, *funcs) -> DataFrame:
    """Runs the given sanitation functions on the data.
    Returns a corrected dataframe.

    Browse the source below for the function catalog."""
    print('Starting with ')
    print(f'Starting with {df.shape[0]} rows')
    for f in funcs:
        df = f(df[:])  # [:] creates a copy to not trigger pandas warnings.
    return df


# ----- Sanitation functions ---------------------------------------------------


def check_genders(df: DataFrame) -> DataFrame:
    """Raises an exception if a gender is no M or F."""
    for x in df.gender.values:
        if x not in {'M', 'F'}:
            raise ValueError(f'Bad gender: {x}')
    return df


def remove_bad_ages(df: DataFrame) -> DataFrame:
    """Removes rows with ages that are not between 0-100."""
    df['age'] = [float_or(x, -1) for x in df['age'].values]
    df_good = df[(df['age'] >= 0) & (df['age'] <= 120)]
    print(f'Removed {len(df) - len(df_good)} rows with bad ages')
    return df_good


def remove_bad_temperatures(df: DataFrame) -> DataFrame:
    """Removes rows with temperatures that are not between 35-43.
    Assigns nan to missing values."""
    df['body_temp'] = [float_or(x, -1) for x in df['body_temp'].values]
    df_good = df[(df['body_temp'] >= 35) & (df['body_temp'] <= 43) |
                 (df['body_temp'] == 0)][:]
    print(f'Removed {len(df) - len(df_good)} rows with bad body temperature')
    n_missing = len(df_good[df_good['body_temp'] == 0])
    df_good.loc[df_good['body_temp'] == 0, 'body_temp'] = float('nan')
    print(n_missing, 'entries missing body temperature')
    return df_good


# ----- Utilities --------------------------------------------------------------

def float_or(a, fallback):
    """Returns a as a float or fallback if failed."""
    try:
        return float(a)
    except ValueError:
        return fallback
