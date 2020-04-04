"""Unifies different form times to one standard format.

Output times are in Israel timezone: 2020-03-30T10:24:59.
"""

import re
from time import strptime, strftime

import pandas as pd

# Supported date formats. Newly encountered forms should be added here.
_templates = [
    '%Y-%m-%dT%H:%M:%S',  # Output format
    '%m/%d/%Y %H:%M:%S',
    '%Y-%m-%dT%H:%M:%S.%f',
]

# Caveat: this only supports digital representation and not verbal fields like
# Jan or Tue.
_regexes = [re.compile('^' + re.compile('%.').sub('\\\\d+', x) + '$')
            for x in _templates]


def unify_time(t: str) -> str:
    """Converts a representation of a time to the form 2020-03-30T10:24:59."""
    for regex, fmt in zip(_regexes, _templates):
        if regex.match(t):
            return strftime(_templates[0], strptime(t, fmt))
    raise ValueError(f'Unsupported time: {t}')


def utc_to_israel(s: pd.Series):
    """Converts timezone-less timestamps that are UTC to timezone-less
    timestamps in Israel time."""
    return pd.to_datetime(s.values).tz_localize('UTC') \
        .tz_convert('Asia/Jerusalem').tz_localize(None) \
        .strftime(_templates[0])
