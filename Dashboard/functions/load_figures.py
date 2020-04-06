import json
import logging
import os
import random
import string

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from Dashboard.data import max_date, color_scale, MAX_CITIES_PIE
from Dashboard.translations import feature_translations
from config import DASH_CACHE_DIR

log_ = logging.getLogger(__name__)


def load_piechart(colorby, selected_date, tag, lang):
    if selected_date is None:
        selected_date = max_date
    if tag == 'city':
        id = tag + "_" + lang
    else:
        id = tag + "_id"
    df = pd.read_feather(os.path.join(DASH_CACHE_DIR, tag + '.feather'), columns=['date', colorby, colorby + '_color', id])
    # df = pd.read_feather(os.path.join(DASH_CACHE_DIR, tag + '.feather'), columns=['date', colorby + '_count', colorby + '_color', id])
    selected_date = pd.to_datetime(selected_date)
    df = df.query("date == @selected_date").sort_values(colorby, ascending=False).head(MAX_CITIES_PIE)
    # df = df.query("date == @selected_date").sort_values(colorby + '_count', ascending=False).head(MAX_CITIES_PIE)
    # fig = go.Figure(data=[go.Pie(labels=df[id],
    #                              values=df[colorby + '_count'],
    #                              opacity=OPACITY)])
    # fig.update_traces(hoverinfo='label+percent',
    #                   # textinfo='value',
    #                   textfont_size=20,
    #                   marker=dict(#colors=df[colorby + '_color'],
    #                               line=dict(color='#000000', width=2))
    #                   )
    fig = go.Figure(data=[go.Bar(
        x=' ' + df[id].astype(str),
        y=df[colorby],
        # y=df[colorby + '_count'],
        marker_color=df[colorby + '_color']  # marker color can be a single color value or an iterable
    )], layout=dict(autosize=True, yaxis_title=feature_translations[colorby][lang + "_short"]))
    return fig


# def load_figure_cities(geolayers, colorby, selected_date, lang):
#     # log_.info('Reading JSON...')
#     # with open(os.path.join(DASH_CACHE_DIR, 'cities_polygons.json')) as json_file:
#     #     polygons = json.load(json_file)
#     log_.info('Importing...')
#     return load_figure(df, geolayers=geolayers, tag='cities', loc_col='city_id', colorby=colorby, color=colorcol, selected_date=selected_date, lang=lang)
#
#
# def load_figure_neighborhoods(geolayers, colorby, selected_date, lang):
#     df = pd.read_feather(os.path.join(DASH_CACHE_DIR, 'neighborhood.feather'), columns=['date', 'neighborhood_id', colorby])
#     df['neighborhood_id'] = df['neighborhood_id'].astype(int)
#     # with open(os.path.join(DASH_CACHE_DIR, 'neighborhoods_polygons.json')) as json_file:
#     #     polygons = json.load(json_file)
#     return load_figure(df, polygons=polygons['cities'], loc_col='neighborhood_id', colorby=colorby, selected_date=selected_date, lang=lang)
