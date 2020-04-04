import json
import logging
import os
import pandas as pd

from dash.dependencies import Output, Input, State

from Dashboard.app import app

from Dashboard.functions.load_figures import load_piechart
from Dashboard.functions.map import load_figure
from Dashboard.layout import html_align_dict
from Dashboard.translations import layout_elements, feature_translations, level_options
from config import DASH_CACHE_DIR

log_ = logging.getLogger(__name__)

log_.info('Reading geo data...')
geolayers_all = {}
centers_all = {}
for level in ['city', 'neighborhood']:
    with open(os.path.join(DASH_CACHE_DIR, level + '_geolayers.json')) as json_file:
        geolayers_all[level] = json.load(json_file)
    centers_all[level] = pd.read_feather(os.path.join(DASH_CACHE_DIR, level + '_centers.feather'))

log_.info('Reading finished...')


@app.callback(
    Output("choropleth", "figure"),
    [Input("level_selector", "value"),
     Input("color_selector", "value"),
     Input('date_selector', 'date'),
     Input('language_tab', 'value'),
     ],
)
def reload_graph(level, colorby, selected_date, lang):
    return load_figure(geolayers=geolayers_all[level], centers=centers_all[level], level=level, colorby=colorby, selected_date=selected_date, lang=lang)


@app.callback(
    Output("piechart", "figure"),
    [
        #Input("level_selector", "value"),
     Input("color_selector", "value"),
     Input('date_selector', 'date'),
     Input('language_tab', 'value'),
     ]
)
def plot_distribution(colorby, selected_date, lang):
    return load_piechart(colorby=colorby, selected_date=selected_date, tag='city', lang=lang)


@app.callback(
    [Output("text_pie_description", "children"),
     Output("text_pie_description", "style")],
    [Input('language_tab', 'value'),
     # Input(component_id="level_selector", component_property='value')
     ]
)
def text_pie_description(lang):#, level):
    return layout_elements['pie_description'][lang], {'textAlign': html_align_dict[lang]}

@app.callback(
    [Output("text_map_description", "children"),
     Output("text_map_description", "style")],
    [Input('language_tab', 'value')]
)
def text_map_description(lang):
    return layout_elements['map_description'][lang], {'textAlign': html_align_dict[lang]}


@app.callback(
    [Output("text_data_resolution", "children"),
     Output("text_data_resolution", "style")],
    [Input('language_tab', 'value')]
)
def text_data_resolution(lang):
    return layout_elements['Data resolution'][lang], {'textAlign': html_align_dict[lang]}


@app.callback(
    [Output("text_date_selector", "children"),
     Output("text_date_selector", "style")],
    [Input('language_tab', 'value')]
)
def text_date_selector(lang):
    return layout_elements['Date selector'][lang], {'textAlign': html_align_dict[lang]}


@app.callback(
    [Output("text_color_selector", "children"),
     Output("text_color_selector", "style")],
    [Input('language_tab', 'value')]
)
def text_color_selector(lang):
    return layout_elements['Color by'][lang], {'textAlign': html_align_dict[lang]}


@app.callback(
    Output("level_selector", "options"),
    [Input('language_tab', 'value')]
)
def options_level_selector(lang):
    return [{'label': level_options[i][lang], 'value': i} for i in ['city', 'neighborhood']]


@app.callback(
    Output("color_selector", "options"),
    [Input('language_tab', 'value')]
)
def options_color_selector(lang):
    return [{'label': feature_translations[f][lang], 'value': f} for f in feature_translations.keys()]


# @app.callback(
#     Output(component_id='pie-container', component_property='style'),
#     [Input(component_id="level_selector", component_property='value')],
#     [State(component_id='pie-container', component_property='style')])
# def show_pie(level, style):
#     if level == 'neighborhoods':
#         return {'display': 'none'}
#         # style['display'] = 'none'
#     return None
