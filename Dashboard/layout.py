import os

import dash_html_components as html
from dash_core_components import Dropdown, RadioItems, Tabs, Tab, Graph, Slider, DatePickerRange, DatePickerSingle

from Dashboard.data import min_date, max_date, LANGUAGE
from Dashboard.translations import layout_elements, level_options, feature_translations

html_align_dict = {'heb': 'right', 'eng': 'left'}

main_layout = html.Div([
    html.Div([
        html.Div(id='level_selector_container',
                 children=[html.H3(id='text_data_resolution'),
                           Dropdown(id='level_selector',
                                    value='city'),
                           html.Br(),
                           ],
                 className="six columns"
                 ),

        html.Div(id='date_selector_container',
                 children=[html.H3(id='text_date_selector'),
                           DatePickerSingle(
                               id='date_selector',
                               min_date_allowed=min_date,
                               max_date_allowed=max_date,
                               date=max_date
                           ),
                           html.Br(),
                           ],
                 className="six columns"
                 ),
    ], className="row"),
    html.Div([
        html.Div(id='color_selector_container',
                         children=[html.H3(id='text_color_selector'),
                                   Dropdown(id='color_selector',
                                            value=list(feature_translations.keys())[-1]
                                            ),
                                   html.Br(),
                                   ],
                         className="six columns"
                         ),
        html.Div(id='language_selector_container',
                 children=[
                    Tabs(id="language_tab", value='heb', children=[
                            Tab(label='English', value='eng'),
                            Tab(label='עברית', value='heb'),
                        ]),
                     # html.H3(layout_elements['Language selector'][LANGUAGE],
                     #               style={'textAlign': html_align}),
                     #       Dropdown(id='language_selector',
                     #                options=[
                     #                    {'label': 'Hebrew', 'value': 'heb'},
                     #                    # {'label': 'English', 'value': 'eng'},
                     #                ],
                     #                value='heb'),
                           html.Br(),
                           ],
                 className="six columns"
                 ),
    ], className="row"),
    html.Div([
        html.Div(
                id="map-container",
                children=[
                    Graph(
                        id="choropleth",
                        # figure=dict(
                        #     # data=[
                        #     #     dict(
                        #     #         lat=df_lat_lon["Latitude "],
                        #     #         lon=df_lat_lon["Longitude"],
                        #     #         text=df_lat_lon["Hover"],
                        #     #         type="scattermapbox",
                        #     #     )
                        #     # ],
                        #     layout=dict(
                        #         mapbox=dict(
                        #             layers=[],
                        #             accesstoken=mapbox_access_token,
                        #             style=mapbox_style,
                        #             center=dict(
                        #                 lat=38.72490, lon=-95.61446
                        #             ),
                        #             pitch=0,
                        #             zoom=3.5,
                        #         ),
                        #         autosize=True,
                        #     ),
                        # ),
                    ),

                ], className="six columns"
            ),
        html.Div(
            children=[
                html.H5(id='text_map_description'),
                html.Div(id='pie-container', children=[
                    Graph(id='piechart'),
                    html.H5(id='text_pie_description'),
                ])
            ], className="six columns"
        ),
    ], className="row"),

])
