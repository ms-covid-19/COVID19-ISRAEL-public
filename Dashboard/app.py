import dash

import dash_html_components as html
from dash_core_components import Dropdown, Tabs, Tab, Graph, DatePickerSingle

from Dashboard.data import min_date, max_date
from Dashboard.translations import feature_translations

app = dash.Dash('COVID19 dash app')
app.layout = html.Div([
    # html.Div(
    #     children=[
    #         html.Div(html.Img(id="wis_logo",
    #                           src='https://www.weizmann.ac.il/pages/sites/default/files/wis_logo_wis_logo_eng_v1_black.png',
    #                           style={'height':'70%', 'width':'70%', 'textAlign': 'left'}),
    #              className="three columns"),
    #         html.Div(html.H2(id='dashboard_title', style={'textAlign': 'center'}),
    #              # className="four columns"
    #                  ),
    #         html.Div(html.Img(id="dash_logo",
    #                           src=app.get_asset_url('dash-logo.bmp'),
    #                           style={'height':'70%', 'width':'70%', 'textAlign': 'right'}),
    #              className="three columns"),
    #               ],
    #     className="row"
    # ),
    html.Div([
        html.Div(id='level_selector_container',
                 children=[html.H4(id='text_data_resolution'),
                           Dropdown(id='level_selector',
                                    value='city'),
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
                     html.Br(),
                 ],
                 className="six columns"
                 ),

    ], className="row"),
    html.Div([
        html.Div(id='color_selector_container',
                         children=[html.H4(id='text_color_selector'),
                                   Dropdown(id='color_selector',
                                            value=list(feature_translations.keys())[-1]
                                            ),
                                   html.Br(),
                                   ],
                         className="six columns"
                         ),
        html.Div(id='date_selector_container',
                 children=[
                     html.H4(id='text_date_selector'),
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
        html.Div(
                id="map-container",
                children=[
                    Graph(
                        id="choropleth",
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

