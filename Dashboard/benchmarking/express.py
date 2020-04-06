import json
import logging
import os

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from Dashboard.data import color_scale, max_date
from Dashboard.functions.map import OPACITY
from Dashboard.translations import feature_translations
from config import DASH_CACHE_DIR
from src.utils.processed_data_class import ANCHOR_DATE

log_ = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

colorby = 'symptom_ratio_weighted'
loc_col = 'neighborhood_id'
selected_date = max_date
df = pd.read_feather(os.path.join(DASH_CACHE_DIR, 'neighborhood.feather'), columns=['date', 'neighborhood_id', colorby, colorby + "_color", colorby + "_count"])
with open(os.path.join(DASH_CACHE_DIR, 'neighborhood_polygons.json')) as json_file:
    polygons = json.load(json_file)

if __name__ == '__main__':
    # log_.info('Filtering the data...')
    # df = df.query('date == @selected_date')
    log_.info('Filtering polygons...')
    polygons['features'] = [polygons['features'][i] for i in range(len(polygons['features'])) if polygons['features'][i]['id'] in df[loc_col].to_list()]
    log_.info('Plotlying...')
    # df['day'] = df['date'].dt.strftime("%Y-%m-%d")
    df['day'] = pd.Categorical(df['date'].dt.strftime("%Y-%m-%d"), categories=df['date'].sort_values().drop_duplicates().dt.strftime("%Y-%m-%d"), ordered=True)
    df['day_num'] = (df['date'] - ANCHOR_DATE).dt.days
    df[colorby + '_color'] = df[colorby + '_color'].str[:7]
    # newcolorby = " "
    # df = df.rename(columns={colorby: newcolorby})
    fig_choropleth = px.choropleth_mapbox(df, #animation_frame="day", animation_group=colorby,
                                          geojson=polygons, locations=loc_col, color=colorby,
                                          color_continuous_scale=color_scale,
                                          zoom=7, center={"lat": 32.0853, "lon": 34.7818},
                                          opacity=OPACITY,
                                          mapbox_style="carto-positron",
                                          )
    #
    # fig_choropleth.update_layout(height=800, autosize=True)
    # log_.info('Figure ready!')
    # fig_choropleth.write_html('c:/users/dkolobok/cache/dash_express.html')
    # log_.info('Dumped!')
    # , color = colorby + "_color"

    # top_20 = df.groupby('city_heb')[colorby + "_count"].mean().sort_values(ascending=False).head(20).index.tolist()
    # fig = px.bar(df.query('city_heb in @top_20').sort_values('day_num'),
    #              x='city_heb', y=colorby, color='city_heb',
    #              animation_frame="day_num", animation_group='city_heb',
    #              # orientation="h"
    #              )
    # fig = px.bar(df.query("city_heb in @top_20").sort_values('day'), y='city_heb', x=colorby, color='city_heb',
    #              animation_frame="day", animation_group='city_heb',
    #              orientation="h")

    # fig = go.Figure(data=[go.Bar(
    #     x=' ' + df[id].astype(str),
    #     y=df[colorby],
    #     # y=df[colorby + '_count'],
    #     marker_color=df[colorby + '_color'].str[:7]  # marker color can be a single color value or an iterable
    # )], layout=dict(autosize=True, yaxis_title=feature_translations[colorby]['heb' + "_short"]))

    fig_choropleth.write_html('c:/users/dkolobok/cache/dash_express.html')
    log_.info('Dumped!')


