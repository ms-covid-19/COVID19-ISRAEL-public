import plotly.graph_objects as go

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
loc_col = 'city_id'
selected_date = max_date
df = pd.read_feather(os.path.join(DASH_CACHE_DIR, 'city.feather'), columns=['date', 'city_id', 'city_heb', colorby, colorby + "_color", colorby + "_count"])

if __name__ == '__main__':
    # log_.info('Filtering the data...')
    # df = df.query('date == @selected_date')
    log_.info('Plotlying...')
    # df['day'] = df['date'].dt.strftime("%Y-%m-%d")
    df['day'] = pd.Categorical(df['date'].dt.strftime("%Y-%m-%d"), categories=df['date'].sort_values().drop_duplicates().dt.strftime("%Y-%m-%d"), ordered=True)
    df['day_num'] = (df['date'] - ANCHOR_DATE).dt.days
    df[colorby + '_color'] = df[colorby + '_color'].str[:7]

    top_20 = df.groupby('city_heb')[colorby + "_count"].mean().sort_values(ascending=False).head(20).index.tolist()
    df = df.query('city_heb in @top_20')

    def custom_fun(gdf):
        day_num = gdf['day_num'].iloc[0]
        gdf = gdf.reindex(top_20)
        gdf['day_num'] = day_num
        return gdf

    df = df.set_index(['city_heb']).groupby('day_num', as_index=False, group_keys=False).apply(custom_fun).reset_index()

    def get_trace_daynum(df, i):
        df_day = df.query("day_num == @i")
        return go.Bar(
            y=df_day['city_heb'],
            x=df_day[colorby],
            orientation='h',
            marker_color=df_day[colorby + '_color'].str[:7]  # marker color can be a single color value or an iterable
        )

    fig = go.Figure(
        data=[get_trace_daynum(df, 0)],
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None])])],
            # xaxis={'categoryorder':'category ascending', 'categoryarray':top_20}
        ),
        # frames=[go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])]),
        #         go.Frame(data=[go.Scatter(x=[1, 4], y=[1, 4])]),
        #         go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])],
        #                  layout=go.Layout(title_text="End Title"))]
    )
    frames = []
    for i in df['day_num'].sort_values().unique():
        frames.append(go.Frame(data=[get_trace_daynum(df, i)],
                               # layout=dict(xaxis={'categoryorder':'category ascending', 'categoryarray':top_20})
                               ))

    fig['frames'] = frames

    fig.write_html('c:/users/dkolobok/cache/dash_barplot.html')


# fig = go.Figure(data=[go.Bar(
#     x=' ' + df[id].astype(str),
#     y=df[colorby],
#     # y=df[colorby + '_count'],
#     marker_color=df[colorby + '_color'].str[:7]  # marker color can be a single color value or an iterable
# )], layout=dict(autosize=True, yaxis_title=feature_translations[colorby]['heb' + "_short"]))


# fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
# fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 'categoryarray':['d','a','c','b']})
