import logging
import os

from Dashboard.data import max_date
import pandas as pd
import plotly.express as px

from config import DASH_CACHE_DIR
from src.utils.CONSTANTS import COLORS

log_ = logging.getLogger(__name__)
OPACITY = 0.5


def build_map_px(df, polygons, loc_col, colorby, color_scale, opacity):
    fig_choropleth = px.choropleth_mapbox(df, geojson=polygons, locations=loc_col, color=colorby,
                                          color_continuous_scale=color_scale,
                                          zoom=6, center={"lat": 32.0853, "lon": 34.7818},
                                          opacity=opacity,
                                          mapbox_style="carto-positron",
                                          )

    fig_choropleth.update_layout(height=800, autosize=True)
    return fig_choropleth


def build_map(centers, geolayers, color_bar_dict):
    data = [
        dict(
            lat=centers["lat"],
            lon=centers["lon"],
            text=centers["hover"],
            type="scattermapbox",
            marker=color_bar_dict,
        )
    ]

    # annotations = [
    #     dict(
    #         showarrow=False,
    #         align="right",
    #         text="<b>Age-adjusted death rate<br>per county per year</b>",
    #         font=dict(color="#2cfec1"),
    #         bgcolor="#1f2630",
    #         x=0.95,
    #         y=0.95,
    #     )
    # ]
    #
    # BINS = [
    #     "0-2",
    #     "2.1-4",
    #     "4.1-6",
    #     "6.1-8",
    #     "8.1-10",
    #     "10.1-12",
    #     "12.1-14",
    #     "14.1-16",
    #     "16.1-18",
    #     "18.1-20",
    #     "20.1-22",
    #     "22.1-24",
    #     "24.1-26",
    #     "26.1-28",
    #     "28.1-30",
    #     ">30",
    # ]
    #
    # DEFAULT_COLORSCALE = [
    #     "#f2fffb",
    #     "#bbffeb",
    #     "#98ffe0",
    #     "#79ffd6",
    #     "#6df0c8",
    #     "#69e7c0",
    #     "#59dab2",
    #     "#45d0a5",
    #     "#31c194",
    #     "#2bb489",
    #     "#25a27b",
    #     "#1e906d",
    #     "#188463",
    #     "#157658",
    #     "#11684d",
    #     "#10523e",
    # ]
    # cm = dict(zip(BINS, DEFAULT_COLORSCALE))
    # for i, bin in enumerate(reversed(BINS)):
    #     color = cm[bin]
    #     annotations.append(
    #         dict(
    #             arrowcolor=color,
    #             text=bin,
    #             x=0.95,
    #             y=0.85 - (i / 20),
    #             ax=-60,
    #             ay=0,
    #             arrowwidth=5,
    #             arrowhead=0,
    #             bgcolor="#1f2630",
    #             font=dict(color="#2cfec1"),
    #         )
    #     )

    # colorbar_trace = go.Scatter(x=[None],
    #                             y=[None],
    #                             mode='markers',
    #                             marker=dict(
    #                                 colorscale='Reds',
    #                                 showscale=True,
    #                                 cmin=-5,
    #                                 cmax=5,
    #                                 colorbar=dict(thickness=10, tickvals=[-5, 5], ticktext=['Low', 'High']),
    #                             ),
    #                             hoverinfo='none'
    #                             )
    #
    # fig.add_trace(colorbar_trace)

    layout = dict(
        mapbox=dict(
            layers=geolayers, #list(geolayers.values()),
            style="carto-positron",
            center=dict(lat=32.0853, lon=34.7818),
            zoom=6,
        ),
        # annotations=annotations,
        hovermode="closest",
        margin=dict(r=0, l=0, t=0, b=0), height=800, autosize=True
    )

    fig = dict(data=data, layout=layout)
    return fig


def load_figure(geolayers, centers, level, colorby, selected_date, lang):
    if selected_date is None:
        selected_date = max_date
    else:
        selected_date = pd.Timestamp(selected_date)
    log_.info('Reading feather...')
    colorcol = colorby + '_color'
    level_col = level + '_id'
    df = pd.read_feather(os.path.join(DASH_CACHE_DIR, level + '.feather'), columns=['date', level_col, colorby, colorcol])
    log_.info('Filtering the data...')
    df = df.query('date == @selected_date').reset_index(drop=True)
    log_.info('Merging the data...')
    centers = pd.merge(centers, df[[level + '_id', colorby]], left_on='id', right_on=level + '_id', how='right')
    if level == 'city':
        centers['hover'] = centers['_'.join([level, lang])] + ' ' + centers[colorby].round(2).fillna(' ').astype(str)
    else:
        centers['hover'] = centers['_'.join([level, 'id'])].astype(str) + ' ' + centers[colorby].round(2).fillna(' ').astype(str)
    log_.info('Filtering polygons and assigning colors...')
    geolayers = [dict(geolayers[str(df[level + '_id'].iloc[i])], color=df[colorcol].iloc[i][:-2]) for i in range(len(df))]
    # geolayers = list(geolayers.values())

    # geolayers = [geolayers[str(i)] for i in range(len(df))]


    # polygons['features'] = [polygons['features'][i] for i in range(len(polygons['features'])) if polygons['features'][i]['id'] in df[loc_col].to_list()]
    # polygons_fn = 'c:/users/dkolobok/cache/' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + '.json'
    # with open(polygons_fn, 'w') as outfile:
    #     json.dump(polygons, outfile)



    # log_.info('Colorbar dict...')
    # COLOR_MAP
    color_bar_dict = dict(
        size=5, color="white", opacity=0,
         colorscale=[[i / (len(COLORS) - 1), COLORS[i]] for i in range(len(COLORS))],
         showscale=True,
         cmin=df[colorby].min(),
         cmax=df[colorby].max(),
         colorbar=dict(thickness=10),
         )
    log_.info('Plotlying...')
    # newcolorby = " "
    # df = df.rename(columns={colorby: newcolorby})
    # fig = build_map_px(df=df, loc_col=loc_col, colorby=newcolorby, color_scale=color_scale, opacity=OPACITY)
    fig = build_map(centers=centers, geolayers=geolayers, color_bar_dict=color_bar_dict)
    log_.info('Done!')
    return fig
