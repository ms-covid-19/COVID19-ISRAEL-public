import json
import logging
import os

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from Dashboard.data import color_scale, max_date
from Dashboard.functions.load_figures import OPACITY
from config import DASH_CACHE_DIR

log_ = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

colorby = 'age'
loc_col = 'city_id'
selected_date = max_date
df = pd.read_feather(os.path.join(DASH_CACHE_DIR, 'city.feather'), columns=['date', 'city_id', colorby])
with open(os.path.join(DASH_CACHE_DIR, 'cities_polygons.json')) as json_file:
    polygons = json.load(json_file)
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNqdnBvNDMyaTAxYzkzeW5ubWdpZ2VjbmMifQ.TXcBE-xg9BFdV2ocecc_7g"
mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"

if __name__ == '__main__':
    log_.info('Filtering the data...')
    df = df.query('date == @selected_date')
    log_.info('Filtering polygons...')
    polygons['features'] = [polygons['features'][i] for i in range(len(polygons['features'])) if polygons['features'][i]['id'] in df[loc_col].to_list()]
    log_.info('Plotlying...')
    newcolorby = " "
    df = df.rename(columns={colorby: newcolorby})

    layout = dict(
        mapbox=dict(
            layers=[],
            accesstoken=mapbox_access_token,
            style=mapbox_style,
            center=dict(lat=32.0853, lon=34.7818),
            zoom=6,
        ),
        hovermode="closest",
        margin=dict(r=0, l=0, t=0, b=0),
        # annotations=annotations,
        dragmode="lasso",
    )

    base_url = "https://raw.githubusercontent.com/jackparmer/mapbox-counties/master/"
    # for bin in BINS:
    geo_layer = dict(
        sourcetype="geojson",
        source=polygons,#base_url + str(year) + "/" + bin + ".geojson",
        type="fill",
        color='red',#cm[bin],
        opacity=0.5,
        # CHANGE THIS
        fill=dict(outlinecolor="#afafaf"),
    )
    layout["mapbox"]["layers"].append(geo_layer)

    fig_choropleth = go.Figure(data=[], layout=layout)


    # fig_choropleth = go.Figure(go.Choroplethmapbox(geojson=polygons, locations=df[loc_col], z=df[newcolorby],
    #                                     colorscale="Viridis", zmin=0, zmax=12,
    #                                     marker_opacity=0.5, marker_line_width=0))
    # fig_choropleth.update_layout(mapbox_style="carto-positron",
    #                   mapbox_zoom=6, mapbox_center={"lat": 32.0853, "lon": 34.7818})

    log_.info('Figure ready!')
    fig_choropleth.write_html('c:/users/dkolobok/cache/dash_opioidlike.html')
    log_.info('Dumped!')
