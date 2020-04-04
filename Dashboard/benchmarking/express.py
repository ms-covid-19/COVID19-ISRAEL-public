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

if __name__ == '__main__':
    log_.info('Filtering the data...')
    df = df.query('date == @selected_date')
    log_.info('Filtering polygons...')
    polygons['features'] = [polygons['features'][i] for i in range(len(polygons['features'])) if polygons['features'][i]['id'] in df[loc_col].to_list()]
    log_.info('Plotlying...')
    newcolorby = " "
    df = df.rename(columns={colorby: newcolorby})
    fig_choropleth = px.choropleth_mapbox(df, geojson=polygons, locations=loc_col, color=newcolorby,
                                          color_continuous_scale=color_scale,
                                          zoom=6, center={"lat": 32.0853, "lon": 34.7818},
                                          opacity=OPACITY,
                                          mapbox_style="carto-positron",
                                          )

    fig_choropleth.update_layout(height=800, autosize=True)
    log_.info('Figure ready!')
    fig_choropleth.write_html('c:/users/dkolobok/cache/dash_express.html')
    log_.info('Dumped!')
