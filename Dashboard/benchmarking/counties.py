import logging
from urllib.request import urlopen
import json

log_ = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


if __name__ == '__main__':
    counties_link = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
    counties_link2 = 'https://raw.githubusercontent.com/dkolobok/dkolobok/master/counties.json'
    with urlopen(counties_link) as response:
        counties = json.load(response)
    counties_fn = 'C:/Users/dkolobok/cache/counties.json'
    with open(counties_fn, 'w') as outfile:
        json.dump(counties, outfile)

    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                       dtype={"fips": str})

    import plotly.express as px

    log_.info('Plotlying...')
    fig = px.choropleth_mapbox(df, geojson=counties_link2, locations='fips', color='unemp',
                               color_continuous_scale="Viridis",
                               range_color=(0, 12),
                               mapbox_style="carto-positron",
                               zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                               opacity=0.5,
                               labels={'unemp':'unemployment rate'}
                              )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    log_.info('Done...')
    fig.write_html('c:/users/dkolobok/cache/counties.html')