import os

import pandas as pd
import plotly.graph_objects as go

from Dashboard.data import max_date
from Dashboard.translations import feature_translations
from config import DASH_CACHE_DIR, NEW_PROCESSED_DATA_DIR
from src.figures.paths import FIGURES_CACHE_DIR

FEATURE = 'symptom_ratio_weighted'
SELECTED_DATE = max_date


class top_bottom_cities(object):
    def __init__(self):
        self.df = pd.read_feather(os.path.join(DASH_CACHE_DIR, 'city.feather'),
                                  columns=['date', 'city_id', 'city_heb', 'city_eng', FEATURE, FEATURE + '_count',
                                           FEATURE + "_color", FEATURE + "_count"])

    @staticmethod
    def plot_bar(df, folder, fn):
        fig = go.Figure(go.Bar(
            y=df['city_heb'],
            x=df[FEATURE],
            orientation='h',
            marker_color=df[FEATURE + '_color'].str[:7]  # marker color can be a single color value or an iterable
            ),
            layout=dict(
                # title=feature_translations[FEATURE]['heb'],
                xaxis_title=feature_translations[FEATURE]['heb_short'],
                # yaxis_title="y Axis Title"
            ))
        fig.write_html(os.path.join(folder, fn + '.html'))
        fig.write_image(os.path.join(folder, fn + '.png'))

    def make_figures(self):
        lddf = self.df.query("date == @SELECTED_DATE")
        lddf = lddf.sort_values(FEATURE).reset_index(drop=True)
        # lddf.query("city_heb == 'אילת'")
        # lddf.query("city_heb == 'טבריה'")
        # cities_list = ['ביתר עילית', 'אשקלון', 'טבריה', 'ירושלים', 'מודיעין-מכבים-רעות', 'מגדל העמק', 'אלעד', 'בית שמש', 'בני ברק', 'מודיעין עילית', 'רעננה', 'אור יהודה']
        # cities_mtd = pd.read_csv(os.path.join(NEW_PROCESSED_DATA_DIR, 'cities_meta.csv'))
        # tmp = cities_mtd.query("SHEM_YISHU in @cities_list")['SHEM_YISHU']
        # found_cities = tmp.tolist()
        # meta_cities =
        # print(len(lddf))
        # cities_list = ['ביתר עילית', 'אשקלון', 'טבריה', 'ירושלים', 'מודיעין-מכבים-רעות*', 'מגדל העמק', 'אלעד', 'בית שמש', 'בני ברק', 'מודיעין עילית', 'רעננה', 'אור יהודה']
        # tmp = lddf.query("city_heb in @cities_list")[['city_heb', 'city_eng', FEATURE, FEATURE + '_count']].sort_values(FEATURE, ascending=False)
        # tmp.to_csv(os.path.join(FIGURES_CACHE_DIR, 'ranking_all.csv'))
        self.plot_bar(lddf.tail(15), folder=FIGURES_CACHE_DIR, fn='top15_cities')
        lddf.tail(15).sort_values(FEATURE, ascending=False).to_csv(os.path.join(FIGURES_CACHE_DIR, 'top15_cities' + '.csv'))
        self.plot_bar(lddf.head(15), folder=FIGURES_CACHE_DIR, fn='bottom15_cities')
        lddf.head(15).to_csv(os.path.join(FIGURES_CACHE_DIR, 'bottom15_cities' + '.csv'))



if __name__ == '__main__':
    tbc = top_bottom_cities()
    tbc.make_figures()