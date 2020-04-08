# %% Imports
import json
from importlib import reload
import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame
import matplotlib
# notebook or inline:
#%matplotlib notebook
#matplotlib.use('Qt5Agg')  # pipenv install PyQt5 --dev
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
from collections import OrderedDict
#from IPython.display import display
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))


notebook_dir = os.path.split(os.getcwd())[0]
base_dir = notebook_dir[:notebook_dir.find('sandbox') - 1]
print("{0}\n{1}".format(notebook_dir, base_dir))
if notebook_dir not in sys.path:
    sys.path.append(notebook_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)

cv_dataset = r'D:\OneDrive - Microsoft\CoronaIsr\WzData\data'

import env_vars_init
from atlitools.envvars import str_rep_env, env_vars, extract_env_prefix
from atlitools.utils.plotutils import drawnow, nplot, qplot, pseudo_signed_log10, PlotAxesGenerator


class CoronaDataset(object):

    def __init__(self, data_date_str: str = '2020-04-02',
                 drop_nan_city_flag: bool = True,
                 city_filter=None):
        self.by_city = pd.read_csv(os.path.join(
            cv_dataset, 'Processed', 'confirmed_patients', 'confirmed_patients_by_day_and_city.csv'))
        if drop_nan_city_flag:
            self.by_city = self.by_city[~self.by_city.City_En.isna()]
        if city_filter is not None:
            self.by_city = self.by_city[self.by_city.City_En.isin(city_filter)]
        self.by_city.set_index('City_En', inplace=True)

        self.wz_data = pd.read_csv(str_rep_env(os.path.join(
            '%CvDatasets%', 'Weizmann_COVID19_Survey_data_{}.csv'.format(data_date_str))))
        if city_filter is not None:
            self.wz_data = self.wz_data[self.wz_data.City_En.isin(city_filter)]
        self.ref_date = datetime.datetime.strptime('2020-03-14', '%Y-%m-%d')
        self.wz_data['datetime'] = self.wz_data.timestamp.map(
            lambda r: datetime.datetime.strptime(r, '%Y-%m-%dT%H:%M:%S'))
        self.wz_data['date_int'] = self.wz_data.datetime.map(lambda r: (r - self.ref_date).days)
        self.wz_data['date_num'] = self.wz_data.datetime.map(
            lambda r: (r - self.ref_date).total_seconds() / (24*3600))

        self.magen = self.load_hamagen_data(
            os.path.join('%CvDatasets%', 'Hamagen', 'Points_2020-04-03-1714.json'))

        # Some pivot tables:
        self.city_date_wz_cnt = pd.pivot_table(
            self.wz_data, index='City_En', columns=['date_int'], values=['symptom_ratio_weighted'],
            aggfunc=np.size)
        self.city_date_wz_cnt[self.city_date_wz_cnt.isna()] = 0.
        self.city_date_wz_symptom_ratio_weighted = pd.pivot_table(
            self.wz_data, index='City_En', columns=['date_int'], values=['symptom_ratio_weighted'],
            aggfunc=np.mean)
        self.city_date_wz_symptom_ratio_weighted = pd.pivot_table(
            self.wz_data, index='City_En', columns=['date_int'], values=['symptom_ratio_weighted'],
            aggfunc=np.mean)

    @staticmethod
    def load_hamagen_data(filename: str) -> DataFrame:
        with open(str_rep_env(filename), 'rb') as json_data:
            points = json.load(json_data)
        points_df = pd.io.json.json_normalize(points['features'])
        # normalize date columns and set thm as index
        points_df['fromTime'] = pd.to_datetime(points_df['properties.fromTime'] // 1000, unit='s')
        points_df['toTime'] = pd.to_datetime(points_df['properties.toTime'] // 1000, unit='s')
        points_df = points_df.set_index(pd.DatetimeIndex(points_df['fromTime']))

        return points_df


# %% Load data
cv_ds = CoronaDataset('2020-04-02')

cities = ['BENE BERAQ', 'JERUSALEM', 'ELAT', 'ARAD', 'NESHER', 'AFULA', 'UMM AL-QUTUF']

t = cv_ds.wz_data.body_temp.map(lambda t: t if isinstance(t, float) else np.nan)
t_avail = ~np.isnan(t)

t = t[t_avail]
lat = cv_ds.wz_data.lat[t_avail]
lng = cv_ds.wz_data.lng[t_avail]
date_num = cv_ds.wz_data.date_int[t_avail]
b = date_num > 10
nplot()
plt.plot(lat[b], lng[b], 'b.')

print(np.sum(t>37.5))

# cv_ds.by_city.info()
# cv_ds.wz_data.info()

# %% Basic stats
nplot()
sns.lineplot(x='date_int', y='symptom_ratio_weighted', hue='City_En', ci=None,
             estimator='mean', data=cv_ds.wz_data[cv_ds.wz_data.City_En.isin(cities)])

fig, axes = plt.subplots(2, 1, sharex='all')
for ax in axes:
    ax.grid(True)
is_norm = False
for city in cities:
    date_int = [k[1] for k in cv_ds.city_date_wz_cnt.columns]
    if is_norm:
        reps_per_day = np.mean(cv_ds.city_date_wz_cnt, axis=0)
        symps_wg_per_day = np.mean(cv_ds.city_date_wz_symptom_ratio_weighted, axis=0)
    else:
        reps_per_day = symps_wg_per_day = 1.
    axes[0].plot(date_int, cv_ds.city_date_wz_cnt.loc[city] / reps_per_day, label=city)
    axes[1].plot(date_int, cv_ds.city_date_wz_symptom_ratio_weighted.loc[city] / symps_wg_per_day, label=city)
plt.legend()


age_ths = [0, 10, 20, 30, 40, 50, 60]
wg_symptoms_per_city = OrderedDict([
    ('above_age', OrderedDict([('before', []), ('after', [])])),
    ('below_age', OrderedDict([('before', []), ('after', [])]))
])
reps_per_city =  OrderedDict([('before', []), ('after', [])])
wg_isolated_contact_with_patient = 1.0  # Default is 1.0  @@@@
for when in ['before', 'after']:
    days_range = range(12, 100) if when == 'after' else range(0, 12)
    for city in cv_ds.by_city.index:
        b = np.logical_and(
                cv_ds.wz_data.City_En == city,
                cv_ds.wz_data.date_int.isin(days_range))
        reps_per_city[when].append(b.sum())
        age = cv_ds.wz_data.age[b]
        sr = cv_ds.wz_data.symptom_ratio_weighted[b]
        #sr = 0 * cv_ds.wz_data.isolation_diagnosed[b] + 1 * cv_ds.wz_data.isolation_contact_with_patient[b]
        sr_weight = (1.0 + (wg_isolated_contact_with_patient - 1.0)
                     * cv_ds.wz_data.isolation_contact_with_patient[b])
        sr_above_age = []
        sr_below_age = []
        for age_th in age_ths:
            ba = (age >= age_th)
            if np.sum(sr_weight[ba]) > 0:
                sr_above_age.append(np.sum(sr_weight[ba] * sr[ba]) / np.sum(sr_weight[ba]))
            else:
                sr_above_age.append(np.nan)
            if np.sum(sr_weight[~ba]) > 0:
                sr_below_age.append(np.sum(sr_weight[~ba] * sr[~ba]) / np.sum(sr_weight[~ba]))
            else:
                sr_below_age.append(np.nan)
        wg_symptoms_per_city['above_age'][when].append(sr_above_age)
        wg_symptoms_per_city['below_age'][when].append(sr_below_age)
for k in ['before', 'after']:
    for aa in wg_symptoms_per_city.keys():
        wg_symptoms_per_city[aa][k] = np.stack(wg_symptoms_per_city[aa][k])
    reps_per_city[k] = np.asarray(reps_per_city[k])

fig, ax = plt.subplots()
i_age = 2
x = wg_symptoms_per_city['above_age']['after'][:, i_age] - 0 * wg_symptoms_per_city['above_age']['before'][:, i_age]
if i_age > 0:
    x -= 0.1 * wg_symptoms_per_city['below_age']['after'][:, i_age] - 0 * wg_symptoms_per_city['below_age']['before'][:, i_age]
y = cv_ds.by_city.number_sick_3103 / (cv_ds.by_city.number_sick_2703 + 1)
# Filter by population
b = np.logical_and(cv_ds.by_city.population >= 50000,
                   reps_per_city['after'] >= 200)
x, y = x[b], y[b]
ax.scatter(x, y, s=cv_ds.by_city.population / 2000)

for i, txt in enumerate(cv_ds.by_city.index[b]):
    ax.annotate(txt, (x[i], y[i]))
ax.set_title('AgeTh = {}, pearson = {}'.format(age_ths[i_age], scipy.stats.pearsonr(x, y)))

drawnow()

print('')

