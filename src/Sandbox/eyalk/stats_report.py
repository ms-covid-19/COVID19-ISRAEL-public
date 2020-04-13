#%% Imports
from collections import OrderedDict
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import seaborn as sns

from config import GOV_COVID19_TESTED_INDIVIDUALS_LATEST
from src.utils.data_loader import load_unified_forms, load_hamagen_data, load_confirmed_by_day_and_city, \
    load_confirmed_patients_by_cities_mar_two_dates, load_lamas_data, REF_DATETIME, load_gov_covid19, \
    get_all_symptoms_list

#% Load data
from src.utils.eval_sensitivity import roc_est_calculator, RocEstCalculatorCfg, CoronaSimCfg, \
    default_p_symptom_dict

df_forms = load_unified_forms()
df_hamagen = load_hamagen_data()
df_lamas_cities = load_lamas_data()
df_city_day = load_confirmed_by_day_and_city()
df_city = load_confirmed_patients_by_cities_mar_two_dates()
df_tested_individuals = load_gov_covid19(GOV_COVID19_TESTED_INDIVIDUALS_LATEST)

df_forms_src_copy = df_forms.copy()  # Keep a copy in memory, in case we want to modify the source below

#% Here select data for the stats
df_forms = df_forms_src_copy.copy()

# body_temp with and without headache
# b = (df_forms.body_temp > 0)
# body_temp = np.arange(35.0, 43.1, 0.1).astype(np.float64)
# t_hc = []
# t_no_hc = []
# for t in body_temp:
#     bb = (np.abs(df_forms.body_temp - t) < 0.05)
#     t_hc.append((bb & (df_forms.symptom_headache | df_forms.symptom_fatigue |
#                        df_forms.symptom_runny_nose)).sum())
#     t_no_hc.append((bb & (~df_forms.symptom_headache) & (~df_forms.symptom_fatigue) &
#                     (~df_forms.symptom_runny_nose)).sum())
# df_temp = pd.DataFrame.from_dict(
#     {'body_temp': body_temp,
#      'cnt_headache': t_hc,
#      'cnt_noheadache': t_no_hc
#      }
# )
# df_temp.to_csv(r'c:\users\eyalk\Documents\COVID19-ISRAEL-public\out\body_temp2.csv')
#plt.figure()
#plt.hist(df_forms.body_temp[b], bins=100)

# age_filter_name corresponds to column in lamas data
age_filter_name = 'Pop_Total'  # 'Pop_Total' 'age_0_14', 'age_15_19', 'age_20_29', 'age_30_64', 'age_65_up'
print('Age filtering: {}'.format(age_filter_name))
if age_filter_name != 'Pop_Total':
    df_forms = df_forms[df_forms.age_group_lms_name == age_filter_name]
df_forms = df_forms[df_forms.source == 'bot']
# df_forms = df_forms[df_forms.age > 18]
df_forms = df_forms[df_forms.date_int >= 34]  # @@@  Take only recent days (31 is 2020-04-01)
# df_forms = df_forms[df_forms.body_temp_measured]
assert df_forms.shape[0] > 0

#% Estimate ROC by simulator
group_by_col, min_reps = 'City_En', 110  # For 110, and date filter from 2020-04-04 we get 100 cities
df_groups = df_forms.groupby(group_by_col).filter(lambda g: len(g) >= min_reps).groupby(group_by_col)

roc_est_cfg = RocEstCalculatorCfg(
    corona_sim_cfg=CoronaSimCfg(
        p_infection=0.05,
        p_report_if_infected_increase_factor=1.0,
        p_asymptomatic_override=0.5,
        p_symptom_dict=default_p_symptom_dict()
    ),
    corona_from_date='2020-03-01',  # Use old dates to take everything
    extract_metric_func='symptom_ratio_weighted',
    epochs_num=1,
    rand_seed=78941
)
fig, ax = plt.subplots(1)
fp_tp_per_p_infection = OrderedDict()
for p_infection in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
    roc_est_cfg.corona_sim_cfg.p_infection = p_infection
    roc_est_cfg.epochs_num = 1 if p_infection == 0. else 5
    fp_vec, tp_vec = roc_est_calculator(df_groups, roc_est_cfg)
    fp_tp_per_p_infection[p_infection] = (fp_vec, tp_vec)
    ax.plot(fp_vec, tp_vec, label='p_infection={}'.format(p_infection))
    plt.draw()
    plt.pause(0.001)
ax.grid()
ax.legend()
print('')

# df_forms.info()roc_est_cfg = RocEstCalculatorCfg(
#     corona_sim_cfg=CoronaSimCfg(
#         p_infection=0.2,
#         p_report_if_infected_increase_factor=1.0,
#         p_asymptomatic_override=0.5,
#         p_symptom_dict=default_p_symptom_dict()
#     ),
#     corona_from_date='2020-03-01',  # Use old dates to take everything
#     extract_metric_func='symptom_ratio_weighted',
#     epochs_num=10,
#     rand_seed=78941
# )
# fig, ax = plt.subplots(1)
# for p_infection in [0.0, 0.01, 0.02, 0.05, 0.1]:
#     fp_vec, tp_vec = roc_est_calculator(df_groups, roc_est_cfg)
#     ax.plot(fp_vec, tp_vec, label='p_infection={}'.format(p_infection))
#     ax.grid()
# df_hamagen.info()
# df_city_day.info()
# df_city.info()

#% General statistics
plt.figure()
df_forms[get_all_symptoms_list()].mean(axis=0).sort_values(ascending=False).plot.bar()
plt.grid(True)
plt.title('Mean symptoms over all data')

df_groups = df_forms[df_forms.symptom_any].groupby('City_En')
# df_groups.describe()
df_grp = df_groups[get_all_symptoms_list() +
                   ['symptom_ratio_weighted', 'symptom_ratio_weighted_sqr',
                    'body_temp', 'body_temp_measured',
                    'condition_any', 'smoking_currently']].mean()
df_grp['reps_num'] = df_groups.age.count()
df_grp['City_En'] = df_grp.index

sns.set(style='ticks')

b = (df_grp.reps_num >= 40)
cols = ['reps_num', 'symptom_ratio_weighted_sqr', 'symptom_ratio_weighted',
        'symptom_fever', 'body_temp', 'body_temp_measured',
        'symptom_fatigue','symptom_cough', 'symptom_cough_dry',
        'symptom_headache', 'symptom_diarrhea', 'symptom_runny_nose']
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H',
           '+', 'x', 'X', 'D', 'd', '|', '_'] * 10
markers = markers[:b.sum()]
sns.pairplot(df_grp.loc[b, ['City_En'] + cols], hue='City_En', markers=markers)

pd.set_option('display.max_columns', 12, 'display.width', 300)
df = df_grp.loc[b, cols].sort_values('symptom_ratio_weighted_sqr')
print(df.tail(40))
df.to_csv(r'c:\users\eyalk\Documents\COVID19-ISRAEL-public\out\temp5.csv')

#% Grouping statistics
fig, axes = plt.subplots(2, 2)
for plt_col, group_column in enumerate(['City_En', 'date']):
    df_groups = df_forms.groupby(group_column)
    #df_groups.describe()
    df_grp = df_groups.age.count().to_frame(name='reps_num')

    if group_column == 'date':
        df_grp['date'] = df_grp.index
    df_grp['symptom_any'] = df_groups.symptom_any.mean()
    df_grp['symptom_ratio_weighted'] = df_groups.symptom_ratio_weighted.mean()
    df_grp['symptom_wg_norm_runny_nose'] = (df_groups.symptom_ratio_weighted.mean() /
                                            df_groups.symptom_runny_nose.mean())

    # The following merge will drop cities which are not found in df_lamas_cities
    if group_column == 'City_En':
        df_grp = pd.merge(df_grp, df_lamas_cities, how='inner', on='City_En', left_index=True)
        norm_by = age_filter_name
        b = np.logical_and(df_grp.Pop_Total >= 20000, df_grp.reps_num >= 400)
    else:
        norm_by = None
        b = (df_grp.reps_num >= 400)

    for plt_row, y_column in enumerate(['symptom_any', 'symptom_ratio_weighted']):  # symptom_wg_norm_runny_nose
        x = df_grp.reps_num[b]
        if norm_by is not None:
            x = x / df_grp[norm_by][b]
        x = np.asarray(x)
        y = np.asarray(df_grp[y_column][b])
        ax = axes[plt_row, plt_col]
        ax.scatter(x, y, s=df_grp.Pop_Total[b] / 4000 if group_column == 'City_En' else None)
        for i, txt in enumerate(df_grp[group_column][b].to_numpy()):
            #print(i, txt)
            ax.annotate(txt, (x[i], y[i]))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        xlr = np.asarray([0., x.max()])
        ax.plot(xlr, intercept + slope * xlr, 'k-')
        ax.set_title('{} a, b= {:.3g}, {:.3g} cor={:.3g}, p={:.3g}'.format(
            age_filter_name, slope, intercept, r_value, p_value))
        ax.set_xlabel({
            'City_En': '(Total forms filled)/(population size)',
            'date': 'Forms filled @ date'
        }[group_column])
        ax.set_ylabel(y_column)
        ax.grid()

    if group_column == 'date':
        df_by_date_summary = df_grp[['reps_num']][b].copy()
        df_by_date_summary['rep_any_symp'] = (df_grp.symptom_any[b] * df_grp.reps_num[b]).astype(np.int)
        df_by_date_summary['rep_no_symp'] = ((1.0 - df_grp.symptom_any[b]) * df_grp.reps_num[b]).astype(np.int)
        df_by_date_summary['P(S|A)'] = df_grp.symptom_any[b]
        print(df_by_date_summary)

        x = np.asarray(df_grp.reps_num[b] * (1. - df_grp.symptom_any[b]))
        y = np.asarray(df_grp.reps_num[b] * df_grp.symptom_any[b])
        _, ax = plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(df_grp[group_column][b].to_numpy()):
            ax.annotate(txt, (x[i], y[i]))
        # (Direct pearson corr is not useful here, as we need to force it pass from (0, 0))
        ax.set_xlabel('Reports non-symptoms')
        ax.set_ylabel('Reports with one or more symptom (any)')
        ax.grid()


plt.draw()
plt.pause(0.001)

#%


#%
print('')