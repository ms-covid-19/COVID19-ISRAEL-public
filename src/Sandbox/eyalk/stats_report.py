#%% Imports
from collections import OrderedDict
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats

from src.utils.data_loader import load_unified_forms, load_hamagen_data, load_confirmed_by_day_and_city, \
    load_confirmed_patients_by_cities_mar_two_dates, load_lamas_data, REF_DATETIME

#% Load data
df_forms = load_unified_forms()
df_hamagen = load_hamagen_data()
df_lamas_cities = load_lamas_data()
df_city_day = load_confirmed_by_day_and_city()
df_city = load_confirmed_patients_by_cities_mar_two_dates()

all_symptoms = [
    'symptom_shortness_of_breath', 'symptom_runny_nose', 'symptom_cough',
    'symptom_fatigue', 'symptom_nausea_vomiting', 'symptom_muscle_pain',
    'symptom_general_pain', 'symptom_sore_throat', 'symptom_cough_dry',
    'symptom_cough_moist', 'symptom_headache', 'symptom_infirmity',
    'symptom_diarrhea', 'symptom_stomach', 'symptom_fever',
    'symptom_chills', 'symptom_confusion', 'symptom_smell_or_taste_loss']
df_forms['symptom_any'] = np.any(df_forms[all_symptoms], axis=1)

# df_forms.info()
# df_hamagen.info()
# df_city_day.info()
# df_city.info()

#% General statistics
plt.figure()
df_forms[all_symptoms].mean(axis=0).sort_values(ascending=False).plot.bar()
plt.grid(True)
plt.title('Mean symptoms over all data')

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
        norm_by = 'Pop_Total'
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
        cor, p = scipy.stats.pearsonr(x, y)
        ax.set_title('Pearson = {:.5g}, {:.5g}'.format(cor, p))
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

print('')