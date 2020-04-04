import os

import pandas as pd

from config import NEW_PROCESSED_DATA_DIR

# data = pd.read_csv(os.path.join(NEW_PROCESSED_DATA_DIR, 'COVID_19-All_with_location.csv'), index_col=0,
#                    low_memory=False)
# data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# tmp = data.query('City_En=="HOLON"')
# tmp = tmp[tmp.Timestamp >= pd.to_datetime('2020-03-23 00:00:00')]
#
# assert tmp.count()['symptoms_ratio_norm'] == 53 ## HOLON 523
# pass


data = pd.read_csv(os.path.join(NEW_PROCESSED_DATA_DIR, 'bot_and_questionnaire_2403.csv'), index_col=0,
                   low_memory=False)
data['timestamp'] = pd.to_datetime(data['timestamp'])
tmp = data[(data['timestamp'] >= pd.to_datetime('2020-03-18 00:00:00')) & (
            data['timestamp'] < pd.to_datetime('2020-03-21 00:00:00'))]
tmp.groupby('city_en')['smoking'].count().sort_values()
