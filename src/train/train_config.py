from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from src.utils.CONSTANTS import CITY_ID_COL, NEIGHBORHOOD_ID_COL
from config import UNIFIED_FORMS_FILE

lower_cut_date = '2020-03-23'
upper_cut_date = '2020-04-01'

train_date = '2020-03-26'
label_date = '2020-03-27'
test_date = '2020-03-28'

agg_col = CITY_ID_COL

city_type = 'city'
neighborhood_type = 'neighbor'
y_col_name = 'symptom_ratio_weighted'

data_file = UNIFIED_FORMS_FILE

MINIMUM_PER_REGION = 100

model_features_list = None

N_splits = 3
kfold = KFold(n_splits=N_splits, random_state=1, shuffle=True)

model = LinearRegression()

save_map = False
