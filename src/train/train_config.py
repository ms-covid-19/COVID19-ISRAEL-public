import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from src.utils.CONSTANTS import CITY_ID_COL, NEIGHBORHOOD_ID_COL
from config import UNIFIED_FORMS_FILE, PROCESSED_DATA_DIR
from src.train.constants import TIME_COL, DATE_COL, PRED_COL, GT_COL


AGGREGATED_DIR = os.path.join(PROCESSED_DATA_DIR,'aggregated')

N_splits = 3
kfold = KFold(n_splits=N_splits, random_state=1, shuffle=True)

MINIMUM_PER_REGION = 50
x_agg_mode = {'mode': 'range', 'n_days': 3, 'min_per_region': 50}
y_agg_mode = {'mode': 'range', 'n_days': 2, 'min_per_region': 15}

city_type = 'city'
neighborhood_type = 'neighbor'

lower_cut_date = '2020-03-21'
upper_cut_date = '2020-04-05'

x_train_date = '2020-03-26'
y_train_date = '2020-03-30'

x_test_date = '2020-03-30'
y_test_date = '2020-04-03'

y_col_name = 'confirmed_cases'
save_map = False
test = True

agg_col = CITY_ID_COL

data_path = UNIFIED_FORMS_FILE
imputation_model = False
model_features_list = None

model = Ridge(alpha=0.1)

col_date = DATE_COL

gender_col = ['gender']

age_col = ['age']

symptoms_cols = ['body_temp', 'symptom_well', 'symptom_not_well', 'symptom_runny_nose', 'symptom_cough',
                 'symptom_fatigue', 'symptom_shortness_of_breath', 'symptom_nausea_vomiting',  'symptom_cough_dry',
                 'symptom_cough_moist', 'symptom_muscle_pain', 'symptom_general_pain', 'symptom_sore_throat',
                 'symptom_headache', 'symptom_infirmity', 'symptom_diarrhea', 'symptom_stomach', 'symptom_fever']

conditions_cols =['condition_any', 'condition_diabetes', 'condition_hypertention',
                  'condition_ischemic_heart_disease', 'condition_asthma', 'condition_lung_disease',
                  'condition_kidney_disease', 'condition_cancer']

smoking_cols = ['smoking_never', 'smoking_past', 'smoking_past_less_than_five_years_ago',
                'smoking_past_more_than_five_years_ago', 'smoking_currently']

isolation_cols = ['isolation_not_isolated', 'isolation_isolated', 'isolation_voluntary',
                  'isolation_back_from_abroad', 'isolation_contact_with_patient', 'isolation_has_symptoms',
                  'isolation_diagnosed']

patient_loc_cols = ['patient_location_none', 'patient_location_home', 'patient_location_hotel',
                    'patient_location_hospital', 'patient_location_recovered']

lamas_cols = ['lms_male_perc', 'lms_female_perc', 'lms_0_14_perc', 'lms_15_19_perc',
              'lms_20_29_perc', 'lms_30_64_prc', 'lms_65_up_perc', 'Pop_Total']

SR_cols = ['SRt', 'SRs']

patients_cols = ['confirmed_cases', 'norm_confirmed_cases']

IDs_cols = ['CITY_ID', 'NEIGHBOR_ID', 'District_Number']

age_groups_cols = ['ag_0_18_symptom_fever', 'ag_18_40_symptom_fever', 'ag_40_60_symptom_fever',
                   'ag_60_120_symptom_fever', 'ag_0_18_symptom_diarrhea', 'ag_18_40_symptom_diarrhea',
                   'ag_40_60_symptom_diarrhea', 'ag_60_120_symptom_diarrhea', 'ag_0_18_symptom_cough_moist',
                   'ag_18_40_symptom_cough_moist', 'ag_40_60_symptom_cough_moist', 'ag_60_120_symptom_cough_moist',
                   'ag_0_18_symptom_cough_dry', 'ag_18_40_symptom_cough_dry', 'ag_40_60_symptom_cough_dry',
                   'ag_60_120_symptom_cough_dry', 'ag_0_18_symptom_nausea_vomiting', 'ag_18_40_symptom_nausea_vomiting',
                   'ag_40_60_symptom_nausea_vomiting', 'ag_60_120_symptom_nausea_vomiting', 'ag_0_18_symptom_cough',
                   'ag_18_40_symptom_cough', 'ag_40_60_symptom_cough', 'ag_60_120_symptom_cough',
                   'ag_0_18_symptom_runny_nose', 'ag_18_40_symptom_runny_nose', 'ag_40_60_symptom_runny_nose',
                   'ag_60_120_symptom_runny_nose',]

only_with_sr_avgs = ['ow_sr_condition_any', 'ow_sr_condition_diabetes', 'ow_sr_condition_hypertention',
                     'ow_sr_condition_ischemic_heart_disease', 'ow_sr_condition_asthma', 'ow_sr_condition_lung_disease',
                     'ow_sr_condition_kidney_disease', 'ow_sr_condition_cancer', 'ow_sr_smoking_never',
                     'ow_sr_smoking_past', 'ow_sr_smoking_past_less_than_five_years_ago',
                     'ow_sr_smoking_past_more_than_five_years_ago', 'ow_sr_smoking_currently',
                     'ow_sr_isolation_not_isolated', 'ow_sr_isolation_isolated', 'ow_sr_isolation_voluntary',
                     'ow_sr_isolation_back_from_abroad', 'ow_sr_isolation_contact_with_patient',
                     'ow_sr_isolation_has_symptoms', 'ow_sr_isolation_diagnosed', 'ow_sr_patient_location_none',
                     'ow_sr_patient_location_home', 'ow_sr_patient_location_hotel', 'ow_sr_patient_location_hospital',
                     'ow_sr_patient_location_recovered',  'ow_sr_symptom_ratio', 'ow_sr_symptom_ratio_weighted',
                     'ow_sr_SRt', 'ow_sr_SRs', 'ow_sr_age', 'ow_sr_gender', 'ow_sr_body_temp', 'ow_sr_symptom_well',
                     'ow_sr_symptom_not_well', 'ow_sr_symptom_shortness_of_breath', 'ow_sr_symptom_runny_nose',
                     'ow_sr_symptom_cough', 'ow_sr_symptom_fatigue', 'ow_sr_symptom_nausea_vomiting',
                     'ow_sr_symptom_muscle_pain', 'ow_sr_symptom_general_pain', 'ow_sr_symptom_sore_throat',
                     'ow_sr_symptom_cough_dry', 'ow_sr_symptom_cough_moist',  'ow_sr_symptom_headache',
                     'ow_sr_symptom_infirmity', 'ow_sr_symptom_diarrhea', 'ow_sr_symptom_stomach',
                     'ow_sr_symptom_fever']

groups_totals = ['N_aggregated', 'N_with_sr', 'With_sr_perc', 'N_ag_0_18', 'N_ag_18_40', 'N_ag_40_60', 'N_ag_60_120',
                 'norm_responses']

city_places = ['bank_count', 'elderly_count', 'food_count', 'hospitals_count', 'post_office_count',
               'religous_sites_count']



AGGREGATION_DIR = os.path.join(PROCESSED_DATA_DIR, 'aggregated')