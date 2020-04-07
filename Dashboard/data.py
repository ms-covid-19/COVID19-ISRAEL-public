import json
import os

import pandas as pd

# selected_features = ['Age', 'Gender', 'Body Temp', 'Smoking history', 'Prior conditions', 'Feel good', 'Nausea and vomiting', 'Muscle pains',
#                      'Runny nose', 'Fatigue', 'Shortness of breath', 'Cough', 'Diarrhea',
#                      'No quarantine', 'Quarantine - contact\travel', 'Quarantine - travel', 'symptoms_ratio_norm']
from config import DASH_CACHE_DIR
from src.utils.CONSTANTS import COLORS

min_date = pd.to_datetime('2020-03-17')
# max_date = pd.Timestamp.today().normalize()
max_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)

LANGUAGE = 'heb'
html_align_dict = {'heb': 'right', 'eng': 'left'}

MIN_OBSERVATIONS_CITY = 500
MIN_OBSERVATIONS_NEIGHBORHOOD = 100
NDAYS_SMOOTHING = 9
MAX_CITIES_PIE = 15

color_scale = [[i / (len(COLORS) - 1), COLORS[i]] for i in range(len(COLORS))]

features_to_perc = ['gender', 'symptom_fever', 'symptom_shortness_of_breath', 'symptom_runny_nose',
                            'symptom_cough',
                            'symptom_fatigue', 'symptom_nausea_vomiting', 'symptom_muscle_pain', 'symptom_sore_throat',
                            'symptom_headache', 'symptom_diarrhea', 'smoking_currently', 'isolation_not_isolated',
                    'symptom_ratio', 'symptom_ratio_weighted']

# selected_features = ['age',
#  'body_temp',
#  'symptom_well',
#  'symptom_not_well',
#  'symptom_shortness_of_breath',
#  'symptom_runny_nose',
#  'symptom_cough',
#  'symptom_fatigue',
#  'symptom_nausea_vomiting',
#  'symptom_muscle_pain',
#  'symptom_general_pain',
#  'symptom_sore_throat',
#  'symptom_cough_dry',
#  'symptom_cough_moist',
#  'symptom_headache',
#  'symptom_infirmity',
#  'symptom_diarrhea',
#  'symptom_stomach',
#  'symptom_fever',
#  'condition_any',
#  'condition_diabetes',
#  'condition_hypertention',
#  'condition_ischemic_heart_disease',
#  'condition_asthma',
#  'condition_lung_disease',
#  'condition_kidney_disease',
#  'condition_cancer',
#  'smoking_never',
#  'smoking_past',
#  'smoking_past_less_than_five_years_ago',
#  'smoking_past_more_than_five_years_ago',
#  'smoking_currently',
#  'isolation_not_isolated',
#  'isolation_isolated',
#  'isolation_voluntary',
#  'isolation_back_from_abroad',
#  'isolation_contact_with_patient',
#  'isolation_has_symptoms',
#  'isolation_diagnosed',
#  'patient_location_none',
#  'patient_location_home',
#  'patient_location_hotel',
#  'patient_location_hospital',
#  'patient_location_recovered',
#  'symptom_ratio',
#  'symptom_ratio_weighted']
