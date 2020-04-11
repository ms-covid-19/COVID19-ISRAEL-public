# Map constants
COLORS = ['#bae4b3', '#74c476', '#08519c', '#c51b8a', '#7a0177']


# Symptoms ratio
SYMPTOM_RATIO = 'symptom_ratio'
SYMPTOM_RATIO_WEIGHTED = 'symptom_ratio_weighted'
AGE_THRESHOLD = 18
BODY_TEMP_THRESHOLD = 38
##############################

# patients col
PATIENT_COL = 'confirmed_cases'

# Column names
FEEL_GOOD = 'Feel good'
NAUSEA = 'Nausea and vomiting'
MUSCLE_PAIN = 'Muscle pains'
RUNNY_NOSE = 'Runny nose'
FATIGUE = 'Fatigue'
BREATH_SHORTNESS = 'Shortness of breath'
COUGH = 'Cough'
DIARRHEA = 'Diarrhea'
NO_ISOLATION = 'No quarantine'
ISOLATION_CONTACT = 'Quarantine - contact\travel'
ISOLATION_TRAVEL = 'Quarantine - travel'
BODY_TEMP = 'Body Temp'
SMOKING = 'Smoking history'
PRIOR_CONDITIONS = 'Prior conditions'
HIGH_FEVER = 'High Fever'
ISOLATION = 'Quarantine'

ADDED_DATA_COLS = [FEEL_GOOD, NAUSEA, MUSCLE_PAIN, RUNNY_NOSE, FATIGUE, BREATH_SHORTNESS, COUGH, DIARRHEA,
                   NO_ISOLATION, ISOLATION_CONTACT, ISOLATION_TRAVEL]

# Lamas data
LAMAS_CITY_NAME_COL = 'SHEM_YISHU'
LAMAS_ID_COL = 'OBJECTID_1'
CITY_ID_COL = 'CITY_ID'
NEIGHBORHOOD_ID_COL = 'NEIGHBOR_ID'

# prevalence meta analysis path
ALL_SYMPTOM_LIST = ['Fever', 'Cough', 'Shortness of breath (SOB)/ Dyspnea', 'Fatigue',
                    'Myalgia (muscle pain)', 'Headache', 'Nasal congestion', 'rhinorrhoea',
                    'Nausea and vomiting', 'Diarrea', 'Confusion', 'Sputum', 'Chest pain',
                    'Abdominal pain', 'Hemoptysis', 'Chills', 'Rash', 'Sore throat',
                    'Palpitation', 'Anorexia']

ALL_DISEASE_LIST = ['Comorbidities', 'HTN', 'DM', 'IHD/ CVD',
                    'COPD', 'Asthma', 'Chronic hepatitis', 'Malignancy', 'Immunodeficency',
                    'CKD', 'Smoking - current', 'Former smoker']

# Patients data
TIME_STAY_HR = 'time_stay'
TIME_STAY_FLT = 'time_stay_float'
