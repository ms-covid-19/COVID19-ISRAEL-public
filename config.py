# write mock - push - ignore
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
GITHUB_DATA_DIR = os.path.join(os.path.dirname(__file__), 'github-data')
DASH_CACHE_DIR = os.path.join(DATA_DIR, 'dashboard_cache')
GENERAL_CACHE_DIR = os.path.join(DATA_DIR, 'general_cache')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'Raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'Processed')
PROCESSED_DATA_MAOZ_DIR = os.path.join(PROCESSED_DATA_DIR, 'maoz')
NEW_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'New processed')

RAW_FORMS_DIR = os.path.join(RAW_DATA_DIR, 'forms')
PROCESSED_FORMS_DIR = os.path.join(PROCESSED_DATA_DIR, 'forms')
UNIFIED_FORMS_FILE = os.path.join(PROCESSED_FORMS_DIR, 'all_forms.csv')

GOV_COVID19_DIR = os.path.join(DATA_DIR, 'GovCovid19')
GOV_COVID19_TESTED_INDIVIDUALS_LATEST = 'corona_tested_individuals_ver_001.xlsx'

GOOGLE_FORM_RAW_DIR = os.path.join(RAW_DATA_DIR, 'google_forms')
GOOGLE_FORM_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, 'google_forms')

BOT_RAW_DIR = os.path.join(RAW_DATA_DIR, 'Bot')
BOT_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, 'Bot')

PATIENTS_RAW_DIR = os.path.join(RAW_DATA_DIR, 'confirmed_patients')
PATIENTS_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, 'confirmed_patients')
PATIENTS_CITY_DATE_FILE = os.path.join(PATIENTS_PROCESSED_DIR,'patients_by_day_and_city_org.csv')
PATIENTS_CITY_DATE_FILE_NON_ZERO = os.path.join(PATIENTS_PROCESSED_DIR,'patients_by_day_and_city_org_no_zero.csv')
PATIENTS_CITY_DATE_FILE_SAAR = os.path.join(PATIENTS_PROCESSED_DIR,'patients_by_day_and_city_saar.csv')

UTILITY_DATA = os.path.join(PROCESSED_DATA_DIR, 'utility')
LAMAS_DATA = os.path.join(DATA_DIR, 'Lamas')

LAMAS_NEIGHBERHOOD_DATA = os.path.join(LAMAS_DATA, 'neighborhoods_lms_features.csv')
LAMAS_CITIES_DATA = os.path.join(LAMAS_DATA, 'cities_lms_features.csv')


HAMAGEN_DATA = os.path.join(DATA_DIR, 'Hamagen')

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')

if not os.path.exists(GENERAL_CACHE_DIR):
    os.makedirs(GENERAL_CACHE_DIR, exist_ok=True)
