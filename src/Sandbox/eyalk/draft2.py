import pandas as pd

from src.utils.data_loader import load_unified_forms, load_hamagen_data, load_confirmed_by_day_and_city, \
    load_confirmed_patients_by_cities_mar_two_dates

df_forms = load_unified_forms()
df_hamagen = load_hamagen_data()
df_city_day = load_confirmed_by_day_and_city()
df_city = load_confirmed_patients_by_cities_mar_two_dates()

df_forms.info()
df_hamagen.info()
df_city_day.info()
df_city.info()

print('')