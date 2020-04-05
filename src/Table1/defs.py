from src.utils.unify_forms import Symptom, Condition, Smoking, Isolation, enum_to_column
from src.utils.CONSTANTS import SYMPTOM_RATIO_WEIGHTED

TABLE_COLS = ['age', 'gender', enum_to_column(Smoking.CURRENTLY), enum_to_column(Condition.ANY),
              enum_to_column(Isolation.ISOLATED), enum_to_column(Isolation.DIAGNOSED),
              'body_temp', enum_to_column(Symptom.FEVER), enum_to_column(Symptom.WELL),
              enum_to_column(Symptom.NAUSEA_VOMITING), enum_to_column(Symptom.MUSCLE_PAIN),
              enum_to_column(Symptom.RUNNY_NOSE), enum_to_column(Symptom.FATIGUE),
              enum_to_column(Symptom.SHORTNESS_OF_BREATH),
              enum_to_column(Symptom.COUGH), enum_to_column(Symptom.DIARRHEA), SYMPTOM_RATIO_WEIGHTED]

TABLE_LABELS = ['Age (years)', 'Sex - Male', 'Smoking (currently)', 'Presence of a chronic medical conditions',
                'Isolation', 'COVID-19 diagnosed',
                'Body temperature (Celsius)', 'Body temperature above 38',
                'No symptoms (Feel good)', 'Nausea and vomiting',
                'Muscle pains', 'Rhinorrhea or nasal congestion',
                'Fatigue', 'Shortness of breath',
                'Cough', 'Diarrhea', 'Symptoms Ratio']

TABEL_COLS_TYPE = ['continuous', 'count_perc', 'count_perc', 'count_perc', 'count_perc', 'count_perc'] + \
                  ['continuous']*2 + ['count_perc']*8 + ['continuous']
