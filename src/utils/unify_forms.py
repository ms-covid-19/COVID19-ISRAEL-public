"""Converts questionnaire CSVs to a unified format.

See main for usage example.
"""
import csv
import time
from enum import Enum, auto
from typing import Dict, Set

from src.utils.CONSTANTS import BODY_TEMP_THRESHOLD
from src.utils.unify_time import unify_time


class Symptom(Enum):
    WELL = auto()
    NOT_WELL = auto()
    SHORTNESS_OF_BREATH = auto()
    RUNNY_NOSE = auto()
    COUGH = auto()
    FATIGUE = auto()
    NAUSEA_VOMITING = auto()
    MUSCLE_PAIN = auto()
    GENERAL_PAIN = auto()
    SORE_THROAT = auto()
    COUGH_DRY = auto()
    COUGH_MOIST = auto()
    HEADACHE = auto()
    INFIRMITY = auto()
    DIARRHEA = auto()
    STOMACH = auto()
    FEVER = auto()
    CHILLS = auto()
    CONFUSION = auto()
    SMELL_OR_TASTE_LOSS = auto()


class Condition(Enum):
    # Any is for the Google forms where people can only respond with yes/no
    # on everything.
    ANY = auto()
    DIABETES = auto()
    HYPERTENTION = auto()
    ISCHEMIC_HEART_DISEASE = auto()
    LUNG_DISEASE = auto()
    KIDNEY_DISEASE = auto()
    CANCER = auto()


class Smoking(Enum):
    NEVER = auto()
    PAST = auto()
    PAST_LESS_THAN_FIVE_YEARS_AGO = auto()
    PAST_MORE_THAN_FIVE_YEARS_AGO = auto()
    CURRENTLY = auto()


class Isolation(Enum):
    NOT_ISOLATED = auto()
    ISOLATED = auto()
    VOLUNTARY = auto()
    BACK_FROM_ABROAD = auto()
    CONTACT_WITH_PATIENT = auto()
    HAS_SYMPTOMS = auto()
    DIAGNOSED = auto()


class PatientLocation(Enum):
    """Where the patient is now."""
    NONE = auto()
    HOME = auto()
    HOTEL = auto()
    HOSPITAL = auto()
    RECOVERED = auto()


def enum_to_column(a: Enum):
    """Converts an enum value to a column name."""
    if not isinstance(a, Enum):
        raise TypeError(f'Bad type: {type(a)}, expected Enum')

    # Break down camel case enum name to words.
    words = []
    begin = 0
    a = str(a)
    for i, c in enumerate(a):
        if i == begin:
            continue
        if 'A' <= c <= 'Z' or c == '.':
            words.append(a[begin:i])
            begin = i
        # After the period it's underscore, so stop splitting.
        if c == '.':
            begin += 1
            break
    words.append(a[begin:len(a)])
    return '_'.join(words).lower()


def _enum_to_columns(enum_class):
    """Converts a series of enum values to column names."""
    return [enum_to_column(x) for x in enum_class]


def _unpack_lists(a):
    """Takes a heterogeneous series which may contain lists,
    and returns the same values with lists unpacked to individual values."""
    return (x for y in a for x in (y if isinstance(y, list) else [y]))


def _enum_columns_from_set(s: Set, enum_class) -> Dict:
    return {_enum_to_column_mapping[x]: (1 if x in s else 0)
            for x in enum_class}


def isfloat(x) -> bool:
    """Checks whether a value can be converted to float."""
    try:
        float(x)
        return True
    except:
        return False


def convert_file(in_file, out_file, convert_dict_func, hebrew=False):
    """Converts a CSV to another CSV using the given dictionary converter."""
    with open(in_file, encoding='utf-8') as in_f:
        rows = csv.DictReader(in_f) if not hebrew else \
            csv.DictReader(in_f, fieldnames=_form_hebrew_columns)
        # Skip Hebrew header because we gave it custom field names.
        if hebrew:
            next(rows)
        parsed = (convert_dict_func(row) for row in rows)
        parsed = (d for d in parsed if d)  # Drop empty results.
        output = ((d.get(f, '') for f in _output_fields)
                  for d in parsed)
        with open(out_file, 'w', encoding='utf-8') as out_f:
            w = csv.writer(out_f)
            w.writerow(_output_fields)
            w.writerows(output)


# TODO: Add form languages.

_output_fields = ['timestamp', 'age', 'gender', 'city', 'street', 'zip_code',
                  'body_temp', 'lat', 'lng', 'source', 'medical_staff_member'] + \
                 _enum_to_columns(Symptom) + \
                 _enum_to_columns(Condition) + \
                 _enum_to_columns(Smoking) + \
                 _enum_to_columns(Isolation) + \
                 _enum_to_columns(PatientLocation)

_enum_to_column_mapping = {
    e: enum_to_column(e)
    for e in
    list(Symptom) + list(Condition) + list(Smoking)
    + list(Isolation) + list(PatientLocation)
}

_gender_mapping = {
    '0': 'M',
    '1': 'F',
    'Male': 'M',
    'Female': 'F',
    'זכר': 'M',
    'נקבה': 'F',
    'Мужской': 'M',
    'Женский': 'F',
}

_form_field_mapping = {
    'Timestamp': 'timestamp',
    'Email Address': 'email_address',
    '': '',  # Avoid key errors on empty columns.
    'Age': 'age',
    'Gender': 'gender',
    'City': 'city',
    'Street': 'street',
    'Zip code': 'zip_code',
    'Are you experiencing any of the following symptoms?': 'symptoms',
    'Have you been diagnosed with any of the following conditions: Diabetes, Hypertension, Ischemic heart disease, Asthma, Chronic lung disease, Chronic kidney disease': 'conditions',
    'I am currently:': 'isolation',
    'Cigarette smoking habits': 'smoking',
    'What is your current body temperature in Celsius?': 'body_temp',

    'גיל': 'age',
    'מין': 'gender',
    'עיר / יישוב מגורים': '[dump]',  # Redundant column.
    'עיר / ישוב מגורים': 'city',
    'רחוב מגורים': 'street',
    'מיקוד כתובת המגורים ': 'zip_code',
    'האם את/ה סובל/ת מאחד התסמינים הבאים': 'symptoms',
    'האם אתה מאובחן כסובל מאחת מהמחלות הבאות: סוכרת, יתר לחץ דם, מחלת לב איסכמית, אסטמה, מחלת ריאות כרונית, אי ספיקת כליות כרונית': 'conditions',
    'עישון': 'smoking',
    'בידוד': 'isolation',
    'מדידת חום': 'body_temp',

    'Возраст': 'age',
    'Пол': 'gender',
    'Город проживания': 'city',
    'Улица': 'street',
    'Почтовый индекс (можно проверить здесь https://tinyurl.com/y5ddf8jg)': 'zip_code',
    'Наблюдаете ли Вы у себя следующие симптомы?': 'symptoms',
    'Был ли Вам поставлен один или более диагноз из списка: сахарный диабет, гипертония, ишемическая болезнь сердца, астма, хроническая болезнь легких, хроническая почечная недостаточность :': 'conditions',
    'В данный момент Вы': 'isolation',
    'Вы': 'smoking',
    'Температура :': 'body_temp',
}

_form_symptom_mapping = {
    'I feel well': Symptom.WELL,
    'Shortness of breath': Symptom.SHORTNESS_OF_BREATH,
    'Runny nose or Nasal congestion': Symptom.RUNNY_NOSE,
    'Cough': Symptom.COUGH,
    'Fatigue': Symptom.FATIGUE,
    'Nausea and vomiting': [Symptom.NAUSEA_VOMITING, Symptom.STOMACH],
    'Muscle pain': [Symptom.MUSCLE_PAIN, Symptom.GENERAL_PAIN],
    'Diarrea': Symptom.DIARRHEA,

    'אני מרגיש/ה טוב': Symptom.WELL,
    'קוצר נשימה': Symptom.SHORTNESS_OF_BREATH,
    'נזלת או גודש באף': Symptom.RUNNY_NOSE,
    'שיעול': Symptom.COUGH,
    'עייפות חריגה': Symptom.FATIGUE,
    'בחילה והקאות': [Symptom.NAUSEA_VOMITING, Symptom.STOMACH],
    'כאבי שרירים': [Symptom.MUSCLE_PAIN, Symptom.GENERAL_PAIN],
    'שלשול': Symptom.DIARRHEA,

    'чувствую себя хорошо': Symptom.WELL,
    'одышка': Symptom.SHORTNESS_OF_BREATH,
    'насморк или заложенность носа': Symptom.RUNNY_NOSE,
    'кашель': Symptom.COUGH,
    'черезмерная усталость': Symptom.FATIGUE,
    'тошнота или рвота': [Symptom.NAUSEA_VOMITING, Symptom.STOMACH],
    'боль в мышцах': [Symptom.MUSCLE_PAIN, Symptom.GENERAL_PAIN],
    'понос': Symptom.DIARRHEA,
    'плохое самочувствие': Symptom.NOT_WELL,

    # TODO: Add other languages.
}

_form_smoking_mapping = {
    'I have never smoked': Smoking.NEVER,
    'I used to smoke': Smoking.PAST,
    'I currently smoke': Smoking.CURRENTLY,

    'מעולם לא עישנתי': Smoking.NEVER,
    'עישנתי בעבר': Smoking.PAST,
    'מעשן/ת': Smoking.CURRENTLY,

    'никогда не курили': Smoking.NEVER,
    'курили раньше': Smoking.PAST,
    'курите сейчас': Smoking.CURRENTLY,
}

_form_isolation_mapping = {
    'Not in isolation': Isolation.NOT_ISOLATED,
    'In isolation due to a recent international travel':
        Isolation.BACK_FROM_ABROAD,
    'In isolation due to a contact with an individual who was infected with '
    'coronavirus or an individual who recently returned from any destination '
    'abroad': Isolation.CONTACT_WITH_PATIENT,

    'לא נמצא בבידוד': Isolation.NOT_ISOLATED,
    'נמצא בבידוד כי חזרתי מחו״ל לאחרונה': Isolation.BACK_FROM_ABROAD,
    'נמצא בבידוד כי הייתי במגע עם אדם שנדבק בנגיף או שחזר מחו״ל לאחרונה':
        Isolation.CONTACT_WITH_PATIENT,

    'Не находитесь на карантине': Isolation.NOT_ISOLATED,
    'Находитесь на карантине': Isolation.ISOLATED,
    'Нахожусь в карантине': Isolation.ISOLATED,
    'недавно вернулся с заграницы': Isolation.BACK_FROM_ABROAD,
    'Находитесь на карантине, недавно вернулся с заграницы':
        Isolation.BACK_FROM_ABROAD,
    'который заразился вирусом или недавно вернулся из-за границы':
        Isolation.BACK_FROM_ABROAD,
    'Нахожусь в карантине, потому что был в контакте с человеком, который '
    'заразился вирусом или недавно вернулся из-за границы':
        Isolation.CONTACT_WITH_PATIENT,
    'потому что был в контакте с человеком': Isolation.CONTACT_WITH_PATIENT,
}

# Hebrew for has duplicated column names (isolation and smoking)
# so we need to rename them.
_form_hebrew_columns = [
    'Timestamp',
    'גיל',
    'מין',
    'מיקוד כתובת המגורים ',
    'מדידת חום',
    'האם את/ה סובל/ת מאחד התסמינים הבאים',
    'עיר / יישוב מגורים',
    'עישון',
    'רחוב מגורים',
    'האם אתה מאובחן כסובל מאחת מהמחלות הבאות: סוכרת, יתר לחץ דם, מחלת לב איסכמית, אסטמה, מחלת ריאות כרונית, אי ספיקת כליות כרונית',
    '',
    'בידוד',
    'עיר / ישוב מגורים',
    'Email Address',
]

_bot_smoking_mapping = {
    '0': Smoking.NEVER,
    '1': [Smoking.PAST_MORE_THAN_FIVE_YEARS_AGO, Smoking.PAST],
    '2': [Smoking.PAST_LESS_THAN_FIVE_YEARS_AGO, Smoking.PAST],
    '3': Smoking.CURRENTLY,
}

_bot_symptom_mapping = {
    'toplevel_symptoms_cough': Symptom.COUGH,
    'toplevel_symptoms_pains': Symptom.GENERAL_PAIN,
    'toplevel_symptoms_tiredness': Symptom.FATIGUE,
    'toplevel_symptoms_stomach': Symptom.STOMACH,
    'symptoms_clogged_nose': Symptom.RUNNY_NOSE,
    'symptoms_sore_throat': Symptom.SORE_THROAT,
    'symptoms_dry_cough': [Symptom.COUGH_DRY, Symptom.COUGH],
    'symptoms_moist_cough': [Symptom.COUGH_MOIST, Symptom.COUGH],
    'symptoms_breath_shortness': Symptom.SHORTNESS_OF_BREATH,
    'symptoms_muscles_pain': [Symptom.MUSCLE_PAIN, Symptom.GENERAL_PAIN],
    'symptoms_headache': Symptom.HEADACHE,
    'symptoms_fatigue': Symptom.FATIGUE,
    'symptoms_infirmity': Symptom.INFIRMITY,
    'symptoms_diarrhea': Symptom.DIARRHEA,
    'symptoms_nausea_and_vomiting': [Symptom.NAUSEA_VOMITING, Symptom.STOMACH],
    'symptoms_chills': Symptom.CHILLS,
    'symptoms_confusion': Symptom.CONFUSION,
    'symptoms_tiredness_or_fatigue': Symptom.FATIGUE,
    'symptoms_smell_taste_loss': Symptom.SMELL_OR_TASTE_LOSS,
    # 'symptoms_other':Symptom.???,
}

_bot_condition_mapping = {
    'diabetes': Condition.DIABETES,
    'hypertension': Condition.HYPERTENTION,
    'ischemic_heart_disease_or_stroke': Condition.ISCHEMIC_HEART_DISEASE,
    'lung_disease': Condition.LUNG_DISEASE,
    'cancer': Condition.CANCER,
    'kidney_failure': Condition.KIDNEY_DISEASE,
}

_bot_isolation_mapping = {
    '0': Isolation.NOT_ISOLATED,
    '1': Isolation.VOLUNTARY,
    '2': Isolation.BACK_FROM_ABROAD,
    '3': Isolation.CONTACT_WITH_PATIENT,
    '4': Isolation.HAS_SYMPTOMS,
    '5': Isolation.DIAGNOSED,
}

_bot_location_mapping = {
    '0': PatientLocation.NONE,
    '1': PatientLocation.HOME,
    '2': PatientLocation.HOTEL,
    '3': PatientLocation.HOSPITAL,
    '4': PatientLocation.HOSPITAL,
    '5': PatientLocation.RECOVERED,
}

# TODO Values are inconsistent in this field.
#  Need to talk with Ishai and have him fix that.
_bot_medical_staff_mapping = {
    '': '',
    '{}': '',
    'true': '1',
    '1': '1',
    '0': '0',
    'false': '0',
}


def convert_form_dict(d: Dict) -> Dict:
    result = {_form_field_mapping[k]: v for k, v in d.items()}
    result['source'] = 'gforms'
    result['gender'] = _gender_mapping[result['gender']]
    result['timestamp'] = unify_time(result['timestamp'])

    symptoms = set(_unpack_lists(_form_symptom_mapping[x] for x
                                 in result['symptoms'].split(', ')
                                 if x))
    if result['body_temp'] and isfloat(result['body_temp']) and \
            float(result['body_temp']) >= 38:
        symptoms.add(Symptom.FEVER)
    smoking = set(_unpack_lists([_form_smoking_mapping[result['smoking']]]))
    isolation = set(_form_isolation_mapping[x] for x
                    in result['isolation'].split(', ')
                    if x)

    # Add general isolated if isolated for any reason.
    if isolation - {Isolation.NOT_ISOLATED}:
        isolation.add(Isolation.ISOLATED)

    result.update(_enum_columns_from_set(symptoms, Symptom))
    result.update(_enum_columns_from_set({Condition.ANY}, Condition))
    result.update(_enum_columns_from_set(smoking, Smoking))
    result.update(_enum_columns_from_set(isolation, Isolation))
    result.update(_enum_columns_from_set(set(), PatientLocation))
    return result


def convert_bot_dict(d: Dict) -> Dict:
    # Bot table has headers scattered along (wtf). Ignore those lines.
    if d['id'] == 'id':
        return {}

    result = d.copy()
    result['source'] = 'bot'
    result['gender'] = _gender_mapping[result['gender']]
    result['timestamp'] = unify_time(result['timestamp'])
    result['body_temp'] = result['temperature']  # Conform with output fields.
    result['medical_staff_member'] = \
        _bot_medical_staff_mapping[result.get('medical_staff_member', '')]

    symptoms = set(_unpack_lists(v for k, v in _bot_symptom_mapping.items()
                                 if result[k] == '1'))
    if result['body_temp'] and isfloat(result['body_temp']) and \
            float(result['body_temp']) >= BODY_TEMP_THRESHOLD:
        symptoms.add(Symptom.FEVER)
    conditions = set(_unpack_lists(v for k, v in _bot_condition_mapping.items()
                                   if result[k] == '1'))
    smoking = set(_unpack_lists([_bot_smoking_mapping[result['smoking']]]))
    isolation = {_bot_isolation_mapping[result['isolation']]}
    patient_location = {_bot_location_mapping[result['diagnosed_location']]}

    if conditions:
        conditions.add(Condition.ANY)
    if isolation - {Isolation.NOT_ISOLATED}:
        isolation.add(Isolation.ISOLATED)

    result.update(_enum_columns_from_set(symptoms, Symptom))
    result.update(_enum_columns_from_set(conditions, Condition))
    result.update(_enum_columns_from_set(smoking, Smoking))
    result.update(_enum_columns_from_set(isolation, Isolation))
    result.update(_enum_columns_from_set(patient_location, PatientLocation))
    return result


def _main():
    t = time.monotonic()

    # Convert google form.
    print('Converting google form')
    convert_file('../../data/Raw/forms/COVID-19-English.csv',
                 '../../data/Raw/forms/test_unify_forms_gform.csv',
                 convert_form_dict)

    # Convert bot file.
    print('Converting bot file')
    convert_file('../../data/Raw/forms/COVID-19-Bot-1004.csv',
                 '../../data/Raw/forms/test_unify_forms_bot.csv',
                 convert_bot_dict)

    print('Took {:.1f}s'.format(time.monotonic() - t))


if __name__ == '__main__':
    _main()
