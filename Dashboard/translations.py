from Dashboard.data import NDAYS_SMOOTHING, MIN_OBSERVATIONS_CITY, MIN_OBSERVATIONS_NEIGHBORHOOD

level_options = {
    'city': {'eng': 'Cities',
               'heb': 'ערים'},
    'neighborhood': {'eng': 'Neighborhoods',
                      'heb': 'שכונות'}
}

layout_elements = {
    'dashboard_title': {'eng': 'COVID-19 Questionnaire Dashboard',
                        'heb': 'COVID-19 Questionnaire Dashboard'},
    'Data resolution': {'eng': 'Level of display',
                        'heb': 'רמת תצוגה'},
    'Color by': {'eng': 'Color by',
                 'heb': 'צביעה לפי'},
    'Date selector': {'eng': 'Date selector. The data are averaged over the last {} days from the selected date'.format(NDAYS_SMOOTHING),
                      'heb': 'בחירת תאריך. הנתונים בתאריך מסויים מייצגים את כמות התסמינים הממוצעת כפי שדווחה {} הימים האחרונים'.format(NDAYS_SMOOTHING)},
    'Language selector': {'eng': 'Language selector',
                          'heb': 'Language selector'},
    'map_description': {
        'eng': """This map shows Israel divided into different regions - by cities or neighborhoods.
Every region is colored according to symptoms reported in the questionnaire. The more people reported a certain symptom - the closer its color is to purple.
The data is averaged over the last {} days. Only cities with at least {} responses and neighborhoods with at least {} responses are presented.
Please note: the map does not show risk of COVID-19 morbidity or incidence state.
Presented here is a report of symptoms described as common in COVID-19 and not specific to the disease (thus these symptoms can also appear in other diseases).""".format(
            NDAYS_SMOOTHING, MIN_OBSERVATIONS_CITY, MIN_OBSERVATIONS_NEIGHBORHOOD),

        'heb': """במפה המוצגת ניתן לראות את מדינת ישראל מחולקת לאזורים שונים - על פי ערים או שכונות. כל אזור צבוע על פי כמות התסמינים אשר דווחה בו. במידה ותסמין מסויים דווח בכמות גבוהה באזור מסויים צבעו יהיה בגוון סגול. לעומת זאת, תסמין אשר דווח ברמה מועטה באזור מסויים יצבע בגוון ירוק. 
שים/י לב! הנתונים בתאריך מסויים מייצגים את כמות התסמינים הממוצעת כפי שדווחה {} הימים האחרונים. בנוסף, רק ערים עם לפחות {} תגובות ושכונות עם לפחות {} תגובות מוצגות. 
חשוב! הנתונים מייצגים את דיווחי התסמינים בלבד. הנתונים אינם מייצגים את מדד הסיכון לתחלואה בנגיף הקורונה (COVID-19) או את מצב התחלואה עצמו. 
כמו כן, על אף שהתסמינים המוצגים נמצאו נפוצים אצל חולים בנגיף הקורונה הם אינם ספציפיים למחלה זו בלבד ויכולים להיות קשורים למחלות נוספות.""".format(
            NDAYS_SMOOTHING, MIN_OBSERVATIONS_CITY, MIN_OBSERVATIONS_NEIGHBORHOOD)},

    'pie_description': {'eng': 'Cities with the highest mean symptom/feature',
                        'heb': 'הערים בהם ממוצע דיווח התסמין הוא הגבוה ביותר'},
}

# feature_translations = {
#     'age': {'eng': 'Average age', 'heb': 'גיל', 'eng_short': ''},
#     'gender': {'eng': 'Gender: % of responses that are male', 'heb': 'מין: % תשובות מגברים'},
#     'symptom_fever': {'eng': 'Fever: % of responses', 'heb': 'חום: % תשובות'},
#     'symptom_shortness_of_breath': {'eng': 'Shortness of breath: % of responses', 'heb': 'קוצר נשימה: % תשובות'},
#     'symptom_runny_nose': {'eng': 'Rhinorrhea (Runny nose) and\or Nasal congestion: % of responses',
#                            'heb': 'נזלת ו/או גודש באף: % תשובות'},
#     'symptom_cough': {'eng': 'Cough: % of responses', 'heb': 'שיעול: % תשובות'},
#     'symptom_fatigue': {'eng': 'Fatigue: % of responses', 'heb': 'עייפות חריגה: % תשובות'},
#     'symptom_nausea_vomiting': {'eng': 'Nausea and\or vomiting: % of responses', 'heb': 'בחילה ו/או הקאות: % תשובות'},
#     'symptom_muscle_pain': {'eng': 'Muscle pain: % of responses', 'heb': 'כאבי שרירים: % תשובות'},
#     'symptom_sore_throat': {'eng': 'Sore throat: % of responses', 'heb': 'כאב גרון: % תשובות'},
#     'symptom_headache': {'eng': 'Headache: % of responses', 'heb': 'כאב ראש: % תשובות'},
#     'symptom_diarrhea': {'eng': 'Diarrhea: % of responses', 'heb': 'שלשול: % תשובות'},
#     'smoking_currently': {'eng': 'Currently smoking: % of responses', 'heb': 'עישון: % תשובות'},
#     'isolation_not_isolated': {'eng': 'Not isolated: % of responses', 'heb': 'לא בבידוד: % תשובות'},
#     'symptom_ratio_weighted': {'eng': 'Weighted average of symptoms (gives more weight to symptoms more common in COVID19 confirmed cases)', 'heb': 'ממוצע תסמינים משוקלל (נותן משקל גדול יותר לתסמינים שכיחים יותר בקרב חולי קורנה)'},
# }
#

feature_translations = {'age': {'eng': 'Average age',
                                  'heb': 'גיל',
                                  'eng_short': 'Average age',
                                  'heb_short': 'גיל'},
 'gender': {'eng': 'Gender: % of responses that are male',
          'heb': 'מין: % תשובות מגברים',
          'eng_short': 'Gender: % of responses that are male',
          'heb_short': 'מין: % תשובות מגברים'},
 'symptom_fever': {'eng': 'Fever: % of responses',
                  'heb': 'חום: % תשובות',
                  'eng_short': 'Fever: % of responses',
                  'heb_short': 'חום: % תשובות'},
 'symptom_shortness_of_breath': {'eng': 'Shortness of breath: % of responses',
                                  'heb': 'קוצר נשימה: % תשובות',
                                  'eng_short': 'Shortness of breath: % of responses',
                                  'heb_short': 'קוצר נשימה: % תשובות'},
 'symptom_runny_nose': {'eng': 'Rhinorrhea (Runny nose) and\\or Nasal congestion: % of responses',
                          'heb': 'נזלת ו/או גודש באף: % תשובות',
                          'eng_short': 'Rhinorrhea (Runny nose) and\\or Nasal congestion: % of responses',
                          'heb_short': 'נזלת ו/או גודש באף: % תשובות'},
 'symptom_cough': {'eng': 'Cough: % of responses',
                  'heb': 'שיעול: % תשובות',
                  'eng_short': 'Cough: % of responses',
                  'heb_short': 'שיעול: % תשובות'},
'symptom_fatigue': {'eng': 'Fatigue: % of responses',
                  'heb': 'עייפות חריגה: % תשובות',
                  'eng_short': 'Fatigue: % of responses',
                  'heb_short': 'עייפות חריגה: % תשובות'},
 'symptom_nausea_vomiting': {'eng': 'Nausea and\\or vomiting: % of responses',
                          'heb': 'בחילה ו/או הקאות: % תשובות',
                          'eng_short': 'Nausea and\\or vomiting: % of responses',
                          'heb_short': 'בחילה ו/או הקאות: % תשובות'},
 'symptom_muscle_pain': {'eng': 'Muscle pain: % of responses',
                      'heb': 'כאבי שרירים: % תשובות',
                      'eng_short': 'Muscle pain: % of responses',
                      'heb_short': 'כאבי שרירים: % תשובות'},
 'symptom_sore_throat': {'eng': 'Sore throat: % of responses',
                      'heb': 'כאב גרון: % תשובות',
                      'eng_short': 'Sore throat: % of responses',
                      'heb_short': 'כאב גרון: % תשובות'},
 'symptom_headache': {'eng': 'Headache: % of responses',
                      'heb': 'כאב ראש: % תשובות',
                      'eng_short': 'Headache: % of responses',
                      'heb_short': 'כאב ראש: % תשובות'},
 'symptom_diarrhea': {'eng': 'Diarrhea: % of responses',
                      'heb': 'שלשול: % תשובות',
                      'eng_short': 'Diarrhea: % of responses',
                      'heb_short': 'שלשול: % תשובות'},
 'smoking_currently': {'eng': 'Currently smoking: % of responses',
                      'heb': 'עישון: % תשובות',
                      'eng_short': 'Currently smoking: % of responses',
                      'heb_short': 'עישון: % תשובות'},
 'isolation_not_isolated': {'eng': 'Not isolated: % of responses',
                          'heb': 'לא בבידוד: % תשובות',
                          'eng_short': 'Not isolated: % of responses',
                          'heb_short': 'לא בבידוד: % תשובות'},
 'symptom_ratio_weighted': {'eng': 'Weighted average of symptoms (gives more weight to symptoms more common in COVID19 confirmed cases)',
                          'heb': 'ממוצע תסמינים משוקלל (נותן משקל גדול יותר לתסמינים שכיחים יותר בקרב חולי קורנה)',
                          'eng_short': 'Weighted average of symptoms',
                          'heb_short': 'ממוצע תסמינים משוקלל'}}



