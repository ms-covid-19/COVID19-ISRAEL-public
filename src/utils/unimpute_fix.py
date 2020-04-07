imputation_cols = ['symptom_well', 'symptom_not_well',
       'symptom_shortness_of_breath', 'symptom_runny_nose', 'symptom_cough',
       'symptom_fatigue', 'symptom_nausea_vomiting', 'symptom_muscle_pain',
       'symptom_general_pain', 'symptom_sore_throat', 'symptom_cough_dry',
       'symptom_cough_moist', 'symptom_headache', 'symptom_infirmity',
       'symptom_diarrhea', 'symptom_stomach', 'symptom_fever',
       'symptom_chills', 'symptom_confusion', 'symptom_smell_or_taste_loss',
       'condition_any', 'condition_diabetes', 'condition_hypertention',
       'condition_ischemic_heart_disease', 'condition_asthma',
       'condition_lung_disease', 'condition_kidney_disease',
       'condition_cancer', 'smoking_never', 'smoking_past',
       'smoking_past_less_than_five_years_ago',
       'smoking_past_more_than_five_years_ago', 'smoking_currently',
       'isolation_not_isolated', 'isolation_isolated', 'isolation_voluntary',
       'isolation_back_from_abroad', 'isolation_contact_with_patient',
       'isolation_has_symptoms', 'isolation_diagnosed',
       'patient_location_none', 'patient_location_home',
       'patient_location_hotel', 'patient_location_hospital',
       'patient_location_recovered']

def get_first_ones_in_cols(df, imputation_cols):
    """
    df: Pandas Dataframe loaded from all_forms.csv
    """
    timestamps = []
    timestamps.append({'What': 'First Timestamp', 'Timestamp':(df.timestamp).min()})

    for col in imputation_cols:
        first_one_in_col = df.loc[df[col]==1, 'timestamp'].min()
        timestamps.append({'What': col, 'Timestamp':first_one_in_col})

    timestamps.append({'What': 'Last Timestamp', 'Timestamp':(df.timestamp).max()})
    ts = pd.DataFrame(timestamps, columns=['What', 'Timestamp']).sort_values('Timestamp')
    return ts


def unimpute_columns(df, imputation_cols):
    """
    Unimputes (puts NaNs) in columns that were introduced to the survey later on
    Does so by taking 1 hour before the first "1" as the first valid time
    
    df: Pandas Dataframe loaded from all_forms.csv
    imputation_cols: columns in df requiring imputation
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    ts = get_first_ones_in_cols(df, imputation_cols)

    for col in imputation_cols:   
        margin_timedelta = pd.to_timedelta('1 h')
        first_valid_time = ts.loc[ts['What']==col,'Timestamp'] - margin_timedelta 
        idx_to_unimpute = df['timestamp'] <= (np.repeat(first_valid_time.values, len(df['timestamp'])))
        df.loc[idx_to_unimpute, col] = np.nan
    return df