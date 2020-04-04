import pandas as pd

from src.utils.unify_forms import Symptom, enum_to_column
from src.utils.CONSTANTS import AGE_THRESHOLD, SYMPTOM_RATIO, SYMPTOM_RATIO_WEIGHTED


# Prevalence reported - will add ref
SYM_DICT_ABOVE_WEIGHTED = {enum_to_column(Symptom.COUGH): 0.58, enum_to_column(Symptom.MUSCLE_PAIN): 0.038,
                            enum_to_column(Symptom.SHORTNESS_OF_BREATH): 0.035, enum_to_column(Symptom.FATIGUE): 0.293,
                            enum_to_column(Symptom.FEVER): 0.791, enum_to_column(Symptom.SORE_THROAT): 0.032,
                           enum_to_column(Symptom.HEADACHE): 0.06, enum_to_column(Symptom.DIARRHEA): 0.057}

SYM_DICT_BELOW_WEIGHTED = {enum_to_column(Symptom.COUGH): 0.478, enum_to_column(Symptom.FEVER): 0.466,
                           enum_to_column(Symptom.SORE_THROAT): 0.4, enum_to_column(Symptom.DIARRHEA): 0.094,
                           enum_to_column(Symptom.NAUSEA_VOMITING): 0.084, enum_to_column(Symptom.RUNNY_NOSE): 0.083,
                           enum_to_column(Symptom.FATIGUE): 0.08}
# no weighting

EVEN_WEIGHTS_ABOVE = [1 / len(SYM_DICT_ABOVE_WEIGHTED.keys())] * len(SYM_DICT_ABOVE_WEIGHTED.keys())
EVEN_WEIGHTS_BELOW = [1 / len(SYM_DICT_BELOW_WEIGHTED.keys())] * len(SYM_DICT_BELOW_WEIGHTED.keys())

SYM_DICT_ABOVE_EVEN_WEIGHTS = {sym: prev for sym, prev in zip(SYM_DICT_ABOVE_WEIGHTED.keys(), EVEN_WEIGHTS_ABOVE)}
SYM_DICT_BELOW_EVEN_WEIGHTS = {sym: prev for sym, prev in zip(SYM_DICT_BELOW_WEIGHTED.keys(), EVEN_WEIGHTS_BELOW)}


def weights_calc(prevalence_dict):
    weights = pd.Series(prevalence_dict) / pd.Series(prevalence_dict).sum()
    return weights


def calc_symtom_ratio(data):

    data[SYMPTOM_RATIO] = 0
    data[SYMPTOM_RATIO_WEIGHTED] = 0

    weights_below = weights_calc(SYM_DICT_BELOW_WEIGHTED)
    weights_above = weights_calc(SYM_DICT_ABOVE_WEIGHTED)
    not_weighted_below = weights_calc(SYM_DICT_BELOW_EVEN_WEIGHTS)
    not_weighted_above = weights_calc(SYM_DICT_ABOVE_EVEN_WEIGHTS)

    below_threshold_idx = data.age <= AGE_THRESHOLD
    above_threshold_idx = data.age > AGE_THRESHOLD

    data.loc[below_threshold_idx, SYMPTOM_RATIO_WEIGHTED] = (
        data.loc[below_threshold_idx, SYM_DICT_BELOW_WEIGHTED.keys()]
            .multiply(weights_below, axis=1).sum(axis=1))

    data.loc[above_threshold_idx, SYMPTOM_RATIO_WEIGHTED] = (
        data.loc[above_threshold_idx, SYM_DICT_ABOVE_WEIGHTED.keys()]
            .multiply(weights_above, axis=1).sum(axis=1))

    data.loc[below_threshold_idx, SYMPTOM_RATIO] = (
        data.loc[below_threshold_idx, SYM_DICT_BELOW_EVEN_WEIGHTS.keys()]
            .multiply(not_weighted_below, axis=1).sum(axis=1))

    data.loc[above_threshold_idx, SYMPTOM_RATIO] = (
        data.loc[above_threshold_idx, SYM_DICT_ABOVE_EVEN_WEIGHTS.keys()]
            .multiply(not_weighted_above, axis=1).sum(axis=1))

    return data
