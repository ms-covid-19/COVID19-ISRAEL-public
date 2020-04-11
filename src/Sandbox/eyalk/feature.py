from typing import Optional, Union, Tuple, List

import numpy as np
from collections import OrderedDict
import pandas as pd
from pandas import DataFrame

from config import OUT_DIR, LAMAS_DATA, UNIFIED_FORMS_FILE
from src.utils.CONSTANTS import LAMAS_ID_COL, CITY_ID_COL, NEIGHBORHOOD_ID_COL, SYMPTOM_RATIO


class LocFeaturesCfg:

    def __init__(self,
                 symptom_scaling: OrderedDict,
                 modulation_features_wg: OrderedDict,
                 age_as_vec_pwl_ths: Union[Tuple, List] = (0., 30., 60., 100.),
                 body_temp_min_th: float = 37.2,
                 body_temp_mid_th: float = 38.0,
                 body_temp_beta: float = 2.0,
                 fscale_body_temp: float = 0.6,
                 fscale_body_temp_is_missing: float = 0.2,
                 fscale_SRt: float = 2.0,
                 fscale_SRs: float = 2.0,
                 ):
        """

        Args:
            symptom_scaling:
            modulation_features_wg:
            age_as_vec_pwl_ths: See age_to_vec
            body_temp_min_th:
            body_temp_mid_th:
            body_temp_beta:
            fscale_body_temp:
            fscale_body_temp_is_missing:
            fscale_SRt:
            fscale_SRs:
        """
        self.age_as_vec_pwl_ths = age_as_vec_pwl_ths
        self.body_temp_min_th = body_temp_min_th
        self.body_temp_mid_th = body_temp_mid_th
        self.body_temp_beta = body_temp_beta
        self.fscale_body_temp = fscale_body_temp
        self.fscale_body_temp_is_missing = fscale_body_temp_is_missing
        self.fscale_SRt = fscale_SRt
        self.fscale_SRs = fscale_SRs

        self.symptom_scaling = symptom_scaling.copy()
        self.modulation_features_wg = modulation_features_wg.copy()

    @staticmethod
    def get_default_symptom_scaling(
            symptom_scaling_default_floor_val: float = 0.1,
            symptom_scaling_override_defaults: Optional[dict] = None) -> OrderedDict:
        """

        Args:
            symptom_scaling_default_floor_val:  When determine default weight of symptoms,
                    add this value to the weight.
                    For symptoms that are overridden in symptom_scaling this is ignored
            symptom_scaling_override_defaults:

        Returns:

        """
        sf = symptom_scaling_default_floor_val
        default_symptom_scaling = OrderedDict([
            ('symptom_ratio', sf + 0.5),
            ('symptom_ratio_weighted', sf + 2.0),
            ('symptom_not_well', np.nan),  # TODO: Not sure if this field is correct in the source Excel
            ('symptom_shortness_of_breath', sf + 0.035),
            ('symptom_runny_nose', sf + 0.28),
            ('symptom_cough', sf + 0.58),
            ('symptom_cough_dry', sf),
            ('symptom_cough_moist', sf),
            ('symptom_fatigue', sf + 0.293),
            ('symptom_nausea_vomiting', sf),
            ('symptom_muscle_pain', sf + 0.26),
            ('symptom_general_pain', sf),
            ('symptom_sore_throat', sf + 0.3),
            ('symptom_headache', sf + 0.24),
            ('symptom_infirmity', sf),
            ('symptom_diarrhea', sf + 0.027),
            ('symptom_stomach', sf),
            ('symptom_fever', sf + 0.791),
            ('symptom_chills', sf),
            ('symptom_confusion', sf),
            # TODO: I put here height weight for smell/taste loss because not participating in calc_sr.py
            ('symptom_smell_or_taste_loss', sf + 0.5)
        ])
        symptom_scaling = default_symptom_scaling.copy()

        # Override default:
        for k, v in symptom_scaling_override_defaults.items():
            assert k in symptom_scaling
            symptom_scaling[k] = v

        return symptom_scaling

    @staticmethod
    def get_default_modulation_features_wg(
            modulation_features_override_defaults: Optional[dict] = None
    ) -> OrderedDict:

        default_modulation_features_wg = OrderedDict([
            ('age_as_vec',      {'ftr_symptom_ratio_weighted': 0.2, '*': 0.02}),
            ('gender',          {'ftr_symptom_ratio_weighted': 0.2}),
            ('condition_*',     {'ftr_symptom_ratio_weighted': 0.1}),
            ('smoking_never',   None),
            ('smoking_past',    None),
            ('smoking_past_more_than_five_years_ago', None),
            ('smoking_past_less_than_five_years_ago', {'ftr_symptom_ratio_weighted': 0.2}),
            ('smoking_currently', {'ftr_symptom_ratio_weighted': 0.2,
                                   'ftr_symptom_cough': 0.2,
                                   'ftr_bias': 0.2}),
            ('isolation_not_isolated', None),
            ('isolation_isolated', {'ftr_symptom_ratio_weighted': 0.2, 'ftr_bias': 0.4}),
            ('isolation_voluntary', {'ftr_symptom_ratio_weighted': 0.2}),
            ('isolation_back_from_abroad', {'ftr_symptom_ratio_weighted': 0.5, 'ftr_bias': 0.2}),
            ('isolation_contact_with_patient', {'ftr_symptom_ratio_weighted': 0.5, 'ftr_bias': 0.4}),
            ('isolation_has_symptoms', {'ftr_symptom_ratio_weighted': 0.5, 'ftr_bias': 0.5}),
            ('isolation_diagnosed', {'ftr_symptom_ratio_weighted': 0.5, 'ftr_bias': 1.0}),
            ('patient_location_none', None),
            ('patient_location_home', None),
            ('patient_location_hotel', {'ftr_symptom_ratio_weighted': 0.5, 'ftr_bias': 1.0}),
            ('patient_location_hospital', {'ftr_symptom_ratio_weighted': 0.5, 'ftr_bias': 1.0}),
            ('patient_location_recovered', {'ftr_symptom_ratio_weighted': 0.5, 'ftr_bias': 1.0})
        ])

        modulation_features_wg = default_modulation_features_wg.copy()

        # Override default:
        for k, v in modulation_features_override_defaults.items():
            assert k in modulation_features_wg
            assert v is None or isinstance(v, dict)
            modulation_features_wg[k] = v

        return modulation_features_wg


def age_to_vec(age: np.ndarray, age_as_vec_pwl_ths: Union[List, np.ndarray]) -> np.ndarray:
    """

    Args:
        age: vector
        age_as_vec_pwl_ths:  As a (modulation) feature, age should be a one-hot style vector,
                    but here we use soft version of it by interpolation between specific age thresholds.
                    Examples for the vector we get for various ages:
                    age    age_as_vec  (where age_as_vec_pwl_ths = (0., 30., 60., 100.))
                            0   30  60   100  (age_as_vec_pwl_ths)
                     30    [0   1   0    0]
                     60    [0   0   1    0]
                     70    [0   0   0.75 0.25]
                     80    [0   0   0.5  0.5]
                     100   [0   0   0    1.0]

    Returns:

    """
    age_as_vec_pwl_ths = np.asarray(age_as_vec_pwl_ths, dtype=np.float64)
    assert np.all(np.diff(age_as_vec_pwl_ths) > 0)  # Must be sorted
    age = np.asarray(age, dtype=np.float64)

    res = np.zeros((age.size, age_as_vec_pwl_ths.size), dtype=np.float64)
    res[age <= age_as_vec_pwl_ths[0], 0] = 1.0
    res[age >= age_as_vec_pwl_ths[-1], -1] = 1.0
    for i in range(age_as_vec_pwl_ths.size - 1):
        b = np.logical_and(age >= age_as_vec_pwl_ths[i], age < age_as_vec_pwl_ths[i + 1])
        res[b, i + 1] = (age[b] - age_as_vec_pwl_ths[i]) / (age_as_vec_pwl_ths[i + 1] - age_as_vec_pwl_ths[i])
        res[b, i] = (1 - res[b, i + 1])

    return res


def calc_features_per_form_row(df_forms: DataFrame, cfg: LocFeaturesCfg):
    """ Add feature columns per row

    Args:
        df_forms:
        cfg:

    Returns:

    """
    df_forms['age_as_vec'] = age_to_vec(df_forms.age.to_numpy(), cfg.age_as_vec_pwl_ths)

    df_forms['ftr_bias'] = np.ones(df_forms.shape[0], np.float64)
    if not np.isnan(cfg.fscale_body_temp):
        df_forms['ftr_by_body_temp'] = cfg.fscale_body_temp * (df_forms.body_temp > cfg.body_temp_min_th) * (
                0.5 + 0.5 * np.tanh(cfg.body_temp_beta * (df_forms.body_temp - cfg.body_temp_mid_th)))
    if not np.isnan(cfg.fscale_body_temp_is_missing):
        # Results is 1.0 if nan or missing, and 0.0 otherwise
        df_forms['ftr_body_temp_is_missing'] = cfg.fscale_body_temp_is_missing * (1. - (
                df_forms.body_temp >= 35.0))
    if not np.isnan(cfg.fscale_SRt):
        df_forms['ftr_SRt'] = cfg.fscale_SRt * df_forms.SRt
    if not np.isnan(cfg.fscale_SRs):
        df_forms['ftr_SRs'] = cfg.fscale_SRt * df_forms.SRs

    for symp_name, symp_scaling in cfg.symptom_scaling.items():
        if not np.isnan(symp_scaling):
            df_forms['ftr_' + symp_name] = symp_scaling * df_forms[symp_name]
