from datetime import datetime
from typing import Optional, Tuple, List
from collections import OrderedDict
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy

from src.utils.calc_sr import calc_symtom_ratio
from src.utils.data_loader import get_all_symptoms_list


def default_p_symptom_dict() -> OrderedDict:
    return OrderedDict([
        ('symptom_fever', 0.791 * 0.75),  # *0.75 Simulating probability of measuring fever
        ('symptom_cough', 0.58),
        ('symptom_fatigue', 0.293),
        ('symptom_muscle_pain', 0.038),
        ('symptom_sore_throat', 0.032),
        ('symptom_headache', 0.06),
        ('symptom_diarrhea', 0.057)
    ]).copy()


class CoronaSimCfg():
    """ Corona generation / symptoms simulator """

    def __init__(
            self,
            p_infection: float = 0.01,
            p_report_if_infected_increase_factor: float = 1.0,
            p_asymptomatic_override: float = 0.5,
            p_symptom_dict: Optional[OrderedDict] = None
    ):
        """
           Default number taken (arbitrarily) from calc_sr, adult

        Args:
            p_infection: Probability of individual to be infected
            p_report_if_infected_increase_factor: Factor on the probability of individual to responsd
                    to questionnaire because he have symptoms
            p_asymptomatic_override: Probability that all symptoms are zero, regardless of other p
                    (If True, set all to zero. Otherwise, each is randomized individually)
            p_symptom_dict: Dictionary from symptom to probability of having it (for sick person)
        """
        self.p_infection = p_infection
        self.p_report_if_infected_increase_factor = p_report_if_infected_increase_factor
        self.p_asymptomatic_override = p_asymptomatic_override
        self.p_symptom_dict = default_p_symptom_dict() if p_symptom_dict is None else p_symptom_dict


def add_rand_infected_simulation(df_forms: DataFrame,
                                 subset_sel: pd.Series,
                                 rand_seed: int,
                                 cfg: CoronaSimCfg
                                 ) -> DataFrame:
    """
    Randomly replaces fraction of healthy report with Corona report

    Current limitations:
        Do not simulate correlation between symptoms
        Do not simulate some of the symptoms
        Ignores age
        body_temp is not modified, but only symptom_fever

    Args:
        df_forms: We don't modify it here, but return a modified copy
        subset_sel: Select subset of df_forms (logical)
            Use it to select city/neighborhoods and dates
        rand_seed:
        cfg:

    Returns:

    """
    df_forms = df_forms.copy()  # We don't modify the source
    rand_stream = np.random.RandomState(seed=rand_seed)

    no_symp_subset =  subset_sel & (~df_forms.symptom_any)
    # p_report_if_infected_increase_factor is used to artificial increase in probability of adding symptoms.
    # This is approximation to the fact that sick individual is more likely to report.
    # The overall process may be complicated, but this approximation is good enough for p_infection << 1.0
    p = cfg.p_infection * (1. - cfg.p_asymptomatic_override) * cfg.p_report_if_infected_increase_factor
    # True for subset that we want to randomly add symptoms (based on cfg)
    modify_to_sick_subset = no_symp_subset & (rand_stream.rand(df_forms.shape[0]) < p)

    for symp, p in cfg.p_symptom_dict.items():
        set_symp_flag = modify_to_sick_subset & (rand_stream.rand(df_forms.shape[0]) < p)
        assert symp in df_forms.columns
        df_forms.loc[set_symp_flag, symp] = 1.0

    df_forms = calc_symtom_ratio(df_forms)
    df_forms.symptom_any = np.any(df_forms[get_all_symptoms_list()], axis=1)
    df_forms['symptom_ratio_weighted_sqr'] = df_forms.symptom_ratio_weighted ** 2

    # TOOD: Do we need to update any other columns ?

    return df_forms


class RocEstCalculatorCfg():

    def __init__(self,
                 corona_sim_cfg: CoronaSimCfg,
                 corona_from_date: str,
                 extract_metric_func = 'symptom_ratio_weighted',
                 epochs_num: int = 1,
                 rand_seed: int = 47891
                 ):
        """

        Args:
            corona_sim_cfg:
            extract_metric_func: What is the function for corona detection per group
                Can be one of:
                * Column used as metric
                * Callback function that calculate the metrics for group.
                    This is used as argument to GroupBy.apply
            corona_from_date_str: e.g., '2020-03-22'. Starting from this day randomize corona symptoms in the
                    groups
            epochs_num: How much time to go through each group with different random seed
            rand_seed:
        """
        self.corona_sim_cfg = corona_sim_cfg
        self.corona_from_date_str = corona_from_date
        self.extract_metric_func = extract_metric_func
        self.epochs_num = epochs_num
        self.rand_seed = rand_seed


def roc_est_calculator(df_groups: DataFrameGroupBy,
                       cfg: RocEstCalculatorCfg) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        df_groups:
        cfg:

    Returns:

    """
    if isinstance(cfg.extract_metric_func, str):
        base_metric = df_groups[cfg.extract_metric_func].mean()
    else:
        raise NotImplementedError('Implemented, but not tested. Test before using it and remove this raise')
        base_metric = df_groups.apply(cfg.extract_metric_func)

    corona_from_datetime = pd.to_datetime(cfg.corona_from_date_str)

    fp_for_tp_list = []
    cur_rand_seed = cfg.rand_seed
    for i_epoch in range(cfg.epochs_num):
        for i_grp, (group_id, df_grp) in enumerate(df_groups):
            cur_rand_seed = cur_rand_seed + 1
            df_grp_plus_corona = add_rand_infected_simulation(
                df_grp, df_grp.datetime >= corona_from_datetime,
                cur_rand_seed, cfg.corona_sim_cfg)
            if isinstance(cfg.extract_metric_func, str):
                cur_metric = df_grp_plus_corona[cfg.extract_metric_func].mean()
            else:
                cur_metric = cfg.extract_metric_func(df_grp_plus_corona)
            # We added corona, hence expect the metric to increase or be the same
            assert cur_metric >= base_metric.iloc[i_grp] - 1e-7
            if cfg.corona_sim_cfg.p_infection == 0.:
                # For probability 0 infection, we don't expect any change:
                assert np.isclose(cur_metric, base_metric.iloc[i_grp])
            tmp = base_metric.copy().drop(index=base_metric.index[i_grp])
            assert tmp.shape[0] == base_metric.shape[0] - 1
            # The following is the minimal FP threshold such that cur_metric will be above threshold
            cur_detected_for_fp_th = np.sum(tmp >= cur_metric) / (base_metric.shape[0] - 1)
            fp_for_tp_list.append(cur_detected_for_fp_th)

    fp_vec = np.asarray(sorted(fp_for_tp_list))
    tp_vec = np.linspace(0.0, fp_vec.size - 0.0, fp_vec.size) / fp_vec.size

    return fp_vec, tp_vec
