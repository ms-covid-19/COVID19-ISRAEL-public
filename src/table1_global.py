import pandas as pd
import numpy as np
import os


def _make_tmp_row(data_all, label, col, agg_action, TX_col, TX_labels, decimals=1, row_type='continuous'):
    tmp = data_all.groupby(TX_col)[col].agg(agg_action).rename(TX_labels)

    if row_type == 'continuous':
        tmp['All patients'] = data_all[col].agg(agg_action)
        tmp = (tmp.round(decimals=decimals).rename(label))
        tmp = tmp.apply(lambda x: '{:,}'.format(x))
    elif row_type == 'count':
        tmp['All patients'] = data_all[col].agg(agg_action)
        tmp = (tmp.rename(label).astype(int))
        tmp = tmp.apply(lambda x: '{:,}'.format(x))
    #         tmp = (tmp.rename(label).astype(int)).astype(str)
    elif row_type == 'percent':
        tmp['All patients'] = data_all[col].agg(agg_action)
        tmp = ((100 * tmp).round(decimals=decimals).rename(label)).astype(str) + '%'
    return tmp


def _add_table1_row(table1, data_all, label, col, TX_col, TX_labels, var_type='continuous', decimals=1):
    if var_type == 'continuous':
        row_type1 = 'continuous'
        row_type2 = 'continuous'
        agg_mean = 'mean'
        agg_std = 'std'
        mean_val = _make_tmp_row(data_all, label, col, agg_mean, TX_col, TX_labels, decimals, row_type1)
        std_val = _make_tmp_row(data_all, label, col, agg_std, TX_col, TX_labels, decimals, row_type2)
        tmp = mean_val + ' (' + std_val + ')'
        table1 = table1.append(tmp)

    elif var_type == 'count_perc':
        row_type1 = 'count'
        row_type2 = 'percent'
        agg_sum = 'sum'
        agg_mean = 'mean'
        sum_val = _make_tmp_row(data_all, label, col, agg_sum, TX_col, TX_labels, decimals, row_type1)
        mean_val = _make_tmp_row(data_all, label, col, agg_mean, TX_col, TX_labels, decimals, row_type2)
        tmp = sum_val + ' (' + mean_val + ')'
        table1 = table1.append(tmp)

    elif var_type == 'count':
        row_type = 'count'
        agg_count = 'count'
        count_val = _make_tmp_row(data_all, label, col, agg_count, TX_col, TX_labels, decimals, row_type)
        table1 = table1.append(count_val.rename(label))

    elif var_type == 'N':
        agg_count = 'count'
        tmp = data_all.groupby(TX_col)[col].agg(agg_count).rename(TX_labels)
        N_tot = data_all[col].agg(agg_count)
        tmp2 = (100 * (tmp / N_tot)).round(decimals=decimals).rename(label).astype(str) + '%'
        tmp = (tmp.rename(label).astype(int))
        tmp = tmp.apply(lambda x: '{:,}'.format(x))
        tmp = tmp + ' (' + tmp2 + ')'
        tmp['All patients'] = '{:,}'.format(N_tot)
        table1 = table1.append(tmp)

    return table1


def make_table1(data_all, cols, labels, col_types, TX_col, TX_labels, decimals=1):
    table1 = pd.DataFrame(columns=list(TX_labels.values()) + ['All patients'])

    label = 'N'
    col = TX_col
    table1 = _add_table1_row(table1, data_all, label, col, TX_col, TX_labels, var_type='N', decimals=decimals)

    for col, label, col_type in list(zip(cols, labels, col_types)):
        table1 = _add_table1_row(table1, data_all, label, col, TX_col, TX_labels, var_type=col_type, decimals=decimals)

    return (table1)
