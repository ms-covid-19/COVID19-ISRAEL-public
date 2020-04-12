import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import statsmodels.api as sm
import shap

from src.individual_level_models.defs import MODELS_DIR


def get_shap_categories(shap_values, features, features_categories, feat_col, cat_col):
    """
    Group shap values into feature category groups

    Parameters
    ----------
    shap_values: SHAP values numpy array
    features: pandas dataframe of X
    features_categories: a pandas df that contains feature names in feat_col and categories in cat_col
    feat_col: name of column with features names
    cat_col: name of column withcategories names


    Returns
    -------
    shap_values_cats: pandas df with grouped shap values into feature category groups

    """
    # Make shap_values a Dataframe with named columns
    shap_values = pd.DataFrame(data=shap_values, columns=features.columns)

    # retain only relevant columns in feature_categories
    features_categories = features_categories[features_categories[feat_col].isin(features.columns)]

    shap_values_cats = pd.DataFrame(index=shap_values.index)
    for cat_val in features_categories[cat_col].unique():
        cols = list(features_categories.loc[features_categories[cat_col] == cat_val, feat_col])
        shap_values_cats.loc[:, cat_val] = shap_values[cols].sum(axis=1)

    #     shap_values_mean_abs = shap_values_cats_df.abs().mean().sort_values(ascending=False).to_frame().reset_index()
    #     shap_values_mean_abs.columns = ['Feature category', 'Mean absolute SHAP']

    return shap_values_cats


def plot_shap_summary_indication(model, X_train_0, X_train_1, X_train_2, model_name='XGB', out_dir=MODELS_DIR):
    shap_values_0 = shap.TreeExplainer(model).shap_values(X_train_0)
    shap_values_1 = shap.TreeExplainer(model).shap_values(X_train_1)
    shap_values_2 = shap.TreeExplainer(model).shap_values(X_train_2)

    shap_values_df_0 = pd.DataFrame(shap_values_0, columns=X_train_0.columns)
    shap_values_df_1 = pd.DataFrame(shap_values_1, columns=X_train_1.columns)
    shap_values_df_2 = pd.DataFrame(shap_values_2, columns=X_train_2.columns)

    data_0 = shap_values_df_0.abs().mean(axis=0).reset_index()
    data_0.columns = ['Feature category', 'Other']

    data_1 = shap_values_df_1.abs().mean(axis=0).reset_index()
    data_1.columns = ['Feature category', 'Abroad']

    data_2 = shap_values_df_2.abs().mean(axis=0).reset_index()
    data_2.columns = ['Feature category', 'Contact with confirmed']

    all_data = pd.merge(data_0, data_1, on='Feature category')
    all_data = pd.merge(all_data, data_2, on='Feature category')
    all_data.sort_values('Contact with confirmed', ascending=False, inplace=True)
    melted_data = pd.melt(all_data, id_vars="Feature category", var_name="Test Indication",
                          value_name="Mean absolute SHAP")

    fig, ax = plt.subplots(1, 1, figsize=(10, len(melted_data) // 2))
    sns.barplot(y='Feature category', x='Mean absolute SHAP', hue='Test Indication', data=melted_data, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '{}_SHAP_test_indication.png'.format(model_name)))
    plt.close()


def plot_shap_summary_ages(model, X_train_young, X_train_old, model_name='XGB', out_dir=MODELS_DIR):
    shap_values_young = shap.TreeExplainer(model).shap_values(X_train_young)
    shap_values_old = shap.TreeExplainer(model).shap_values(X_train_old)
    shap_values_df_young = pd.DataFrame(shap_values_young, columns=X_train_young.columns)
    shap_values_df_old = pd.DataFrame(shap_values_old, columns=X_train_old.columns)

    data_young = shap_values_df_young.abs().mean(axis=0).reset_index()
    data_young.columns = ['Feature category', 'Age - under 60']

    data_old = shap_values_df_old.abs().mean(axis=0).reset_index()
    data_old.columns = ['Feature category', 'Age - above 60']

    all_data = pd.merge(data_old, data_young, on='Feature category')
    all_data.sort_values('Age - above 60', ascending=False, inplace=True)
    melted_data = pd.melt(all_data, id_vars="Feature category", var_name="Age", value_name="Mean absolute SHAP")

    fig, ax = plt.subplots(1, 1, figsize=(10, len(melted_data) // 2))
    sns.barplot(y='Feature category', x='Mean absolute SHAP', hue='Age', data=melted_data, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '{}_SHAP_ages.png'.format(model_name)))
    plt.close()


def plot_shap_summary(model, X_train, model_name='XGB'):
    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    shap_values_df = pd.DataFrame(shap_values, columns=X_train.columns)
    plot_shap_summary_bar(shap_values_df, model_name)


def plot_shap_summary_bar(shap_values_df, model_name, ax=None, sz=14, out_dir=MODELS_DIR):
    """
    Plot bar plot of mean abosulute SHAP values

    Parameters
    ----------
    shap_values_df: pandas dataframe of SHAP values
    ax: matplotlib axes
    sz: size for fonts
    out_dir: path to direrctory were plots will be saved

    Returns
    -------
    ax: matplotlib axes

    """
    data = shap_values_df.abs().mean(axis=0).reset_index()
    data.columns = ['Feature category', 'Mean absolute SHAP']
    data.sort_values('Mean absolute SHAP', ascending=False, inplace=True)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, len(data) // 2))

    sns.barplot(y='Feature category', x='Mean absolute SHAP', data=data, color=sns.color_palette('coolwarm')[0], ax=ax)

    axis_color = '#333333'
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(bottom=True, left=False)
    ax.tick_params('x', labelsize=sz)
    ax.tick_params('y', labelsize=sz)
    ax.set_ylabel('')
    ax.set_xlabel('Mean absolute Shapely value', fontsize=sz)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '{}_SHAP_vals.png'.format(model_name)))
    plt.close()


def convert_shap_values(y, base_shap, y_transform=None):
    """
    Transform SHAP values from log-odds scale to other scales

    Parameters
    ----------
    y: shap_values array
    base_shap: For use with convert_to_risk. Available through
        shap.TreeExplainer(predictor).expected_value
    y_transform: transformation of the logodds SHAP vlaues to one of the following:
                'relative risk', 'odds', 'probability', or log relative risk'

    Returns
    -------
    y, y_lbl: transformed y and the corresponding label of the transform

    """

    # Convert y
    if y_transform == 'log odds':
        # retain original SHAP value in y (log Odds)
        y_lbl = 'Shapely Value'
    elif y_transform == 'relative risk':
        y_lbl = 'Relative Risk'
        y = expit(base_shap + y) / expit(base_shap)
    elif y_transform == 'odds':
        y_lbl = 'Odds'
        y = np.exp(y)
    elif y_transform == 'probability':
        y_lbl = 'Probabilty'
        tmp = np.exp(y)
        y = tmp / (1 + tmp)
    elif y_transform == 'log relative risk':
        y_lbl = 'Log Relative Risk'
        y = np.log(expit(base_shap + y) / expit(base_shap))
    else:
        # retain original SHAP value in y (log Odds)
        y_lbl = 'Shapely Value'
    return y, y_lbl


def _add_bottom_hist(ax, x, bins=None, hist_color='k'):
    ax2 = ax.twinx()
    sns.distplot(x, bins=bins, kde=False, norm_hist=True,
                 ax=ax2, hist_kws={'alpha': 0.2}, color=hist_color)
    # Visuals
    ax2.get_yaxis().set_visible(False)
    # make room in the bottom of original plot
    ax.set_ylim(ax.get_ylim()[0] - ((ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.25), ax.get_ylim()[1])
    # shrink histogram height
    ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1] * 5)


def dep_plot(feature_name, shap_values, features, shap_columns=None, ax=None,
             feature_text=None, shap_text=None,
             base_shap=None, y_transform=None, add_nan=False, nan_label='NaN',
             x_min=None, x_max=None, x_round=None, manual_bins=None,
             x_quantile_min=None, x_quantile_max=None,
             plot_hist=True, plot_type=None,
             SZ=16, color=sns.color_palette()[0], hist_color='k', smoother_color='r'):
    """
    Custom version of SHAP's dependence plot

    Parameters
    ----------
    feature_name: relevant column in features and in shap_values
    shap_values: SHAP values numpy array
    features: pandas dataframe of X
    shap_columns: if sum of multiple shap columns is wanted on the y axis
    ax: axis
    feature_text: xlabel
    shap_text: ylabel
    base_shap: For use with convert_to_risk. Available through
        shap.TreeExplainer(predictor).expected_value
    y_transform: transformation of the logodds SHAP vlaues to one of the following:
                'relative risk', 'odds', 'probability', or log relative risk'
    add_nan: add shap values for nan features on the left of the plot
    nan_label: label on x axis for nans
    x_min, x_max: minimal/maximal feature value to plot
    x_round: rounding order of magnitude. Use some rounding in continuous features when using lineplot
    manual_bins: set manual bins for plotting lineplot width
    x_quantile_min, x_quantile_max: filter min/max feature value by feature quantile given here
    plot_hist: plot feature value histogram at bottom of plot
    plot_type: None or  'scatter' plots all datapoints in a scatterplot. 'lowess' adds a smoother, 'lineplot' usese seaborn lineplot.
    SZ: label fontsize
    color: color
    hist_color: color for histogram

    Examples of usage:
    -----------------
    TBD

    """
    # Make shap_values a Dataframe with named columns
    shap_values = pd.DataFrame(data=shap_values, columns=features.columns)

    if ax is None:
        ax = plt.gca()

    # Define xy for plotting
    x_raw = features[feature_name].rename('x')
    # copy original x for processing
    x = x_raw
    # y can be summed from multiple columns or not
    if shap_columns is None:
        y = shap_values[feature_name].rename('y')
    else:
        # intersect of existing shap columns
        shap_columns = list(set((features.columns).intersection(shap_columns)))
        y = shap_values[shap_columns].sum(axis=1).rename('y')

    # Transform y according to inputs
    y, y_lbl = convert_shap_values(y, base_shap, y_transform)

    # Alter x according to inputs
    if x_quantile_max is not None:
        x = x[x <= x.quantile(x_quantile_max)]
    if x_quantile_min is not None:
        x = x[x >= x.quantile(x_quantile_min)]
    if x_min is not None:
        x = x[(x >= x_min)]
    if x_max is not None:
        x = x[(x <= x_max)]
    if x_round is not None:
        # round x points
        x = x.round(x_round)

    # For given manual bins of x
    if (x_round is None) & (manual_bins is not None):
        # if a single number N is given, then build N bins
        if type(manual_bins) == int:
            # create bins
            cuts, bins = pd.cut(x, bins=manual_bins, retbins=True)
        # if a list is given, then build bins from list
        else:
            bins = manual_bins
        bins_mid_x = np.array([(a + b) / 2 for a, b, in zip(bins[:-1], bins[1:])])
        # find closest bin middle and transform x
        x = x.apply(lambda xx: bins_mid_x[np.abs(xx - bins_mid_x).argmin()])

    # Plot type
    if plot_type == 'lowess':
        ax.scatter(x_raw, y, s=8, color=color, facecolors='none')
        lowess = sm.nonparametric.lowess
        z = lowess(y, x_raw, frac=1 / 8)
        ax.plot(z[:, 0], z[:, 1], color=smoother_color, lw=4, alpha=0.5)
    elif plot_type == 'lineplot':
        sns.lineplot(data=pd.concat((x, y), axis=1), x='x', y='y', ci=None, color=color, ax=ax)
        ax.get_legend().remove()
    else:
        ax.scatter(x_raw, y, s=8, color=color, facecolors='none')

    if feature_text is not None:
        xlabel = feature_text
    else:
        xlabel = "Feature Value"

    if shap_text is None:
        ylabel = y_lbl
    else:
        ylabel = f'{y_lbl} for {shap_text}'

    # Adjust plot
    ax.set_xlabel(xlabel, size=SZ)
    ax.set_ylabel(ylabel, size=SZ)
    sns.despine(right=True, top=True, bottom=False, ax=ax)
    ax.set_xlim(x.min(), x.max())
    if y_transform == 'relative risk':
        ax.axhline(y=1.0, color='k', ls='--', lw=1, alpha=0.4)

    # plot bottom histogram if required
    if plot_hist:
        _add_bottom_hist(ax, x, bins=manual_bins, hist_color=hist_color)

    # Add NaN item if requested
    if add_nan:

        if shap_columns is None:
            y_nan = shap_values[feature_name][features[feature_name].isna()].rename('y')
        else:
            y_nan = shap_values[shap_columns].sum(axis=1)[features[feature_name].isna()].rename('y')
        y_nan, _ = convert_shap_values(y_nan, base_shap, y_transform)

        xlim = ax.get_xlim()
        xticks = ax.get_xticks()
        xticklabels = ax.get_xticklabels()
        x_nan = [xlim[0] - 0.06 * (xlim[1] - xlim[0]), xlim[0] - 0.01 * (xlim[1] - xlim[0])]
        xticklabels = [nan_label] + [str(item) for item in xticks[1:]]
        xticks = [(x_nan[0] + x_nan[1]) / 2] + list(xticks)[1:]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.plot(x_nan, [y_nan.mean(), y_nan.mean()], color=color, lw=2)
        ax.fill_between(x_nan,
                        [y_nan.mean() - y_nan.std(), y_nan.mean() - y_nan.std()],
                        [y_nan.mean() + y_nan.std(), y_nan.mean() + y_nan.std()], alpha=0.1, color=color)
        ax.axvline(xlim[0], color='k', lw=1)
        ax.set_xlim(xlim[0] - 0.07 * (xlim[1] - xlim[0]), xlim[1])
    ax.tick_params(labelsize=SZ - 2)
