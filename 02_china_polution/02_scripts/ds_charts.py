import math
import itertools
from numpy import arange, ndarray, newaxis, set_printoptions, isnan,datetime64
from pandas import DataFrame, concat, unique
import matplotlib.pyplot as plt
from matplotlib.dates import _reset_epoch_test_example, set_epoch, AutoDateLocator, AutoDateFormatter
from warnings import simplefilter
import sklearn.metrics as metrics
import config as cfg
from datetime import datetime
from sklearn.tree import export_graphviz
import matplotlib.font_manager as fm
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


FONT_TEXT = fm.FontProperties(size=6)
TEXT_MARGIN = 0.05

_reset_epoch_test_example()
set_epoch('0000-12-31T00:00:00')  # old epoch (pre MPL 3.3)

simplefilter("ignore")
NR_COLUMNS: int = 3
HEIGHT: int = 4
WIDTH_PER_VARIABLE: int = 0.5


def choose_grid(nr):
    if nr < NR_COLUMNS:
        return 1, nr
    else:
        return (nr // NR_COLUMNS, NR_COLUMNS) if nr % NR_COLUMNS == 0 else (nr // NR_COLUMNS + 1, NR_COLUMNS)


def set_elements(ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '', percentage: bool = False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    return ax


def set_locators(xvalues: list, ax: plt.Axes = None, rotation: bool=False):
    if isinstance(xvalues[0], datetime):
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(AutoDateFormatter(locator, defaultfmt='%Y-%m-%d'))
        return None
    elif isinstance(xvalues[0], str):
        ax.set_xticks(arange(len(xvalues)))
        if rotation:
            ax.set_xticklabels(xvalues, rotation='90', fontsize='small', ha='center')
        else:
            ax.set_xticklabels(xvalues, fontsize='small', ha='center')
        return None
    else:
        ax.set_xlim(xvalues[0], xvalues[-1])
        ax.set_xticks(xvalues)
        return None


def plot_line(xvalues: list, yvalues: list, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '',
              percentage: bool = False, rotation: bool = False):
    ax = set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    set_locators(xvalues, ax=ax, rotation=rotation)
    ax.plot(xvalues, yvalues, c=cfg.LINE_COLOR)


def multiple_line_chart(xvalues: list, yvalues: dict, ax: plt.Axes = None, title: str = '', xlabel: str = '',
                        ylabel: str = '', percentage: bool = False, rotation: bool = False):
    ax = set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    set_locators(xvalues, ax=ax, rotation=rotation)
    legend: list = []
    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend)


def bar_chart(xvalues: list, yvalues: list, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '',
              percentage: bool = False, rotation: bool = False):
    ax = set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    set_locators(xvalues, ax=ax, rotation=rotation)
    ax.bar(xvalues, yvalues, edgecolor=cfg.LINE_COLOR, color=cfg.FILL_COLOR, tick_label=xvalues)
    for i in range(len(yvalues)):
        ax.text(i, yvalues[i] + TEXT_MARGIN, f'{yvalues[i]:.2f}', ha='center', fontproperties=FONT_TEXT)


def multiple_bar_chart(xvalues: list, yvalues: dict, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '',
                       percentage: bool = False):
    ax = set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    ngroups = len(xvalues)
    nseries = len(yvalues)
    pos_group = arange(ngroups)
    width = 0.8 / nseries
    pos_center = pos_group + (nseries-1)*width/2
    ax.set_xticks(pos_center)
    ax.set_xticklabels(xvalues)
    i = 0
    legend = []
    for metric in yvalues:
        ax.bar(pos_group, yvalues[metric], width=width, edgecolor=cfg.LINE_COLOR, color=cfg.ACTIVE_COLORS[i])
        values = yvalues[metric]
        legend.append(metric)
        for k in range(len(values)):
            ax.text(pos_group[k], values[k] + TEXT_MARGIN, f'{values[k]:.2f}', ha='center', fontproperties=FONT_TEXT)
        pos_group = pos_group + width
        i += 1
    ax.legend(legend, fontsize='x-small', title_fontsize='small')


def plot_evaluation_results(labels: ndarray, trn_y, prd_trn, tst_y, prd_tst):
    cnf_mtx_trn = metrics.confusion_matrix(y_true=trn_y, y_pred=prd_trn, labels=labels)
    tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
    cnf_mtx_tst = metrics.confusion_matrix(y_true=tst_y, y_pred=prd_tst, labels=labels)
    tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()

    evaluation = {'Accuracy': [(tn_trn + tp_trn) / (tn_trn + tp_trn + fp_trn + fn_trn),
                               (tn_tst + tp_tst) / (tn_tst + tp_tst + fp_tst + fn_tst)],
                  'Recall': [tp_trn / (tp_trn + fn_trn), tp_tst / (tp_tst + fn_tst)],
                  'Specificity': [tn_trn / (tn_trn + fp_trn), tn_tst / (tn_tst + fp_tst)],
                  'Precision': [tp_trn / (tp_trn + fp_trn), tp_tst / (tp_tst + fp_tst)]}

    fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets",
                       percentage=True)
    plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1], title='Test')


def horizontal_bar_chart(elements: list, values: list, error: list, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = ''):
    ax = set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    y_pos = arange(len(elements))

    ax.barh(y_pos, values, xerr=error, align='center', error_kw={'lw': 0.5, 'ecolor': 'r'})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(elements)
    ax.invert_yaxis()  # labels read top-to-bottom


def plot_confusion_matrix(cnf_matrix: ndarray, classes_names: ndarray, ax: plt.Axes = None, normalize: bool = False, title=''):
    if ax is None:
        ax = plt.gca()
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, newaxis]
        cm = cnf_matrix.astype('float') / total
        title += " - normalized confusion matrix"
    else:
        cm = cnf_matrix
        title += ' - confusion matrix'
    set_printoptions(precision=2)

    set_elements(ax=ax, title=title, xlabel='Predicted label', ylabel='True label', percentage=False)
    tick_marks = arange(0, len(classes_names), 1)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=cfg.cmap_blues)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), color='w', horizontalalignment="center")


def plot_roc_chart(models: dict, tstX: ndarray, tstY: ndarray, ax: plt.Axes = None, target: str = 'class'):
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    set_elements(ax=ax, title=f'ROC chart for {target}', xlabel='FP rate', ylabel='TP rate')

    ax.plot([0, 1], [0, 1], color='navy', label='random', linewidth=1, linestyle='--',  marker='')
    for clf in models.keys():
        metrics.plot_roc_curve(models[clf], tstX, tstY, ax=ax, marker='', linewidth=1)
    ax.legend(loc="lower right")


def plot_tree(tree, variables: list, labels: list, out_filename: str):
    export_graphviz(tree, out_file=out_filename+'.dot', feature_names=variables, class_names=labels,
                    filled=True, rounded=True, special_characters=True)
    # Convert to png
    from subprocess import call
    call(['dot', '-Tpng', out_filename+'.dot', '-o', out_filename+'.png', '-Gdpi=600'])
    plt.figure(figsize=(14, 18))
    plt.imshow(plt.imread(out_filename+'.png'))
    plt.axis('off')


def plot_clusters(data, var1st, var2nd, clusters, centers: list, n_clusters: int, title: str,  ax: plt.Axes = None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(data.iloc[:, var1st], data.iloc[:, var2nd], c=clusters, alpha=0.5, cmap=cfg.cmap_active)
    if centers is not None:
        for k, col in zip(range(n_clusters), cfg.ACTIVE_COLORS):
            cluster_center = centers[k]
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('var' + str(var1st), fontsize=8)
    ax.set_ylabel('var' + str(var2nd), fontsize=8)


def compute_centroids(data: DataFrame, labels: ndarray) -> list:
    n_vars = data.shape[1]
    ext_data = concat([data, DataFrame(labels)], axis=1)
    ext_data.columns = list(data.columns) + ['cluster']
    clusters = unique(labels)
    n_clusters = len(clusters)
    centers = [0] * n_clusters
    for k in range(-1, n_clusters):
        if k != -1:
            cluster = ext_data[ext_data['cluster'] == k]
            centers[k] = list(cluster.sum(axis=0))
            centers[k] = [centers[k][j]/len(cluster) if len(cluster) > 0 else 0 for j in range(n_vars)]
        else:
            centers[k] = [0]*n_vars

    return centers


def compute_mse(X: ndarray, labels: list, centroids: list) -> float:
    n = len(X)
    centroid_per_record = [centroids[labels[i]] for i in range(n)]
    partial = X - centroid_per_record
    partial = list(partial * partial)
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return math.sqrt(partial) / (n-1)


def two_scales(ax1, time, data1, data2, c1, c2, xlabel='', ylabel1='', ylabel2=''):
    ax2 = ax1.twinx()
    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel(ylabel2)
    return ax1, ax2


def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df


def get_variable_types(df):
    NR_SYMBOLS = 10
    variable_types = {'binary': [], 'numeric': [], 'date': [], 'symbolic': []}
    for c in df.columns:
        mv = df[c].isna().sum()
        uniques = df[c].unique()

        if mv == 0:
            if len(uniques) == 2:
                variable_types['binary'].append(c)
                df[c].astype('bool')
            elif df[c].dtype == 'datetime64[ns]':
                variable_types['date'].append(c)
            elif len(uniques) < NR_SYMBOLS:
                df[c].astype('category')
                variable_types['symbolic'].append(c)
            elif str(df[c].dtype) == 'category':
                variable_types['symbolic'].append(c)
            else:
                variable_types['numeric'].append(c)
        else:
            uniques = [v for v in uniques if not pd.isnull(v)]
            values = [v for v in uniques if isinstance(v,str)]
            if len(uniques) == 2:
                variable_types['binary'].append(c)
            elif len(values) == len(uniques):
                df[c].astype('category')
                variable_types['symbolic'].append(c)
            else:
                values = [v for v in uniques if isinstance(v, datetime64)]
                if len(values) == len(uniques):
                    variable_types['date'].append(c)
                else:
                    variable_types['numeric'].append(c)
    return variable_types

def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, pct=True):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    plt.figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=pct)
    plt.savefig('../03_images/overfitting_{name}.png')