import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config as cfg
from pandas import concat
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from ds_charts import multiple_bar_chart
from math import sqrt

NR_COLUMNS: int = 3
HEIGHT: int = 4

PREDICTION_MEASURES = {
    'MSE': mean_squared_error,
    'MAE': mean_absolute_error,
    'MAPE': mean_absolute_percentage_error,
    'R2': r2_score
    }

def create_temporal_dataset(df, target, nr_instants, filename):
    N = len(df)
    index = df.index.name
    df2 = df.copy()
    cols = []
    for i in range(nr_instants+1):
        col = df2.copy()
        col = col.iloc[i:N-nr_instants+i]
        col = col.reset_index()
        col.drop(index, axis=1, inplace=True)
        cols.append(col)

    new_df = concat(cols, axis=1, ignore_index=True)
    new_df.columns = [f'T{i}' for i in range(1, nr_instants+1)] + [target]
    new_df.index = df.index[nr_instants:]
    new_df.index.name = index
    new_df.to_csv(filename)

    return new_df

def split_temporal_data(X, y, trn_pct=0.70):
    trn_size = int(len(X) * trn_pct)
    trnX, trnY = X[:trn_size], y[:trn_size]
    tstX, tstY = X[trn_size:], y[trn_size:]
    return trnX, tstX, trnY, tstY


def plot_evaluation_results(trn_y, prd_trn, tst_y, prd_tst, figname):
    eval1 = {
        'RMSE': [sqrt(PREDICTION_MEASURES['MSE'](trn_y, prd_trn)), sqrt(PREDICTION_MEASURES['MSE'](tst_y, prd_tst))],
        'MAE': [PREDICTION_MEASURES['MAE'](trn_y, prd_trn), PREDICTION_MEASURES['MAE'](tst_y, prd_tst)]
        }
    eval2 = {
        'MAPE': [sqrt(PREDICTION_MEASURES['MAPE'](trn_y, prd_trn)), sqrt(PREDICTION_MEASURES['MAPE'](tst_y, prd_tst))],
        'R2': [PREDICTION_MEASURES['R2'](trn_y, prd_trn), PREDICTION_MEASURES['R2'](tst_y, prd_tst)]
    }

    print(eval1, eval2)
    fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    multiple_bar_chart(['Train', 'Test'], eval1, ax=axs[0], title="Predictor's performance over Train and Test sets", percentage=False)
    multiple_bar_chart(['Train', 'Test'], eval2, ax=axs[1], title="Predictor's performance over Train and Test sets", percentage=True)
    fig.savefig(figname)

def plot_series(series, ax: plt.Axes = None, title: str = '', x_label: str = '', y_label: str = '',
                percentage=False, show_std=False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    if isinstance(series, dict):
        legend: list = []
        i = 0
        for name in series.keys():
            y = series[name]
            ax.set_xlim(y.index[0], y.index[-1])
            std = y.std()
            ax.plot(y, c=cfg.ACTIVE_COLORS[i], label=name)
            if show_std:
                y1 = y.add(-std)
                y2 = y.add(std)
                ax.fill_between(y.index, y1.values, y2.values, color=cfg.ACTIVE_COLORS[i], alpha=0.2)
            i += 1
            legend.append(name)
        ax.legend(legend)
    else:
        ax.plot(series)

    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)