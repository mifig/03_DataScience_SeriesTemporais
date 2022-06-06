from matplotlib.pyplot import subplots, Axes, gca
import matplotlib.dates as mdates
import config as cfg
from pandas import concat, Series
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ds_charts import multiple_bar_chart
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from numpy import ndarray, array

NR_COLUMNS: int = 3
HEIGHT: int = 4

PREDICTION_MEASURES = {
    'MSE': mean_squared_error,
    'MAE': mean_absolute_error,
    'R2': r2_score
    }

def sliding_window(df, seq_length = 4):
    df = df.copy()
    x = []
    y = []
    for i in range(len(df)-seq_length-1):
        _x = df.iloc[i:(i+seq_length), :]
        _y = df.iloc[i+seq_length,:]
        x.append(_x)
        y.append(_y)
    return array(x), array(y)

def shift_target(data, variable, target_name, forecast_lead=15):
    df = data.copy()
    df[target_name] = data[variable].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]

    return df

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

def split_temporal_data(data, target, trn_pct=0.70):
    df = data.copy()
    trn_size = int(len(df) * trn_pct)

    y: ndarray = df.pop(target).values
    X: ndarray = df.values

    trnX, trnY = X[:trn_size], y[:trn_size]
    tstX, tstY = X[trn_size:], y[trn_size:]
    return trnX, tstX, trnY, tstY

def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train = df_cp.iloc[:trn_size, :]
    test = df_cp.iloc[trn_size:]
    return train, test

def plot_evaluation_results(trn_y, prd_trn, tst_y, prd_tst, figname):
    eval1 = {
        'RMSE': [sqrt(PREDICTION_MEASURES['MSE'](trn_y, prd_trn)), sqrt(PREDICTION_MEASURES['MSE'](tst_y, prd_tst))],
        'MAE': [PREDICTION_MEASURES['MAE'](trn_y, prd_trn), PREDICTION_MEASURES['MAE'](tst_y, prd_tst)]
        }
    eval2 = {
        'R2': [PREDICTION_MEASURES['R2'](trn_y, prd_trn), PREDICTION_MEASURES['R2'](tst_y, prd_tst)]
    }

    print(eval1, eval2)
    _, axs = subplots(1, 2)
    multiple_bar_chart(['Train', 'Test'], eval1, ax=axs[0], title="Predictor's performance", percentage=False)
    multiple_bar_chart(['Train', 'Test'], eval2, ax=axs[1], title="Predictor's performance", percentage=False)

def plot_forecasting_series(trn, tst, prd_trn, prd_tst, figname: str, x_label: str = 'time', y_label:str =''):
    _, ax = subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(figname)
    ax.plot(trn.index, trn, label='train', color='b')
    ax.plot(trn.index, prd_trn, '--y', label='train prediction')
    ax.plot(tst.index, tst, label='test', color='g')
    ax.plot(tst.index, prd_tst, '--r', label='test prediction')
    ax.legend(prop={'size': 5})

def plot_series(series, ax: Axes = None, title: str = '', x_label: str = '', y_label: str = '',
                percentage=False, show_std=False):
    if ax is None:
        ax = gca()
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

def plot_components(series: Series,  x_label: str = 'time', y_label:str =''):
    decomposition = seasonal_decompose(series, model = "add")
    lst = [('Observed', series), ('trend', decomposition.trend), ('seasonal', decomposition.seasonal), ('residual', decomposition.resid)]
    _, axs = subplots(len(lst), 1, figsize=(3*HEIGHT, HEIGHT*len(lst)))
    for i in range(len(lst)):
        axs[i].set_title(lst[i][0])
        axs[i].set_ylabel(y_label)
        axs[i].set_xlabel(x_label)
        axs[i].plot(lst[i][1])
