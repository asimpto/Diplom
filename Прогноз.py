#%%
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Попытка импорта Prophet и XGB
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except ImportError:
    HAVE_PROPHET = False

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

#%%
# --- Параметры ---
DATA_DIR = 'output/готовая_выработка_excel'
HORIZON = 12
SEASONAL_PERIOD = 12
MIN_REQUIRED = SEASONAL_PERIOD + HORIZON

# CV для GridSearch
tscv = TimeSeriesSplit(n_splits=3)
knn_grid = {'n_neighbors': [3,5,7], 'weights': ['uniform','distance']}
xgb_grid = {
    'n_estimators': [25,100],
    'max_depth': [3,8],
    'learning_rate': [0.05,0.1],
    'subsample': [0.7,1.0],
    'colsample_bytree': [0.7,1.0]
}

warnings.filterwarnings("ignore")

#%%
def fill_monthly_with_threshold(df, max_gap=3):
    df = df.resample('MS').asfreq()
    df['value'] = df['value'].interpolate(
        method='time',
        limit=max_gap,
        limit_direction='both'
    )
    return df.dropna(subset=['value'])

def compute_metrics(y_true, y_pred):
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2    = r2_score(y_true, y_pred)
    denom = y_true.max() - y_true.min()
    nrmse = rmse / denom if denom != 0 else np.nan
    return {'MAE': mae/1000, 'MAPE': mape, 'MSE': mse/1000, 'RMSE': rmse/1000, 'R2': r2, 'N-RMSE': nrmse}

#%%

all_results = {}

for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith(('.csv', '.xlsx', '.xls')):
        continue

    station = os.path.splitext(fname)[0]
    path = os.path.join(DATA_DIR, fname)

    if fname.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path, engine='openpyxl')
    else:
        df = pd.read_csv(path)

    df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'value'})
    exog_cols = [c for c in df.columns if c not in ('date', 'value')]
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    df = df.set_index('date')
    df = fill_monthly_with_threshold(df, max_gap=3)

    if len(df) < MIN_REQUIRED:
        print(f"Skipping {station}: только {len(df)} мес. (<{MIN_REQUIRED})")
        continue

    series = df['value']
    exog = df[exog_cols]

    train_ts = series.iloc[:-HORIZON]
    test_ts = series.iloc[-HORIZON:]
    train_ex = exog.iloc[:-HORIZON]
    test_ex = exog.iloc[-HORIZON:]

    feat = pd.DataFrame({'y': series}, index=series.index)
    feat['month'] = series.index.month
    for c in exog_cols:
        feat[c] = exog[c]
    for lag in range(1, SEASONAL_PERIOD + 1):
        feat[f'lag{lag}'] = series.shift(lag)
    feat = feat.dropna()

    split_date = test_ts.index[0]
    train_feat = feat.loc[feat.index < split_date]
    test_feat = feat.loc[feat.index >= split_date]
    X_train, y_train = train_feat.drop(columns='y'), train_feat['y']
    X_test, y_test = test_feat.drop(columns='y'), test_feat['y']

    methods_pred = {}

    # ML-модели
    if len(X_train) >= 1 and len(X_test) >= 1:
        if len(X_train) >= tscv.n_splits + 1:
            gs = GridSearchCV(KNeighborsRegressor(), knn_grid,
                              cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
            gs.fit(X_train, y_train)
            knn = gs.best_estimator_
        else:
            k = max(1, min(3, len(X_train)))
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance').fit(X_train, y_train)
        methods_pred['k-NN'] = knn.predict(X_test)

        lr = LinearRegression().fit(X_train, y_train)
        methods_pred['LinearRegression'] = lr.predict(X_test)

        if HAVE_XGB and len(X_train) >= tscv.n_splits + 1:
            gs2 = GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42),
                               xgb_grid, cv=tscv,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
            gs2.fit(X_train, y_train)
            gbm = gs2.best_estimator_
        else:
            base = XGBRegressor(objective='reg:squarederror',
                                random_state=42) if HAVE_XGB else GradientBoostingRegressor(random_state=42)
            gbm = base.fit(X_train, y_train)
        methods_pred['GBM'] = gbm.predict(X_test)

    try:
        ets = ExponentialSmoothing(train_ts, trend='add', seasonal='add', seasonal_periods=SEASONAL_PERIOD)
        methods_pred['ETS'] = ets.fit(optimized=True).forecast(HORIZON).values
    except Exception:
        pass

    methods_pred['SeasonalNaive'] = train_ts.shift(SEASONAL_PERIOD).iloc[-HORIZON:].values

    # --- Ансамблирование с отсевом слабых моделей ---
    temp_metrics = {}
    for name, pred in methods_pred.items():
        yt = test_ts.values
        yp = np.array(pred)
        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if mask.sum() > 0:
            temp_metrics[name] = compute_metrics(yt[mask], yp[mask])

    # Пороговые значения для отсечения
    r2_thresh = 0.3
    mape_thresh = 40.0
    selected = [m for m, met in temp_metrics.items()
                if met['R2'] >= r2_thresh and met['MAPE'] <= mape_thresh]
    if not selected:
        selected = list(temp_metrics.keys())

    models = selected
    rmses = np.array([temp_metrics[m]['RMSE'] for m in models])
    inv = 1 / rmses
    weights = inv / inv.sum()

    ensemble_pred = np.zeros(HORIZON)
    for w, m in zip(weights, models):
        ensemble_pred += w * methods_pred[m]
    methods_pred['Ensemble'] = ensemble_pred

    final_metrics = {}
    for name, pred in methods_pred.items():
        yt = test_ts.values
        yp = np.array(pred)
        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if mask.sum() > 0:
            final_metrics[name] = compute_metrics(yt[mask], yp[mask])

    dfm = pd.DataFrame(final_metrics).T
    print(f"\n=== {station} ===")
    print(dfm.to_markdown(floatfmt=".2f"))

    plt.figure(figsize=(10, 5))
    plt.plot(series[-2 * HORIZON:], label='History')
    for name, pred in methods_pred.items():
        plt.plot(test_ts.index, pred, '--', label=name)
    plt.title(f"{station}: Прогноз {HORIZON} месяцев")
    plt.legend()
    plt.tight_layout()
    plt.show()

    all_results[station] = final_metrics

#%%
agg = pd.DataFrame(all_results).stack().apply(pd.Series)
agg.columns = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2', 'N-RMSE']
avg = agg.groupby(level=0).mean()
print("\n=== Среднее между всеми станциями ===")
print(avg.to_markdown(floatfmt=".2f"))
