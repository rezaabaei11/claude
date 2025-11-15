# راهنمای جامع Pandas برای ساخت ربات تریدر فارکس با LightGBM
## استانداردهای 2025 - از مبتدی تا پیشرفته

**تاریخ تهیه**: اکتبر 2025  
**نسخه**: 2.0  
**مخصوص**: ساخت ربات تریدر اتومات با Pandas + LightGBM + Scikit-learn + NumPy + SHAP + Optuna

---

## 📋 فهرست مطالب

1. [راه‌اندازی محیط توسعه (2025 Setup)](#بخش-0-راه-اندازی-محیط-توسعه-2025)
2. [مبانی Pandas برای تریدینگ](#بخش-1-مبانی-pandas-برای-تریدینگ)
3. [پردازش داده‌های Time Series فارکس](#بخش-2-پردازش-داده-های-time-series-فارکس)
4. [Feature Engineering برای تریدینگ](#بخش-3-feature-engineering-برای-تریدینگ)
5. [یکپارچه‌سازی با LightGBM](#بخش-4-یکپارچه-سازی-با-lightgbm)
6. [Pipeline ساخت مدل کامل](#بخش-5-pipeline-ساخت-مدل-کامل)
7. [بهینه‌سازی و تست](#بخش-6-بهینه-سازی-و-تست)
8. [دیپلویمنت و اجرای زنده](#بخش-7-دیپلویمنت-و-اجرای-زنده)

---

## بخش 0: راه‌اندازی محیط توسعه (2025)

### نصب کتابخانه‌ها

```python
# نصب کتابخانه‌های اصلی
pip install pandas==2.2.0  # آخرین نسخه پایدار با CoW
pip install numpy>=1.26.0
pip install lightgbm>=4.5.0
pip install scikit-learn>=1.5.0
pip install optuna>=3.6.0
pip install shap>=0.45.0

# کتابخانه‌های کمکی برای فارکس
pip install pandas-market-calendars>=4.4.0
pip install pandas-ta>=0.3.14b  # برای Technical Indicators
pip install pyarrow>=15.0.0  # برای بهینه‌سازی حافظه

# کتابخانه‌های دیتا و API
pip install yfinance>=0.2.40
pip install requests>=2.31.0
```

### تنظیمات اولیه با استانداردهای 2025

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import optuna
import shap
import warnings

# فعال‌سازی Copy-on-Write (استاندارد 2025)
pd.options.mode.copy_on_write = True

# تنظیم PyArrow برای بهینه‌سازی حافظه
pd.options.mode.dtype_backend = 'pyarrow'

# تنظیمات نمایش
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 6)
pd.set_option('display.float_format', '{:.6f}'.format)

warnings.filterwarnings('ignore')

print(f"Pandas Version: {pd.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"LightGBM Version: {lgb.__version__}")
```

---

## بخش 1: مبانی Pandas برای تریدینگ

### 1.1 خواندن داده‌های فارکس

```python
# ✅ خواندن داده‌های OHLCV از CSV
def load_forex_data(filepath, pair='EURUSD'):
    """
    بارگذاری داده‌های فارکس با بهینه‌سازی حافظه
    """
    df = pd.read_csv(
        filepath,
        parse_dates=['datetime'],
        index_col='datetime',
        dtype_backend='pyarrow',  # استاندارد 2025
        usecols=['datetime', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # اطمینان از ترتیب زمانی
    df = df.sort_index()
    
    # بررسی Missing Values
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"Data Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# مثال: خواندن داده
df_forex = load_forex_data('data/EURUSD_1H.csv')
print(df_forex.head())
```

### 1.2 مدیریت Timezone برای فارکس (24/7 Market)

```python
import pandas_market_calendars as mcal
from pytz import timezone

def prepare_forex_timezone(df, tz='UTC'):
    """
    مدیریت timezone برای بازار فارکس که 24/7 است
    """
    # تنظیم timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    # تبدیل به timezone دلخواه
    df.index = df.index.tz_convert(tz)
    
    # افزودن ویژگی‌های زمانی
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # جلسات معاملاتی فارکس
    df['session'] = pd.cut(
        df['hour'],
        bins=[0, 7, 15, 21, 24],
        labels=['Asian', 'London', 'NY', 'Pacific'],
        include_lowest=True
    )
    
    return df

# استفاده
df_forex = prepare_forex_timezone(df_forex, tz='America/New_York')
print(df_forex[['open', 'close', 'hour', 'session']].head())
```

### 1.3 مدیریت Missing Data در داده‌های فارکس

```python
def handle_forex_missing_data(df, method='ffill'):
    """
    مدیریت حرفه‌ای Missing Data در فارکس
    
    Parameters:
    -----------
    method : 'ffill', 'interpolate', 'drop'
    """
    print(f"Missing data before: {df.isnull().sum().sum()}")
    
    if method == 'ffill':
        # Forward Fill - مناسب برای قیمت‌ها
        df = df.ffill(limit=5)  # حداکثر 5 مقدار پیاپی
        
    elif method == 'interpolate':
        # Interpolation - برای داده‌های پیوسته
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(
            method='time',  # براساس زمان
            limit_direction='forward',
            limit=5
        )
        
    elif method == 'drop':
        # حذف - برای داده‌های کمتر از 1%
        df = df.dropna()
    
    print(f"Missing data after: {df.isnull().sum().sum()}")
    
    return df

# استفاده
df_forex = handle_forex_missing_data(df_forex, method='interpolate')
```

---

## بخش 2: پردازش داده‌های Time Series فارکس

### 2.1 Resampling داده‌ها (تبدیل Timeframe)

```python
def resample_forex_data(df, timeframe='4H', method='ohlc'):
    """
    تبدیل timeframe داده‌های فارکس
    
    Timeframes: '1T', '5T', '15T', '1H', '4H', 'D', 'W', 'M'
    """
    
    if method == 'ohlc':
        # OHLC Resampling
        resampled = df['close'].resample(timeframe).ohlc()
        resampled['volume'] = df['volume'].resample(timeframe).sum()
        
    elif method == 'last':
        # آخرین مقدار
        resampled = df.resample(timeframe).last()
        
    elif method == 'mean':
        # میانگین
        resampled = df.resample(timeframe).mean()
    
    # حذف NaN
    resampled = resampled.dropna()
    
    print(f"Original shape: {df.shape}")
    print(f"Resampled shape: {resampled.shape}")
    
    return resampled

# مثال: تبدیل از 1H به 4H
df_4h = resample_forex_data(df_forex, timeframe='4H')
print(df_4h.head())
```

### 2.2 Rolling Windows برای Technical Indicators

```python
def calculate_rolling_features(df, windows=[5, 10, 20, 50, 200]):
    """
    محاسبه ویژگی‌های Rolling Window
    """
    df = df.copy()
    
    for window in windows:
        # Moving Averages
        df[f'sma_{window}'] = df['close'].rolling(
            window=window, 
            min_periods=1
        ).mean()
        
        df[f'ema_{window}'] = df['close'].ewm(
            span=window, 
            adjust=False
        ).mean()
        
        # Volatility
        df[f'std_{window}'] = df['close'].rolling(
            window=window,
            min_periods=1
        ).std()
        
        # High-Low Range
        df[f'range_{window}'] = (
            df['high'].rolling(window).max() - 
            df['low'].rolling(window).min()
        )
        
        # Volume Moving Average
        df[f'volume_ma_{window}'] = df['volume'].rolling(
            window=window
        ).mean()
    
    return df

# استفاده
df_forex = calculate_rolling_features(df_forex)
print(df_forex[['close', 'sma_20', 'ema_20', 'std_20']].tail())
```

### 2.3 Shift & Lag Features (ویژگی‌های تاخیری)

```python
def create_lag_features(df, lags=[1, 2, 3, 5, 10], columns=['close', 'volume']):
    """
    ساخت ویژگی‌های Lag برای مدل ML
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            # Lag features
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Percentage change
            df[f'{col}_pct_change_{lag}'] = df[col].pct_change(lag)
            
            # Difference
            df[f'{col}_diff_{lag}'] = df[col].diff(lag)
    
    return df

# استفاده
df_forex = create_lag_features(
    df_forex, 
    lags=[1, 2, 3, 5, 10, 20],
    columns=['close', 'volume']
)
```

### 2.4 Expanding Window Features

```python
def calculate_expanding_features(df):
    """
    محاسبه ویژگی‌های Expanding (تجمعی)
    """
    df = df.copy()
    
    # Expanding Statistics
    df['cumsum_return'] = df['close'].pct_change().expanding().sum()
    df['cummax'] = df['close'].expanding().max()
    df['cummin'] = df['close'].expanding().min()
    
    # Drawdown
    df['drawdown'] = (df['close'] - df['cummax']) / df['cummax']
    df['max_drawdown'] = df['drawdown'].expanding().min()
    
    # Expanding Volatility
    df['expanding_vol'] = df['close'].pct_change().expanding().std()
    
    return df

df_forex = calculate_expanding_features(df_forex)
```

---

## بخش 3: Feature Engineering برای تریدینگ

### 3.1 Technical Indicators با pandas_ta

```python
import pandas_ta as ta

def add_technical_indicators(df):
    """
    افزودن اندیکاتورهای تکنیکال با pandas_ta
    """
    df = df.copy()
    
    # Trend Indicators
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True)
    
    # Momentum Indicators
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.stoch(append=True)  # Stochastic
    
    # Volatility Indicators
    df.ta.bbands(length=20, std=2, append=True)  # Bollinger Bands
    df.ta.atr(length=14, append=True)  # Average True Range
    
    # Volume Indicators
    df.ta.obv(append=True)  # On-Balance Volume
    df.ta.ad(append=True)   # Accumulation/Distribution
    
    # Support/Resistance
    df.ta.pivots(append=True)
    
    return df

# استفاده
df_forex = add_technical_indicators(df_forex)
print(df_forex.columns.tolist())
```

### 3.2 Custom Technical Indicators

```python
def calculate_custom_indicators(df):
    """
    اندیکاتورهای سفارشی برای فارکس
    """
    df = df.copy()
    
    # 1. Price Action Features
    df['body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_doji'] = (df['body'] < (df['high'] - df['low']) * 0.1).astype(int)
    
    # 2. Momentum Features
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    
    # 3. Volatility Ratio
    df['volatility_ratio'] = (
        df['close'].rolling(5).std() / 
        df['close'].rolling(20).std()
    )
    
    # 4. Volume Profile
    df['volume_ratio'] = (
        df['volume'] / 
        df['volume'].rolling(20).mean()
    )
    
    # 5. Price Distance from MA
    df['distance_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['distance_ema50'] = (df['close'] - df['ema_50']) / df['ema_50']
    
    # 6. Trend Strength
    df['trend_strength'] = (
        df['close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
    )
    
    return df

df_forex = calculate_custom_indicators(df_forex)
```

### 3.3 Multi-Timeframe Features

```python
def create_multitimeframe_features(df_1h, timeframes=['4H', 'D']):
    """
    ساخت ویژگی‌های چند تایم‌فریم
    """
    df = df_1h.copy()
    
    for tf in timeframes:
        # Resample به تایم‌فریم بالاتر
        df_higher = resample_forex_data(df, timeframe=tf)
        
        # محاسبه اندیکاتورها
        df_higher['rsi'] = ta.rsi(df_higher['close'], length=14)
        df_higher['macd'] = ta.macd(df_higher['close'])['MACD_12_26_9']
        df_higher['trend'] = np.where(
            df_higher['close'] > df_higher['close'].rolling(20).mean(),
            1, -1
        )
        
        # Merge با داده اصلی
        df_higher = df_higher.add_suffix(f'_{tf}')
        df = df.join(df_higher, how='left', rsuffix=f'_{tf}')
        
        # Forward fill
        df = df.ffill()
    
    return df

# استفاده
df_forex = create_multitimeframe_features(df_forex, timeframes=['4H', 'D'])
```

### 3.4 Target Variable (Label) ساخت

```python
def create_target_variable(df, method='classification', horizon=5, threshold=0.001):
    """
    ساخت متغیر هدف برای مدل ML
    
    Parameters:
    -----------
    method : 'classification', 'regression'
    horizon : تعداد دوره‌های آینده
    threshold : آستانه تغییر قیمت (برای classification)
    """
    df = df.copy()
    
    # محاسبه بازده آینده
    df['future_return'] = df['close'].pct_change(horizon).shift(-horizon)
    
    if method == 'classification':
        # برچسب‌گذاری: BUY (1), SELL (-1), HOLD (0)
        df['target'] = np.where(
            df['future_return'] > threshold, 1,
            np.where(df['future_return'] < -threshold, -1, 0)
        )
        
        print(f"Target Distribution:\n{df['target'].value_counts()}")
        
    elif method == 'regression':
        # پیش‌بینی مستقیم بازده
        df['target'] = df['future_return']
    
    return df

# استفاده برای Classification
df_forex = create_target_variable(
    df_forex, 
    method='classification',
    horizon=5,
    threshold=0.0015  # 0.15% برای فارکس
)
```

### 3.5 Feature Selection با Pandas

```python
def select_features(df, target_col='target', method='correlation', threshold=0.05):
    """
    انتخاب ویژگی‌های مهم
    """
    # حذف ستون‌های غیرعددی
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop([target_col], errors='ignore')
    
    if method == 'correlation':
        # بر اساس همبستگی با target
        correlations = df[numeric_cols].corrwith(df[target_col]).abs()
        selected_features = correlations[correlations > threshold].index.tolist()
        
        print(f"Selected {len(selected_features)} features")
        print(f"Top 10 correlations:\n{correlations.nlargest(10)}")
        
    elif method == 'variance':
        # حذف ویژگی‌های با واریانس کم
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df[numeric_cols])
        
        selected_features = numeric_cols[selector.get_support()].tolist()
        print(f"Selected {len(selected_features)} features")
    
    return selected_features

# استفاده
selected_features = select_features(df_forex, target_col='target', threshold=0.02)
```

---

## بخش 4: یکپارچه‌سازی با LightGBM

### 4.1 آماده‌سازی داده برای LightGBM

```python
def prepare_data_for_lgbm(df, target_col='target', test_size=0.2):
    """
    آماده‌سازی داده برای LightGBM
    """
    # حذف NaN
    df = df.dropna()
    
    # جداسازی Features و Target
    feature_cols = [col for col in df.columns if col not in [
        target_col, 'future_return', 'open', 'high', 'low', 'close', 'volume'
    ]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split Train/Test بر اساس زمان (مهم!)
    split_index = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, feature_cols

# استفاده
X_train, X_test, y_train, y_test, features = prepare_data_for_lgbm(df_forex)
```

### 4.2 ساخت مدل LightGBM پایه

```python
def train_lgbm_baseline(X_train, y_train, X_test, y_test, task='classification'):
    """
    آموزش مدل LightGBM پایه
    """
    if task == 'classification':
        params = {
            'objective': 'multiclass',
            'num_class': 3,  # BUY, SELL, HOLD
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 7,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'force_row_wise': True,  # برای سرعت بیشتر
        }
        
    elif task == 'regression':
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
    
    # ساخت Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # آموزش
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model

# آموزش
model = train_lgbm_baseline(X_train, y_train, X_test, y_test)
```

### 4.3 بهینه‌سازی Hyperparameter با Optuna

```python
def optimize_lgbm_with_optuna(X_train, y_train, X_test, y_test, n_trials=100):
    """
    بهینه‌سازی LightGBM با Optuna
    """
    def objective(trial):
        # فضای جستجوی پارامترها
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'verbose': -1
        }
        
        # Dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # آموزش
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
        
        # ارزیابی
        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred_class)
        
        return accuracy
    
    # بهینه‌سازی
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("Best trial:")
    print(f"  Accuracy: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    return study.best_params

# بهینه‌سازی
best_params = optimize_lgbm_with_optuna(
    X_train, y_train, X_test, y_test, 
    n_trials=50
)
```

### 4.4 تفسیر مدل با SHAP

```python
def explain_model_with_shap(model, X_test, feature_names, max_display=20):
    """
    تفسیر مدل LightGBM با SHAP
    """
    import matplotlib.pyplot as plt
    
    # ساخت Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature Importance
    shap_importance = np.abs(shap_values).mean(axis=0).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)
    
    print("Top 20 Important Features (SHAP):")
    print(feature_importance_df.head(20))
    
    return feature_importance_df

# استفاده
shap_importance = explain_model_with_shap(model, X_test, features)
```

---

## بخش 5: Pipeline ساخت مدل کامل

### 5.1 ساخت Pipeline با Scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

def create_trading_pipeline(numeric_features, categorical_features=None):
    """
    ساخت Pipeline کامل برای پیش‌پردازش و مدل‌سازی
    """
    # پیش‌پردازش
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())  # مناسب‌تر برای outliers
    ])
    
    # ColumnTransformer
    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ]
        )
    
    # Pipeline کامل
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    # فعال‌سازی set_output برای Pandas (2025 feature)
    pipeline.set_output(transform="pandas")
    
    return pipeline

# استفاده
numeric_features = [col for col in features if 'session' not in col]
pipeline = create_trading_pipeline(numeric_features)

# Transform
X_train_scaled = pipeline.fit_transform(X_train)
X_test_scaled = pipeline.transform(X_test)
```

### 5.2 Walk-Forward Optimization

```python
def walk_forward_optimization(df, feature_cols, target_col, 
                               train_window=1000, test_window=200, 
                               step=100):
    """
    Walk-Forward Optimization برای ارزیابی واقعی‌تر
    """
    results = []
    models = []
    
    # تعداد splits
    n_splits = (len(df) - train_window) // step
    
    for i in range(n_splits):
        # تعریف بازه‌های train و test
        train_start = i * step
        train_end = train_start + train_window
        test_start = train_end
        test_end = min(test_start + test_window, len(df))
        
        if test_end - test_start < 50:  # حداقل 50 نمونه test
            break
        
        # جداسازی داده
        X_train_wf = df.iloc[train_start:train_end][feature_cols]
        y_train_wf = df.iloc[train_start:train_end][target_col]
        X_test_wf = df.iloc[test_start:test_end][feature_cols]
        y_test_wf = df.iloc[test_start:test_end][target_col]
        
        # آموزش مدل
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train_wf, label=y_train_wf)
        model_wf = lgb.train(params, train_data, num_boost_round=200)
        
        # پیش‌بینی
        y_pred_wf = model_wf.predict(X_test_wf)
        y_pred_class = np.argmax(y_pred_wf, axis=1)
        
        # ارزیابی
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test_wf, y_pred_class)
        precision = precision_score(y_test_wf, y_pred_class, average='weighted')
        recall = recall_score(y_test_wf, y_pred_class, average='weighted')
        
        results.append({
            'fold': i,
            'train_end': train_end,
            'test_end': test_end,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        })
        
        models.append(model_wf)
        
        print(f"Fold {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}")
    
    # خلاصه نتایج
    results_df = pd.DataFrame(results)
    print("\nWalk-Forward Results Summary:")
    print(results_df.describe())
    
    return results_df, models

# استفاده
wf_results, wf_models = walk_forward_optimization(
    df_forex, features, 'target',
    train_window=5000, test_window=1000, step=500
)
```

### 5.3 Ensemble Models

```python
def create_ensemble_model(models, X_test):
    """
    ساخت Ensemble از چند مدل LightGBM
    """
    predictions = []
    
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # میانگین پیش‌بینی‌ها
    ensemble_pred = np.mean(predictions, axis=0)
    ensemble_class = np.argmax(ensemble_pred, axis=1)
    
    return ensemble_class, ensemble_pred

# استفاده
ensemble_predictions, ensemble_probs = create_ensemble_model(wf_models[:5], X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, ensemble_predictions))
```

---

## بخش 6: بهینه‌سازی و تست

### 6.1 Backtesting استراتژی

```python
def backtest_trading_strategy(df, predictions, initial_capital=10000, 
                               pip_value=10, spread=2):
    """
    Backtesting استراتژی تریدینگ
    
    Parameters:
    -----------
    predictions : array سیگنال‌های تریدینگ (1=BUY, -1=SELL, 0=HOLD)
    pip_value : ارزش هر pip
    spread : اسپرد (به pip)
    """
    df = df.copy()
    df['signal'] = predictions
    
    # محاسبه سود/زیان
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    # کاهش spread
    df['strategy_returns'] = df['strategy_returns'] - (spread * 0.0001)
    
    # محاسبه Equity
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    df['equity'] = initial_capital * df['cumulative_returns']
    
    # محاسبه Drawdown
    df['cummax_equity'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['cummax_equity']) / df['cummax_equity']
    
    # محاسبه Metrics
    total_return = (df['equity'].iloc[-1] - initial_capital) / initial_capital
    max_drawdown = df['drawdown'].min()
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252 * 24)  # فارکس 24/5
    
    # تعداد معاملات
    df['position_change'] = df['signal'].diff()
    num_trades = (df['position_change'] != 0).sum()
    
    # Win Rate
    winning_trades = (df['strategy_returns'] > 0).sum()
    win_rate = winning_trades / num_trades if num_trades > 0 else 0
    
    print("="*50)
    print("Backtest Results:")
    print("="*50)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Equity: ${df['equity'].iloc[-1]:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print("="*50)
    
    return df

# استفاده
df_backtest = backtest_trading_strategy(
    df_forex.loc[X_test.index], 
    ensemble_predictions,
    initial_capital=10000
)

# رسم Equity Curve
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(df_backtest.index, df_backtest['equity'], label='Strategy Equity')
plt.axhline(y=10000, color='r', linestyle='--', label='Initial Capital')
plt.title('Equity Curve - Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('equity_curve.png', dpi=300)
plt.show()
```

### 6.2 Risk Management

```python
def calculate_position_size(equity, risk_percent, stop_loss_pips, pip_value=10):
    """
    محاسبه سایز پوزیشن بر اساس مدیریت ریسک
    
    Parameters:
    -----------
    equity : موجودی حساب
    risk_percent : درصد ریسک (معمولا 1-2%)
    stop_loss_pips : تعداد pip استاپ لاس
    pip_value : ارزش هر pip
    """
    risk_amount = equity * (risk_percent / 100)
    position_size = risk_amount / (stop_loss_pips * pip_value)
    
    return position_size

def add_risk_management(df, predictions, equity=10000, risk_percent=1.5):
    """
    اضافه کردن Stop Loss و Take Profit
    """
    df = df.copy()
    df['signal'] = predictions
    
    # محاسبه ATR برای Stop Loss پویا
    df['atr'] = df.ta.atr(length=14)
    
    # Stop Loss: 2 * ATR
    df['stop_loss_pips'] = df['atr'] * 10000 * 2  # تبدیل به pip
    
    # Take Profit: 3 * ATR (Risk-Reward 1:1.5)
    df['take_profit_pips'] = df['atr'] * 10000 * 3
    
    # محاسبه Position Size
    df['position_size'] = df.apply(
        lambda row: calculate_position_size(
            equity, risk_percent, row['stop_loss_pips']
        ),
        axis=1
    )
    
    return df

# استفاده
df_forex_risk = add_risk_management(df_forex, ensemble_predictions)
print(df_forex_risk[['close', 'signal', 'atr', 'stop_loss_pips', 'position_size']].tail())
```

### 6.3 Performance Analysis

```python
def analyze_trading_performance(df):
    """
    تحلیل جامع عملکرد تریدینگ
    """
    results = {}
    
    # 1. Return Metrics
    results['total_return'] = df['strategy_returns'].sum()
    results['mean_return'] = df['strategy_returns'].mean()
    results['std_return'] = df['strategy_returns'].std()
    
    # 2. Risk Metrics
    results['sharpe_ratio'] = (
        results['mean_return'] / results['std_return'] * np.sqrt(252 * 24)
    )
    results['max_drawdown'] = df['drawdown'].min()
    results['calmar_ratio'] = (
        results['total_return'] / abs(results['max_drawdown'])
    )
    
    # 3. Trade Metrics
    df['trade'] = (df['signal'] != df['signal'].shift()).astype(int)
    results['num_trades'] = df['trade'].sum()
    
    # Winning/Losing Trades
    trades_returns = df[df['trade'] == 1]['strategy_returns']
    results['num_winning'] = (trades_returns > 0).sum()
    results['num_losing'] = (trades_returns < 0).sum()
    results['win_rate'] = results['num_winning'] / results['num_trades']
    
    # Average Win/Loss
    results['avg_win'] = trades_returns[trades_returns > 0].mean()
    results['avg_loss'] = trades_returns[trades_returns < 0].mean()
    results['profit_factor'] = abs(results['avg_win'] / results['avg_loss'])
    
    # 4. Exposure
    results['exposure'] = (df['signal'] != 0).sum() / len(df)
    
    # نمایش نتایج
    print("\n" + "="*60)
    print("TRADING PERFORMANCE ANALYSIS")
    print("="*60)
    
    for key, value in results.items():
        if 'ratio' in key or 'rate' in key or 'factor' in key:
            print(f"{key.replace('_', ' ').title():<30}: {value:>10.4f}")
        elif 'return' in key:
            print(f"{key.replace('_', ' ').title():<30}: {value:>10.4%}")
        else:
            print(f"{key.replace('_', ' ').title():<30}: {value:>10.2f}")
    
    print("="*60 + "\n")
    
    return pd.Series(results)

# استفاده
performance_metrics = analyze_trading_performance(df_backtest)
```

---

## بخش 7: دیپلویمنت و اجرای زنده

### 7.1 ذخیره و بارگذاری مدل

```python
import joblib
import json

def save_trading_model(model, pipeline, feature_names, params, filepath='models/'):
    """
    ذخیره مدل و تنظیمات
    """
    import os
    os.makedirs(filepath, exist_ok=True)
    
    # ذخیره مدل LightGBM
    model.save_model(f'{filepath}lgbm_model.txt')
    
    # ذخیره Pipeline
    joblib.dump(pipeline, f'{filepath}pipeline.pkl')
    
    # ذخیره Feature Names و Params
    metadata = {
        'feature_names': feature_names,
        'params': params,
        'num_features': len(feature_names)
    }
    
    with open(f'{filepath}metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model saved to {filepath}")

def load_trading_model(filepath='models/'):
    """
    بارگذاری مدل و تنظیمات
    """
    # بارگذاری مدل
    model = lgb.Booster(model_file=f'{filepath}lgbm_model.txt')
    
    # بارگذاری Pipeline
    pipeline = joblib.load(f'{filepath}pipeline.pkl')
    
    # بارگذاری Metadata
    with open(f'{filepath}metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Model loaded from {filepath}")
    print(f"Number of features: {metadata['num_features']}")
    
    return model, pipeline, metadata

# ذخیره
save_trading_model(model, pipeline, features, best_params)

# بارگذاری
model_loaded, pipeline_loaded, metadata = load_trading_model()
```

### 7.2 Real-Time Prediction

```python
class ForexTradingBot:
    """
    ربات تریدر فارکس با LightGBM
    """
    def __init__(self, model_path='models/'):
        self.model, self.pipeline, self.metadata = load_trading_model(model_path)
        self.feature_names = self.metadata['feature_names']
        
    def preprocess_live_data(self, df_live):
        """
        پیش‌پردازش داده زنده
        """
        # محاسبه همه فیچرها
        df = df_live.copy()
        df = calculate_rolling_features(df)
        df = add_technical_indicators(df)
        df = calculate_custom_indicators(df)
        df = create_lag_features(df)
        
        # انتخاب آخرین ردیف
        df_latest = df[self.feature_names].iloc[[-1]]
        
        return df_latest
    
    def predict(self, df_live):
        """
        پیش‌بینی سیگنال تریدینگ
        """
        # پیش‌پردازش
        X = self.preprocess_live_data(df_live)
        
        # Transform با Pipeline
        X_scaled = self.pipeline.transform(X)
        
        # پیش‌بینی
        pred_proba = self.model.predict(X_scaled)
        pred_class = np.argmax(pred_proba, axis=1)[0]
        
        # Mapping
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal = signal_map.get(pred_class, 'HOLD')
        
        confidence = pred_proba[0][pred_class]
        
        return {
            'signal': signal,
            'confidence': float(confidence),
            'probabilities': {
                'SELL': float(pred_proba[0][0]),
                'HOLD': float(pred_proba[0][1]),
                'BUY': float(pred_proba[0][2])
            }
        }
    
    def generate_trading_signal(self, df_live, min_confidence=0.6):
        """
        تولید سیگنال تریدینگ با فیلتر confidence
        """
        prediction = self.predict(df_live)
        
        if prediction['confidence'] >= min_confidence:
            return prediction
        else:
            return {
                'signal': 'HOLD',
                'confidence': prediction['confidence'],
                'reason': 'Low confidence'
            }

# استفاده
bot = ForexTradingBot(model_path='models/')

# فرض: دریافت داده زنده
df_live = df_forex.iloc[-100:]  # 100 کندل آخر

# پیش‌بینی
signal = bot.generate_trading_signal(df_live, min_confidence=0.65)
print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']:.2%}")
print(f"Probabilities: {signal.get('probabilities', {})}")
```

### 7.3 Live Trading Integration

```python
import time
from datetime import datetime

def live_trading_loop(bot, api_client, pair='EURUSD', 
                       timeframe='1H', check_interval=60):
    """
    حلقه تریدینگ زنده
    
    Parameters:
    -----------
    api_client : کلاینت API بروکر (مثلا MetaTrader, OANDA)
    check_interval : فاصله چک کردن (ثانیه)
    """
    print(f"Starting live trading for {pair} - {timeframe}")
    print(f"Check interval: {check_interval}s")
    print("-" * 60)
    
    while True:
        try:
            # دریافت داده آخرین 200 کندل
            df_live = api_client.get_historical_data(
                pair=pair,
                timeframe=timeframe,
                count=200
            )
            
            # تولید سیگنال
            signal = bot.generate_trading_signal(df_live, min_confidence=0.65)
            
            # Log
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Signal: {signal['signal']} | "
                  f"Confidence: {signal['confidence']:.2%}")
            
            # اجرای معامله
            if signal['signal'] == 'BUY':
                # محاسبه Stop Loss و Take Profit
                current_price = df_live['close'].iloc[-1]
                atr = df_live['atr'].iloc[-1]
                
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)
                
                # ارسال سفارش
                order = api_client.place_order(
                    pair=pair,
                    side='BUY',
                    volume=0.01,  # Lot size
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                print(f"  --> BUY order placed: {order['order_id']}")
                print(f"      SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                
            elif signal['signal'] == 'SELL':
                # مشابه BUY
                current_price = df_live['close'].iloc[-1]
                atr = df_live['atr'].iloc[-1]
                
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)
                
                order = api_client.place_order(
                    pair=pair,
                    side='SELL',
                    volume=0.01,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                print(f"  --> SELL order placed: {order['order_id']}")
                print(f"      SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            
            # انتظار تا چک بعدی
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\nStopping live trading...")
            break
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(check_interval)

# توجه: نیاز به پیاده‌سازی api_client دارید
# مثال: MT5Client, OANDAClient, etc.
```

### 7.4 Monitoring و Logging

```python
import logging
from datetime import datetime

def setup_trading_logger(log_file='logs/trading.log'):
    """
    راه‌اندازی Logger برای تریدینگ
    """
    import os
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.INFO)
    
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# استفاده
logger = setup_trading_logger()

logger.info("Trading bot started")
logger.info(f"Model loaded: LightGBM with {len(features)} features")
logger.warning("Low confidence signal detected")
logger.error("API connection failed")
```

---

## بخش 8: نکات پیشرفته و Best Practices

### 8.1 Memory Optimization برای Big Data

```python
def optimize_dataframe_memory(df):
    """
    بهینه‌سازی مصرف حافظه DataFrame
    """
    print(f"Memory before: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Integer
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    
            # Float
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        # Categorical
        else:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    print(f"Memory after: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# استفاده
df_forex = optimize_dataframe_memory(df_forex)
```

### 8.2 GPU Acceleration برای LightGBM

```python
def train_lgbm_gpu(X_train, y_train, X_test, y_test):
    """
    آموزش LightGBM با GPU
    """
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'device': 'gpu',  # فعال‌سازی GPU
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    
    # آموزش
    import time
    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50)]
    )
    
    training_time = time.time() - start_time
    print(f"GPU Training time: {training_time:.2f}s")
    
    return model

# توجه: نیاز به نصب LightGBM با پشتیبانی GPU
```

### 8.3 Feature Store

```python
class FeatureStore:
    """
    ذخیره و مدیریت فیچرها
    """
    def __init__(self, cache_dir='feature_cache/'):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_features(self, df, name):
        """ذخیره فیچرها"""
        filepath = f"{self.cache_dir}{name}.parquet"
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        print(f"Features saved: {filepath}")
    
    def load_features(self, name):
        """بارگذاری فیچرها"""
        filepath = f"{self.cache_dir}{name}.parquet"
        df = pd.read_parquet(filepath, engine='pyarrow')
        print(f"Features loaded: {filepath}")
        return df
    
    def update_features(self, df_new, name):
        """به‌روزرسانی فیچرها"""
        try:
            df_old = self.load_features(name)
            df_combined = pd.concat([df_old, df_new]).drop_duplicates()
            self.save_features(df_combined, name)
        except FileNotFoundError:
            self.save_features(df_new, name)

# استفاده
feature_store = FeatureStore()
feature_store.save_features(df_forex, 'EURUSD_1H_features')
```

---

## بخش 9: خلاصه و چک‌لیست نهایی

### ✅ چک‌لیست کامل پروژه

#### **مرحله 1: راه‌اندازی**
- [ ] نصب تمام کتابخانه‌ها با نسخه‌های 2025
- [ ] فعال‌سازی Copy-on-Write در Pandas
- [ ] تنظیم PyArrow برای بهینه‌سازی حافظه
- [ ] راه‌اندازی محیط توسعه (Jupyter / VS Code)

#### **مرحله 2: دریافت و پردازش داده**
- [ ] دریافت داده‌های تاریخی فارکس (حداقل 2-3 سال)
- [ ] مدیریت Timezone و ساعت‌های معاملاتی
- [ ] پاک‌سازی و مدیریت Missing Data
- [ ] Resampling به timeframe دلخواه

#### **مرحله 3: Feature Engineering**
- [ ] محاسبه Technical Indicators (RSI, MACD, BB, ATR)
- [ ] ساخت Rolling & Expanding Features
- [ ] ایجاد Lag Features (1-20 دوره)
- [ ] Multi-Timeframe Features
- [ ] Custom Indicators
- [ ] Target Variable (Classification / Regression)

#### **مرحله 4: مدل‌سازی**
- [ ] تقسیم Train/Test بر اساس زمان
- [ ] ساخت Pipeline با ColumnTransformer
- [ ] آموزش مدل LightGBM پایه
- [ ] بهینه‌سازی Hyperparameter با Optuna (50+ trials)
- [ ] Feature Importance با SHAP
- [ ] Walk-Forward Optimization

#### **مرحله 5: ارزیابی**
- [ ] Backtesting با سرمایه واقعی
- [ ] محاسبه Metrics (Sharpe, Max DD, Win Rate)
- [ ] Risk Management (Stop Loss, Take Profit, Position Sizing)
- [ ] Ensemble Models
- [ ] Performance Analysis

#### **مرحله 6: دیپلویمنت**
- [ ] ذخیره مدل و Pipeline
- [ ] ساخت کلاس Trading Bot
- [ ] پیاده‌سازی Real-Time Prediction
- [ ] یکپارچه‌سازی با API بروکر
- [ ] Logging و Monitoring
- [ ] تست در حساب Demo

#### **مرحله 7: نگهداری**
- [ ] Re-training مدل هر 1-3 ماه
- [ ] به‌روزرسانی Features
- [ ] بررسی Performance
- [ ] بهینه‌سازی مجدد با Optuna

---

## منابع و لینک‌های مفید

### 📚 مستندات رسمی
- **Pandas**: https://pandas.pydata.org/docs/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Optuna**: https://optuna.readthedocs.io/
- **SHAP**: https://shap.readthedocs.io/

### 🔗 GitHub Repositories
- Pandas Source: https://github.com/pandas-dev/pandas
- LightGBM Examples: https://github.com/microsoft/LightGBM/tree/master/examples
- Trading Strategies: https://github.com/topics/algorithmic-trading

### 📖 کتاب‌ها
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Python for Finance" - Yves Hilpisch  
- "Machine Learning for Algorithmic Trading" - Stefan Jansen

### 🎓 دوره‌ها
- QuantConnect - https://www.quantconnect.com/
- QuantInsti - https://www.quantinsti.com/
- DataCamp: Machine Learning for Finance

---

## نکات نهایی و هشدارها ⚠️

### ⚠️ **اخطارهای مهم**

1. **این راهنما فقط جنبه آموزشی دارد**
   - هرگونه تریدینگ واقعی ریسک دارد
   - ابتدا در حساب Demo تست کنید
   - سرمایه‌ای که حاضر به از دست دادن نیستید ریسک نکنید

2. **هیچ مدل ML بدون ریسک نیست**
   - Performance گذشته تضمین آینده نیست
   - همیشه مدیریت ریسک داشته باشید
   - از Leverage بالا اجتناب کنید

3. **Overfitting بزرگترین خطر است**
   - از Walk-Forward Optimization استفاده کنید
   - مدل را روی داده Out-of-Sample تست کنید
   - تعداد فیچرها را محدود کنید

4. **بازار دائما تغییر می‌کند**
   - مدل را مرتبا Re-train کنید
   - Performance را Monitor کنید
   - آماده توقف ربات باشید

### 💡 **توصیه‌های حرفه‌ای**

1. **شروع کوچک**: با یک pair و یک timeframe شروع کنید
2. **Logging کامل**: تمام تصمیمات و معاملات را Log کنید
3. **Diversification**: روی چند pair مختلف کار کنید
4. **Automation**: هرچه بیشتر خودکار کنید تا احساسات دخیل نشود
5. **Continuous Learning**: همیشه در حال یادگیری باشید

---

## نتیجه‌گیری

این راهنما جامع‌ترین منبع فارسی برای ساخت ربات تریدر فارکس با Pandas و LightGBM است. با دنبال کردن این مراحل می‌توانید:

✅ یک سیستم تریدینگ کاملا خودکار بسازید  
✅ از جدیدترین تکنیک‌های ML استفاده کنید  
✅ مدل‌های قابل اعتماد و قابل توضیح داشته باشید  
✅ ریسک خود را مدیریت کنید  

**موفق و پرسود باشید! 🚀📈**

---

**نسخه**: 2.0  
**تاریخ**: اکتبر 2025  
**سازگار با**: Pandas 2.2+, LightGBM 4.5+, Python 3.10+

---

*توجه: این راهنما با تحقیق گسترده از منابع معتبر شامل مستندات رسمی Pandas، LightGBM، Scikit-learn و مقالات علمی سال 2025 تهیه شده است.*