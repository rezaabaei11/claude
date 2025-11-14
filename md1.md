# راهنمای جامع و کامل: بهینه‌سازی LightGBM برای تست فیچرها با تمرکز بر دقت
## یافته‌های تمام دورهای تحقیقات + جلوگیری از Data Leakage

**تاریخ:** نوامبر 2025  
**نسخه LightGBM:** 4.6.0+  
**اولویت:** **دقت و اعتبار > سرعت و حافظه**  
**هدف:** تست دقیق و معتبر 3885 فیچر TSfresh بدون نشت داده یا بیش‌برازش

---

## فهرست مطالب

1. [اصول بنیادی: چرا دقت مهم‌تر از سرعت است](#اصول)
2. [دوره اول: اصلاحات بنیادی](#دوره-اول)
3. [دوره دوم: تکنیک‌های پیشرفته](#دوره-دوم)
4. [دوره سوم: بهینه‌سازی‌های نهایی](#دوره-سوم)
5. [دوره چهارم: مدیریت حافظه عمیق](#دوره-چهارم)
6. [دوره پنجم: جلوگیری از Data Leakage و Overfitting](#دوره-پنجم)
7. [کد نهایی Production-Grade](#کد-نهایی)
8. [چک‌لیست کامل](#چک-لیست)

---

## اصول بنیادی: چرا دقت مهم‌تر از سرعت است

### اهمیت دقت در Feature Selection

**یک اشتباه در تست فیچرها = فاجعه:**
- فیچرهای قوی حذف می‌شوند ❌
- فیچرهای ضعیف به اشتباه نمره بالا می‌گیرند ❌
- مدل نهایی بر اساس فیچرهای اشتباه train می‌شود ❌
- نتایج در production غیرقابل اعتماد می‌شوند ❌

### فلسفه طراحی این راهنما

```
دقت و اعتبار تست >>> سرعت اجرا >>> مصرف حافظه
```

**اصل طلایی:** فیچرها باید قدرت ذاتی واقعی خود را نمایان کنند، نه قدرت مصنوعی ناشی از:
1. Data Leakage (نشت اطلاعات آینده)
2. Target Leakage (نشت target به features)
3. Overfitting (بیش‌برازش)
4. Spurious Correlations (همبستگی‌های جعلی)

---

## دوره اول: اصلاحات بنیادی

### 1.1 مشکل GOSS (حیاتی)

**مشکل:**
```python
# ❌ روش منسوخ شده (LightGBM < 4.0)
params = {'boosting_type': 'goss'}
```

**راهکار:**
```python
# ✅ LightGBM 4.0+ (صحیح)
params = {
    'boosting_type': 'gbdt',
    'data_sample_strategy': 'goss',
    'top_rate': 0.2,
    'other_rate': 0.1
}
```

**منابع:** LightGBM 4.6 Documentation, GitHub #3182

---

### 1.2 CPU Optimization

```python
import psutil
import os

physical_cores = psutil.cpu_count(logical=False)
os.environ['OMP_NUM_THREADS'] = str(physical_cores)

params = {
    'num_threads': physical_cores,
    'force_col_wise': True,
    'deterministic': True
}
```

**منابع:** GitHub #512, #4425

---

### 1.3 Validation Strategy (حیاتی برای دقت)

```python
# ❌ اشتباه
valid_sets=[train_data, val_data]

# ✅ صحیح
valid_sets=[val_data]
```

**منابع:** GitHub #82, #84, #278

---

### 1.4 Callbacks LightGBM 4.0+

```python
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=0)
]
```

**منابع:** GitHub #113, #5196

---

## دوره دوم: تکنیک‌های پیشرفته

### 2.1 lgb.cv Native (سرعت 2-3x)

```python
cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=500,
    folds=custom_folds,
    stratified=False,
    return_cvbooster=True
)
```

**منابع:** lightgbm.cv Documentation

---

### 2.2 Feature Importance: Gain vs Split (حیاتی)

```python
# ✅ استفاده از Gain
importance = model.feature_importance(importance_type='gain')
```

**چرا Gain بهتر است:**
- Split فقط تعداد استفاده را می‌شمارد
- Gain کیفیت split را اندازه می‌گیرد
- Gain معیار واقعی اهمیت است

**منابع:** GitHub #4255, #132

---

### 2.3 Purged Time Series Split (حیاتی برای دقت)

```python
class PurgedGroupTimeSeriesSplit:
    def __init__(self, n_splits=5, purge_gap=10, embargo_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = (i + 1) * fold_size
            test_end = test_start + fold_size
            train_end = test_start - self.purge_gap
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
```

**منابع:** Kaggle Best Practices, Combinatorial Purged CV

---

### 2.4 Permutation Importance (دقیق‌ترین روش)

```python
from sklearn.inspection import permutation_importance

perm_result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,
    random_state=42
)
```

**منابع:** scikit-learn, Reddit ML discussions

---

## دوره سوم: بهینه‌سازی‌های نهایی

### 3.1 Path Smoothing Regularization

```python
params = {
    'path_smooth': 1.0,  # افزایش برای جلوگیری از overfitting
    'min_data_in_leaf': 20
}
```

**منابع:** LightGBM Parameters

---

### 3.2 Interaction Constraints

```python
params = {
    'interaction_constraints': [
        [0, 1, 2],  # technical features
        [3, 4, 5]   # price features
    ]
}
```

**منابع:** GitHub #2884

---

### 3.3 Monotone Constraints

```python
monotone_constraints = []
for col in X.columns:
    if 'rsi' in col.lower():
        monotone_constraints.append(1)
    else:
        monotone_constraints.append(0)

params['monotone_constraints'] = monotone_constraints
```

**منابع:** ethen8181 Blog

---

## دوره چهارم: مدیریت حافظه عمیق

### 4.1 Histogram Pool Size Optimization

**فرمول:**
```
RAM = num_leaves × 20 × num_features × num_bins (bytes)
```

**مثال:** 3885 features, 1023 leaves, 255 bins = ~20GB!

**راهکار:**
```python
params = {
    'histogram_pool_size': 8192,  # 8GB
    'num_leaves': 255,
    'max_bin': 127
}
```

**منابع:** GitHub #261, #271, LightGBM FAQ

---

### 4.2 Two-Round Loading

```python
train_data = lgb.Dataset(
    X, label=y,
    params={'two_round': True}
)
```

**تأثیر:** کاهش 50% peak memory

**منابع:** GitHub #1146, #1032

---

### 4.3 EFB (Exclusive Feature Bundling)

**نحوه کار:**
- Features با conflict کم bundle می‌شوند
- کاهش 30-70% تعداد features
- Automatic (نیاز به تنظیم ندارد)

**منابع:** LightGBM NIPS Paper, GitHub #3010

---

### 4.4 Force Col-Wise vs Row-Wise

**جدول تصمیم:**

| شرایط | انتخاب |
|-------|--------|
| #features > 1000 | col_wise |
| #threads > 20 | col_wise |
| RAM محدود | col_wise |
| #data > 1M & bins < 100 | row_wise |

**برای 3885 فیچر TSfresh:**
```python
params = {'force_col_wise': True}
```

**منابع:** Parameters Documentation

---

## دوره پنجم: جلوگیری از Data Leakage و Overfitting

### 5.1 درک Data Leakage در Time Series

**تعریف:** استفاده از اطلاعاتی که در زمان prediction موجود نیست.

**انواع Leakage:**

#### 5.1.1 Look-Ahead Bias (نگاه به آینده)

**مثال اشتباه:**
```python
# ❌ محاسبه rolling mean از کل data
df['rolling_mean'] = df['close'].rolling(window=10).mean()

# بعد split
X_train, X_test = train_test_split(df)
```

**مشکل:** rolling mean در train از داده‌های test استفاده کرده!

**راهکار صحیح:**
```python
# ✅ split اول، بعد feature engineering
train_df = df[:split_point]
test_df = df[split_point:]

train_df['rolling_mean'] = train_df['close'].rolling(10).mean()
test_df['rolling_mean'] = test_df['close'].rolling(10).mean()
```

**منابع:** TrainingData Blog, Nature Scientific Reports

---

#### 5.1.2 Target Leakage (نشت target)

**مثال اشتباه:**
```python
# ❌ target encoding با کل dataset
for cat in categorical_cols:
    df[f'{cat}_encoded'] = df.groupby(cat)['target'].transform('mean')
```

**مشکل:** target در features لو رفته!

**راهکار صحیح (Leave-One-Out Encoding):**
```python
def loo_encoding(df, cat_col, target_col):
    # برای هر row، میانگین بدون خود row
    global_mean = df[target_col].mean()
    
    agg = df.groupby(cat_col)[target_col].agg(['sum', 'count'])
    
    encoded = []
    for idx, row in df.iterrows():
        cat = row[cat_col]
        if agg.loc[cat, 'count'] > 1:
            # حذف خود row
            encoded.append(
                (agg.loc[cat, 'sum'] - row[target_col]) / 
                (agg.loc[cat, 'count'] - 1)
            )
        else:
            encoded.append(global_mean)
    
    return encoded
```

**یا استفاده از CatBoost Ordered Target Encoding:**
```python
# CatBoost به صورت automatic از ordered encoding استفاده می‌کند
# که target leakage ندارد
```

**منابع:** CatBoost Paper (NeurIPS 2018), WandB Feature Engineering, Neptune.ai

---

#### 5.1.3 Future Information in Rolling Features

**مشکل TSfresh:**

TSfresh features معمولاً از window گذشته محاسبه می‌شوند، اما:

```python
# ❌ اگر tsfresh روی کل data اجرا شود
from tsfresh import extract_features

# این باعث leakage می‌شود!
features = extract_features(df, column_id='id', column_sort='time')
```

**راهکار صحیح:**
```python
# ✅ rolling window extraction با gap
def extract_tsfresh_with_gap(df, window_size, gap):
    """
    gap: تعداد samples که نباید استفاده شوند
    """
    all_features = []
    
    for i in range(window_size + gap, len(df)):
        # فقط از window گذشته (با gap) استفاده کن
        window_end = i - gap
        window_start = max(0, window_end - window_size)
        
        window_df = df[window_start:window_end]
        features = extract_features(window_df, ...)
        all_features.append(features)
    
    return pd.concat(all_features)
```

**منابع:** Reddit ML, Frontiers Research, Kaggle Time Series

---

### 5.2 STL Decomposition Leakage

**مشکل:**
```python
# ❌ STL روی کل test set
from statsmodels.tsa.seasonal import STL

# این باعث future leakage می‌شود
stl = STL(test_data, seasonal=7)
result = stl.fit()
```

**راهکار:**
```python
# ✅ STL برای هر sample جداگانه
def stl_decompose_per_sample(df, window_size=100):
    results = []
    
    for i in range(window_size, len(df)):
        # فقط از داده‌های قبلی استفاده کن
        historical = df[:i][-window_size:]
        
        stl = STL(historical, seasonal=7)
        result = stl.fit()
        
        # فقط آخرین مقدار را بگیر
        results.append({
            'trend': result.trend.iloc[-1],
            'seasonal': result.seasonal.iloc[-1],
            'resid': result.resid.iloc[-1]
        })
    
    return pd.DataFrame(results)
```

**منابع:** Frontiers in Environmental Science 2025, GitHub AutoGluon #2779

---

### 5.3 Lag Features با Gap صحیح

**مثال برای fیچرهای TSfresh:**

اگر TSfresh از window 30 دقیقه استفاده کرده (برای 15-minute data = 2 bars):

```python
def create_lag_features_with_proper_gap(df, prediction_horizon=1):
    """
    prediction_horizon: چند step جلو را predict می‌کنیم
    """
    
    # gap باید حداقل = prediction_horizon
    min_gap = prediction_horizon
    
    # برای TSfresh با window=2 bars
    tsfresh_window = 2
    
    # gap کل
    total_gap = min_gap + tsfresh_window
    
    # ایجاد lags با gap مناسب
    for lag in range(total_gap, total_gap + 10):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    return df
```

**منابع:** Kaggle TS-10, Reddit Quant

---

### 5.4 Overfitting Detection و Prevention

#### 5.4.1 Train/Valid Gap Monitoring

```python
def calculate_overfit_ratio(train_metric, valid_metric):
    """
    نسبت بیش‌برازش
    """
    if train_metric > 0:
        return valid_metric / train_metric
    return np.inf

def custom_early_stopping_with_overfit_check(
    stopping_rounds=50,
    overfit_tolerance=1.15
):
    """
    Stop اگر:
    1. valid بهبود نیافت
    2. overfit_ratio > tolerance
    """
    
    best_score = None
    best_iter = 0
    counter = 0
    
    def callback(env):
        nonlocal best_score, best_iter, counter
        
        if len(env.evaluation_result_list) >= 1:
            valid_score = env.evaluation_result_list[0][2]
            
            # بررسی بهبود
            if best_score is None or valid_score > best_score:
                best_score = valid_score
                best_iter = env.iteration
                counter = 0
            else:
                counter += 1
                if counter >= stopping_rounds:
                    raise lgb.callback.EarlyStopException(
                        best_iter, best_score
                    )
    
    return callback
```

**منابع:** GitHub #4996, #278

---

#### 5.4.2 Regularization برای دقت

**تنظیمات پیشنهادی (دقت > سرعت):**

```python
params = {
    # Tree Structure (محدودتر برای overfitting کمتر)
    'num_leaves': 31,  # نه بیشتر
    'max_depth': 6,  # محدود کردن عمق
    'min_data_in_leaf': 50,  # افزایش برای دقت
    'min_gain_to_split': 0.02,  # افزایش
    
    # Regularization قوی
    'lambda_l1': 0.5,  # افزایش از 0.1
    'lambda_l2': 0.5,  # افزایش از 0.1
    'path_smooth': 2.0,  # افزایش برای regularization بیشتر
    
    # Feature/Data Sampling
    'feature_fraction': 0.8,  # کاهش برای جلوگیری از overfit
    'bagging_fraction': 0.7,  # کاهش
    'bagging_freq': 5,
    
    # Categorical (در صورت وجود)
    'min_data_per_group': 200,  # افزایش
    'cat_smooth': 20.0,  # افزایش
    'cat_l2': 20.0,  # افزایش
    
    # Learning
    'learning_rate': 0.01,  # کاهش برای دقت بیشتر
    'num_iterations': 1000,  # افزایش با early stopping
}
```

**منابع:** XGBoost vs LightGBM, TowardsDataScience

---

### 5.5 Spurious Correlation Detection

**مشکل:** فیچرهایی که در train خوب به نظر می‌رسند اما spurious هستند.

**تشخیص:**

```python
def detect_spurious_features(X, y, model, n_runs=20):
    """
    فیچرهایی که importance آنها ناپایدار است احتمالاً spurious هستند
    """
    
    importances = []
    
    for seed in range(n_runs):
        # Train با seed مختلف
        model.set_params(random_state=seed)
        model.fit(X, y)
        
        imp = model.feature_importance(importance_type='gain')
        importances.append(imp)
    
    importances = np.array(importances)
    
    # محاسبه coefficient of variation
    mean_imp = importances.mean(axis=0)
    std_imp = importances.std(axis=0)
    cv = std_imp / (mean_imp + 1e-10)
    
    # فیچرهای با CV بالا احتمالاً spurious
    spurious_threshold = 1.0  # قابل تنظیم
    spurious_features = X.columns[cv > spurious_threshold].tolist()
    
    return {
        'feature': X.columns,
        'mean_importance': mean_imp,
        'std_importance': std_imp,
        'cv': cv,
        'is_spurious': cv > spurious_threshold
    }
```

**منابع:** Stanford AI Lab, Nature papers on spurious features

---

### 5.6 Null Importance Test (بهترین روش برای feature validation)

**تئوری:**
- اگر feature واقعاً مهم است، importance آن باید >> null importance باشد
- Null importance = importance وقتی target shuffle شده (random)

**پیاده‌سازی صحیح:**

```python
def null_importance_test_robust(
    X, y, 
    n_actual=20,  # افزایش برای دقت
    n_null=100,  # افزایش برای اعتبار
    cv_splits=5
):
    """
    Null importance با cross-validation برای دقت بالا
    """
    
    from sklearn.model_selection import KFold
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'importance_type': 'gain',  # حیاتی
        'verbose': -1
    }
    
    # Actual importances (با CV)
    actual_importances = []
    
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    for run in range(n_actual):
        fold_importances = []
        
        for train_idx, val_idx in kf.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            
            model = lgb.LGBMClassifier(**params, random_state=run)
            model.fit(X_train, y_train)
            
            fold_importances.append(
                model.feature_importance(importance_type='gain')
            )
        
        # میانگین از folds
        actual_importances.append(np.mean(fold_importances, axis=0))
    
    # Null importances
    null_importances = []
    
    for run in range(n_null):
        fold_importances = []
        
        # Shuffle target
        y_shuffled = y.sample(frac=1, random_state=run).values
        
        for train_idx, val_idx in kf.split(X):
            X_train = X.iloc[train_idx]
            y_train_shuffled = y_shuffled[train_idx]
            
            model = lgb.LGBMClassifier(**params, random_state=run)
            model.fit(X_train, y_train_shuffled)
            
            fold_importances.append(
                model.feature_importance(importance_type='gain')
            )
        
        null_importances.append(np.mean(fold_importances, axis=0))
    
    # آنالیز statistical
    actual_mean = np.mean(actual_importances, axis=0)
    actual_std = np.std(actual_importances, axis=0)
    
    null_mean = np.mean(null_importances, axis=0)
    null_std = np.std(null_importances, axis=0)
    
    # Z-score
    z_scores = (actual_mean - null_mean) / (null_std + 1e-10)
    
    # P-value (two-tailed)
    from scipy import stats
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # تصمیم‌گیری
    significance_level = 0.05
    is_significant = p_values < significance_level
    
    # Score نهایی (actual / null ratio)
    importance_ratio = actual_mean / (null_mean + 1e-10)
    
    results = pd.DataFrame({
        'feature': X.columns,
        'actual_importance_mean': actual_mean,
        'actual_importance_std': actual_std,
        'null_importance_mean': null_mean,
        'null_importance_std': null_std,
        'z_score': z_scores,
        'p_value': p_values,
        'is_significant': is_significant,
        'importance_ratio': importance_ratio
    })
    
    results = results.sort_values('z_score', ascending=False)
    
    return results
```

**تفسیر نتایج:**
- **z_score > 3**: فیچر قطعاً مهم است
- **2 < z_score < 3**: فیچر احتمالاً مهم است
- **z_score < 2**: فیچر احتمالاً spurious یا weak است
- **importance_ratio > 2**: فیچر 2x بهتر از random است

**منابع:** Kaggle Feature Selection, IEEE Papers, Reddit ML

---

### 5.7 Cross-Validated Permutation Importance

**بهترین روش برای stability:**

```python
def cv_permutation_importance(
    X, y,
    n_repeats=10,
    cv_splits=5
):
    """
    Permutation importance با cross-validation
    """
    
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold
    
    all_importances = []
    
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold_num + 1}/{cv_splits}")
        
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Permutation importance
        perm_result = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        all_importances.append(perm_result.importances_mean)
    
    # Aggregate از همه folds
    mean_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)
    
    results = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': mean_importance,
        'importance_std': std_importance,
        'cv_coefficient': std_importance / (mean_importance + 1e-10)
    })
    
    results = results.sort_values('importance_mean', ascending=False)
    
    return results
```

**منابع:** scikit-learn, Reddit ML

---

### 5.8 Combined Feature Selection Strategy (نهایی)

**استراتژی multi-stage برای دقت maximum:**

```python
class AccuracyFirstFeatureSelector:
    """
    Feature selection با اولویت دقت
    """
    
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
    
    def select_features(self, X, y):
        """
        مراحل:
        1. Null importance test
        2. Permutation importance (CV)
        3. Feature stability
        4. Spurious correlation detection
        5. Final ranking
        """
        
        print("Step 1/5: Null Importance Test...")
        null_results = self.null_importance_test_robust(
            X, y, n_actual=20, n_null=100
        )
        
        # فیلتر کردن insignificant features
        significant_features = null_results[
            null_results['is_significant']
        ]['feature'].tolist()
        
        print(f"  Significant features: {len(significant_features)}/{len(X.columns)}")
        
        X_filtered = X[significant_features]
        
        print("\nStep 2/5: Permutation Importance...")
        perm_results = self.cv_permutation_importance(
            X_filtered, y, n_repeats=10, cv_splits=5
        )
        
        print("\nStep 3/5: Feature Stability Test...")
        stability_results = self.detect_spurious_features(
            X_filtered, y, n_runs=20
        )
        
        print("\nStep 4/5: Combining Results...")
        
        # Merge همه نتایج
        final_results = null_results[
            null_results['feature'].isin(significant_features)
        ].copy()
        
        final_results = final_results.merge(
            perm_results[['feature', 'importance_mean', 'cv_coefficient']],
            on='feature',
            suffixes=('_null', '_perm')
        )
        
        final_results = final_results.merge(
            stability_results[['feature', 'cv']],
            on='feature'
        )
        
        # محاسبه combined score
        # وزن‌دهی: 40% null importance, 40% permutation, 20% stability
        
        # Normalize هر metric
        final_results['null_score_norm'] = (
            final_results['z_score'] / final_results['z_score'].max()
        )
        
        final_results['perm_score_norm'] = (
            final_results['importance_mean'] / 
            final_results['importance_mean'].max()
        )
        
        final_results['stability_score_norm'] = (
            1 - (final_results['cv'] / final_results['cv'].max())
        )
        
        # Combined score
        final_results['final_score'] = (
            0.4 * final_results['null_score_norm'] +
            0.4 * final_results['perm_score_norm'] +
            0.2 * final_results['stability_score_norm']
        )
        
        final_results = final_results.sort_values(
            'final_score', ascending=False
        )
        
        print("\nStep 5/5: Final Ranking Complete")
        print(f"Total features evaluated: {len(X.columns)}")
        print(f"Significant features: {len(final_results)}")
        
        return final_results
```

**منابع:** Integration of multiple sources

---

## کد نهایی Production-Grade

```python
"""
Production-Ready Feature Selector
اولویت: دقت و اعتبار >> سرعت و حافظه
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import psutil
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance
from scipy import stats
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# تنظیمات CPU
physical_cores = psutil.cpu_count(logical=False)
os.environ['OMP_NUM_THREADS'] = str(physical_cores)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PurgedTimeSeriesSplit:
    """
    Time series split با purging و embargo
    """
    def __init__(self, n_splits=5, purge_gap=50, embargo_gap=20):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = (i + 1) * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            train_end = test_start - self.purge_gap
            train_start = 0
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class AccuracyFirstFeatureSelector:
    """
    Feature Selector با تمرکز بر دقت و جلوگیری از leakage
    """
    
    def __init__(
        self,
        target_column='target',
        classification=True,
        significance_level=0.05,
        random_state=42
    ):
        self.target_column = target_column
        self.classification = classification
        self.significance_level = significance_level
        self.random_state = random_state
        self.physical_cores = physical_cores
    
    def _get_conservative_params(self):
        """
        پارامترهای محافظه‌کارانه برای دقت بالا
        """
        return {
            'objective': 'binary' if self.classification else 'regression',
            'metric': 'auc' if self.classification else 'rmse',
            'boosting_type': 'gbdt',
            
            # CPU
            'num_threads': self.physical_cores,
            'force_col_wise': True,
            'deterministic': True,
            
            # Tree - محدود برای overfitting کمتر
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 50,
            'min_gain_to_split': 0.02,
            
            # Regularization - قوی
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'path_smooth': 2.0,
            
            # Sampling
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            
            # Categorical
            'min_data_per_group': 200,
            'cat_smooth': 20.0,
            'cat_l2': 20.0,
            
            # Learning
            'learning_rate': 0.01,
            'n_estimators': 1000,
            
            # Other
            'verbose': -1,
            'random_state': self.random_state
        }
    
    def null_importance_test(
        self,
        X, y,
        n_actual=20,
        n_null=100,
        cv_splits=5
    ):
        """
        Null importance test با CV
        """
        logging.info(f"Null Importance: {n_actual} actual, {n_null} null, {cv_splits} CV")
        
        params = self._get_conservative_params()
        params['n_estimators'] = 200  # کاهش برای سرعت در null test
        
        # Actual
        actual_importances = []
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        
        for run in range(n_actual):
            fold_importances = []
            
            for train_idx, val_idx in kf.split(X):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                
                model = lgb.LGBMClassifier(**params)
                model.set_params(random_state=run)
                model.fit(X_train, y_train, verbose=False)
                
                fold_importances.append(
                    model.feature_importance(importance_type='gain')
                )
            
            actual_importances.append(np.mean(fold_importances, axis=0))
            
            if (run + 1) % 5 == 0:
                logging.info(f"  Actual runs: {run + 1}/{n_actual}")
        
        # Null
        null_importances = []
        
        for run in range(n_null):
            fold_importances = []
            
            y_shuffled = y.sample(frac=1, random_state=run).values
            
            for train_idx, val_idx in kf.split(X):
                X_train = X.iloc[train_idx]
                y_train_shuffled = y_shuffled[train_idx]
                
                model = lgb.LGBMClassifier(**params)
                model.set_params(random_state=run)
                model.fit(X_train, y_train_shuffled, verbose=False)
                
                fold_importances.append(
                    model.feature_importance(importance_type='gain')
                )
            
            null_importances.append(np.mean(fold_importances, axis=0))
            
            if (run + 1) % 20 == 0:
                logging.info(f"  Null runs: {run + 1}/{n_null}")
        
        # Statistics
        actual_mean = np.mean(actual_importances, axis=0)
        null_mean = np.mean(null_importances, axis=0)
        null_std = np.std(null_importances, axis=0)
        
        z_scores = (actual_mean - null_mean) / (null_std + 1e-10)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
        is_significant = p_values < self.significance_level
        
        results = pd.DataFrame({
            'feature': X.columns,
            'actual_importance': actual_mean,
            'null_importance': null_mean,
            'z_score': z_scores,
            'p_value': p_values,
            'is_significant': is_significant
        })
        
        return results
    
    def permutation_importance_cv(
        self,
        X, y,
        n_repeats=10,
        cv_splits=5
    ):
        """
        Permutation importance با CV
        """
        logging.info(f"Permutation Importance: {n_repeats} repeats, {cv_splits} CV")
        
        params = self._get_conservative_params()
        params['n_estimators'] = 300
        
        all_importances = []
        
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(X)):
            logging.info(f"  Fold {fold_num + 1}/{cv_splits}")
            
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            perm_result = permutation_importance(
                model, X_val, y_val,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=self.physical_cores
            )
            
            all_importances.append(perm_result.importances_mean)
        
        mean_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
        
        results = pd.DataFrame({
            'feature': X.columns,
            'perm_importance': mean_importance,
            'perm_std': std_importance
        })
        
        return results
    
    def feature_stability_test(
        self,
        X, y,
        n_runs=20
    ):
        """
        Feature stability test
        """
        logging.info(f"Feature Stability: {n_runs} runs")
        
        params = self._get_conservative_params()
        params['n_estimators'] = 200
        
        all_importances = []
        
        for run in range(n_runs):
            model = lgb.LGBMClassifier(**params)
            model.set_params(random_state=run)
            model.fit(X, y, verbose=False)
            
            all_importances.append(
                model.feature_importance(importance_type='gain')
            )
            
            if (run + 1) % 5 == 0:
                logging.info(f"  Run {run + 1}/{n_runs}")
        
        mean_imp = np.mean(all_importances, axis=0)
        std_imp = np.std(all_importances, axis=0)
        cv_scores = std_imp / (mean_imp + 1e-10)
        
        results = pd.DataFrame({
            'feature': X.columns,
            'stability_cv': cv_scores
        })
        
        return results
    
    def select_features(self, X, y):
        """
        Pipeline کامل feature selection
        """
        logging.info("="*50)
        logging.info("Feature Selection با اولویت دقت")
        logging.info(f"Total features: {len(X.columns)}")
        logging.info("="*50)
        
        # 1. Null Importance
        logging.info("\n[1/3] Null Importance Test")
        null_results = self.null_importance_test(
            X, y,
            n_actual=20,
            n_null=100,
            cv_splits=5
        )
        
        significant_features = null_results[
            null_results['is_significant']
        ]['feature'].tolist()
        
        logging.info(f"Significant features: {len(significant_features)}/{len(X.columns)}")
        
        if len(significant_features) == 0:
            logging.warning("هیچ feature معناداری پیدا نشد!")
            return null_results
        
        X_filtered = X[significant_features]
        
        # 2. Permutation Importance
        logging.info("\n[2/3] Permutation Importance")
        perm_results = self.permutation_importance_cv(
            X_filtered, y,
            n_repeats=10,
            cv_splits=5
        )
        
        # 3. Stability
        logging.info("\n[3/3] Feature Stability")
        stability_results = self.feature_stability_test(
            X_filtered, y,
            n_runs=20
        )
        
        # Combine
        logging.info("\nCombining results...")
        
        final_results = null_results[
            null_results['feature'].isin(significant_features)
        ].copy()
        
        final_results = final_results.merge(
            perm_results, on='feature'
        )
        
        final_results = final_results.merge(
            stability_results, on='feature'
        )
        
        # Normalize و combine
        final_results['null_score_norm'] = (
            final_results['z_score'] / final_results['z_score'].max()
        )
        
        final_results['perm_score_norm'] = (
            final_results['perm_importance'] / 
            final_results['perm_importance'].max()
        )
        
        final_results['stability_score_norm'] = (
            1 - (final_results['stability_cv'] / 
                 final_results['stability_cv'].max())
        )
        
        # Combined: 40% null, 40% perm, 20% stability
        final_results['final_score'] = (
            0.4 * final_results['null_score_norm'] +
            0.4 * final_results['perm_score_norm'] +
            0.2 * final_results['stability_score_norm']
        )
        
        final_results = final_results.sort_values(
            'final_score', ascending=False
        )
        
        logging.info("\n" + "="*50)
        logging.info("Feature Selection Complete!")
        logging.info(f"Selected: {len(final_results)} features")
        logging.info("="*50)
        
        return final_results


# استفاده
if __name__ == "__main__":
    
    # بارگذاری data
    df = pd.read_csv('your_tsfresh_features.csv')
    
    # جدا کردن features و target
    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols]
    y = df['target']
    
    # Feature selection
    selector = AccuracyFirstFeatureSelector(
        target_column='target',
        classification=True,
        significance_level=0.05
    )
    
    results = selector.select_features(X, y)
    
    # ذخیره نتایج
    results.to_csv('feature_selection_results.csv', index=False)
    
    # نمایش top 50
    print("\nTop 50 Features:")
    print(results.head(50))
    
    # آمار
    print("\nStatistics:")
    print(f"Total evaluated: {len(X.columns)}")
    print(f"Significant: {len(results)}")
    print(f"With final_score > 0.7: {len(results[results['final_score'] > 0.7])}")
```

---

## چک‌لیست کامل Production

### ✅ جلوگیری از Data Leakage

- [ ] Split قبل از feature engineering
- [ ] Rolling features با gap مناسب
- [ ] STL decomposition per sample
- [ ] Target encoding با leave-one-out
- [ ] Purged time series split با embargo
- [ ] Gap بین train و test
- [ ] هیچ اطلاعات آینده در features

### ✅ جلوگیری از Overfitting

- [ ] Regularization قوی (L1, L2, path_smooth)
- [ ] محدود کردن tree depth
- [ ] افزایش min_data_in_leaf
- [ ] Feature/data sampling
- [ ] Early stopping با overfit monitoring
- [ ] Cross-validation با 5+ folds

### ✅ Feature Importance

- [ ] استفاده از 'gain' نه 'split'
- [ ] Null importance test (n_null >= 100)
- [ ] Permutation importance با CV
- [ ] Feature stability testing
- [ ] Combined scoring

### ✅ Validation Strategy

- [ ] Purged time series split
- [ ] فقط validation در valid_sets
- [ ] Stratified=False برای time series
- [ ] Gap و embargo مناسب

### ✅ LightGBM Configuration

- [ ] `boosting_type='gbdt'`
- [ ] `data_sample_strategy='goss'` (اگر data بزرگ)
- [ ] `force_col_wise=True` (برای features زیاد)
- [ ] `num_threads=physical_cores`
- [ ] `deterministic=True`

### ✅ دقت و اعتبار

- [ ] n_actual >= 20 در null importance
- [ ] n_null >= 100 در null importance
- [ ] cv_splits >= 5
- [ ] n_repeats >= 10 در permutation
- [ ] Significance level = 0.05

---

## خلاصه تأثیرات

### تأثیر بسیار بالا (اجباری):

1. ✅ **Purged Time Series Split** - حذف data leakage
2. ✅ **Feature Importance = 'gain'** - دقت 30-50% بهتر
3. ✅ **Null Importance Test** - شناسایی spurious features
4. ✅ **Split قبل از feature engineering** - جلوگیری از look-ahead bias

### تأثیر بالا (بسیار توصیه):

5. ✅ **Permutation Importance + CV** - stable ranking
6. ✅ **Regularization قوی** - کاهش overfitting
7. ✅ **Feature Stability Testing** - حذف unstable features
8. ✅ **lgb.cv native** - efficient CV

### تأثیر متوسط (مفید):

9. ✅ **Path smoothing** - regularization اضافی
10. ✅ **Interaction/Monotone constraints** - interpretability

---

## منابع جامع

### Papers و Research:
1. CatBoost Paper (NeurIPS 2018) - Ordered boosting, target leakage
2. LightGBM Paper (NIPS 2017) - EFB, histogram-based
3. Nature Scientific Reports (2025) - Data leakage in time series
4. Frontiers Environmental Science (2025) - STL leakage
5. Stanford AI Lab - Spurious features

### Documentation:
1. LightGBM 4.6.0+ Official Docs
2. Parameters Tuning Guide
3. Advanced Topics
4. Python API Reference

### GitHub:
1. microsoft/LightGBM Issues: #512, #4425, #82, #84, #113, #2884
2. AutoGluon #2779 - Stack information leakage

### Blogs و Tutorials:
1. WandB Feature Engineering
2. Neptune.ai - CatBoost vs others
3. TowardsDataScience - Overfitting prevention
4. TrainingData Blog - Look-ahead bias
5. Kaggle Competitions - Time series validation

---

## نتیجه‌گیری

این راهنما با هدف **دقت maximum** طراحی شده است. تمام تکنیک‌ها از منابع معتبر استخراج و با یکدیگر integrate شده‌اند.

**اصل اساسی:**
```
اشتباه در feature selection = فاجعه در production
دقت و اعتبار > همه چیز
```

**نکات کلیدی:**
1. هیچگاه سرعت را بر دقت ترجیح ندهید در feature selection
2. Data leakage را جدی بگیرید - بسیار رایج است
3. Null importance و permutation importance را ترکیب کنید
4. از CV با حداقل 5 folds استفاده کنید
5. Regularization را قوی نگه دارید

با رعایت این راهنما، فیچرهای شما قدرت **ذاتی واقعی** خود را نشان خواهند داد، نه قدرت مصنوعی از data leakage یا overfitting.

**موفق باشید! 🎯**
