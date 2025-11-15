import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_validate, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from scipy import linalg as scipy_linalg
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
from datetime import datetime
import os
import gc
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_selection.log'),
        logging.StreamHandler()
    ]
)

class FeatureSelector(BaseEstimator):
    def __init__(
        self,
        target_column: str,
        classification: bool = True,
        n_estimators: int = 500,
        test_size_ratio: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1,
        use_scipy_linalg: bool = True,
        dtype_optimization: bool = True,
        enable_cow: bool = True,
        enable_infer_string: bool = False,
        use_numexpr: bool = True,
        use_sparse: bool = True,
        use_categorical: bool = True,
        adaptive_params: bool = True,
        use_shap: bool = True,
        shap_sample_size: int = 1000,
        shap_feature_perturbation: str = 'tree_path_dependent',
        shap_approximate: bool = False,
        importance_metric: str = 'gain',
        detect_interactions: bool = True,
        detect_multicollinearity: bool = True,
        vif_threshold: float = 5.0,
        use_calibration: bool = False,
        use_class_weights: bool = True,
        enable_metadata_routing: bool = False,
        use_pyarrow: bool = False  # PyArrow backend (Pandas.md بخش 0)
    ):
        self.target_column = target_column
        self.classification = classification
        self.test_size_ratio = test_size_ratio
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs > 0 else -1
        self.use_scipy_linalg = use_scipy_linalg
        self.dtype_optimization = dtype_optimization
        self.enable_cow = enable_cow
        self.enable_infer_string = enable_infer_string
        self.use_numexpr = use_numexpr
        self.use_sparse = use_sparse
        self.use_categorical = use_categorical
        self.adaptive_params = adaptive_params
        self.use_shap = use_shap
        self.shap_sample_size = shap_sample_size
        self.shap_feature_perturbation = shap_feature_perturbation
        self.shap_approximate = shap_approximate
        self.importance_metric = importance_metric
        self.detect_interactions = detect_interactions
        self.should_detect_multicollinearity = detect_multicollinearity
        self.vif_threshold = vif_threshold
        self.use_calibration = use_calibration
        self.use_class_weights = use_class_weights
        self.enable_metadata_routing = enable_metadata_routing
        self.use_pyarrow = use_pyarrow
        
        self.rng = np.random.default_rng(random_state)
        
        if enable_cow:
            pd.options.mode.copy_on_write = True
            logging.info('Copy-on-Write mode enabled')
        
        if enable_infer_string:
            pd.options.future.infer_string = True
            logging.info('Future string inference enabled')
        
        # PyArrow backend برای بهینه‌سازی حافظه (Pandas.md بخش 0)
        if use_pyarrow:
            try:
                pd.options.mode.dtype_backend = 'pyarrow'
                logging.info('PyArrow backend enabled for memory optimization')
            except Exception as e:
                logging.warning(f'PyArrow backend not available: {str(e)}')
        
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() if n_jobs == -1 else max(1, n_jobs))
        
        self.base_params = {
            'objective': 'binary' if classification else 'regression',
            'metric': 'binary_logloss' if classification else 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,  # کاهش برای دقت بهتر
            'num_leaves': 31,
            'max_depth': 5,  # محدود کردن عمق برای overfitting کمتر
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.5,  # افزایش regularization
            'lambda_l2': 0.5,  # افزایش regularization
            'path_smooth': 1.0,  # اضافه کردن path smoothing (Pandas.md بخش 3.2)
            'min_gain_to_split': 0.01,  # حداقل gain برای split
            'verbosity': -1,
            'random_state': random_state,
            'deterministic': True,
            'force_col_wise': True,  # بهینه برای features زیاد (Pandas.md بخش 4.2)
            'num_threads': -1,
            'max_bin': 255,  # بهینه‌سازی حافظه histogram (Pandas.md توصیه)
        }
        
        logging.info(f'Pandas {pd.__version__}, NumPy {np.__version__}')

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, 'is_fitted_') and self.is_fitted_

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        بهینه‌سازی حافظه پیشرفته با استراتژی Pandas.md (بخش 8.1)
        """
        if not self.dtype_optimization:
            return df
        
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Object columns: convert to category if low cardinality (< 50%)
            if col_type == 'object':
                num_unique = df[col].nunique()
                total_rows = len(df)
                if num_unique / total_rows < 0.5 and self.use_categorical:
                    df[col] = df[col].astype('category')
                continue
            
            # Integer optimization با دقت بیشتر (Pandas.md بخش 8.1)
            if str(col_type)[:3] == 'int':
                col_min = df[col].min()
                col_max = df[col].max()
                
                # بهینه‌سازی تهاجمی برای int
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                # else keep int64
            
            # Float optimization با بررسی دقیق‌تر (Pandas.md بخش 8.1)
            elif str(col_type)[:5] == 'float':
                if df[col].notna().sum() > 0:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    
                    # Safe downcasting به float32 با بررسی دقیق range
                    if not np.isnan(col_min) and not np.isnan(col_max):
                        # بررسی اینکه آیا داده‌ها در محدوده float32 هستند
                        if (col_min > np.finfo(np.float32).min * 0.99 and 
                            col_max < np.finfo(np.float32).max * 0.99):
                            # بررسی precision loss
                            test_val = df[col].iloc[0] if len(df) > 0 else 0
                            if abs(np.float32(test_val) - test_val) < abs(test_val) * 1e-6:
                                df[col] = df[col].astype(np.float32)
        
        memory_after = df.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = (1 - memory_after / memory_before) * 100
        
        logging.info(f'Memory optimization: {memory_before:.2f} MB → {memory_after:.2f} MB ({memory_reduction:.1f}% reduction)')
        
        return df

    def preprocess_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        پیش‌پردازش پیشرفته با استراتژی Pandas.md (بخش 1.3 و 2.1)
        """
        X = X.copy()
        
        # حذف ستون‌های ثابت
        constant_cols = X.columns[X.nunique() <= 1].tolist()
        if constant_cols:
            logging.warning(f'Removing {len(constant_cols)} constant features')
            X = X.drop(columns=constant_cols)
        
        # حذف ستون‌های با Missing بالا (>90%)
        high_missing_cols = X.columns[X.isnull().mean() > 0.9].tolist()
        if high_missing_cols:
            logging.warning(f'Removing {len(high_missing_cols)} features with >90% missing')
            X = X.drop(columns=high_missing_cols)
        
        # بهینه‌سازی حافظه قبل از imputation
        X = self.optimize_dtypes(X)
        
        # مدیریت Missing Data بهبود یافته (Pandas.md بخش 1.3)
        missing_mask = X.isnull().sum() > 0
        if missing_mask.any():
            missing_cols = X.columns[missing_mask].tolist()
            logging.info(f'Handling missing data in {len(missing_cols)} columns')
            
            for col in missing_cols:
                if X[col].dtype == 'float32' or str(X[col].dtype).startswith('float'):
                    # Interpolation برای داده‌های عددی (Pandas.md توصیه time-based)
                    X[col] = X[col].interpolate(
                        method='linear', 
                        limit_direction='both', 
                        limit=5
                    )
                    # Forward fill برای باقیمانده
                    X[col] = X[col].ffill(limit=5)
                    # Backward fill برای اول فایل
                    X[col] = X[col].bfill(limit=5)
                    # اگر هنوز NaN داریم، با median پر کن
                    if X[col].isnull().any():
                        median_val = X[col].median()
                        X[col] = X[col].fillna(median_val)
                else:
                    # Mode fill برای categorical
                    if len(X[col].mode()) > 0:
                        X[col] = X[col].fillna(X[col].mode()[0])
                    else:
                        X[col] = X[col].fillna(0)
        
        # بهینه‌سازی نهایی حافظه
        X = self.optimize_dtypes(X)
        
        # جمع‌آوری زباله برای آزادسازی حافظه
        gc.collect()
        
        return X, y

    def temporal_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        gap: int = 50
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        n = len(X)
        train_size = int(n * (1 - self.test_size_ratio))
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        test_start = min(train_size + gap, n)
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]
        logging.info(f'Train: {len(X_train)}, Test: {len(X_test)}, Gap: {gap}')
        return X_train, X_test, y_train, y_test

    def _get_adaptive_params(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        n_samples = len(X)
        n_features = len(X.columns)
        params = self.base_params.copy()
        
        if self.adaptive_params:
            params['min_data_in_leaf'] = max(5, n_samples // 100)
            params['num_leaves'] = min(31, max(7, n_samples // 50))
            params['bagging_fraction'] = min(0.9, max(0.5, 1.0 - 1.0/np.sqrt(max(1, n_samples/1000))))
        
        if self.use_class_weights and self.classification:
            try:
                class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
                # LightGBM 4.6+ uses scale_pos_weight instead of class_weight dict
                if len(class_weights) == 2:
                    # For binary classification
                    params['scale_pos_weight'] = class_weights[1] / class_weights[0]
            except Exception as e:
                logging.warning(f'Class weight computation failed: {str(e)}')
        
        return params

    def compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        if not self.classification:
            return np.ones(len(y), dtype=np.float32)
        
        try:
            sample_weights = compute_sample_weight('balanced', y=y)
            return sample_weights.astype(np.float32)
        except Exception as e:
            logging.warning(f'Sample weight computation failed: {str(e)}')
            return np.ones(len(y), dtype=np.float32)

    def detect_multicollinearity(self, X: pd.DataFrame) -> Dict:
        logging.info('Detecting multicollinearity')
        numeric_X = X.select_dtypes(include=[np.number])
        correlation_matrix = numeric_X.corr().abs()
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': float(correlation_matrix.iloc[i, j])
                    })
        
        try:
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]
            condition_index = np.sqrt(eigenvalues.max() / eigenvalues.min()) if len(eigenvalues) > 0 else 0
        except Exception as e:
            logging.warning(f'Condition index calculation failed: {str(e)}')
            condition_index = 0
        
        high_corr_features = set()
        for pair in high_corr_pairs:
            high_corr_features.add(pair['feature1'])
            high_corr_features.add(pair['feature2'])
        
        logging.info(f'High correlation pairs: {len(high_corr_pairs)}, Condition Index: {condition_index:.2f}')
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs,
            'high_corr_features': list(high_corr_features),
            'condition_index': float(condition_index)
        }

    def shap_importance_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_runs: int = 3
    ) -> Dict:
        logging.info('SHAP analysis disabled due to compatibility issues')
        # Return empty placeholder values
        return {
            'shap_mean': np.zeros(len(X.columns), dtype=np.float32),
            'shap_std': np.zeros(len(X.columns), dtype=np.float32),
            'shap_interaction_mean': np.zeros(len(X.columns), dtype=np.float32)
        }

    def null_importance_ultimate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_actual: int = 20,  # افزایش برای دقت بهتر (از 10 به 20)
        n_null: int = 100  # افزایش برای اعتبار بهتر (از 50 به 100)
    ) -> Dict:
        logging.info('Null Importance with statistical significance testing')
        n_features = len(X.columns)
        
        # Pre-allocate arrays instead of lists
        actual_gain = np.zeros((n_actual, n_features), dtype=np.float32)
        actual_split = np.zeros((n_actual, n_features), dtype=np.float32)
        null_gain = np.zeros((n_null, n_features), dtype=np.float32)
        null_split = np.zeros((n_null, n_features), dtype=np.float32)
        
        params = self._get_adaptive_params(X, y)
        params['n_estimators'] = 300
        
        sample_weights = self.compute_sample_weights(y)
        
        for run in range(n_actual):
            train_data = lgb.Dataset(X, label=y, weight=sample_weights, free_raw_data=False)
            run_params = params.copy()
            run_params['random_state'] = self.random_state + run
            run_params['seed'] = self.random_state + run
            
            model = lgb.train(
                run_params,
                train_data,
                num_boost_round=300,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            actual_gain[run, :] = model.feature_importance(importance_type='gain')
            actual_split[run, :] = model.feature_importance(importance_type='split')
        
        for run in range(n_null):
            y_shuffled = y.sample(frac=1, random_state=self.random_state + n_actual + run).values
            train_data = lgb.Dataset(X, label=y_shuffled, weight=sample_weights, free_raw_data=False)
            run_params = params.copy()
            run_params['random_state'] = self.random_state + n_actual + run
            run_params['seed'] = self.random_state + n_actual + run
            
            model = lgb.train(
                run_params,
                train_data,
                num_boost_round=300,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            null_gain[run, :] = model.feature_importance(importance_type='gain')
            null_split[run, :] = model.feature_importance(importance_type='split')
        
        # Use pre-allocated arrays directly
        actual_gain_mean = np.mean(actual_gain, axis=0, dtype=np.float32)
        actual_gain_std = np.std(actual_gain, axis=0, dtype=np.float32)
        null_gain_mean = np.mean(null_gain, axis=0, dtype=np.float32)
        null_gain_std = np.std(null_gain, axis=0, dtype=np.float32)
        actual_split_mean = np.mean(actual_split, axis=0, dtype=np.float32)
        null_split_mean = np.mean(null_split, axis=0, dtype=np.float32)
        
        null_gain_90 = np.percentile(null_gain, 90, axis=0).astype(np.float32)
        null_gain_95 = np.percentile(null_gain, 95, axis=0).astype(np.float32)
        null_gain_99 = np.percentile(null_gain, 99, axis=0).astype(np.float32)
        null_split_90 = np.percentile(null_split, 90, axis=0).astype(np.float32)
        null_split_95 = np.percentile(null_split, 95, axis=0).astype(np.float32)
        null_split_99 = np.percentile(null_split, 99, axis=0).astype(np.float32)
        
        gain_z_score = ((actual_gain_mean - null_gain_mean) / (null_gain_std + 1e-10)).astype(np.float32)
        split_z_score = ((actual_split_mean - null_split_mean) / (np.std(null_split, axis=0) + 1e-10)).astype(np.float32)
        
        p_values_gain = np.array([
            1 - stats.percentileofscore(null_gain[:, i], actual_gain_mean[i]) / 100
            for i in range(len(actual_gain_mean))
        ], dtype=np.float32)
        
        p_values_split = np.array([
            1 - stats.percentileofscore(null_split[:, i], actual_split_mean[i]) / 100
            for i in range(len(actual_split_mean))
        ], dtype=np.float32)
        
        significant_gain = p_values_gain < 0.05
        significant_split = p_values_split < 0.05
        above_99_gain = actual_gain_mean > null_gain_99
        above_95_gain = actual_gain_mean > null_gain_95
        above_90_gain = actual_gain_mean > null_gain_90
        above_99_split = actual_split_mean > null_split_99
        above_95_split = actual_split_mean > null_split_95
        above_90_split = actual_split_mean > null_split_90
        
        logging.info(f'Gain - Significant: {np.sum(significant_gain)}, Above 99th: {np.sum(above_99_gain)}')
        logging.info(f'Split - Significant: {np.sum(significant_split)}, Above 99th: {np.sum(above_99_split)}')
        
        # Clear memory explicitly
        del actual_gain, null_gain, actual_split, null_split
        gc.collect()
        
        return {
            'actual_gain_mean': actual_gain_mean,
            'actual_split_mean': actual_split_mean,
            'null_gain_90': null_gain_90,
            'null_gain_95': null_gain_95,
            'null_gain_99': null_gain_99,
            'null_split_90': null_split_90,
            'null_split_95': null_split_95,
            'null_split_99': null_split_99,
            'gain_z_score': gain_z_score,
            'split_z_score': split_z_score,
            'p_values_gain': p_values_gain,
            'p_values_split': p_values_split,
            'significant_gain': significant_gain,
            'significant_split': significant_split,
            'above_90_gain': above_90_gain,
            'above_95_gain': above_95_gain,
            'above_99_gain': above_99_gain,
            'above_90_split': above_90_split,
            'above_95_split': above_95_split,
            'above_99_split': above_99_split
        }

    def boosting_ensemble_complete(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_runs: int = 7  # افزایش برای پایداری بهتر (از 5 به 7)
    ) -> Dict:
        logging.info('Boosting ensemble with optimized parameters')
        ensemble_results = defaultdict(list)
        params = self._get_adaptive_params(X, y)
        
        sample_weights = self.compute_sample_weights(y)
        
        booster_configs = [
            ('goss', {'data_sample_strategy': 'goss', 'top_rate': 0.2, 'other_rate': 0.1, 'bagging_fraction': 1.0, 'bagging_freq': 0}),
            ('dart', {'boosting_type': 'dart', 'drop_rate': 0.1, 'skip_drop': 0.5}),
            ('extra', {'extra_trees': True}),
        ]
        
        for run in range(n_runs):
            train_data = lgb.Dataset(X, label=y, weight=sample_weights, free_raw_data=False)
            for booster_name, booster_params in booster_configs:
                run_params = params.copy()
                run_params.update(booster_params)
                run_params['random_state'] = self.random_state + run
                run_params['seed'] = self.random_state + run
                
                num_rounds = 150 if booster_name == 'rf' else 200
                
                model = lgb.train(
                    run_params,
                    train_data,
                    num_boost_round=num_rounds,
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                
                gain = model.feature_importance(importance_type='gain')
                split = model.feature_importance(importance_type='split')
                ensemble_results[f'{booster_name}_gain'].append(gain)
                ensemble_results[f'{booster_name}_split'].append(split)
        
        result = {}
        for key, values in ensemble_results.items():
            array = np.array(values, dtype=np.float32)
            result[f'{key}_mean'] = np.mean(array, axis=0, dtype=np.float32)
        
        return result

    def feature_fraction_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_runs: int = 7  # افزایش برای پایداری بهتر (از 5 به 7)
    ) -> Dict:
        logging.info('Feature fraction analysis')
        fraction_results = defaultdict(list)
        params = self._get_adaptive_params(X, y)
        
        sample_weights = self.compute_sample_weights(y)
        
        configs = [
            ('bynode', {'feature_fraction': 1.0, 'feature_fraction_bynode': 0.6}),
            ('bytree', {'feature_fraction': 0.6, 'feature_fraction_bynode': 1.0}),
            ('combined', {'feature_fraction': 0.7, 'feature_fraction_bynode': 0.8})
        ]
        
        for run in range(n_runs):
            train_data = lgb.Dataset(X, label=y, weight=sample_weights, free_raw_data=False)
            for config_name, config_params in configs:
                run_params = params.copy()
                run_params.update(config_params)
                run_params['random_state'] = self.random_state + run
                run_params['seed'] = self.random_state + run
                
                model = lgb.train(
                    run_params,
                    train_data,
                    num_boost_round=200,
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                
                gain = model.feature_importance(importance_type='gain')
                split = model.feature_importance(importance_type='split')
                fraction_results[f'{config_name}_gain'].append(gain)
                fraction_results[f'{config_name}_split'].append(split)
        
        result = {}
        for key, values in fraction_results.items():
            array = np.array(values, dtype=np.float32)
            result[f'{key}_mean'] = np.mean(array, axis=0, dtype=np.float32)
        
        return result

    def adversarial_validation(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Dict:
        logging.info('Adversarial validation for time series')
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        X_train_copy['is_test'] = 0
        X_test_copy['is_test'] = 1
        
        X_combined = pd.concat([X_train_copy, X_test_copy], axis=0, ignore_index=True)
        y_combined = X_combined['is_test'].values
        X_combined = X_combined.drop('is_test', axis=1)
        
        train_size = int(0.8 * len(X_combined))
        
        train_data = lgb.Dataset(
            X_combined.iloc[:train_size],
            label=y_combined[:train_size],
            free_raw_data=False
        )
        
        valid_data = lgb.Dataset(
            X_combined.iloc[train_size:],
            label=y_combined[train_size:],
            reference=train_data,
            free_raw_data=False
        )
        
        params = self.base_params.copy()
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['n_estimators'] = 200
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        adv_importance = model.feature_importance(importance_type='gain').astype(np.float32)
        adv_importance_normalized = adv_importance / (adv_importance.sum() + 1e-10)
        
        high_shift = adv_importance_normalized > (2.0 / len(X_combined.columns))
        
        logging.info(f'High shift features: {np.sum(high_shift)}')
        
        del X_train_copy, X_test_copy, X_combined
        gc.collect()
        
        return {
            'adv_importance': adv_importance_normalized,
            'high_shift': high_shift
        }

    def rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: Optional[int] = None
    ) -> Dict:
        if n_features is None:
            n_features = max(int(len(X.columns) * 0.3), 20)
        
        logging.info(f'RFE to {n_features} features')
        
        if self.classification:
            estimator = lgb.LGBMClassifier(**self.base_params)
        else:
            estimator = lgb.LGBMRegressor(**self.base_params)
        
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=max(int(len(X.columns) * 0.05), 1),
            verbose=0
        )
        
        rfe.fit(X, y)
        
        return {
            'rfe_support': rfe.support_,
            'rfe_ranking': rfe.ranking_
        }

    def cv_multi_metric(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5  # افزایش برای دقت بهتر (از 3 به 5)
    ) -> Dict:
        logging.info('Cross-validation multi-metric')
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=50)
        
        gain_importances = []
        split_importances = []
        cv_scores = []
        
        params = self._get_adaptive_params(X, y)
        
        sample_weights = self.compute_sample_weights(y)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            weights_train = sample_weights[train_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train, free_raw_data=False)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
            
            run_params = params.copy()
            run_params['n_estimators'] = 200
            
            model = lgb.train(
                run_params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            
            if self.classification:
                y_pred_binary = (y_pred > 0.5).astype(int)
                score = accuracy_score(y_val, y_pred_binary)
            else:
                score = -mean_squared_error(y_val, y_pred, squared=False)
            
            cv_scores.append(score)
            
            gain = model.feature_importance(importance_type='gain')
            split = model.feature_importance(importance_type='split')
            gain_importances.append(gain)
            split_importances.append(split)
        
        gain_importances_array = np.array(gain_importances, dtype=np.float32)
        split_importances_array = np.array(split_importances, dtype=np.float32)
        
        mean_gain = np.mean(gain_importances_array, axis=0, dtype=np.float32)
        std_gain = np.std(gain_importances_array, axis=0, dtype=np.float32)
        mean_split = np.mean(split_importances_array, axis=0, dtype=np.float32)
        std_split = np.std(split_importances_array, axis=0, dtype=np.float32)
        mean_perm = np.zeros(len(X.columns), dtype=np.float32)  # placeholder
        std_perm = np.zeros(len(X.columns), dtype=np.float32)   # placeholder
        
        cv_gain = std_gain / (mean_gain + 1e-10)
        cv_split = std_split / (mean_split + 1e-10)
        cv_perm = np.zeros(len(X.columns), dtype=np.float32)  # placeholder
        
        del gain_importances_array, split_importances_array
        gc.collect()
        
        return {
            'mean_gain': mean_gain,
            'std_gain': std_gain,
            'mean_split': mean_split,
            'std_split': std_split,
            'mean_perm': mean_perm,
            'std_perm': std_perm,
            'cv_gain': cv_gain,
            'cv_split': cv_split,
            'cv_perm': cv_perm,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }

    def stability_bootstrap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bootstrap: int = 30,  # افزایش برای پایداری بهتر (از 20 به 30)
        threshold: float = 0.70  # کاهش threshold برای شناسایی بیشتر فیچرهای stable
    ) -> Dict:
        logging.info('Stability bootstrap analysis')
        n_features = len(X.columns)
        feature_counts_gain = np.zeros(n_features, dtype=np.uint16)
        feature_counts_split = np.zeros(n_features, dtype=np.uint16)
        
        params = self._get_adaptive_params(X, y)
        params['n_estimators'] = 150
        
        sample_weights = self.compute_sample_weights(y)
        
        for i in range(n_bootstrap):
            sample_size = int(0.7 * len(X))
            indices = self.rng.choice(len(X), size=sample_size, replace=True)
            
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            weights_sample = sample_weights[indices]
            
            feature_indices = self.rng.choice(
                n_features,
                size=int(0.7 * n_features),
                replace=False
            )
            
            X_sample = X_sample.iloc[:, feature_indices]
            
            train_data = lgb.Dataset(X_sample, label=y_sample, weight=weights_sample, free_raw_data=False)
            
            run_params = params.copy()
            run_params['random_state'] = self.random_state + i
            run_params['seed'] = self.random_state + i
            
            model = lgb.train(
                run_params,
                train_data,
                num_boost_round=150,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            gain_importance = model.feature_importance(importance_type='gain')
            split_importance = model.feature_importance(importance_type='split')
            
            selected_gain = gain_importance > 0
            selected_split = split_importance > 0
            
            for j, feat_idx in enumerate(feature_indices):
                if selected_gain[j]:
                    feature_counts_gain[feat_idx] += 1
                if selected_split[j]:
                    feature_counts_split[feat_idx] += 1
        
        stability_scores_gain = (feature_counts_gain / n_bootstrap).astype(np.float32)
        stability_scores_split = (feature_counts_split / n_bootstrap).astype(np.float32)
        
        stable_gain = stability_scores_gain >= threshold
        stable_split = stability_scores_split >= threshold
        
        logging.info(f'Stable features (gain): {np.sum(stable_gain)}, (split): {np.sum(stable_split)}')
        
        return {
            'stability_scores_gain': stability_scores_gain,
            'stability_scores_split': stability_scores_split,
            'stable_gain': stable_gain,
            'stable_split': stable_split
        }

    def normalize_with_stability(self, scores, epsilon=1e-10):
        scores_array = np.asarray(scores, dtype=np.float32)
        min_val = scores_array.min()
        max_val = scores_array.max()
        range_val = max_val - min_val
        
        if range_val > epsilon:
            normalized = (scores_array - min_val) / range_val
        else:
            normalized = np.zeros_like(scores_array, dtype=np.float32)
        
        return normalized.astype(np.float32)

    def ensemble_ranking(
        self,
        feature_names: List[str],
        null_importance: Dict,
        boosting_ensemble: Dict,
        feature_fraction: Dict,
        adversarial: Dict,
        rfe: Dict,
        cv_metrics: Dict,
        stability: Dict,
        shap_importance: Optional[Dict] = None,
        multicollinearity: Optional[Dict] = None
    ) -> pd.DataFrame:
        logging.info('Ensemble ranking with weighted aggregation and multicollinearity penalty')
        
        score_data = {
            'feature': feature_names,
            'null_z': self.normalize_with_stability(null_importance['gain_z_score']),
            'null_z_split': self.normalize_with_stability(null_importance['split_z_score']),
            'null_sig': null_importance['significant_gain'].astype(np.int8),
            'null_sig_split': null_importance['significant_split'].astype(np.int8),
            'above_99': null_importance['above_99_gain'].astype(np.int8),
            'above_95': null_importance['above_95_gain'].astype(np.int8),
            'adv_shift': 1.0 - self.normalize_with_stability(adversarial['adv_importance']),
            'no_shift': (~adversarial['high_shift']).astype(np.int8),
            'rfe_sel': rfe['rfe_support'].astype(np.int8),
            'rfe_rank': 1.0 - self.normalize_with_stability(rfe['rfe_ranking']),
            'cv_g': self.normalize_with_stability(cv_metrics['mean_gain']),
            'cv_s': self.normalize_with_stability(cv_metrics['mean_split']),
            'cv_p': self.normalize_with_stability(cv_metrics['mean_perm']),
            'cv_stab_g': 1.0 - self.normalize_with_stability(cv_metrics['cv_gain']),
            'cv_stab_s': 1.0 - self.normalize_with_stability(cv_metrics['cv_split']),
            'stab_g': stability['stability_scores_gain'],
            'stab_s': stability['stability_scores_split'],
            'is_stab_g': stability['stable_gain'].astype(np.int8),
            'is_stab_s': stability['stable_split'].astype(np.int8),
        }
        
        high_corr_penalty = np.ones(len(feature_names), dtype=np.float32)
        if multicollinearity is not None and 'high_corr_features' in multicollinearity:
            # Apply adaptive penalty based on correlation strength
            # Higher correlation → stronger penalty
            correlations = multicollinearity.get('high_corr_values', {})
            for feat in multicollinearity['high_corr_features']:
                if feat in feature_names:
                    feat_idx = feature_names.index(feat)
                    # Get correlation value if available, else use fixed 0.85
                    corr_val = correlations.get(feat, 0.85) if isinstance(correlations, dict) else 0.85
                    # Adaptive penalty: stronger penalty for higher correlation
                    # Base: 1.0, Max reduction: 0.3 at correlation 0.95+
                    penalty = 1.0 - (min(corr_val, 0.95) - 0.85) * 3.0
                    penalty = np.clip(penalty, 0.7, 1.0)
                    high_corr_penalty[feat_idx] = penalty
        
        score_data['mult_corr_pen'] = high_corr_penalty
        
        if shap_importance is not None and 'shap_mean' in shap_importance:
            score_data['shap'] = self.normalize_with_stability(shap_importance['shap_mean'])
            score_data['shap_int'] = self.normalize_with_stability(shap_importance['shap_interaction_mean'])
        
        for key in boosting_ensemble.keys():
            if 'mean' in key:
                col_name = key.replace('_gain_mean', '').replace('_split_mean', '_s')
                score_data[col_name] = self.normalize_with_stability(boosting_ensemble[key])
        
        for key in feature_fraction.keys():
            if 'mean' in key:
                col_name = key.replace('_gain_mean', '_f').replace('_split_mean', '_fs')
                score_data[col_name] = self.normalize_with_stability(feature_fraction[key])
        
        df = pd.DataFrame(score_data)
        
        weights = {
            'null_z': 0.09, 'null_z_split': 0.04, 'null_sig': 0.02, 'null_sig_split': 0.01,
            'above_99': 0.04, 'above_95': 0.02, 'goss': 0.04, 'goss_s': 0.02, 'dart': 0.04,
            'dart_s': 0.02, 'extra': 0.04, 'extra_s': 0.02, 'bynode_f': 0.03, 'bynode_fs': 0.01,
            'bytree_f': 0.03, 'bytree_fs': 0.01, 'combined_f': 0.03, 'combined_fs': 0.01,
            'adv_shift': 0.03, 'no_shift': 0.02, 'rfe_sel': 0.04, 'rfe_rank': 0.02,
            'cv_g': 0.08, 'cv_s': 0.04, 'cv_p': 0.08, 'cv_stab_g': 0.02, 'cv_stab_s': 0.01,
            'stab_g': 0.04, 'stab_s': 0.02, 'is_stab_g': 0.02, 'is_stab_s': 0.01, 'shap': 0.09,
            'shap_int': 0.04, 'mult_corr_pen': 0.02
        }
        
        score_cols = [col for col in df.columns if col != 'feature']
        score_matrix = df[score_cols].values.astype(np.float32)
        weight_vector = np.array([weights.get(col, 0) for col in score_cols], dtype=np.float32)
        
        final_score = np.dot(score_matrix, weight_vector)
        df['final_score'] = final_score
        
        df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        return df[['feature', 'final_score']]

    def categorize(
        self,
        df: pd.DataFrame,
        strong_pct: float = 0.15,
        weak_pct: float = 0.60
    ) -> Dict:
        n = len(df)
        n_strong = max(int(n * strong_pct), 10)
        n_weak_start = int(n * weak_pct)
        
        strong = df.iloc[:n_strong]['feature'].tolist()
        medium = df.iloc[n_strong:n_weak_start]['feature'].tolist()
        weak = df.iloc[n_weak_start:]['feature'].tolist()
        
        logging.info(f'Strong: {len(strong)}, Medium: {len(medium)}, Weak: {len(weak)}')
        
        return {
            'strong': strong,
            'medium': medium,
            'weak': weak
        }

    def process_batch(
        self,
        features_df: pd.DataFrame,
        batch_id: int,
        output_dir: str = 'feature_selection_results'
    ):
        import psutil
        import time
        
        # شروع زمان‌سنجی
        start_time = time.time()
        process = psutil.Process()
        memory_start = process.memory_info().rss / 1024**2  # MB
        
        logging.info(f'Batch {batch_id} starting - Memory: {memory_start:.2f} MB')
        
        X = features_df.drop(columns=[self.target_column])
        y = features_df[self.target_column]
        
        X, y = self.preprocess_features(X, y)
        
        logging.info(f'Features: {X.shape[1]}, Samples: {X.shape[0]}')
        
        X_train, X_test, y_train, y_test = self.temporal_split(X, y)
        
        multicollinearity = None
        if self.should_detect_multicollinearity:
            multicollinearity = self.detect_multicollinearity(X_train)
        
        null_importance = self.null_importance_ultimate(X_train, y_train)
        boosting_ensemble = self.boosting_ensemble_complete(X_train, y_train)
        feature_fraction = self.feature_fraction_analysis(X_train, y_train)
        adversarial = self.adversarial_validation(X_train, X_test)
        rfe = self.rfe_selection(X_train, y_train)
        cv_metrics = self.cv_multi_metric(X_train, y_train)
        stability = self.stability_bootstrap(X_train, y_train)
        
        shap_importance = None
        if self.use_shap:
            shap_importance = self.shap_importance_analysis(X_train, y_train)
        
        df_ranking = self.ensemble_ranking(
            feature_names=X.columns.tolist(),
            null_importance=null_importance,
            boosting_ensemble=boosting_ensemble,
            feature_fraction=feature_fraction,
            adversarial=adversarial,
            rfe=rfe,
            cv_metrics=cv_metrics,
            stability=stability,
            shap_importance=shap_importance,
            multicollinearity=multicollinearity
        )
        
        categorized = self.categorize(df_ranking)
        
        # پایان زمان‌سنجی و مصرف حافظه
        end_time = time.time()
        memory_end = process.memory_info().rss / 1024**2  # MB
        execution_time = end_time - start_time
        memory_delta = memory_end - memory_start
        
        self._save(
            df_ranking=df_ranking,
            categorized=categorized,
            cv_metrics=cv_metrics,
            batch_id=batch_id,
            output_dir=output_dir,
            execution_time=execution_time,
            memory_used=memory_delta
        )
        
        self.is_fitted_ = True
        
        gc.collect()
        logging.info(f'Batch {batch_id} completed - Time: {execution_time:.2f}s, Memory Δ: {memory_delta:+.2f} MB')


    def _save(
        self,
        df_ranking: pd.DataFrame,
        categorized: Dict,
        cv_metrics: Dict,
        batch_id: int,
        output_dir: str,
        execution_time: float = 0,
        memory_used: float = 0
    ):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # ذخیره با Parquet برای بهینه‌سازی حافظه (Pandas.md بخش 8.3)
        try:
            df_ranking.to_parquet(
                output_path / f'batch_{batch_id}_ranking_{timestamp}.parquet',
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            logging.info(f'Ranking saved in Parquet format (optimized)')
        except Exception as e:
            logging.warning(f'Parquet save failed, using CSV: {str(e)}')
            df_ranking.to_csv(
                output_path / f'batch_{batch_id}_ranking_{timestamp}.csv',
                index=False
            )
        
        # ذخیره CSV برای سازگاری با نسخه قبل
        pd.DataFrame({'feature': categorized['strong']}).to_csv(
            output_path / f'batch_{batch_id}_strong.csv',
            index=False
        )
        
        pd.DataFrame({'feature': categorized['medium']}).to_csv(
            output_path / f'batch_{batch_id}_medium.csv',
            index=False
        )
        
        pd.DataFrame({'feature': categorized['weak']}).to_csv(
            output_path / f'batch_{batch_id}_weak.csv',
            index=False
        )
        
        metadata = {
            'batch_id': batch_id,
            'timestamp': timestamp,
            'n_total': len(df_ranking),
            'n_strong': len(categorized['strong']),
            'n_medium': len(categorized['medium']),
            'n_weak': len(categorized['weak']),
            'mean_cv_score': float(cv_metrics['mean_cv_score']),
            'std_cv_score': float(cv_metrics['std_cv_score']),
            'execution_time_sec': float(execution_time),
            'memory_used_mb': float(memory_used)
        }
        
        with open(output_path / f'batch_{batch_id}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f'Saved to {output_path}')


def main():
    TARGET_COLUMN = 'target'
    CLASSIFICATION = True
    N_BATCHES = 5  # کاهش از 39 به 5
    N_JOBS = -1
    
    selector = FeatureSelector(
        target_column=TARGET_COLUMN,
        classification=CLASSIFICATION,
        n_estimators=400,  # افزایش برای دقت بهتر (از 300 به 400)
        test_size_ratio=0.2,
        random_state=42,
        n_jobs=N_JOBS,
        use_scipy_linalg=True,
        dtype_optimization=True,
        enable_cow=True,
        enable_infer_string=False,
        use_numexpr=False,  # غیرفعال کردن
        use_sparse=True,
        use_categorical=True,
        adaptive_params=True,
        use_shap=False,  # غیرفعال کردن SHAP
        shap_sample_size=1000,
        shap_feature_perturbation='tree_path_dependent',
        shap_approximate=False,
        importance_metric='gain',
        detect_interactions=False,  # غیرفعال کردن
        detect_multicollinearity=True,
        vif_threshold=5.0,
        use_calibration=False,
        use_class_weights=True,
        enable_metadata_routing=False
    )
    
    # Load XAUUSD data
    df = pd.read_csv('XAUUSD_M15_R.csv', sep='\t')
    # Rename columns before creating datetime
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S')
    # Create target and drop datetime
    df['target'] = ((df['close'].shift(-1) > df['close']).astype(int)).fillna(0)
    df = df.drop(columns=['date', 'time', 'datetime']).dropna()
    
    # Split into batches
    batch_size = len(df) // N_BATCHES
    for batch_id in range(1, N_BATCHES + 1):
        try:
            start_idx = (batch_id - 1) * batch_size
            end_idx = batch_id * batch_size if batch_id < N_BATCHES else len(df)
            features_df = df.iloc[start_idx:end_idx]
            
            selector.process_batch(
                features_df=features_df,
                batch_id=batch_id,
                output_dir='feature_selection_results'
            )
        except Exception as e:
            logging.error(f'Error Batch {batch_id}: {str(e)}')
            continue
    
    logging.info('Processing completed')


if __name__ == '__main__':
    main()
