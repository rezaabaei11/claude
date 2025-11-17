import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from collections import OrderedDict
from sklearn.model_selection import TimeSeriesSplit, cross_validate, cross_val_predict, train_test_split
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
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
import sys
try:
    from packaging import version as _version
except Exception:
    try:
        from pkg_resources import parse_version as _parse_version
        class _DummyVersionModule:
            @staticmethod
            def parse(v):
                return _parse_version(v)
        _version = _DummyVersionModule
    except Exception:
        _version = None
from datetime import datetime
import os
import gc
from collections import defaultdict
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
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
        use_numexpr: bool = True,
        use_sparse: bool = True,
        use_categorical: bool = True,
        adaptive_params: bool = True,
        enable_cow: bool = True,
        enable_infer_string: bool = False,
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
        use_pyarrow: bool = False,
        ensure_reproducible: bool = True,
        categorical_unique_ratio_threshold: float = 0.5,
        perm_top_k: int = 300,
        max_threads_for_cpu: int = 16,
        enable_dataset_cache: bool = True,
        max_cache_size: int = 32,
        use_mutual_information: bool = True,
        use_stability_selection: bool = True,
        use_nested_cv: bool = True,
        use_confidence_intervals: bool = True,
        n_bootstrap_ci: int = 50,
        stability_selection_iterations: int = 30,
    ):
        self.target_column = target_column
        self.max_cache_size = max_cache_size
        self.use_mutual_information = use_mutual_information
        self.use_stability_selection = use_stability_selection
        self.use_nested_cv = use_nested_cv
        self.use_confidence_intervals = use_confidence_intervals
        self.n_bootstrap_ci = n_bootstrap_ci
        self.stability_selection_iterations = stability_selection_iterations
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
        if self.enable_metadata_routing:
            try:
                from sklearn import set_config
                set_config(enable_metadata_routing=True, transform_output='pandas')
                logging.info('Metadata routing and pandas output ENABLED (scikit-learn >= 1.3)')
            except Exception as e:
                logging.warning(f'Failed to enable metadata routing: {e}')
        self.use_pyarrow = use_pyarrow
        self.ensure_reproducible = ensure_reproducible
        self.categorical_unique_ratio_threshold = categorical_unique_ratio_threshold
        self.perm_top_k = perm_top_k
        self.max_threads_for_cpu = max_threads_for_cpu
        self.enable_dataset_cache = enable_dataset_cache
        self.rng = np.random.default_rng(random_state)
        self.seed_sequence = np.random.SeedSequence(random_state)
        if self.ensure_reproducible:
            os.environ['PYTHONHASHSEED'] = str(self.random_state)
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            import random
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            logging.info('Reproducible mode ON: Generator API with SeedSequence (no global state pollution)')
        if self.ensure_reproducible:
            try:
                import psutil
                physical_cores = psutil.cpu_count(logical=False)
                self.num_threads = min(4, physical_cores if physical_cores else os.cpu_count())
            except Exception:
                self.num_threads = 2
            self.deterministic_mode = True
        else:
            try:
                import psutil
                physical_cores = psutil.cpu_count(logical=False)
                self.num_threads = physical_cores if physical_cores else os.cpu_count()
            except Exception:
                self.num_threads = os.cpu_count() or 4
            self.deterministic_mode = False
        self.lgb_defaults = {
            'deterministic': True,
            'seed': self.random_state,
            'num_threads': self.num_threads,
            'force_col_wise': True
        }
        if enable_cow:
            pd.options.mode.copy_on_write = True
            logging.info('Copy-on-Write mode enabled')
        if enable_infer_string:
            pd.options.future.infer_string = True
            logging.info('Future string inference enabled')
        if use_pyarrow:
            try:
                pd.options.mode.dtype_backend = 'pyarrow'
                logging.info('PyArrow backend enabled for memory optimization')
            except Exception as e:
                logging.warning(f'PyArrow backend not available: {str(e)}')
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False)
            if physical_cores:
                if self.ensure_reproducible:
                    self.num_threads = 1
                    os.environ['OMP_NUM_THREADS'] = '1'
                    os.environ['MKL_NUM_THREADS'] = '1'
                    os.environ['OPENBLAS_NUM_THREADS'] = '1'
                    logging.info(f'Reproducible mode: using {self.num_threads} thread (deterministic=True)')
                else:
                    os.environ['OMP_NUM_THREADS'] = str(physical_cores)
                    self.num_threads = min(physical_cores, self.max_threads_for_cpu)
                    logging.info(f'Using {physical_cores} physical CPU cores')
            else:
                self.num_threads = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
                os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        except ImportError:
            self.num_threads = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
            os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
            logging.warning('psutil not available, using logical cores')
        self.base_params = {
            'objective': 'binary' if classification else 'regression',
            'metric': 'binary_logloss' if classification else 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,  # FIXED: Reduced from 80 to prevent overfitting
            'max_depth': 8,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,  # FIXED: Increased from 30 for regularization
            'lambda_l1': 1.0,  # FIXED: Increased from 0.3 for L1 regularization
            'lambda_l2': 3.0,  # FIXED: Increased from 2.0 for L2 regularization
            'path_smooth': 10.0,
            'min_gain_to_split': 0.02,
            'verbosity': -1,
            'random_state': random_state,
            'deterministic': True,
            'force_col_wise': True,
            'num_threads': self.num_threads,
            'feature_fraction_seed': self.random_state,
            'bagging_seed': self.random_state,
            'data_random_seed': self.random_state,
            'max_bin': 255,
            'min_data_in_bin': 5,
            'histogram_pool_size': None,
            'min_sum_hessian_in_leaf': 1.0,
            'early_stopping_rounds': 50,  # FIXED: Added early stopping to prevent overfitting
        }
        logging.info(f'Pandas {pd.__version__}, NumPy {np.__version__}')
        self._validate_tree_params()
        if self.ensure_reproducible:
            self._ensure_full_reproducibility()
        self._dataset_cache = OrderedDict() if self.enable_dataset_cache else None
    def _ensure_full_reproducibility(self):
        try:
            os.environ['PYTHONHASHSEED'] = str(self.random_state)
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            np.random.seed(self.random_state)
            import random as _py_random
            _py_random.seed(self.random_state)
            self.base_params.update({
                'seed': self.random_state,
                'deterministic': True,
                'force_col_wise': True,
                'num_threads': 1,
                'feature_fraction_seed': self.random_state,
                'bagging_seed': self.random_state,
                'data_random_seed': self.random_state,
            })
            logging.info('Full reproducibility configuration applied (num_threads=1)')
        except Exception as e:
            logging.warning(f'Failed to set full reproducibility flags: {e}')
    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, 'is_fitted_') and self.is_fitted_
    def _get_feature_selection_params_default(self, classification: bool, random_state: int, num_threads: int) -> Dict:
        return {
            'objective': 'binary' if classification else 'regression',
            'metric': 'binary_logloss' if classification else 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,  # FIXED: Optimized for stability
            'max_depth': 6,
            'min_data_in_leaf': 50,  # FIXED: Higher minimum for regularization
            'feature_fraction': 0.6,
            'feature_fraction_bynode': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 1.0,  # FIXED: Increased L1 regularization
            'lambda_l2': 3.0,  # FIXED: Increased L2 regularization
            'min_gain_to_split': 0.05,
            'path_smooth': 5.0,
            'verbosity': -1,
            'seed': random_state,
            'random_state': random_state,
            'deterministic': True,
            'force_col_wise': True,
            'num_threads': num_threads,
            'feature_fraction_seed': random_state,
            'bagging_seed': random_state,
            'data_random_seed': random_state,
            'max_bin': 255,
            'min_data_in_bin': 5,
            'early_stopping_rounds': 50,  # FIXED: Early stopping for overfitting prevention
            'histogram_pool_size': None,
            'min_sum_hessian_in_leaf': 1.0,
        }
    def _validate_tree_params(self):
        num_leaves = self.base_params.get('num_leaves', 31)
        max_depth = self.base_params.get('max_depth', -1)
        if max_depth > 0:
            max_possible_leaves = 2 ** max_depth
            recommended_max = int(max_possible_leaves * 0.7)
            if num_leaves > recommended_max:
                logging.warning(
                    f'num_leaves={num_leaves} is high for max_depth={max_depth}. '
                    f'Recommended: num_leaves <= {recommended_max} (0.7 * 2^{max_depth})'
                )
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.dtype_optimization:
            return df
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        object_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(object_cols) > 0 and self.use_categorical:
            for col in object_cols:
                if df[col].nunique(dropna=True) / len(df) < self.categorical_unique_ratio_threshold:
                    df[col] = df[col].astype('category')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_type = df[col].dtype
            if str(col_type)[:3] == 'int':
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == 'float':
                if df[col].notna().any():
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if not pd.isna(c_min) and not pd.isna(c_max):
                        if (c_min > np.finfo(np.float32).min * 0.99 and 
                            c_max < np.finfo(np.float32).max * 0.99):
                            sample_vals = df[col].dropna().sample(min(100, len(df[col].dropna())), random_state=42)
                            precision_ok = (np.abs(sample_vals.astype(np.float32) - sample_vals) < np.abs(sample_vals) * 1e-6).all()
                            if precision_ok:
                                df[col] = df[col].astype(np.float32)
        memory_after = df.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = ((memory_before - memory_after) / memory_before) * 100 if memory_before > 0 else 0
        logging.info(f'Memory optimization: {memory_before:.2f} MB -> {memory_after:.2f} MB ({memory_reduction:.1f}% reduction)')
        return df
    def preprocess_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        X = X.copy()
        new_cols = []
        for col in X.columns:
            new_col = col.replace('"', '').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace(',', '_').replace(' ', '_')
            new_cols.append(new_col)
        if new_cols != list(X.columns):
            X.columns = new_cols
            logging.info(f'Sanitized feature names for LightGBM compatibility')
        constant_mask = X.nunique() <= 1
        constant_cols = X.columns[constant_mask].tolist()
        if constant_cols:
            logging.warning(f'Removing {len(constant_cols)} constant features')
            X = X.drop(columns=constant_cols)
        missing_ratios = X.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > 0.9].index.tolist()
        if high_missing_cols:
            logging.warning(f'Removing {len(high_missing_cols)} features with >90% missing')
            X = X.drop(columns=high_missing_cols)
        X = self.optimize_dtypes(X)
        missing_mask = X.isnull().any()
        if missing_mask.any():
            missing_cols = X.columns[missing_mask].tolist()
            logging.info(f'Handling missing data in {len(missing_cols)} columns')
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            missing_numeric = [col for col in missing_cols if col in numeric_cols]
            missing_categorical = [col for col in missing_cols if col not in numeric_cols]
            if missing_numeric:
                for col in missing_numeric:
                    X[col] = (X[col]
                             .interpolate(method='linear', limit_direction='both', limit=5)
                             .ffill(limit=5)
                             .bfill(limit=5))
                    if X[col].isnull().any():
                        median_val = X[col].median()
                        X[col] = X[col].fillna(median_val)
            if missing_categorical:
                for col in missing_categorical:
                    mode_val = X[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                    X[col] = X[col].fillna(fill_val)
        X = self.optimize_dtypes(X)
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
        X_train = X.iloc[:train_size].copy()
        y_train = y.iloc[:train_size].copy()
        test_start = min(train_size + gap, n)
        X_test = X.iloc[test_start:].copy()
        y_test = y.iloc[test_start:].copy()
        logging.info(f'Train: {len(X_train)}, Test: {len(X_test)}, Gap: {gap}')
        return X_train, X_test, y_train, y_test
    def _get_adaptive_params(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        n_samples = len(X)
        n_features = len(X.columns)
        params = self.base_params.copy()
        try:
            strategy = self._determine_optimal_histogram_strategy(n_samples, n_features)
            if strategy == 'col_wise':
                params['force_col_wise'] = True
                params['force_row_wise'] = False
            else:
                params['force_col_wise'] = False
                params['force_row_wise'] = True
        except Exception:
            pass
        if self.adaptive_params:
            params['min_data_in_leaf'] = max(5, n_samples // 100)
            params['num_leaves'] = min(31, max(7, n_samples // 50))
            params['bagging_fraction'] = min(0.9, max(0.5, 1.0 - 1.0/np.sqrt(max(1, n_samples/1000))))
            try:
                opt_threads = self._optimize_num_threads_for_dataset(n_samples, n_features)
                params['num_threads'] = opt_threads
                self.num_threads = opt_threads
            except Exception:
                pass
            try:
                params['min_data_in_leaf'] = self._calculate_optimal_min_data_in_leaf(n_samples, n_features, params.get('num_leaves', 31))
            except Exception:
                pass
            try:
                feat_cfg = self._optimize_feature_fraction_strategy(n_features)
                params.update(feat_cfg)
            except Exception:
                pass
        if params.get('histogram_pool_size') is None:
            try:
                params['histogram_pool_size'] = self._calculate_optimal_histogram_pool(
                    n_features=n_features, n_samples=n_samples
                )
            except Exception:
                estimated_pool_mb = (params.get('num_leaves', 31) * 20 * n_features * 
                                    params.get('max_bin', 255)) // (1024**2)
                params['histogram_pool_size'] = max(256, min(2048, estimated_pool_mb))
            logging.debug(f'Adaptive histogram_pool_size: {params["histogram_pool_size"]} MB (features: {n_features})')
        if self.use_class_weights and self.classification:
            try:
                class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
                if len(class_weights) == 2:
                    params['scale_pos_weight'] = class_weights[1] / class_weights[0]
            except Exception as e:
                logging.warning(f'Class weight computation failed: {str(e)}')
        return params
    def _determine_optimal_histogram_strategy(self, n_samples: int, n_features: int) -> str:
        """
        Adaptive histogram strategy based on LightGBM optimization guidelines:
        - force_col_wise=True: Best when n_features >> n_samples (high-dimensional data)
        - force_row_wise=True: Best when n_samples >> n_features (many samples, few features)
        Reference: LightGBM documentation and benchmarks
        """
        max_bin = int(self.base_params.get('max_bin', 255))
        total_bins = n_features * max_bin
        ratio = n_features / max(1, n_samples)

        # High-dimensional data (many features like tsfresh) → col_wise
        if n_features > 1000 or ratio > 0.5:
            strategy = 'col_wise'
            reason = 'high-dimensional data (many features)'
        # Large datasets with fewer features → row_wise
        elif n_samples > 50000 and ratio < 0.1:
            strategy = 'row_wise'
            reason = 'large dataset with fewer features'
        # Large total bins → col_wise for better cache utilization
        elif total_bins > 100000:
            strategy = 'col_wise'
            reason = 'large total bins'
        # Many threads available and moderate features → col_wise
        elif self.num_threads and self.num_threads > 8 and n_features > 100:
            strategy = 'col_wise'
            reason = 'multi-threaded with moderate features'
        # Default: col_wise is generally safer for reproducibility
        else:
            strategy = 'col_wise'
            reason = 'default (safer for reproducibility)'

        logging.info(f"Histogram strategy: {strategy} ({reason}), samples={n_samples}, features={n_features}, ratio={ratio:.4f}")
        return strategy
    def _optimize_num_threads_for_dataset(self, n_samples: int, n_features: int) -> int:
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False) or 1
            logical_cores = psutil.cpu_count(logical=True) or physical_cores
        except Exception:
            physical_cores = os.cpu_count() or 1
            logical_cores = physical_cores
        if self.ensure_reproducible:
            return 1
        min_samples_per_thread = 5000
        max_threads_by_samples = max(1, n_samples // min_samples_per_thread)
        if n_samples < 10000:
            optimal_threads = min(4, physical_cores)
        elif n_samples < 50000:
            optimal_threads = min(8, physical_cores)
        elif n_samples < 200000:
            optimal_threads = physical_cores
        else:
            optimal_threads = min(logical_cores, int(physical_cores * 1.5))
        optimal_threads = int(min(optimal_threads, max_threads_by_samples))
        if n_features > 2000:
            optimal_threads = min(physical_cores, int(optimal_threads * 1.2))
        optimal_threads = max(1, optimal_threads)
        logging.info(
            f"Optimal threads: {optimal_threads} (physical={physical_cores}, samples={n_samples}, features={n_features})"
        )
        return optimal_threads
    def _calculate_optimal_min_data_in_leaf(self, n_samples: int, n_features: int, num_leaves: int) -> int:
        safety_factor = 8
        base_value = max(1, n_samples // (max(1, num_leaves) * safety_factor))
        if n_samples < 5000:
            min_leaf = max(5, base_value)
        elif n_samples < 20000:
            min_leaf = max(20, base_value)
        elif n_samples < 100000:
            min_leaf = max(50, base_value)
        else:
            min_leaf = max(100, base_value)
        if n_features > 1000:
            min_leaf = int(min_leaf * 1.5)
        min_leaf = min(min_leaf, max(1, n_samples // 10))
        logging.info(f"Optimal min_data_in_leaf: {min_leaf} (samples={n_samples}, leaves={num_leaves})")
        return int(min_leaf)
    def _optimize_feature_fraction_strategy(self, n_features: int) -> Dict:
        configs = {}
        if n_features > 3000:
            configs['feature_fraction'] = 0.4
            configs['feature_fraction_bynode'] = 0.5
        elif n_features > 1500:
            configs['feature_fraction'] = 0.5
            configs['feature_fraction_bynode'] = 0.6
        elif n_features > 500:
            configs['feature_fraction'] = 0.6
            configs['feature_fraction_bynode'] = 0.7
        else:
            configs['feature_fraction'] = 0.8
            configs['feature_fraction_bynode'] = 0.9
        logging.info(f"Feature sampling: tree={configs['feature_fraction']}, node={configs['feature_fraction_bynode']} (features={n_features})")
        return configs
    def _optimize_memory_usage(self, n_samples: int, n_features: int) -> Dict:
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
        except Exception:
            available_gb = 4.0
        num_leaves = int(self.base_params.get('num_leaves', 31))
        max_bin = int(self.base_params.get('max_bin', 255))
        estimated_gb = (num_leaves * n_features * max_bin * 12) / (1024 ** 3)
        logging.info(f"Memory: available={available_gb:.2f}GB, estimated={estimated_gb:.2f}GB")
        if estimated_gb > available_gb * 0.6:
            optimizations = {}
            if max_bin > 127:
                optimizations['max_bin'] = 127
                logging.info('Reduced max_bin: 255 → 127')
            if num_leaves > 31:
                optimizations['num_leaves'] = 31
                logging.info('Reduced num_leaves: 80 → 31')
            pool_size = self.base_params.get('histogram_pool_size', 1024)
            if pool_size > 512:
                optimizations['histogram_pool_size'] = 512
                logging.info('Reduced histogram_pool_size → 512MB')
            optimizations['free_raw_data'] = True
            return optimizations
        return {}
    def _parallel_cv_strategy(self, n_splits: int, n_threads_available: int) -> Tuple[int, int]:
        if n_splits >= n_threads_available:
            strategy = 'sequential_cv'
            cv_n_jobs = 1
            model_n_threads = n_threads_available
        else:
            strategy = 'parallel_cv'
            cv_n_jobs = n_splits
            model_n_threads = max(1, n_threads_available // n_splits)
        logging.info(f"CV strategy: {strategy} (cv_n_jobs={cv_n_jobs}, model_threads={model_n_threads})")
        return cv_n_jobs, model_n_threads
    def _get_optimized_boosting_configs(self, n_samples: int, n_features: int):
        configs = []
        configs.append(('gbdt', {
            'boosting_type': 'gbdt',
            'bagging_freq': 5,
            'bagging_fraction': 0.8
        }))
        if n_samples > 50000:
            configs.append(('goss', {
                'boosting_type': 'goss',
                'top_rate': 0.2,
                'other_rate': 0.1,
                'bagging_fraction': 1.0,
                'bagging_freq': 0
            }))
        configs.append(('dart', {
            'boosting_type': 'dart',
            'drop_rate': 0.15,
            'skip_drop': 0.5,
            'max_drop': 50,
            'uniform_drop': False
        }))
        if n_features < 1000:
            configs.append(('rf', {
                'boosting_type': 'rf',
                'bagging_freq': 1,
                'bagging_fraction': 0.9,
                'feature_fraction': 0.8
            }))
        return configs
    def _calculate_optimal_histogram_pool(self, n_features: int, n_samples: int) -> int:
        max_bin = int(self.base_params.get('max_bin', 255))
        num_leaves = int(self.base_params.get('num_leaves', 31))
        bytes_per_bin = 12
        estimated_bytes = num_leaves * n_features * max_bin * bytes_per_bin
        estimated_mb = estimated_bytes / (1024 ** 2)
        min_pool = 256
        max_pool = min(4096, max(512, n_samples // 100))
        optimal_pool = int(max(min_pool, min(max_pool, estimated_mb * 1.2)))
        logging.debug(f'Optimal histogram_pool_size estimation: {optimal_pool} MB (estimated: {estimated_mb:.2f} MB)')
        return optimal_pool
    def _create_dataset(self, X: pd.DataFrame, y: pd.Series, weight=None, reference=None):
        n_features = len(X.columns)
        try:
            adaptive_pool = self._calculate_optimal_histogram_pool(n_features, len(X))
        except Exception:
            estimated_pool_mb = (31 * 20 * n_features * 255) // (1024**2)
            adaptive_pool = max(256, min(2048, estimated_pool_mb))
        dataset_params = {
            'force_col_wise': True,
            'feature_pre_filter': True,
            'max_bin': int(self.base_params.get('max_bin', 255)),
            'min_data_in_bin': int(self.base_params.get('min_data_in_bin', 5)),
            'histogram_pool_size': adaptive_pool
        }
        dataset_kwargs = dict(
            data=X,
            label=y,
            weight=weight,
            params=dataset_params,
            free_raw_data=False
        )
        if reference is not None:
            dataset_kwargs['reference'] = reference
        try:
            return lgb.Dataset(**dataset_kwargs)
        except Exception as e:
            dataset_kwargs.pop('params', None)
            try:
                return lgb.Dataset(data=X, label=y, weight=weight, reference=reference, free_raw_data=False)
            except Exception:
                return lgb.Dataset(X, label=y, weight=weight)
    def _create_dataset_cached(self, X: pd.DataFrame, y: pd.Series, weight=None, cache_key=None, reference=None):
        if self._dataset_cache is None:
            return self._create_dataset(X, y, weight=weight, reference=reference)
        if cache_key is not None and cache_key in self._dataset_cache:
            cached = self._dataset_cache[cache_key]
            if weight is not None:
                cached.set_weight(weight)
            return cached
        dataset = self._create_dataset(X, y, weight=weight, reference=reference)
        if cache_key is not None and self._dataset_cache is not None:
            max_cache_size = getattr(self, 'max_cache_size', 32)
            if len(self._dataset_cache) >= max_cache_size:
                oldest_key = next(iter(self._dataset_cache))
                del self._dataset_cache[oldest_key]
                logging.debug(f'Dataset cache evicted: {oldest_key} (max_size={max_cache_size})')
            self._dataset_cache[cache_key] = dataset
        return dataset
    def _get_binned_reference(self, X: pd.DataFrame, cache_key: Optional[str] = None):
        if self._dataset_cache is None:
            return self._create_dataset(X, pd.Series(np.zeros(len(X))), weight=None)
        if cache_key is not None and cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]
        try:
            dataset = self._create_dataset(X, pd.Series(np.zeros(len(X))), weight=None)
        except Exception:
            dataset = self._create_dataset(X, pd.Series(np.zeros(len(X))), weight=None)
        if cache_key is not None and self._dataset_cache is not None:
            self._dataset_cache[cache_key] = dataset
        return dataset
    def _clear_dataset_cache(self):
        self._dataset_cache.clear()
        gc.collect()
    def _train_with_fallback(self, params: Dict, train_data, num_boost_round: int, valid_sets=None, callbacks=None):
        if valid_sets is None:
            valid_sets = []
        if callbacks is None:
            callbacks = []
        safe_params = params.copy()
        try:
            current_ver = _version.parse(lgb.__version__)
            if current_ver < _version.parse('4.6.0'):
                for p in ['use_quantized_grad', 'min_sum_hessian_in_leaf', 'path_smooth']:
                    safe_params.pop(p, None)
        except Exception:
            pass
        try:
            if safe_params.get('use_quantized_grad', False):
                safe_params.setdefault('num_grad_quant_bins', 16)
            return lgb.train(
                safe_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks
            )
        except TypeError as e:
            logging.warning(f'LGB train TypeError, retrying without advanced params: {e}')
            for p in ['use_quantized_grad', 'min_sum_hessian_in_leaf', 'path_smooth', 'histogram_pool_size']:
                safe_params.pop(p, None)
            return lgb.train(
                safe_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks
            )
    def _early_stopping_callback(self, stopping_rounds: int = 100, min_delta: float = 1e-4, first_metric_only: bool = True, verbose: bool = False):
        try:
            import inspect
            sig = inspect.signature(lgb.early_stopping)
            supports_min_delta = 'min_delta' in sig.parameters
        except Exception:
            try:
                supports_min_delta = _version.parse(lgb.__version__) >= _version.parse('4.0.0')
            except Exception:
                supports_min_delta = False
        if supports_min_delta:
            return lgb.early_stopping(stopping_rounds=stopping_rounds, min_delta=min_delta, first_metric_only=first_metric_only, verbose=verbose)
        if min_delta and min_delta > 0:
            logging.warning('LightGBM %s does not support `min_delta` in early_stopping; min_delta will be ignored', getattr(lgb, '__version__', '<unknown>'))
        return lgb.early_stopping(stopping_rounds=stopping_rounds, first_metric_only=first_metric_only, verbose=verbose)
    def _adaptive_early_stopping(self, n_estimators: int = 500, context: str = 'default'):
        stopping_configs = {
            'null_importance': {
                'stopping_rounds': max(30, n_estimators // 10),
                'min_delta': 0.001
            },
            'cv': {
                'stopping_rounds': max(50, n_estimators // 5),
                'min_delta': 0.0001
            },
            'rfe': {
                'stopping_rounds': max(20, n_estimators // 15),
                'min_delta': 0.0005
            },
            'default': {
                'stopping_rounds': max(50, n_estimators // 8),
                'min_delta': 0.0001
            }
        }
        cfg = stopping_configs.get(context, stopping_configs['default'])
        return self._early_stopping_callback(stopping_rounds=cfg['stopping_rounds'], min_delta=cfg['min_delta'], first_metric_only=True, verbose=False)
    def log_environment(self, output_dir: str = 'feature_selection_results') -> Dict:
        try:
            import importlib.metadata as importlib_metadata
        except Exception:
            import importlib_metadata
        import sys
        packages = ['numpy', 'pandas', 'lightgbm', 'scikit-learn', 'scipy']
        env = {}
        env['python'] = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
        for pkg in packages:
            try:
                env[pkg] = importlib_metadata.version(pkg)
            except Exception:
                try:
                    import pkg_resources
                    env[pkg] = pkg_resources.get_distribution(pkg).version
                except Exception:
                    env[pkg] = None
        env['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        with open(output_path / f'environment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(env, f, indent=2)
        logging.info(f'Logged environment versions to {output_path}')
        return env
    def get_accuracy_focused_params_2025(self) -> Dict:
        params = self.base_params.copy()
        params.update({
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'max_bin': 255,
            'num_threads': 1 if self.ensure_reproducible else self.num_threads,
            'feature_fraction': 0.6,
            'feature_fraction_seed': self.random_state,
            'min_data_in_leaf': 30,
            'min_child_samples': 30,
            'lambda_l1': 0.3,
            'lambda_l2': 2.0,
            'path_smooth': 10.0,
            'use_quantized_grad': False,
            'num_grad_quant_bins': 16,
            'min_sum_hessian_in_leaf': 1.0,
            'min_data_in_bin': 5,
            'bagging_seed': self.random_state,
        })
        return params
    def train_and_save_final_model(self, X: pd.DataFrame, y: pd.Series, output_dir: str, batch_id: int = 0):
        params = self.get_accuracy_focused_params_2025()
        sample_weights = self.compute_sample_weights(y)
        train_data = self._create_dataset(X, y, weight=sample_weights)
        model = self._train_with_fallback(params, train_data, num_boost_round=params.get('n_estimators', 1000))
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        model_path = out / f'batch_{batch_id}_final_model.txt'
        model.save_model(str(model_path))
        meta = {
            'model_path': str(model_path),
            'params': params,
            'seed': self.random_state
        }
        with open(out / f'batch_{batch_id}_final_model_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        logging.info('Saved final model to %s', model_path)
        return model_path
    def smoke_test_reproducibility(self, n_samples: int = 200, n_features: int = 20) -> bool:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=self.random_state)
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
        y = pd.Series(y)
        params = self.get_accuracy_focused_params_2025()
        sample_weights = self.compute_sample_weights(y)
        train_data = self._create_dataset(X, y, weight=sample_weights)
        model1 = self._train_with_fallback(params, train_data, num_boost_round=100)
        imp1 = model1.feature_importance(importance_type='gain')
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        train_data = self._create_dataset(X, y, weight=sample_weights)
        model2 = self._train_with_fallback(params, train_data, num_boost_round=100)
        imp2 = model2.feature_importance(importance_type='gain')
        same = np.allclose(imp1, imp2)
        logging.info('Smoke reproducibility test: %s', 'PASS' if same else 'FAIL')
        return bool(same)
    def compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        if not self.classification:
            return np.ones(len(y), dtype=np.float32)
        try:
            sample_weights = compute_sample_weight('balanced', y=y)
            return sample_weights.astype(np.float32)
        except Exception as e:
            logging.warning(f'Sample weight computation failed: {str(e)}')
            return np.ones(len(y), dtype=np.float32)
    def _calculate_optimal_shap_sample_size(self, n_features: int, n_samples: int) -> int:
            try:
                recommended = int(20 * np.sqrt(max(1, n_features)))
            except Exception:
                recommended = 1000
            max_allowed = max(1, int(0.3 * n_samples))
            min_required = min(int(0.5 * n_samples), 500)
            opt = max(min_required, recommended)
            opt = min(opt, max_allowed)
            logging.info(f"SHAP sample size set to {opt} (recommended={recommended}, max_allowed={max_allowed})")
            return max(1, opt)
    def _calculate_optimal_ts_cv_params(self, n_samples: int) -> Dict:
        if n_samples < 500:
            n_splits = 2
            gap = max(0, n_samples // 20)
            test_size = max(20, n_samples // 10)
        elif n_samples < 2000:
            n_splits = 3
            gap = max(10, n_samples // 50)
            test_size = max(50, n_samples // 10)
        elif n_samples < 10000:
            n_splits = 5
            gap = max(50, n_samples // 100)
            test_size = max(100, n_samples // 10)
        else:
            n_splits = 5
            gap = max(96, n_samples // 200)
            test_size = max(500, n_samples // 20)
        logging.info(f"Optimal TS-CV: n_splits={n_splits}, gap={gap}, test_size={test_size} (n_samples={n_samples})")
        return {'n_splits': n_splits, 'gap': gap, 'test_size': test_size}
    def _shap_interaction_for_top_features(self, X: pd.DataFrame, y: pd.Series, shap_mean: np.ndarray, n_top_features: int = 50) -> np.ndarray:
            import shap
            n_top = min(n_top_features, X.shape[1])
            top_idx = np.argsort(shap_mean)[-n_top:][::-1]
            top_feats = X.columns[top_idx]
            sample_size = min(300, len(X))
            idx = self.rng.choice(len(X), size=sample_size, replace=False)
            X_sample = X.iloc[idx][top_feats]
            y_sample = y.iloc[idx]
            if self.classification:
                model = lgb.LGBMClassifier(**self.base_params)
            else:
                model = lgb.LGBMRegressor(**self.base_params)
            model.fit(X_sample, y_sample)
            explainer = shap.TreeExplainer(model)
            try:
                interactions = explainer.shap_interaction_values(X_sample)
            except Exception as e:
                logging.warning(f'SHAP interaction computation failed: {e}')
                return np.zeros(X.shape[1], dtype=np.float32)
            if isinstance(interactions, list):
                interactions = interactions[1]
            interaction_strength = np.zeros(n_top, dtype=np.float32)
            for i in range(n_top):
                off_diag = np.abs(interactions[:, i, :]).sum(axis=1) - np.abs(interactions[:, i, i])
                interaction_strength[i] = float(np.mean(off_diag))
            full_interaction = np.zeros(X.shape[1], dtype=np.float32)
            for i, fi in enumerate(top_idx):
                full_interaction[fi] = interaction_strength[i]
            return full_interaction
    def detect_multicollinearity(self, X: pd.DataFrame) -> Dict:
        logging.info('Detecting multicollinearity (NumPy optimized)')
        numeric_X = X.select_dtypes(include=[np.number])
        feature_names = numeric_X.columns.tolist()
        Xvalues = np.ascontiguousarray(numeric_X.values.astype(np.float32))
        corr_matrix = np.corrcoef(Xvalues.T).astype(np.float32)
        np.abs(corr_matrix, out=corr_matrix)
        upper_indices = np.triu_indices_from(corr_matrix, k=1)
        high_corr_mask = corr_matrix[upper_indices] > 0.9
        high_corr_pairs = []
        high_corr_features = set()
        if high_corr_mask.any():
            pair_indices = np.where(high_corr_mask)[0]
            for idx in pair_indices:
                i, j = upper_indices[0][idx], upper_indices[1][idx]
                high_corr_pairs.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': float(corr_matrix[i, j])
                })
                high_corr_features.add(feature_names[i])
                high_corr_features.add(feature_names[j])
        try:
            if self.use_scipy_linalg:
                try:
                    from scipy.sparse.linalg import eigsh
                    eigenvalues_max = eigsh(corr_matrix, k=1, which='LA', return_eigenvectors=False, maxiter=500)
                    eigenvalues_min = eigsh(corr_matrix, k=1, which='SA', return_eigenvectors=False, maxiter=500)
                    condition_index = np.sqrt(np.float64(eigenvalues_max[0]) / np.float64(eigenvalues_min[0]))
                except Exception as e:
                    logging.debug(f'Sparse eigenvalue computation failed: {e}, falling back to full')
                    from scipy import linalg as scipy_linalg
                    eigenvalues = scipy_linalg.eigvals(corr_matrix)
                    eigenvalues = np.real(eigenvalues)
                    eigenvalues = eigenvalues[eigenvalues > 1e-10]
                    if len(eigenvalues) > 0:
                        condition_index = np.sqrt(np.float64(eigenvalues.max()) / np.float64(eigenvalues.min()))
                    else:
                        condition_index = 0.0
            else:
                eigenvalues = np.linalg.eigvals(corr_matrix)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                if len(eigenvalues) > 0:
                    condition_index = np.sqrt(np.float64(eigenvalues.max()) / np.float64(eigenvalues.min()))
                else:
                    condition_index = 0.0
        except Exception as e:
            logging.warning(f'Condition index calculation failed: {str(e)}')
            condition_index = 0.0
        logging.info(f'High correlation pairs: {len(high_corr_pairs)}, Condition Index: {condition_index:.2f}')
        return {
            'correlation_matrix': corr_matrix,
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
        logging.info('SHAP analysis started (deterministic sample)')
        try:
            import shap
            recommended = self._calculate_optimal_shap_sample_size(len(X.columns), len(X))
            sample_size = min(self.shap_sample_size or recommended, recommended)
            n_runs = max(1, int(n_runs))
            try:
                from joblib import Parallel, delayed
                use_joblib = True
            except Exception:
                use_joblib = False
            def _single_shap_run(run_idx: int):
                seed = int(self.random_state + run_idx)
                rng_run = np.random.default_rng(seed)
                idx = rng_run.choice(len(X), size=min(sample_size, len(X)), replace=False)
                X_sample = X.iloc[idx]
                y_sample = y.iloc[idx]
                if self.classification:
                    model_params = self.base_params.copy()
                    model_params.pop('random_state', None)
                    model = lgb.LGBMClassifier(**model_params, random_state=seed)
                else:
                    model_params = self.base_params.copy()
                    model_params.pop('random_state', None)
                    model = lgb.LGBMRegressor(**model_params, random_state=seed)
                model.fit(X_sample, y_sample)
                explainer = shap.TreeExplainer(
                    model,
                    feature_perturbation=self.shap_feature_perturbation,
                    model_output='raw'
                )
                try:
                    shap_values = explainer.shap_values(X_sample, check_additivity=True)
                except Exception as e:
                    logging.debug(f'SHAP additivity check failed (run {run_idx}): {e}; retrying without check')
                    shap_values = explainer.shap_values(X_sample, check_additivity=False)
                if isinstance(shap_values, list):
                    shap_vals = np.mean(np.abs(shap_values[1]), axis=0)
                else:
                    shap_vals = np.mean(np.abs(shap_values), axis=0)
                return shap_vals.astype(np.float32)
            if n_runs > 1 and use_joblib:
                n_jobs = min(max(1, self.num_threads // 2), 8)
                results = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(_single_shap_run)(run) for run in range(n_runs)
                )
                shap_values_runs = np.vstack(results)
            else:
                shap_values_runs = np.zeros((n_runs, len(X.columns)), dtype=np.float32)
                for run in range(n_runs):
                    shap_values_runs[run, :] = _single_shap_run(run)
            shap_mean = np.mean(shap_values_runs, axis=0).astype(np.float32)
            shap_std = np.std(shap_values_runs, axis=0).astype(np.float32)
            shap_cv = (shap_std / (shap_mean + 1e-10)).astype(np.float32)
            try:
                shap_interaction = self._shap_interaction_for_top_features(X, y, shap_mean)
            except Exception as e:
                logging.debug(f'SHAP interaction calc failed: {e}')
                shap_interaction = np.zeros(len(X.columns), dtype=np.float32)
            return {
                'shap_mean': shap_mean,
                'shap_std': shap_std,
                'shap_cv': shap_cv,
                'shap_interaction_mean': shap_interaction
            }
        except Exception as e:
            logging.warning('SHAP not available or failed: %s', str(e))
            return {
                'shap_mean': np.zeros(len(X.columns), dtype=np.float32),
                'shap_std': np.zeros(len(X.columns), dtype=np.float32),
                'shap_interaction_mean': np.zeros(len(X.columns), dtype=np.float32)
            }
    def shap_importance_analysis_round2(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_runs: int = 5,
        use_sample_weights: bool = True,
        cache_explainer: bool = True,
        add_feature_names: bool = True,
        model_output: Optional[str] = None,
        n_top_interaction: int = 50
    ) -> Dict:
        try:
            import shap
        except Exception as e:
            logging.warning('SHAP not available or failed to import: %s', e)
            return {
                'shap_mean': np.zeros(len(X.columns), dtype=np.float32),
                'shap_std': np.zeros(len(X.columns), dtype=np.float32),
                'shap_cv': np.zeros(len(X.columns), dtype=np.float32),
                'shap_interaction_mean': np.zeros(len(X.columns), dtype=np.float32),
                'n_runs': 0,
                'n_stable_features': 0,
                'sample_size': 0
            }
        recommended = self._calculate_optimal_shap_sample_size(len(X.columns), len(X))
        sample_size = min(self.shap_sample_size or recommended, recommended)
        n_runs = max(1, int(n_runs))
        n_features = len(X.columns)
        try:
            from joblib import Parallel, delayed
            use_joblib = True
        except Exception:
            use_joblib = False
        base_explainer = None
        if cache_explainer:
            try:
                logging.info('Building SHAP base model for explainer (cache_explainer=True)')
                base_sample_weights = self.compute_sample_weights(y) if use_sample_weights else None
                if self.classification:
                    base_model = lgb.LGBMClassifier(**self.base_params)
                else:
                    base_model = lgb.LGBMRegressor(**self.base_params)
                base_model.fit(X, y, sample_weight=base_sample_weights)
                model_output_to_use = model_output
                if model_output_to_use is None:
                    if self.shap_feature_perturbation == 'tree_path_dependent':
                        model_output_to_use = 'raw'
                    else:
                        model_output_to_use = 'probability' if self.classification else 'raw'
                base_explainer = shap.TreeExplainer(
                    base_model,
                    feature_perturbation=self.shap_feature_perturbation,
                    model_output=model_output_to_use,
                    feature_names=X.columns.tolist() if add_feature_names else None
                )
                expected_val = base_explainer.expected_value
                if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
                    logging.info(f'SHAP expected_value (multiclass): {[float(v) for v in expected_val[:3]]}...')
                else:
                    ev = float(expected_val[0]) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val)
                    logging.info(f'SHAP expected_value: {ev:.6f}')
            except Exception as e:
                logging.warning(f'Failed to build cached explainer: {e}')
                base_explainer = None
        def _single_shap_run(run_idx: int):
            seed = int(self.random_state + run_idx)
            rng_run = np.random.default_rng(seed)
            idx = rng_run.choice(len(X), size=min(sample_size, len(X)), replace=False)
            X_sample = X.iloc[idx]
            y_sample = y.iloc[idx]
            weights_sample = self.compute_sample_weights(y_sample) if use_sample_weights else None
            if base_explainer is None:
                model_params = self.base_params.copy()
                model_params.pop('random_state', None)
                if self.classification:
                    model = lgb.LGBMClassifier(**model_params, random_state=seed)
                else:
                    model = lgb.LGBMRegressor(**model_params, random_state=seed)
                model.fit(X_sample, y_sample, sample_weight=weights_sample)
                model_output_to_use = model_output if model_output is not None else ('probability' if self.classification else 'raw')
                if self.shap_feature_perturbation == 'tree_path_dependent':
                    model_output_to_use = 'raw'
                explainer = shap.TreeExplainer(
                    model,
                    feature_perturbation=self.shap_feature_perturbation,
                    model_output=model_output_to_use,
                    feature_names=X.columns.tolist() if add_feature_names else None
                )
            else:
                explainer = base_explainer
            try:
                shap_values = explainer.shap_values(X_sample, check_additivity=True, tree_limit=-1)
            except Exception as e:
                logging.debug(f'SHAP additivity check failed (run {run_idx}): {e}; retrying without check')
                shap_values = explainer.shap_values(X_sample, check_additivity=False, tree_limit=-1)
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_vals = np.mean(np.abs(shap_values[1]), axis=0)
                else:
                    shap_vals = np.mean(np.abs(np.vstack(shap_values)), axis=0)
            else:
                shap_vals = np.mean(np.abs(shap_values), axis=0)
            return shap_vals.astype(np.float32)
        if n_runs > 1 and use_joblib:
            n_jobs = min(max(1, self.num_threads // 2), 8)
            results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
                delayed(_single_shap_run)(run) for run in range(n_runs)
            )
            shap_values_runs = np.vstack(results)
        else:
            shap_values_runs = np.zeros((n_runs, n_features), dtype=np.float32)
            for run in range(n_runs):
                shap_values_runs[run, :] = _single_shap_run(run)
        shap_mean = np.mean(shap_values_runs, axis=0).astype(np.float32)
        shap_std = np.std(shap_values_runs, axis=0).astype(np.float32)
        shap_cv = (shap_std / (shap_mean + 1e-10)).astype(np.float32)
        shap_min = np.min(shap_values_runs, axis=0).astype(np.float32)
        shap_max = np.max(shap_values_runs, axis=0).astype(np.float32)
        try:
            shap_interaction = self._shap_interaction_for_top_features(X, y, shap_mean, n_top_features=n_top_interaction)
        except Exception as e:
            logging.debug(f'SHAP interaction calc failed: {e}')
            shap_interaction = np.zeros(n_features, dtype=np.float32)
        n_stable = int(np.sum(shap_cv < 0.1))
        stable_pct = 100.0 * n_stable / n_features
        logging.info(f'SHAP - Runs: {n_runs}, Mean: {np.mean(shap_mean):.6f}, CV: {np.mean(shap_cv):.6f} - Stable features (<10% cv): {n_stable}/{n_features} ({stable_pct:.1f}%)')
        logging.info(f'SHAP stability - Median CV: {np.median(shap_cv):.6f}, Max CV: {np.max(shap_cv):.6f}')
        return {
            'shap_mean': shap_mean,
            'shap_std': shap_std,
            'shap_cv': shap_cv,
            'shap_min': shap_min,
            'shap_max': shap_max,
            'shap_interaction_mean': shap_interaction,
            'n_runs': n_runs,
            'n_stable_features': n_stable,
            'sample_size': sample_size
        }
    def null_importance_ultimate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_actual: int = 15,
        n_null: int = 75
    ) -> Dict:
        logging.info('Null Importance with statistical significance testing (optimized iterations)')
        n_features = len(X.columns)
        actual_gain = np.zeros((n_actual, n_features), dtype=np.float32, order='F')
        actual_split = np.zeros((n_actual, n_features), dtype=np.float32, order='F')
        actual_cover = np.zeros((n_actual, n_features), dtype=np.float32, order='F')
        null_gain = np.zeros((n_null, n_features), dtype=np.float32, order='F')
        null_split = np.zeros((n_null, n_features), dtype=np.float32, order='F')
        null_cover = np.zeros((n_null, n_features), dtype=np.float32, order='F')
        params = self._get_adaptive_params(X, y)
        params['n_estimators'] = 500
        sample_weights = self.compute_sample_weights(y)
        n_samples = len(X)
        val_split = int(0.2 * n_samples)
        X_train, X_val = X.iloc[val_split:], X.iloc[:val_split]
        y_train, y_val = y.iloc[val_split:], y.iloc[:val_split]
        w_train = sample_weights[val_split:] if sample_weights is not None else None
        w_val = sample_weights[:val_split] if sample_weights is not None else None
        try:
            ref_train = self._get_binned_reference(X_train, cache_key=f'null_bins_train_{len(X_train)}_{len(X_train.columns)}')
            ref_val = self._get_binned_reference(X_val, cache_key=f'null_bins_val_{len(X_val)}_{len(X_val.columns)}')
        except Exception:
            ref_train = None
            ref_val = None
        for run in range(n_actual):
            train_data = self._create_dataset(X_train, y_train, weight=w_train, reference=ref_train)
            val_data = self._create_dataset(X_val, y_val, weight=w_val, reference=train_data)
            run_params = params.copy()
            run_params['random_state'] = self.random_state + run
            run_params['seed'] = self.random_state + run
            model = self._train_with_fallback(
                run_params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    self._adaptive_early_stopping(n_estimators=run_params.get('n_estimators', 500), context='null_importance'),
                    lgb.log_evaluation(period=0)
                ]
            )
            actual_gain[run, :] = model.feature_importance(importance_type='gain')
            actual_split[run, :] = model.feature_importance(importance_type='split')
            try:
                actual_cover[run, :] = model.feature_importance(importance_type='cover')
            except Exception:
                actual_cover[run, :] = 0
        for run in range(n_null):
            # Block Bootstrap: Use circular shift to break X-y relationship while preserving temporal structure
            # This avoids temporal leakage that random shuffle would cause
            rng_null = np.random.default_rng(self.random_state + n_actual + run)
            block_size = max(1, int(np.sqrt(len(y_train))))  # Block size based on sqrt(n)
            shift_amount = rng_null.integers(block_size, len(y_train) - block_size)
            y_shuffled = np.roll(y_train.values, shift_amount)
            train_data = self._create_dataset(X_train, y_shuffled, weight=w_train, reference=ref_train)
            # Same block shift for validation to maintain consistency
            val_shift = rng_null.integers(block_size, max(block_size + 1, len(y_val) - block_size))
            val_shuffled = np.roll(y_val.values, val_shift)
            val_data = self._create_dataset(X_val, val_shuffled, weight=w_val, reference=ref_train)
            run_params = params.copy()
            run_params['random_state'] = self.random_state + n_actual + run
            run_params['seed'] = self.random_state + n_actual + run
            model = self._train_with_fallback(
                run_params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    self._adaptive_early_stopping(n_estimators=run_params.get('n_estimators', 500), context='null_importance'),
                    lgb.log_evaluation(period=0)
                ]
            )
            null_gain[run, :] = model.feature_importance(importance_type='gain')
            null_split[run, :] = model.feature_importance(importance_type='split')
            try:
                null_cover[run, :] = model.feature_importance(importance_type='cover')
            except Exception:
                null_cover[run, :] = 0
        actual_gain_mean = np.mean(actual_gain, axis=0, dtype=np.float64).astype(np.float32)
        actual_gain_std = np.std(actual_gain, axis=0, dtype=np.float64).astype(np.float32)
        null_gain_mean = np.mean(null_gain, axis=0, dtype=np.float64).astype(np.float32)
        null_gain_std = np.std(null_gain, axis=0, dtype=np.float64).astype(np.float32)
        actual_split_mean = np.mean(actual_split, axis=0, dtype=np.float64).astype(np.float32)
        null_split_mean = np.mean(null_split, axis=0, dtype=np.float64).astype(np.float32)
        actual_cover_mean = np.mean(actual_cover, axis=0, dtype=np.float64).astype(np.float32)
        null_cover_mean = np.mean(null_cover, axis=0, dtype=np.float64).astype(np.float32)
        null_gain_percentiles = np.percentile(null_gain, [90, 95, 99], axis=0).astype(np.float32)
        null_gain_90, null_gain_95, null_gain_99 = null_gain_percentiles[0], null_gain_percentiles[1], null_gain_percentiles[2]
        null_split_percentiles = np.percentile(null_split, [90, 95, 99], axis=0).astype(np.float32)
        null_split_90, null_split_95, null_split_99 = null_split_percentiles[0], null_split_percentiles[1], null_split_percentiles[2]
        null_cover_percentiles = np.percentile(null_cover, [90, 99], axis=0).astype(np.float32)
        null_cover_90, null_cover_99 = null_cover_percentiles[0], null_cover_percentiles[1]
        gain_z_score = np.zeros(n_features, dtype=np.float32)
        np.divide(
            actual_gain_mean - null_gain_mean,
            null_gain_std,
            out=gain_z_score,
            where=null_gain_std > 1e-10,
            casting='unsafe'
        )
        null_split_std = np.std(null_split, axis=0, dtype=np.float64).astype(np.float32)
        split_z_score = np.zeros(n_features, dtype=np.float32)
        np.divide(
            actual_split_mean - null_split_mean,
            null_split_std,
            out=split_z_score,
            where=null_split_std > 1e-10,
            casting='unsafe'
        )
        null_cover_std = np.std(null_cover, axis=0, dtype=np.float64).astype(np.float32)
        cover_z_score = np.zeros(n_features, dtype=np.float32)
        np.divide(
            actual_cover_mean - null_cover_mean,
            null_cover_std,
            out=cover_z_score,
            where=null_cover_std > 1e-10,
            casting='unsafe'
        )
        null_gain_sorted = np.sort(null_gain, axis=0)
        null_split_sorted = np.sort(null_split, axis=0)
        null_cover_sorted = np.sort(null_cover, axis=0)
        p_values_gain = np.zeros(n_features, dtype=np.float32)
        p_values_split = np.zeros(n_features, dtype=np.float32)
        p_values_cover = np.zeros(n_features, dtype=np.float32)
        batch_size = 64
        for batch_start in range(0, n_features, batch_size):
            batch_end = min(batch_start + batch_size, n_features)
            for i in range(batch_start, batch_end):
                idx_gain = np.searchsorted(null_gain_sorted[:, i], actual_gain_mean[i])
                p_values_gain[i] = np.float32(1 - (idx_gain / len(null_gain)))
                idx_split = np.searchsorted(null_split_sorted[:, i], actual_split_mean[i])
                p_values_split[i] = np.float32(1 - (idx_split / len(null_split)))
                idx_cover = np.searchsorted(null_cover_sorted[:, i], actual_cover_mean[i])
                p_values_cover[i] = np.float32(1 - (idx_cover / len(null_cover)))
        try:
            from scipy import stats as _sps
            p_values_gain_raw = 1.0 - _sps.norm.cdf(gain_z_score)
            p_values_split_raw = 1.0 - _sps.norm.cdf(split_z_score)
            p_values_cover_raw = 1.0 - _sps.norm.cdf(cover_z_score)
            if HAS_STATSMODELS:
                rejected_gain_fdr, p_values_gain_fdr, _, _ = multipletests(
                    p_values_gain_raw, alpha=0.05, method='fdr_bh', is_sorted=False
                )
                rejected_split_fdr, p_values_split_fdr, _, _ = multipletests(
                    p_values_split_raw, alpha=0.05, method='fdr_bh', is_sorted=False
                )
                rejected_cover_fdr, p_values_cover_fdr, _, _ = multipletests(
                    p_values_cover_raw, alpha=0.05, method='fdr_bh', is_sorted=False
                )
                p_values_gain = p_values_gain_fdr.astype(np.float32)
                p_values_split = p_values_split_fdr.astype(np.float32)
                p_values_cover = p_values_cover_fdr.astype(np.float32)
                logging.info(f'FDR Control (Benjamini-Hochberg): Applied - more powerful than Bonferroni')
            else:
                adjusted_gain = np.minimum(p_values_gain_raw * n_features, 1.0)
                adjusted_split = np.minimum(p_values_split_raw * n_features, 1.0)
                adjusted_cover = np.minimum(p_values_cover_raw * n_features, 1.0)
                p_values_gain = adjusted_gain.astype(np.float32)
                p_values_split = adjusted_split.astype(np.float32)
                p_values_cover = adjusted_cover.astype(np.float32)
                logging.warning('statsmodels not installed - using Bonferroni correction (install statsmodels for FDR)')
        except Exception as e:
            logging.warning(f'FDR correction failed: {e}, using raw p-values')
            pass
        significant_gain = p_values_gain < 0.05
        significant_split = p_values_split < 0.05
        significant_cover = p_values_cover < 0.05
        above_99_gain = actual_gain_mean > null_gain_99
        above_95_gain = actual_gain_mean > null_gain_95
        above_90_gain = actual_gain_mean > null_gain_90
        above_99_split = actual_split_mean > null_split_99
        above_95_split = actual_split_mean > null_split_95
        above_90_split = actual_split_mean > null_split_90
        above_99_cover = actual_cover_mean > null_cover_99
        above_90_cover = actual_cover_mean > null_cover_90
        logging.info(f'Gain - Significant: {np.sum(significant_gain)}, Above 99th: {np.sum(above_99_gain)}')
        logging.info(f'Split - Significant: {np.sum(significant_split)}, Above 99th: {np.sum(above_99_split)}')
        logging.info(f'Cover - Significant: {np.sum(significant_cover)}, Above 99th: {np.sum(above_99_cover)}')
        del actual_gain, null_gain, actual_split, null_split, actual_cover, null_cover
        gc.collect()
        return {
            'actual_gain_mean': actual_gain_mean,
            'actual_split_mean': actual_split_mean,
            'actual_cover_mean': actual_cover_mean,
            'null_gain_90': null_gain_90,
            'null_gain_95': null_gain_95,
            'null_gain_99': null_gain_99,
            'null_split_90': null_split_90,
            'null_split_95': null_split_95,
            'null_split_99': null_split_99,
            'null_cover_90': null_cover_90,
            'null_cover_99': null_cover_99,
            'gain_z_score': gain_z_score,
            'split_z_score': split_z_score,
            'cover_z_score': cover_z_score,
            'p_values_gain': p_values_gain,
            'p_values_split': p_values_split,
            'p_values_cover': p_values_cover,
            'significant_gain': significant_gain,
            'significant_split': significant_split,
            'significant_cover': significant_cover,
            'above_90_gain': above_90_gain,
            'above_95_gain': above_95_gain,
            'above_99_gain': above_99_gain,
            'above_90_split': above_90_split,
            'above_95_split': above_95_split,
            'above_99_split': above_99_split,
            'above_90_cover': above_90_cover,
            'above_99_cover': above_99_cover
        }
    def boosting_ensemble_complete(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_runs: int = 7
    ) -> Dict:
        logging.info('Boosting ensemble with optimized parameters')
        ensemble_results = defaultdict(list)
        params = self._get_adaptive_params(X, y)
        sample_weights = self.compute_sample_weights(y)
        booster_configs = self._get_optimized_boosting_configs(n_samples=len(X), n_features=len(X.columns))
        for run in range(n_runs):
            train_data = self._create_dataset(X, y, weight=sample_weights)
            for booster_name, booster_params in booster_configs:
                run_params = self.base_params.copy()
                run_params.update(booster_params)
                run_params['random_state'] = self.random_state + run
                run_params['seed'] = self.random_state + run
                if booster_name == 'rf':
                    num_rounds = 150
                elif booster_name == 'dart':
                    num_rounds = 250
                else:
                    num_rounds = 200
                model = self._train_with_fallback(
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
        n_runs: int = 7
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
            train_data = self._create_dataset(X, y, weight=sample_weights)
            for config_name, config_params in configs:
                run_params = params.copy()
                run_params.update(config_params)
                run_params['random_state'] = self.random_state + run
                run_params['seed'] = self.random_state + run
                model = self._train_with_fallback(
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
        X_test: pd.DataFrame = None
    ) -> Dict:
        logging.info('Adversarial validation for time series (no data leakage)')
        # DATA LEAKAGE PREVENTION: Use only training data for adversarial validation
        # Split training data into pseudo-train (early) and pseudo-test (late) periods
        # This checks for temporal drift without seeing actual future test data
        if X_test is None:
            # Proper implementation: split training data temporally
            split_point = int(0.7 * len(X_train))
            X_early = X_train.iloc[:split_point].copy()
            X_late = X_train.iloc[split_point:].copy()
            logging.info(f'Adversarial validation: early period ({len(X_early)} samples) vs late period ({len(X_late)} samples)')
        else:
            # Legacy mode: use provided test set (NOT RECOMMENDED - causes data leakage)
            logging.warning('Adversarial validation using actual test data - potential data leakage!')
            X_early = X_train.copy()
            X_late = X_test.copy()

        X_early['is_test'] = 0
        X_late['is_test'] = 1
        X_combined = pd.concat([X_early, X_late], axis=0, ignore_index=True)
        y_combined = X_combined['is_test'].values
        X_combined = X_combined.drop('is_test', axis=1)
        train_size = int(0.8 * len(X_combined))
        train_data = self._create_dataset(X_combined.iloc[:train_size], y_combined[:train_size])
        valid_data = self._create_dataset(X_combined.iloc[train_size:], y_combined[train_size:], reference=train_data)
        params = self.base_params.copy()
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['n_estimators'] = 200
        model = self._train_with_fallback(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[valid_data],
            callbacks=[
                self._adaptive_early_stopping(n_estimators=params.get('n_estimators', 200), context='cv'),
                lgb.log_evaluation(period=0)
            ]
        )
        adv_importance = model.feature_importance(importance_type='gain').astype(np.float32)
        adv_importance_normalized = adv_importance / (adv_importance.sum() + 1e-10)
        high_shift = adv_importance_normalized > (2.0 / len(X_combined.columns))
        logging.info(f'High shift features: {np.sum(high_shift)}')
        del X_early, X_late, X_combined
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
        n_total = len(X.columns)
        if n_features is None:
            n_features = max(int(n_total * 0.3), 20)
        est_params = self.base_params.copy()
        est_params.update({'n_estimators': 150, 'num_threads': max(1, self.num_threads // 2)})
        if self.classification:
            base_estimator = lgb.LGBMClassifier(**est_params)
            scoring = 'roc_auc'
        else:
            base_estimator = lgb.LGBMRegressor(**est_params)
            scoring = 'neg_root_mean_squared_error'
        cv_splitter = TimeSeriesSplit(n_splits=3, gap=50)
        if n_total > 1000:
            logging.info(f'RFE: 2-phase approach (Phase 1: quick elimination, Phase 2: fine-tuning)')
            step_phase1 = max(int(n_total * 0.05), 50)
            n_intermediate = max(int(n_total * 0.4), 500)
            try:
                rfecv_phase1 = RFECV(
                    estimator=base_estimator,
                    step=step_phase1,
                    cv=cv_splitter,
                    scoring=scoring,
                    min_features_to_select=n_intermediate,
                    n_jobs=max(1, self.num_threads // 2),
                    verbose=0
                )
                sample_weights_rfecv = self.compute_sample_weights(y)
                rfecv_phase1.fit(X, y, sample_weight=sample_weights_rfecv)
                logging.info(f'Phase 1: {n_total} → {np.sum(rfecv_phase1.support_)} features')
                X_phase2 = X.loc[:, rfecv_phase1.support_]
                step_phase2 = max(int(len(X_phase2.columns) * 0.02), 5)
                rfecv_phase2 = RFECV(
                    estimator=base_estimator,
                    step=step_phase2,
                    cv=cv_splitter,
                    scoring=scoring,
                    min_features_to_select=max(10, n_features // 2),
                    n_jobs=max(1, self.num_threads // 2),
                    verbose=0
                )
                rfecv_phase2.fit(X_phase2, y, sample_weight=sample_weights_rfecv)
                logging.info(f'Phase 2: {len(X_phase2.columns)} → {np.sum(rfecv_phase2.support_)} features')
                final_support = np.zeros(n_total, dtype=bool)
                phase1_indices = np.where(rfecv_phase1.support_)[0]
                phase2_selected_indices = phase1_indices[rfecv_phase2.support_]
                final_support[phase2_selected_indices] = True
                results = {
                    'rfe_support': final_support,
                    'rfe_ranking': np.ones(n_total),
                    'optimal_n_features': np.sum(final_support),
                    'phase1_features': np.sum(rfecv_phase1.support_),
                    'phase2_features': np.sum(rfecv_phase2.support_)
                }
                logging.info(f'RFE 2-phase complete: optimal features = {np.sum(final_support)}')
                return results
            except Exception as e:
                logging.warning(f'2-phase RFE failed ({e}), falling back to single-phase RFE')
        try:
            step = max(1, int(n_total * 0.05))
            rfecv = RFECV(
                estimator=base_estimator,
                step=step,
                cv=cv_splitter,
                scoring=scoring,
                min_features_to_select=max(10, n_features // 2),
                n_jobs=max(1, self.num_threads // 2),
                verbose=0
            )
            sample_weights_rfecv = self.compute_sample_weights(y)
            rfecv.fit(X, y, sample_weight=sample_weights_rfecv)
            results = {
                'rfe_support': rfecv.support_,
                'rfe_ranking': rfecv.ranking_,
                'optimal_n_features': rfecv.n_features_
            }
            logging.info(f'RFE: optimal features = {rfecv.n_features_}')
            return results
        except Exception as e:
            logging.warning(f'RFECV failed ({e}), falling back to basic RFE')
            est = base_estimator
            rfe = RFE(estimator=est, n_features_to_select=n_features, step=max(int(n_total * 0.05), 1))
            rfe.fit(X, y)
            return {'rfe_support': rfe.support_, 'rfe_ranking': rfe.ranking_}
    def cv_multi_metric(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        gap: int = 96
    ) -> Dict:
        logging.info('Cross-validation multi-metric (optimized: adaptive splits)')
        n = len(X)
        ts_params = self._calculate_optimal_ts_cv_params(n)
        n_splits = ts_params['n_splits']
        gap = ts_params['gap']
        test_size = ts_params['test_size']
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
        cv_n_jobs, model_n_threads = self._parallel_cv_strategy(n_splits=n_splits, n_threads_available=self.num_threads)
        n_features = len(X.columns)
        gain_importances = np.zeros((n_splits, n_features), dtype=np.float32, order='F')
        split_importances = np.zeros((n_splits, n_features), dtype=np.float32, order='F')
        perm_importances = np.zeros((n_splits, n_features), dtype=np.float32, order='F')
        cv_scores = np.zeros(n_splits, dtype=np.float32)
        params = self._get_adaptive_params(X, y)
        sample_weights = self.compute_sample_weights(y)
        if cv_n_jobs > 1:
            from joblib import Parallel, delayed
            def _run_fold(fold, train_idx, val_idx):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]
                weights_train = sample_weights[train_idx]
                train_data = self._create_dataset(X_train, y_train, weight=weights_train)
                val_data = self._create_dataset(X_val, y_val, reference=train_data)
                local_params = params.copy()
                local_params['n_estimators'] = 200
                local_params['num_threads'] = model_n_threads
                model = self._train_with_fallback(
                    local_params,
                    train_data,
                    num_boost_round=200,
                    valid_sets=[val_data],
                    callbacks=[
                        self._adaptive_early_stopping(n_estimators=local_params.get('n_estimators', 200), context='cv'),
                        lgb.log_evaluation(period=0)
                    ]
                )
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                if self.classification:
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    score = accuracy_score(y_val, y_pred_binary)
                else:
                    score = -mean_squared_error(y_val, y_pred, squared=False)
                gains = model.feature_importance(importance_type='gain')
                splits = model.feature_importance(importance_type='split')
                perm = np.zeros_like(gains)
                if fold < 3:
                    try:
                        wrapper = type('W', (), {})()
                        wrapper.booster = model
                        wrapper.is_clf = self.classification
                        def predict(X_):
                            return wrapper.booster.predict(X_, num_iteration=wrapper.booster.best_iteration)
                        wrapper.predict = predict
                        perm_result = permutation_importance(wrapper, X_val, y_val, n_repeats=5, random_state=self.random_state + fold, n_jobs=min(4, model_n_threads))
                        perm = perm_result.importances_mean
                    except Exception:
                        perm = np.zeros_like(gains)
                return {'fold': fold, 'score': score, 'gain': gains, 'split': splits, 'perm': perm}
            results_parallel = Parallel(n_jobs=cv_n_jobs, backend='threading', verbose=0)(
                delayed(_run_fold)(fold, train_idx, val_idx) for fold, (train_idx, val_idx) in enumerate(tscv.split(X))
            )
            for res in results_parallel:
                fold = res['fold']
                cv_scores[fold] = res['score']
                gain_importances[fold, :] = res['gain']
                split_importances[fold, :] = res['split']
                perm_importances[fold, :] = res['perm']
        else:
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]
                weights_train = sample_weights[train_idx]
                train_data = self._create_dataset(X_train, y_train, weight=weights_train)
                val_data = self._create_dataset(X_val, y_val, reference=train_data)
                run_params = params.copy()
                run_params['n_estimators'] = 200
                model = self._train_with_fallback(
                run_params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[
                    self._adaptive_early_stopping(n_estimators=run_params.get('n_estimators', 200), context='cv'),
                    lgb.log_evaluation(period=0)
                ]
            )
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            if self.classification:
                y_pred_binary = (y_pred > 0.5).astype(int)
                score = accuracy_score(y_val, y_pred_binary)
            else:
                score = -mean_squared_error(y_val, y_pred, squared=False)
            cv_scores[fold] = score
            gain_importances[fold, :] = model.feature_importance(importance_type='gain')
            split_importances[fold, :] = model.feature_importance(importance_type='split')
            if fold < 3:
                try:
                    class _BoosterWrapper:
                        def __init__(self, booster, is_clf):
                            self.booster = booster
                            self.is_clf = is_clf
                        def fit(self, X, y):
                            return self
                        def predict(self, X):
                            preds = self.booster.predict(X, num_iteration=self.booster.best_iteration)
                            if self.is_clf:
                                return (preds > 0.5).astype(int)
                            return preds
                        def predict_proba(self, X):
                            if not self.is_clf:
                                raise AttributeError('predict_proba only for classifiers')
                            preds = self.booster.predict(X, num_iteration=self.booster.best_iteration)
                            preds = np.clip(preds, 0.0, 1.0)
                            return np.vstack([1 - preds, preds]).T
                        def score(self, X, y):
                            try:
                                from sklearn.metrics import accuracy_score, mean_squared_error
                                preds = self.predict(X)
                                if self.is_clf:
                                    return float(accuracy_score(y, preds))
                                else:
                                    return -float(mean_squared_error(y, preds, squared=False))
                            except Exception:
                                return 0.0
                    wrapper = _BoosterWrapper(model, self.classification)
                    n_features = X_val.shape[1]
                    if n_features > 1000:
                        top_k = min(self.perm_top_k, n_features)
                        gain_imp = model.feature_importance(importance_type='gain')
                        top_idx = np.argsort(gain_imp)[-top_k:][::-1]
                        perm_imp_fold = np.zeros(n_features, dtype=np.float32)
                        rng = np.random.default_rng(self.random_state + fold)
                        try:
                            baseline_score = wrapper.score(X_val, y_val)
                        except Exception:
                            baseline_score = 0.0
                        n_repeats_local = 5
                        for col_idx in top_idx:
                            vals = []
                            for r in range(n_repeats_local):
                                X_perm = X_val.copy()
                                X_perm.iloc[:, col_idx] = rng.permutation(X_perm.iloc[:, col_idx].values)
                                try:
                                    score_perm = wrapper.score(X_perm, y_val)
                                except Exception:
                                    score_perm = 0.0
                                vals.append(baseline_score - score_perm)
                            perm_imp_fold[col_idx] = np.mean(vals)
                        perm_importances[fold, :] = perm_imp_fold
                    else:
                        perm_result = permutation_importance(
                            wrapper, X_val, y_val, n_repeats=5, random_state=self.random_state + fold, n_jobs=min(4, self.num_threads)
                        )
                        perm_importances[fold, :] = perm_result.importances_mean
                except Exception as e:
                    logging.warning(f'Permutation importance calculation failed: {e}')
                    perm_importances[fold, :] = 0
            else:
                if fold >= 3:
                    perm_importances[fold, :] = np.mean(perm_importances[max(0, fold-3):fold], axis=0)
        gain_importances_array = gain_importances
        split_importances_array = split_importances
        perm_importances_array = perm_importances
        mean_gain = np.mean(gain_importances_array, axis=0, dtype=np.float32)
        std_gain = np.std(gain_importances_array, axis=0, dtype=np.float32)
        mean_split = np.mean(split_importances_array, axis=0, dtype=np.float32)
        std_split = np.std(split_importances_array, axis=0, dtype=np.float32)
        mean_perm = np.mean(perm_importances_array, axis=0, dtype=np.float32)
        std_perm = np.std(perm_importances_array, axis=0, dtype=np.float32)
        cv_gain = std_gain / (mean_gain + 1e-10)
        cv_split = std_split / (mean_split + 1e-10)
        cv_perm = std_perm / (mean_perm + 1e-10)
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
    def cv_repeated(self, X: pd.DataFrame, y: pd.Series, seeds: List[int] = [42, 43, 44], n_splits: int = 10, gap: int = 96) -> Dict:
        results = {'seeds': [], 'mean_scores': [], 'std_scores': []}
        for seed in seeds:
            self.random_state = seed
            params = self._get_adaptive_params(X, y)
            cv_metrics = self.cv_multi_metric(X, y, n_splits=n_splits, gap=gap)
            results['seeds'].append(seed)
            results['mean_scores'].append(float(cv_metrics['mean_cv_score']))
            results['std_scores'].append(float(cv_metrics['std_cv_score']))
        results['overall_mean'] = float(np.mean(results['mean_scores']))
        results['overall_std'] = float(np.std(results['mean_scores']))
        return results
    def _block_bootstrap_indices(self, n_samples: int, sample_size: int, block_size: int = None, rng=None) -> np.ndarray:
        if rng is None:
            rng = self.rng
        if block_size is None:
            block_size = max(1, int(np.sqrt(n_samples)))
        indices = []
        p = 1.0 / block_size
        while len(indices) < sample_size:
            block_len = rng.geometric(p)
            start_idx = rng.integers(0, n_samples)
            block_indices = [(start_idx + j) % n_samples for j in range(block_len)]
            indices.extend(block_indices)
        return np.array(indices[:sample_size])
    def stability_bootstrap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bootstrap: int = 30,
        threshold: float = 0.70,
        use_block_bootstrap: bool = True
    ) -> Dict:
        logging.info('Stability bootstrap analysis (optimized with reduced iterations and parallel processing)')
        if use_block_bootstrap:
            logging.info('Using Block Bootstrap (preserves temporal dependencies for time series)')
        n_features = len(X.columns)
        feature_counts_gain = np.zeros(n_features, dtype=np.uint16)
        feature_counts_split = np.zeros(n_features, dtype=np.uint16)
        params = self._get_adaptive_params(X, y)
        params['n_estimators'] = 150
        sample_weights = self.compute_sample_weights(y)
        n_bootstrap_optimized = min(max(1, n_bootstrap), 50)
        block_size = max(1, int(np.sqrt(len(X))))
        try:
            from joblib import Parallel, delayed
            use_parallel = True
            n_jobs = min(4, max(1, self.num_threads // 2))
        except ImportError:
            use_parallel = False
            logging.warning('joblib not available, using sequential bootstrap')
            n_jobs = 1
        if use_parallel:
            def _single_bootstrap(i, n_features, X, y, sample_weights, params, rng_seed, use_block, block_sz):
                rng = np.random.default_rng(rng_seed + i)
                sample_size = int(0.7 * len(X))
                if use_block:
                    indices = []
                    p = 1.0 / block_sz
                    while len(indices) < sample_size:
                        block_len = rng.geometric(p)
                        start_idx = rng.integers(0, len(X))
                        block_indices = [(start_idx + j) % len(X) for j in range(block_len)]
                        indices.extend(block_indices)
                    indices = np.array(indices[:sample_size])
                else:
                    indices = rng.choice(len(X), size=sample_size, replace=True)
                X_sample = X.iloc[indices]
                y_sample = y.iloc[indices]
                weights_sample = sample_weights[indices]
                feature_indices = rng.choice(n_features, size=int(0.7 * n_features), replace=False)
                X_sample_feat = X_sample.iloc[:, feature_indices]
                train_data = self._create_dataset(X_sample_feat, y_sample, weight=weights_sample)
                run_params = params.copy()
                run_params['random_state'] = self.random_state + i
                run_params['seed'] = self.random_state + i
                model = self._train_with_fallback(
                    run_params,
                    train_data,
                    num_boost_round=150,
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                gain_importance = model.feature_importance(importance_type='gain')
                split_importance = model.feature_importance(importance_type='split')
                selected_gain = gain_importance > 0
                selected_split = split_importance > 0
                feature_counts_gain_local = np.zeros(n_features, dtype=np.uint16)
                feature_counts_split_local = np.zeros(n_features, dtype=np.uint16)
                for j, feat_idx in enumerate(feature_indices):
                    if selected_gain[j]:
                        feature_counts_gain_local[feat_idx] += 1
                    if selected_split[j]:
                        feature_counts_split_local[feat_idx] += 1
                return feature_counts_gain_local, feature_counts_split_local
            try:
                results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
                    delayed(_single_bootstrap)(i, n_features, X, y, sample_weights, params, self.random_state, use_block_bootstrap, block_size)
                    for i in range(n_bootstrap_optimized)
                )
                for gain_counts, split_counts in results:
                    feature_counts_gain += gain_counts
                    feature_counts_split += split_counts
                logging.info(f'Bootstrap completed with {n_jobs} parallel jobs')
            except Exception as e:
                logging.warning(f'Parallel bootstrap failed: {e}. Falling back to sequential processing')
                use_parallel = False
        else:
            for i in range(n_bootstrap_optimized):
                sample_size = int(0.7 * len(X))
                if use_block_bootstrap:
                    indices = self._block_bootstrap_indices(len(X), sample_size, block_size, self.rng)
                else:
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
                train_data = self._create_dataset(X_sample, y_sample, weight=weights_sample)
                run_params = params.copy()
                run_params['random_state'] = self.random_state + i
                run_params['seed'] = self.random_state + i
                model = self._train_with_fallback(
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
        stability_scores_gain = (feature_counts_gain / n_bootstrap_optimized).astype(np.float32)
        stability_scores_split = (feature_counts_split / n_bootstrap_optimized).astype(np.float32)
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
        if isinstance(scores, np.ndarray):
            if scores.dtype != np.float32:
                scores_array = scores.astype(np.float32)
            else:
                scores_array = scores.copy()
        else:
            scores_array = np.asarray(scores, dtype=np.float32)
        min_val = scores_array.min()
        max_val = scores_array.max()
        range_val = max_val - min_val
        if range_val > epsilon:
            np.subtract(scores_array, min_val, out=scores_array)
            np.divide(scores_array, range_val, out=scores_array)
        else:
            scores_array.fill(0)
        return scores_array
    def compute_confidence_intervals(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        importance_type: str = 'gain'
    ) -> Dict:
        logging.info(f'Computing {confidence_level*100:.0f}% confidence intervals for importance scores (n_bootstrap={n_bootstrap})')
        n_features = len(X.columns)
        n_samples = len(X)
        bootstrap_importances = np.zeros((n_bootstrap, n_features), dtype=np.float32)
        params = self._get_adaptive_params(X, y)
        params['n_estimators'] = 200
        sample_weights = self.compute_sample_weights(y)
        for b in range(n_bootstrap):
            rng_b = np.random.default_rng(self.random_state + b * 1000)
            boot_idx = rng_b.choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[boot_idx].reset_index(drop=True)
            y_boot = y.iloc[boot_idx].reset_index(drop=True)
            w_boot = sample_weights[boot_idx] if sample_weights is not None else None
            val_size = int(0.2 * n_samples)
            X_train_b, X_val_b = X_boot.iloc[val_size:], X_boot.iloc[:val_size]
            y_train_b, y_val_b = y_boot.iloc[val_size:], y_boot.iloc[:val_size]
            w_train_b = w_boot[val_size:] if w_boot is not None else None
            w_val_b = w_boot[:val_size] if w_boot is not None else None
            train_data = self._create_dataset(X_train_b, y_train_b, weight=w_train_b)
            val_data = self._create_dataset(X_val_b, y_val_b, weight=w_val_b, reference=train_data)
            run_params = params.copy()
            run_params['random_state'] = self.random_state + b
            run_params['seed'] = self.random_state + b
            model = self._train_with_fallback(
                run_params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[
                    self._adaptive_early_stopping(n_estimators=200, context='confidence_intervals'),
                    lgb.log_evaluation(period=0)
                ]
            )
            bootstrap_importances[b, :] = model.feature_importance(importance_type=importance_type)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_importances, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_importances, upper_percentile, axis=0)
        ci_mean = np.mean(bootstrap_importances, axis=0)
        ci_std = np.std(bootstrap_importances, axis=0)
        ci_median = np.median(bootstrap_importances, axis=0)
        ci_width = ci_upper - ci_lower
        threshold = np.percentile(ci_mean, 10)
        significant_features = ci_lower > threshold
        logging.info(f'CI computed: Mean width={np.mean(ci_width):.4f}, Significant features: {np.sum(significant_features)}/{n_features}')
        return {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_mean': ci_mean,
            'ci_std': ci_std,
            'ci_median': ci_median,
            'ci_width': ci_width,
            'significant_features': significant_features,
            'n_significant': int(np.sum(significant_features)),
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap
        }
    def mutual_information_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        logging.info('Computing Mutual Information scores for non-linear relationships')
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        n_features = len(X.columns)
        if self.classification:
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression
        mi_scores = mi_func(
            X, y,
            discrete_features=False,
            n_neighbors=5,
            random_state=self.random_state
        )
        mi_max = np.max(mi_scores)
        if mi_max > 0:
            mi_normalized = mi_scores / mi_max
        else:
            mi_normalized = mi_scores
        mi_rank = np.argsort(-mi_scores)
        threshold_75 = np.percentile(mi_scores, 75)
        high_mi_features = mi_scores >= threshold_75
        logging.info(f'MI scores: Max={np.max(mi_scores):.4f}, Mean={np.mean(mi_scores):.4f}, High MI features: {np.sum(high_mi_features)}/{n_features}')
        return {
            'mi_scores': mi_scores,
            'mi_normalized': mi_normalized,
            'mi_rank': mi_rank,
            'high_mi_features': high_mi_features,
            'n_high_mi': int(np.sum(high_mi_features))
        }
    def stability_selection_framework(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iterations: int = 50,
        sample_fraction: float = 0.5,
        threshold: float = 0.6
    ) -> Dict:
        logging.info(f'Stability Selection Framework: {n_iterations} iterations, sample_fraction={sample_fraction}, threshold={threshold}')
        n_features = len(X.columns)
        n_samples = len(X)
        subsample_size = int(n_samples * sample_fraction)
        selection_frequency = np.zeros(n_features, dtype=np.float32)
        params = self._get_adaptive_params(X, y)
        params['n_estimators'] = 150
        sample_weights = self.compute_sample_weights(y)
        top_k = min(max(int(np.sqrt(n_features)), 10), n_features // 2)
        for iteration in range(n_iterations):
            rng_iter = np.random.default_rng(self.random_state + iteration * 100)
            sub_idx = rng_iter.choice(n_samples, size=subsample_size, replace=False)
            X_sub = X.iloc[sub_idx].reset_index(drop=True)
            y_sub = y.iloc[sub_idx].reset_index(drop=True)
            w_sub = sample_weights[sub_idx] if sample_weights is not None else None
            val_size = int(0.2 * len(X_sub))
            X_train_sub = X_sub.iloc[val_size:]
            y_train_sub = y_sub.iloc[val_size:]
            X_val_sub = X_sub.iloc[:val_size]
            y_val_sub = y_sub.iloc[:val_size]
            w_train_sub = w_sub[val_size:] if w_sub is not None else None
            w_val_sub = w_sub[:val_size] if w_sub is not None else None
            train_data = self._create_dataset(X_train_sub, y_train_sub, weight=w_train_sub)
            val_data = self._create_dataset(X_val_sub, y_val_sub, weight=w_val_sub, reference=train_data)
            run_params = params.copy()
            run_params['random_state'] = self.random_state + iteration
            run_params['seed'] = self.random_state + iteration
            model = self._train_with_fallback(
                run_params,
                train_data,
                num_boost_round=150,
                valid_sets=[val_data],
                callbacks=[
                    self._adaptive_early_stopping(n_estimators=150, context='stability_selection'),
                    lgb.log_evaluation(period=0)
                ]
            )
            importance = model.feature_importance(importance_type='gain')
            top_k_indices = np.argsort(-importance)[:top_k]
            selection_frequency[top_k_indices] += 1
        selection_probability = selection_frequency / n_iterations
        stable_features_mask = selection_probability >= threshold
        n_stable = np.sum(stable_features_mask)
        expected_false_discoveries = ((1 / threshold) ** 2) * (top_k ** 2) / n_features
        logging.info(f'Stability Selection: Stable features={n_stable}/{n_features} (threshold>={threshold}), Expected FD<={expected_false_discoveries:.2f}')
        return {
            'selection_probability': selection_probability,
            'stable_features': stable_features_mask,
            'n_stable': int(n_stable),
            'threshold': threshold,
            'n_iterations': n_iterations,
            'sample_fraction': sample_fraction,
            'expected_false_discoveries': float(expected_false_discoveries)
        }
    def nested_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_outer_splits: int = 5,
        n_inner_splits: int = 3
    ) -> Dict:
        logging.info(f'Nested Cross-Validation: outer_splits={n_outer_splits}, inner_splits={n_inner_splits}')
        n_samples = len(X)
        outer_params = self._calculate_optimal_ts_cv_params(n_samples)
        outer_n_splits = min(n_outer_splits, outer_params['n_splits'])
        outer_gap = outer_params['gap']
        outer_test_size = outer_params['test_size']
        outer_cv = TimeSeriesSplit(n_splits=outer_n_splits, gap=outer_gap, test_size=outer_test_size)
        outer_scores = []
        feature_importances_outer = []
        models_per_fold = []
        sample_weights = self.compute_sample_weights(y)
        for fold_idx, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X)):
            X_train_outer = X.iloc[train_outer_idx].reset_index(drop=True)
            y_train_outer = y.iloc[train_outer_idx].reset_index(drop=True)
            X_test_outer = X.iloc[test_outer_idx].reset_index(drop=True)
            y_test_outer = y.iloc[test_outer_idx].reset_index(drop=True)
            w_train_outer = sample_weights[train_outer_idx] if sample_weights is not None else None
            inner_params = self._calculate_optimal_ts_cv_params(len(X_train_outer))
            inner_n_splits = min(n_inner_splits, inner_params['n_splits'])
            inner_gap = inner_params['gap']
            inner_test_size = inner_params['test_size']
            inner_cv = TimeSeriesSplit(n_splits=inner_n_splits, gap=inner_gap, test_size=inner_test_size)
            best_inner_score = -np.inf
            best_n_estimators = 300
            for n_est_candidate in [200, 300, 500]:
                inner_scores = []
                for train_inner_idx, val_inner_idx in inner_cv.split(X_train_outer):
                    X_train_inner = X_train_outer.iloc[train_inner_idx]
                    y_train_inner = y_train_outer.iloc[train_inner_idx]
                    X_val_inner = X_train_outer.iloc[val_inner_idx]
                    y_val_inner = y_train_outer.iloc[val_inner_idx]
                    w_train_inner = w_train_outer[train_inner_idx] if w_train_outer is not None else None
                    w_val_inner = w_train_outer[val_inner_idx] if w_train_outer is not None else None
                    params = self._get_adaptive_params(X_train_inner, y_train_inner)
                    params['n_estimators'] = n_est_candidate
                    train_data = self._create_dataset(X_train_inner, y_train_inner, weight=w_train_inner)
                    val_data = self._create_dataset(X_val_inner, y_val_inner, weight=w_val_inner, reference=train_data)
                    model = self._train_with_fallback(
                        params,
                        train_data,
                        num_boost_round=n_est_candidate,
                        valid_sets=[val_data],
                        callbacks=[
                            self._adaptive_early_stopping(n_estimators=n_est_candidate, context='nested_cv_inner'),
                            lgb.log_evaluation(period=0)
                        ]
                    )
                    if self.classification:
                        y_pred_inner = model.predict(X_val_inner)
                        y_pred_binary = (y_pred_inner > 0.5).astype(int)
                        score = accuracy_score(y_val_inner, y_pred_binary)
                    else:
                        y_pred_inner = model.predict(X_val_inner)
                        score = -mean_squared_error(y_val_inner, y_pred_inner)
                    inner_scores.append(score)
                mean_inner_score = np.mean(inner_scores)
                if mean_inner_score > best_inner_score:
                    best_inner_score = mean_inner_score
                    best_n_estimators = n_est_candidate
            params = self._get_adaptive_params(X_train_outer, y_train_outer)
            params['n_estimators'] = best_n_estimators
            val_size_outer = int(0.15 * len(X_train_outer))
            X_train_final = X_train_outer.iloc[:-val_size_outer]
            y_train_final = y_train_outer.iloc[:-val_size_outer]
            X_val_final = X_train_outer.iloc[-val_size_outer:]
            y_val_final = y_train_outer.iloc[-val_size_outer:]
            w_train_final = w_train_outer[:-val_size_outer] if w_train_outer is not None else None
            w_val_final = w_train_outer[-val_size_outer:] if w_train_outer is not None else None
            train_data = self._create_dataset(X_train_final, y_train_final, weight=w_train_final)
            val_data = self._create_dataset(X_val_final, y_val_final, weight=w_val_final, reference=train_data)
            final_model = self._train_with_fallback(
                params,
                train_data,
                num_boost_round=best_n_estimators,
                valid_sets=[val_data],
                callbacks=[
                    self._adaptive_early_stopping(n_estimators=best_n_estimators, context='nested_cv_outer'),
                    lgb.log_evaluation(period=0)
                ]
            )
            if self.classification:
                y_pred_outer = final_model.predict(X_test_outer)
                y_pred_binary = (y_pred_outer > 0.5).astype(int)
                outer_score = accuracy_score(y_test_outer, y_pred_binary)
            else:
                y_pred_outer = final_model.predict(X_test_outer)
                outer_score = -mean_squared_error(y_test_outer, y_pred_outer)
            outer_scores.append(outer_score)
            feature_importances_outer.append(final_model.feature_importance(importance_type='gain'))
            models_per_fold.append({
                'best_n_estimators': best_n_estimators,
                'inner_score': best_inner_score,
                'outer_score': outer_score
            })
        feature_importances_mean = np.mean(feature_importances_outer, axis=0)
        feature_importances_std = np.std(feature_importances_outer, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            feature_cv = np.where(feature_importances_mean > 0,
                                  feature_importances_std / feature_importances_mean,
                                  0)
        mean_outer_score = np.mean(outer_scores)
        std_outer_score = np.std(outer_scores)
        logging.info(f'Nested CV: Unbiased score={mean_outer_score:.4f} +/- {std_outer_score:.4f}')
        logging.info(f'Nested CV: Feature importance CV mean={np.mean(feature_cv):.4f}')
        return {
            'outer_scores': outer_scores,
            'mean_score': float(mean_outer_score),
            'std_score': float(std_outer_score),
            'feature_importances_mean': feature_importances_mean,
            'feature_importances_std': feature_importances_std,
            'feature_cv': feature_cv,
            'models_per_fold': models_per_fold,
            'n_outer_splits': outer_n_splits,
            'n_inner_splits': inner_n_splits
        }
    def conditional_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model=None,
        correlation_threshold: float = 0.7,
        n_repeats: int = 10
    ) -> Dict:
        logging.info(f'Conditional Permutation Importance (correlation_threshold={correlation_threshold}, n_repeats={n_repeats})')
        n_features = len(X.columns)
        corr_matrix = X.corr().abs()
        feature_groups = []
        used_features = set()
        for feat_idx, feat in enumerate(X.columns):
            if feat in used_features:
                continue
            group = [feat]
            used_features.add(feat)
            for other_feat in X.columns:
                if other_feat not in used_features:
                    if corr_matrix.loc[feat, other_feat] > correlation_threshold:
                        group.append(other_feat)
                        used_features.add(other_feat)
            feature_groups.append(group)
        logging.info(f'Identified {len(feature_groups)} feature groups (avg size: {n_features/len(feature_groups):.1f})')
        if model is None:
            params = self._get_adaptive_params(X, y)
            sample_weights = self.compute_sample_weights(y)
            train_data = self._create_dataset(X, y, weight=sample_weights)
            model = self._train_with_fallback(
                params,
                train_data,
                num_boost_round=200,
                callbacks=[lgb.log_evaluation(period=0)]
            )
        if self.classification:
            y_pred = model.predict(X)
            baseline_score = accuracy_score(y, (y_pred > 0.5).astype(int))
        else:
            y_pred = model.predict(X)
            baseline_score = -mean_squared_error(y, y_pred)
        rng = np.random.default_rng(self.random_state)
        group_importances = []
        for group_idx, group in enumerate(feature_groups):
            group_imp = []
            for repeat in range(n_repeats):
                X_perm = X.copy()
                perm_indices = rng.permutation(len(X))
                for col in group:
                    X_perm[col] = X_perm[col].iloc[perm_indices].values
                if self.classification:
                    y_pred_perm = model.predict(X_perm)
                    perm_score = accuracy_score(y, (y_pred_perm > 0.5).astype(int))
                else:
                    y_pred_perm = model.predict(X_perm)
                    perm_score = -mean_squared_error(y, y_pred_perm)
                importance = baseline_score - perm_score
                group_imp.append(importance)
            group_importances.append({
                'group': group,
                'group_size': len(group),
                'mean_importance': float(np.mean(group_imp)),
                'std_importance': float(np.std(group_imp))
            })
        group_importances.sort(key=lambda x: x['mean_importance'], reverse=True)
        feature_importances = np.zeros(n_features)
        for group_info in group_importances:
            group = group_info['group']
            imp_per_feature = group_info['mean_importance'] / len(group)
            for feat in group:
                feat_idx = X.columns.get_loc(feat)
                feature_importances[feat_idx] = imp_per_feature
        logging.info(f'Conditional PI: Top group importance={group_importances[0]["mean_importance"]:.6f}')
        return {
            'feature_importances': feature_importances,
            'group_importances': group_importances,
            'n_groups': len(feature_groups),
            'baseline_score': baseline_score,
            'correlation_threshold': correlation_threshold
        }
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
        multicollinearity: Optional[Dict] = None,
        mutual_info: Optional[Dict] = None,
        stability_selection: Optional[Dict] = None,
        nested_cv: Optional[Dict] = None,
        confidence_intervals: Optional[Dict] = None
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
            feat_index = {name: idx for idx, name in enumerate(feature_names)}
            correlations = multicollinearity.get('high_corr_values', {})
            for feat, corr_val in correlations.items():
                if feat in feat_index:
                    idx = feat_index[feat]
                    penalty = np.clip(1.0 - (corr_val - 0.85) * 3.0, 0.7, 1.0)
                    high_corr_penalty[idx] = penalty
        score_data['mult_corr_pen'] = high_corr_penalty
        if shap_importance is not None and 'shap_mean' in shap_importance:
            score_data['shap'] = self.normalize_with_stability(shap_importance['shap_mean'])
            score_data['shap_int'] = self.normalize_with_stability(shap_importance['shap_interaction_mean'])
        if mutual_info is not None and 'mi_normalized' in mutual_info:
            score_data['mi_score'] = mutual_info['mi_normalized'].astype(np.float32)
            score_data['mi_high'] = mutual_info['high_mi_features'].astype(np.int8)
        if stability_selection is not None and 'selection_probability' in stability_selection:
            score_data['stab_sel_prob'] = stability_selection['selection_probability'].astype(np.float32)
            score_data['stab_sel_stable'] = stability_selection['stable_features'].astype(np.int8)
        if nested_cv is not None and 'feature_importances_mean' in nested_cv:
            score_data['nested_cv_imp'] = self.normalize_with_stability(nested_cv['feature_importances_mean'])
            score_data['nested_cv_stab'] = 1.0 - self.normalize_with_stability(nested_cv['feature_cv'])
        if confidence_intervals is not None and 'ci_mean' in confidence_intervals:
            score_data['ci_mean'] = self.normalize_with_stability(confidence_intervals['ci_mean'])
            score_data['ci_certainty'] = 1.0 - self.normalize_with_stability(confidence_intervals['ci_width'])
            score_data['ci_significant'] = confidence_intervals['significant_features'].astype(np.int8)
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
            'null_z': 0.08, 'null_z_split': 0.03, 'null_sig': 0.02, 'null_sig_split': 0.01,
            'above_99': 0.03, 'above_95': 0.02, 'goss': 0.03, 'goss_s': 0.01, 'dart': 0.03,
            'dart_s': 0.01, 'extra': 0.03, 'extra_s': 0.01, 'bynode_f': 0.02, 'bynode_fs': 0.01,
            'bytree_f': 0.02, 'bytree_fs': 0.01, 'combined_f': 0.02, 'combined_fs': 0.01,
            'adv_shift': 0.02, 'no_shift': 0.01, 'rfe_sel': 0.03, 'rfe_rank': 0.02,
            'cv_g': 0.06, 'cv_s': 0.03, 'cv_p': 0.06, 'cv_stab_g': 0.02, 'cv_stab_s': 0.01,
            'stab_g': 0.03, 'stab_s': 0.02, 'is_stab_g': 0.02, 'is_stab_s': 0.01, 'shap': 0.08,
            'shap_int': 0.03, 'mult_corr_pen': 0.02,
            'mi_score': 0.04, 'mi_high': 0.02,
            'stab_sel_prob': 0.05, 'stab_sel_stable': 0.03,
            'nested_cv_imp': 0.06, 'nested_cv_stab': 0.03,
            'ci_mean': 0.04, 'ci_certainty': 0.03, 'ci_significant': 0.02
        }
        score_cols = [col for col in df.columns if col != 'feature']
        score_matrix = df[score_cols].values.astype(np.float32)
        weight_vector = np.array([weights.get(col, 0) for col in score_cols], dtype=np.float32)
        final_score = np.einsum('ij,j->i', score_matrix, weight_vector, optimize=True)
        df['final_score'] = final_score
        sorted_indices = np.argsort(-final_score)
        df = df.iloc[sorted_indices].reset_index(drop=True)
        return df[['feature', 'final_score']]
    def categorize(
        self,
        df: pd.DataFrame,
        strong_pct: float = 0.15,
        weak_pct: float = 0.60
    ) -> Dict:
        features_array = df['feature'].values
        n = len(df)
        n_strong = max(int(n * strong_pct), 10)
        n_weak_start = int(n * weak_pct)
        strong = features_array[:n_strong]
        medium = features_array[n_strong:n_weak_start]
        weak = features_array[n_weak_start:]
        logging.info(f'Strong: {len(strong)}, Medium: {len(medium)}, Weak: {len(weak)}')
        return {
            'strong': strong.tolist(),
            'medium': medium.tolist(),
            'weak': weak.tolist()
        }
    def quick_prefilter(self, X: pd.DataFrame, y: pd.Series, n_estimators: int = 100) -> tuple:
        dropped_features = []
        for col in X.columns:
            if X[col].var() < 1e-8:
                dropped_features.append(col)
                logging.info(f'Dropped constant/quasi-constant: {col}')
        missing_ratio = X.isnull().sum() / len(X)
        for col in X.columns:
            if col not in dropped_features and missing_ratio[col] > 0.9:
                dropped_features.append(col)
                logging.info(f'Dropped high-missing: {col} ({missing_ratio[col]:.1%})')
        X_temp = X.drop(columns=dropped_features, errors='ignore')
        if X_temp.shape[1] > 1000:
            logging.info(f'Quick pre-filtering {X_temp.shape[1]} features via single-model importance')
            try:
                if y.dtype in ['int64', 'int32', 'int8', 'int16']:
                    model = lgb.LGBMClassifier(
                        n_estimators=n_estimators,
                        num_leaves=31,
                        max_depth=6,
                        random_state=self.random_state,
                        num_threads=1,
                        verbose=-1
                    )
                else:
                    model = lgb.LGBMRegressor(
                        n_estimators=n_estimators,
                        num_leaves=31,
                        max_depth=6,
                        random_state=self.random_state,
                        num_threads=1,
                        verbose=-1
                    )
                model.fit(X_temp, y)
                importances = model.feature_importances_
                n_keep = max(int(X_temp.shape[1] * 0.5), 100)
                threshold = np.partition(importances, len(importances) - n_keep - 1)[len(importances) - n_keep - 1]
                weak_cols = X_temp.columns[importances < threshold].tolist()
                dropped_features.extend(weak_cols)
                logging.info(f'Dropped {len(weak_cols)} weak features by model importance (threshold={threshold:.4f})')
            except Exception as e:
                logging.warning(f'Quick pre-filter model failed: {e}, skipping importance filtering')
        X_filtered = X.drop(columns=dropped_features, errors='ignore')
        logging.info(f'Quick pre-filter: {X.shape[1]} -> {X_filtered.shape[1]} features, dropped {len(dropped_features)}')
        return X_filtered, dropped_features
    def process_batch(
        self,
        features_df: pd.DataFrame,
        batch_id: int,
        output_dir: str = 'feature_selection_results'
        , save_final_model: bool = False
    ):
        try:
            import psutil
        except Exception:
            psutil = None
        import time
        start_time = time.time()
        if psutil is not None:
            process = psutil.Process()
            memory_start = process.memory_info().rss / 1024**2
        else:
            logging.warning('psutil not installed—memory stats not available')
            process = None
            memory_start = 0.0
        logging.info(f'Batch {batch_id} starting - Memory: {memory_start:.2f} MB')
        if self.ensure_reproducible:
            try:
                is_repro = self.smoke_test_reproducibility(n_samples=min(100, len(features_df)), n_features=min(10, len(features_df.columns)-1))
                if not is_repro:
                    logging.warning('Reproducibility smoke test FAILED - results may vary between runs')
                else:
                    logging.info('Reproducibility smoke test PASSED')
            except Exception as e:
                logging.warning(f'Reproducibility smoke test error: {e}')
        X = features_df.drop(columns=[self.target_column])
        y = features_df[self.target_column]
        logging.info('Performing temporal split BEFORE preprocessing (data leakage prevention)')
        X_train_raw, X_test_raw, y_train, y_test = self.temporal_split(X, y)
        X_train_filtered, dropped_features = self.quick_prefilter(X_train_raw, y_train)
        kept_features = X_train_filtered.columns.tolist()
        X_test_filtered = X_test_raw[kept_features]
        logging.info(f'Applied train-set filtering decisions to test set (no leakage)')
        try:
            optimal_threads = self._optimize_num_threads_for_dataset(len(X_train_filtered), len(X_train_filtered.columns))
            self.num_threads = optimal_threads
            self.base_params['num_threads'] = optimal_threads
            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
        except Exception:
            pass
        try:
            mem_opts = self._optimize_memory_usage(len(X_train_filtered), len(X_train_filtered.columns))
            if mem_opts:
                self.base_params.update(mem_opts)
                logging.info(f"Applied memory optimizations: {mem_opts}")
        except Exception:
            pass
        X_train, y_train = self.preprocess_features(X_train_filtered, y_train)
        X_test = X_test_filtered.copy()
        X_test.columns = X_train.columns
        logging.info(f'Features: {X_train.shape[1]}, Train Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}')
        logging.info(f'DATA LEAKAGE PREVENTION: Preprocessing based on training data only')
        multicollinearity = None
        original_shap_perturbation = self.shap_feature_perturbation
        if self.should_detect_multicollinearity:
            multicollinearity = self.detect_multicollinearity(X_train)
            if multicollinearity is not None and self.use_shap:
                n_high_corr = len(multicollinearity.get('high_corr_features', []))
                if n_high_corr > 10 and self.shap_feature_perturbation == 'tree_path_dependent':
                    # AUTO-SWITCH to interventional for high correlation (prevents SHAP bias)
                    logging.warning(
                        f'HIGH MULTICOLLINEARITY DETECTED: {n_high_corr} highly correlated feature pairs. '
                        f'AUTO-SWITCHING SHAP from tree_path_dependent to interventional for unbiased results. '
                        f'Reference: FastPD Study (2024) - OpenReview'
                    )
                    self.shap_feature_perturbation = 'interventional'
        null_importance = self.null_importance_ultimate(X_train, y_train)
        boosting_ensemble = self.boosting_ensemble_complete(X_train, y_train)
        feature_fraction = self.feature_fraction_analysis(X_train, y_train)
        adversarial = self.adversarial_validation(X_train)  # No X_test to prevent data leakage
        rfe = self.rfe_selection(X_train, y_train)
        cv_metrics = self.cv_multi_metric(X_train, y_train)
        stability = self.stability_bootstrap(X_train, y_train)
        shap_importance = None
        if self.use_shap:
            shap_importance = self.shap_importance_analysis_round2(
                X_train, y_train,
                n_runs=5,
                use_sample_weights=True,
                cache_explainer=True
            )
            # Restore original SHAP perturbation setting after computation
            self.shap_feature_perturbation = original_shap_perturbation
        mutual_info = None
        if self.use_mutual_information:
            try:
                mutual_info = self.mutual_information_scores(X_train, y_train)
            except Exception as e:
                logging.warning(f'Mutual Information calculation failed: {e}')
        stability_selection = None
        if self.use_stability_selection:
            try:
                stability_selection = self.stability_selection_framework(
                    X_train, y_train,
                    n_iterations=self.stability_selection_iterations,
                    sample_fraction=0.5,
                    threshold=0.6
                )
            except Exception as e:
                logging.warning(f'Stability Selection failed: {e}')
        nested_cv_results = None
        if self.use_nested_cv:
            try:
                nested_cv_results = self.nested_cross_validation(
                    X_train, y_train,
                    n_outer_splits=3,
                    n_inner_splits=2
                )
            except Exception as e:
                logging.warning(f'Nested CV failed: {e}')
        confidence_intervals = None
        if self.use_confidence_intervals:
            try:
                confidence_intervals = self.compute_confidence_intervals(
                    X_train, y_train,
                    n_bootstrap=self.n_bootstrap_ci,
                    confidence_level=0.95,
                    importance_type='gain'
                )
            except Exception as e:
                logging.warning(f'Confidence Intervals calculation failed: {e}')
        df_ranking = self.ensemble_ranking(
            feature_names=X_train.columns.tolist(),
            null_importance=null_importance,
            boosting_ensemble=boosting_ensemble,
            feature_fraction=feature_fraction,
            adversarial=adversarial,
            rfe=rfe,
            cv_metrics=cv_metrics,
            stability=stability,
            shap_importance=shap_importance,
            multicollinearity=multicollinearity,
            mutual_info=mutual_info,
            stability_selection=stability_selection,
            nested_cv=nested_cv_results,
            confidence_intervals=confidence_intervals
        )
        if dropped_features:
            df_dropped = pd.DataFrame({
                'feature': dropped_features,
                'score': 0.0,
                'category': 'dropped_prefilter',
                'category_rank': -1
            })
            df_ranking = pd.concat([df_ranking, df_dropped], ignore_index=True)
            logging.info(f'Added {len(dropped_features)} dropped features to ranking with score=0')
        categorized = self.categorize(df_ranking)
        end_time = time.time()
        if process is not None:
            memory_end = process.memory_info().rss / 1024**2
        else:
            memory_end = 0.0
        execution_time = end_time - start_time
        memory_delta = memory_end - memory_start
        self._save(
            df_ranking=df_ranking,
            categorized=categorized,
            cv_metrics=cv_metrics,
            batch_id=batch_id,
            output_dir=output_dir,
            execution_time=execution_time,
            memory_used=memory_delta,
            memory_start=memory_start,
            memory_end=memory_end,
            process_pid=process.pid if process is not None else None
        )
        if save_final_model:
            try:
                self.train_and_save_final_model(X_train, y_train, output_dir, batch_id=batch_id)
            except Exception as e:
                logging.warning('Failed to save final model: %s', str(e))
        self.is_fitted_ = True
        gc.collect()
        logging.info(f'Batch {batch_id} completed - Time: {execution_time:.2f}s, Memory delta: {memory_delta:+.2f} MB')
        try:
            if self._dataset_cache is not None:
                self._clear_dataset_cache()
        except Exception:
            pass
    def _save(self, df_ranking: pd.DataFrame, categorized: Dict, cv_metrics: Dict, batch_id: int, output_dir: str, execution_time: float = 0, memory_used: float = 0, memory_start: Optional[float] = None, memory_end: Optional[float] = None, process_pid: Optional[int] = None):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            df_ranking.to_parquet(output_path / f'batch_{batch_id}_ranking_{timestamp}.parquet', engine='pyarrow', compression='snappy', index=False)
            logging.info(f'Ranking saved in Parquet format')
        except Exception as e:
            logging.warning(f'Parquet save failed, using CSV: {str(e)}')
            df_ranking.to_csv(output_path / f'batch_{batch_id}_ranking_{timestamp}.csv', index=False)
        pd.DataFrame({'feature': categorized['strong']}).to_csv(output_path / f'batch_{batch_id}_strong.csv', index=False)
        pd.DataFrame({'feature': categorized['medium']}).to_csv(output_path / f'batch_{batch_id}_medium.csv', index=False)
        pd.DataFrame({'feature': categorized['weak']}).to_csv(output_path / f'batch_{batch_id}_weak.csv', index=False)
        metadata = {'batch_id': batch_id, 'timestamp': timestamp, 'n_total': len(df_ranking), 'n_strong': len(categorized['strong']), 'n_medium': len(categorized['medium']), 'n_weak': len(categorized['weak']), 'mean_cv_score': float(cv_metrics['mean_cv_score']), 'std_cv_score': float(cv_metrics['std_cv_score']), 'execution_time_sec': float(execution_time), 'memory_used_mb': float(memory_used), 'memory_start_mb': float(memory_start) if memory_start is not None else None, 'memory_end_mb': float(memory_end) if memory_end is not None else None, 'process_pid': int(process_pid) if process_pid is not None else None}
        with open(output_path / f'batch_{batch_id}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f'Saved to {output_path}')
        try:
            env = self.log_environment(output_dir)
            with open(output_path / f'batch_{batch_id}_env.json', 'w') as f:
                json.dump(env, f, indent=2)
        except Exception:
            pass
class FeatureSelectorCPUOptimized(FeatureSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_all_cpu_optimizations()
    def _apply_all_cpu_optimizations(self):
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False) or 1
        except Exception:
            physical_cores = os.cpu_count() or 1
        if not self.ensure_reproducible:
            self.num_threads = min(16, physical_cores)
        else:
            self.num_threads = 1
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.num_threads)
        cpu_params = self._get_feature_selection_params_default(self.classification, self.random_state, self.num_threads)
        cpu_params.update({
            'force_col_wise': True,
            'max_bin': 127,
            'feature_fraction': 1.0,
            'feature_fraction_bynode': 0.4,
            'num_threads': self.num_threads,
        })
        self.base_params.update(cpu_params)
        self._dataset_cache = OrderedDict() if self.enable_dataset_cache else None
        logging.info(f"CPU optimizations complete: threads={self.num_threads}, max_bin=127, num_leaves={self.base_params.get('num_leaves', 31)}, force_col_wise={self.base_params.get('force_col_wise')}")
    def _save(
        self,
        df_ranking: pd.DataFrame,
        categorized: Dict,
        cv_metrics: Dict,
        batch_id: int,
        output_dir: str,
        execution_time: float = 0,
        memory_used: float = 0
        , memory_start: Optional[float] = None, memory_end: Optional[float] = None, process_pid: Optional[int] = None
    ):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
            'memory_used_mb': float(memory_used),
            'memory_start_mb': float(memory_start) if memory_start is not None else None,
            'memory_end_mb': float(memory_end) if memory_end is not None else None,
            'process_pid': int(process_pid) if process_pid is not None else None
        }
        with open(output_path / f'batch_{batch_id}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f'Saved to {output_path}')
        try:
            env = self.log_environment(output_dir)
            with open(output_path / f'batch_{batch_id}_env.json', 'w') as f:
                json.dump(env, f, indent=2)
        except Exception:
            pass
def main():
    logging.info('Loading F.parquet...')
    df = pd.read_parquet('F.parquet')
    logging.info(f'F.parquet loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    if 'target' not in df.columns:
        logging.error('No target column found in F.parquet')
        return
    logging.info(f'Target column found. Features: {df.shape[1]-1}, Samples: {df.shape[0]}')
    logging.info(f'Target distribution: {df["target"].value_counts().to_dict()}')
    fs = FeatureSelector(
        target_column='target',
        classification=True,
        random_state=42,
        ensure_reproducible=True,
        enable_metadata_routing=False,
        use_shap=True,
        shap_sample_size=None,
        enable_dataset_cache=True,
        max_cache_size=32
    )
    logging.info('Starting Feature Selection Pipeline...')
    fs.process_batch(
        features_df=df,
        batch_id=0,
        output_dir='feature_selection_results',
        save_final_model=False
    )
    logging.info('Feature Selection Pipeline completed!')
if __name__ == '__main__':
    main()
