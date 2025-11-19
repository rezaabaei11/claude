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
class PurgedKFold:
    def __init__(
        self,
        n_splits: int = 5,
        samples_info_sets: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
        label_horizon: int = 0,
        embargo_safety_factor: float = 2.0
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not (0 <= pct_embargo < 1):
            raise ValueError(f"pct_embargo must be in [0, 1), got {pct_embargo}")
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.label_horizon = label_horizon if label_horizon else 0
        self.embargo_safety_factor = embargo_safety_factor
        if self.label_horizon > 0:
            self._embargo_absolute = int(self.label_horizon * self.embargo_safety_factor)
            self.pct_embargo = pct_embargo
            logging.debug(f"[C4-FIX] Embargo: {self._embargo_absolute} samples (label_horizon={label_horizon} * safety={embargo_safety_factor})")
        else:
            self.pct_embargo = pct_embargo
            self._embargo_absolute = None
    def split(self, X, y=None, groups=None):
        if self.samples_info_sets is None:
            indices = np.arange(len(X))
        else:
            indices = np.array(self.samples_info_sets.index)
        n_samples = len(indices)
        test_fold_size = n_samples // self.n_splits
        if hasattr(self, '_embargo_absolute') and self._embargo_absolute is not None:
            embargo_size = self._embargo_absolute
        else:
            embargo_size = int(test_fold_size * self.pct_embargo)
        for fold_idx in range(self.n_splits):
            test_start = fold_idx * test_fold_size
            test_end = (fold_idx + 1) * test_fold_size
            if fold_idx == self.n_splits - 1:
                test_end = n_samples
            test_indices_pos = np.arange(test_start, test_end)
            if self.samples_info_sets is None:
                train_before = np.arange(0, test_start)
                train_after = np.arange(min(test_end + embargo_size, n_samples), n_samples)
                train_indices_pos = np.concatenate([train_before, train_after])
            else:
                train_indices_pos = []
                test_start_time = indices[test_start] if test_start < n_samples else indices[-1]
                test_end_time = indices[test_end - 1] if test_end <= n_samples else indices[-1]
                for idx_pos in range(n_samples):
                    if test_start <= idx_pos < test_end:
                        continue
                    if test_end <= idx_pos < min(test_end + embargo_size, n_samples):
                        continue
                    idx = indices[idx_pos]
                    if idx in self.samples_info_sets.index:
                        sample_end_time = self.samples_info_sets.loc[idx]
                        if sample_end_time <= test_start_time:
                            train_indices_pos.append(idx_pos)
                        elif idx_pos >= test_end + embargo_size:
                            train_indices_pos.append(idx_pos)
                train_indices_pos = np.array(train_indices_pos, dtype=int)
            yield train_indices_pos, test_indices_pos
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
class CombinatorialPurgedCV:
    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        pct_embargo: float = 0.01,
        label_horizon: int = 0,
        embargo_safety_factor: float = 2.0
    ):
        from itertools import combinations
        if n_test_groups >= n_splits:
            raise ValueError(f"n_test_groups ({n_test_groups}) must be < n_splits ({n_splits})")
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.label_horizon = label_horizon if label_horizon else 0
        self.embargo_safety_factor = embargo_safety_factor
        if self.label_horizon > 0:
            self._embargo_absolute = int(self.label_horizon * self.embargo_safety_factor)
            logging.debug(f"[C4-FIX] CPCV embargo: {self._embargo_absolute} samples")
        else:
            self._embargo_absolute = None
        self.pct_embargo = pct_embargo
        self.combinations = combinations
    def generate_paths(self, n_observations: int):
        indices = np.arange(n_observations)
        fold_size = n_observations // self.n_splits
        if hasattr(self, '_embargo_absolute') and self._embargo_absolute is not None:
            embargo_size = self._embargo_absolute
        else:
            embargo_size = int(fold_size * self.pct_embargo)
        fold_indices = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_observations
            fold_indices.append(indices[start:end])
        for test_combo in self.combinations(range(self.n_splits), self.n_test_groups):
            test_idx = np.concatenate([fold_indices[i] for i in test_combo])
            train_folds = []
            for train_fold_idx in range(self.n_splits):
                if train_fold_idx in test_combo:
                    continue
                min_distance = min(abs(train_fold_idx - test_fold)
                                 for test_fold in test_combo)
                if min_distance > 0:
                    train_folds.append(train_fold_idx)
            if len(train_folds) > 0:
                train_idx = np.concatenate([fold_indices[i] for i in train_folds])
                yield train_idx, test_idx
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
        enable_metadata_routing: bool = True,
        use_pyarrow: bool = False,
        ensure_reproducible: bool = True,
        categorical_unique_ratio_threshold: float = 0.5,
        perm_top_k: int = 300,
        max_threads_for_cpu: int = 16,
        enable_dataset_cache: bool = False,
        max_cache_size: int = 32,
        use_mutual_information: bool = True,
        use_stability_selection: bool = True,
        use_nested_cv: bool = True,
        use_confidence_intervals: bool = True,
        n_bootstrap_ci: int = 50,
        stability_selection_iterations: int = 30,
        check_lookahead_bias: bool = True,
        check_stationarity: bool = True,
        label_horizon: int = None,
        temporal_decay: float = None,
        sample_weight_half_life: int = 252,
    ):
        self.target_column = target_column
        self.check_lookahead_bias = check_lookahead_bias
        self.check_stationarity = check_stationarity
        self.label_horizon = label_horizon
        self.temporal_decay = temporal_decay
        self.sample_weight_half_life = sample_weight_half_life
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
        self.shap_feature_perturbation = 'interventional'
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
            logging.debug('Reproducible mode ON: Generator API with SeedSequence (no global state pollution)')
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
            logging.debug('Copy-on-Write mode enabled')
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
            'num_leaves': 80,
            'max_depth': 8,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 30,
            'lambda_l1': 0.3,
            'lambda_l2': 2.0,
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
        }
        logging.debug(f'Pandas {pd.__version__}, NumPy {np.__version__}')
        self._validate_tree_params()
        if self.ensure_reproducible:
            self._ensure_full_reproducibility()
        if self.enable_dataset_cache:
            logging.warning("[ISSUE#15] Dataset caching ENABLED - Risk of train/test contamination!")
            logging.warning("[ISSUE#15] For production/research: Set enable_dataset_cache=False")
            self._dataset_cache = OrderedDict()
        else:
            self._dataset_cache = None
            logging.info("[ISSUE#15] Dataset caching DISABLED (safe default)")
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
            logging.debug('Full reproducibility configuration applied (num_threads=1)')
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
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.6,
            'feature_fraction_bynode': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.5,
            'lambda_l2': 3.0,
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
        logging.debug(f'Memory optimization: {memory_before:.2f} MB -> {memory_after:.2f} MB ({memory_reduction:.1f}% reduction)')
        return df
    def fit_preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        X = X.copy()
        self._preprocess_params = {}
        new_cols = []
        for col in X.columns:
            new_col = col.replace('"', '').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace(',', '_').replace(' ', '_')
            new_cols.append(new_col)
        if new_cols != list(X.columns):
            X.columns = new_cols
            logging.info(f'[PREPROCESS-FIT] Sanitized feature names')
        constant_mask = X.nunique() <= 1
        constant_cols = X.columns[constant_mask].tolist()
        self._preprocess_params['constant_cols'] = constant_cols
        if constant_cols:
            logging.warning(f'[PREPROCESS-FIT] Removing {len(constant_cols)} constant features')
            X = X.drop(columns=constant_cols)
        missing_ratios = X.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > 0.9].index.tolist()
        self._preprocess_params['high_missing_cols'] = high_missing_cols
        if high_missing_cols:
            logging.warning(f'[PREPROCESS-FIT] Removing {len(high_missing_cols)} features with >90% missing')
            X = X.drop(columns=high_missing_cols)
        X = self.optimize_dtypes(X)
        self._preprocess_params['fill_values'] = {}
        missing_mask = X.isnull().any()
        if missing_mask.any():
            missing_cols = X.columns[missing_mask].tolist()
            logging.info(f'[PREPROCESS-FIT] Handling missing data in {len(missing_cols)} columns')
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            missing_numeric = [col for col in missing_cols if col in numeric_cols]
            missing_categorical = [col for col in missing_cols if col not in numeric_cols]
            for col in missing_numeric:
                X[col] = (X[col]
                         .interpolate(method='linear', limit_direction='forward', limit=5)
                         .ffill(limit=5))
                if X[col].isnull().any():
                    median_val = X[col].median()
                    self._preprocess_params['fill_values'][col] = median_val
                    X[col] = X[col].fillna(median_val)
            for col in missing_categorical:
                mode_val = X[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                self._preprocess_params['fill_values'][col] = fill_val
                X[col] = X[col].fillna(fill_val)
        self._preprocess_params['final_columns'] = X.columns.tolist()
        X = self.optimize_dtypes(X)
        gc.collect()
        return X, y
    def transform_preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, '_preprocess_params'):
            raise ValueError("Must call fit_preprocess first!")
        X = X.copy()
        new_cols = []
        for col in X.columns:
            new_col = col.replace('"', '').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace(',', '_').replace(' ', '_')
            new_cols.append(new_col)
        X.columns = new_cols
        constant_cols = self._preprocess_params['constant_cols']
        X = X.drop(columns=[c for c in constant_cols if c in X.columns], errors='ignore')
        high_missing_cols = self._preprocess_params['high_missing_cols']
        X = X.drop(columns=[c for c in high_missing_cols if c in X.columns], errors='ignore')
        X = self.optimize_dtypes(X)
        fill_values = self._preprocess_params['fill_values']
        for col, fill_val in fill_values.items():
            if col in X.columns and X[col].isnull().any():
                X[col] = (X[col]
                         .interpolate(method='linear', limit_direction='forward', limit=5)
                         .ffill(limit=5))
                X[col] = X[col].fillna(fill_val)
        final_columns = self._preprocess_params['final_columns']
        extra_cols = [c for c in X.columns if c not in final_columns]
        if extra_cols:
            X = X.drop(columns=extra_cols)
        missing_cols = [c for c in final_columns if c not in X.columns]
        for col in missing_cols:
            X[col] = 0
        X = X[final_columns]
        X = self.optimize_dtypes(X)
        gc.collect()
        logging.debug(f'[PREPROCESS-TRANSFORM] Applied train preprocessing to {len(X)} samples')
        return X
    def validate_no_lookahead_bias(self, X: pd.DataFrame) -> bool:
        warnings_found = []
        suspicious_keywords = ['future', 'next', 'forward', 'ahead', 'lead', 'tomorrow', 'later']
        for col in X.columns:
            col_lower = str(col).lower()
            for keyword in suspicious_keywords:
                if keyword in col_lower:
                    warnings_found.append(f"Suspicious feature name: {col} (contains '{keyword}')")
                    break
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_values = X[col].values
            trailing_nans = 0
            for val in reversed(col_values):
                if pd.isna(val):
                    trailing_nans += 1
                else:
                    break
            if trailing_nans > 0:
                pct = (trailing_nans / len(X)) * 100
                if pct > 1.0:
                    warnings_found.append(
                        f"Feature '{col}' has {trailing_nans} trailing NaNs ({pct:.1f}%) - possible lookahead bias"
                    )
        if warnings_found:
            logging.error(f'\n{"="*70}')
            logging.error('[LOOKAHEAD BIAS DETECTED] Critical Issues Found:')
            logging.error(f'{"="*70}')
            for i, warning in enumerate(warnings_found, 1):
                logging.error(f'  {i}. {warning}')
            logging.error(f'{"="*70}')
            raise ValueError(f"Lookahead bias detected! Found {len(warnings_found)} suspicious features. Fix these before continuing.")
        logging.info(f'[LOOKAHEAD CHECK] PASSED - No lookahead bias detected in {len(X.columns)} features')
        return True
    def validate_target_no_lookahead(self, df: pd.DataFrame, target_col: str = 'target') -> bool:
        logging.info(f'[TARGET-LEAKAGE-CHECK] Validating target creation for "{target_col}"...')
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        issues_found = []
        future_keywords = ['future', 'next', 'ahead', 'forward', 'tomorrow', 'next_', 'lead']
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in future_keywords):
                if col != target_col:
                    issues_found.append(f"Feature '{col}' contains future keyword")
        if hasattr(df, 'attrs') and 'metadata' in df.attrs:
            metadata = df.attrs['metadata']
            if target_col in metadata:
                target_meta = metadata[target_col]
                if 'operations' in target_meta:
                    for op in target_meta['operations']:
                        if 'shift' in op.lower():
                            shift_val = int(op.split('shift')[-1].strip('()'))
                            if shift_val < 0:
                                issues_found.append(f"Target uses shift({shift_val}) - LOOKAHEAD!")
        if len(df) > 100:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_numeric = df[target_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(target_numeric) > 50:
                for col in numeric_cols:
                    if col == target_col:
                        continue
                    max_check_lag = min(20, len(df) // 10)
                    for lag in range(1, max_check_lag):
                        feature_shifted = df[col].shift(-lag).dropna()
                        target_subset = target_numeric.iloc[:len(feature_shifted)]
                        if len(target_subset) > 10 and feature_shifted.std() > 1e-10:
                            corr = np.corrcoef(
                                target_subset.fillna(0).values,
                                feature_shifted.fillna(0).values
                            )[0, 1]
                            if abs(corr) > 0.3:
                                issues_found.append(
                                    f"Target correlates {corr:.3f} with FUTURE {col}[t+{lag}] - LOOKAHEAD!"
                                )
                                break
        if issues_found:
            logging.error(f"\n[TARGET-LEAKAGE-CHECK] [WARNING] CRITICAL ISSUES FOUND:")
            for issue in issues_found[:5]:
                logging.error(f"  [X] {issue}")
            if len(issues_found) > 5:
                logging.error(f"  ... and {len(issues_found)-5} more issues")
            raise ValueError(f"Target creation has lookahead bias! {issues_found[0]}")
        else:
            logging.info(f'[TARGET-LEAKAGE-CHECK] [OK] PASSED - Target appears clean (no future data)')
            return True
    def calculate_target_safe(self, df: pd.DataFrame,
                              price_col: str = 'close',
                              horizon: int = 5,
                              method: str = 'return',
                              threshold: float = 0.0) -> pd.Series:
        if price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame")
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {horizon}")
        n = len(df)
        if n <= horizon:
            raise ValueError(f"DataFrame too short ({n} rows) for horizon={horizon}")
        prices = df[price_col].copy()
        if method == 'return':
            future_price = prices.shift(-horizon)
            target = (future_price - prices) / prices
            if threshold != 0.0:
                target = (target > threshold).astype(int)
        elif method == 'direction':
            future_price = prices.shift(-horizon)
            target = (future_price > prices).astype(int)
        elif method == 'volatility':
            returns = prices.pct_change()
            target = returns.shift(-horizon).rolling(window=horizon).std()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'return', 'direction', or 'volatility'")
        target.iloc[-horizon:] = np.nan
        assert target.iloc[-horizon:].isna().all(), "Last rows should be NaN!"
        n_valid = target.notna().sum()
        n_removed = horizon
        logging.info(f"[TARGET-SAFE-ISSUE#2] Created target with method='{method}', horizon={horizon}")
        logging.info(f"[TARGET-SAFE-ISSUE#2] Valid samples: {n_valid}, Removed (future): {n_removed}")
        logging.warning(f"[TARGET-SAFE-ISSUE#2] [WARNING] MUST remove NaN rows before train/test split!")
        return target
    def validate_target_causality(
        self,
        df: pd.DataFrame,
        target_col: str,
        price_col: str = 'close',
        horizon: int = None,
        max_lag_test: int = 20
    ) -> Dict:
        logging.info(f'\n{"="*70}')
        logging.info('[CRITICAL-FIX-1] TARGET CAUSALITY VALIDATION')
        logging.info(f'{"="*70}')
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        if price_col not in df.columns:
            logging.warning(f"Price column '{price_col}' not found - skipping price-based tests")
            price_col = None
        if horizon is None:
            horizon = self.label_horizon or 1
        results = {
            'passed': True,
            'warnings': [],
            'critical_issues': [],
            'test_results': {}
        }
        if price_col is not None:
            logging.info(f'[TEST-1] Checking correlation with future prices...')
            future_prices = df[price_col].shift(-horizon)
            valid_mask = df[target_col].notna() & future_prices.notna()
            if valid_mask.sum() > 10:
                correlation = df[target_col][valid_mask].corr(future_prices[valid_mask])
                results['test_results']['future_price_correlation'] = float(correlation)
                logging.info(f'  Correlation with t+{horizon} price: {correlation:.4f}')
                if abs(correlation) > 0.3:
                    msg = (f"CRITICAL: Target has correlation {correlation:.3f} with future prices! "
                           f"This indicates SEVERE look-ahead bias.")
                    results['critical_issues'].append(msg)
                    results['passed'] = False
                    logging.error(f'   {msg}')
                elif abs(correlation) > 0.15:
                    msg = f"WARNING: Target has moderate correlation {correlation:.3f} with future prices"
                    results['warnings'].append(msg)
                    logging.warning(f'    {msg}')
                else:
                    logging.info(f'   PASSED: Low correlation with future prices')
        logging.info(f'[TEST-2] Multi-horizon forward correlation test...')
        if price_col is not None:
            max_forward_corr = 0.0
            critical_lag = None
            for lag in range(1, min(max_lag_test, len(df)//10)):
                future_data = df[price_col].shift(-lag)
                valid_mask = df[target_col].notna() & future_data.notna()
                if valid_mask.sum() > 10:
                    corr = df[target_col][valid_mask].corr(future_data[valid_mask])
                    if abs(corr) > abs(max_forward_corr):
                        max_forward_corr = corr
                        critical_lag = lag
            results['test_results']['max_forward_correlation'] = float(max_forward_corr)
            results['test_results']['critical_forward_lag'] = critical_lag
            logging.info(f'  Max forward correlation: {max_forward_corr:.4f} at lag={critical_lag}')
            if abs(max_forward_corr) > 0.25:
                msg = f"CRITICAL: High forward correlation {max_forward_corr:.3f} at lag={critical_lag}"
                results['critical_issues'].append(msg)
                results['passed'] = False
                logging.error(f'   {msg}')
            elif abs(max_forward_corr) > 0.12:
                msg = f"WARNING: Moderate forward correlation {max_forward_corr:.3f}"
                results['warnings'].append(msg)
                logging.warning(f'    {msg}')
            else:
                logging.info(f'   PASSED: Low forward correlation across all lags')
        logging.info(f'[TEST-3] Target statistics sanity check...')
        target_values = df[target_col].dropna()
        if len(target_values) > 0:
            last_n = min(horizon * 2, len(df) // 10)
            nan_ratio_end = df[target_col].iloc[-last_n:].isna().sum() / last_n
            results['test_results']['nan_ratio_at_end'] = float(nan_ratio_end)
            logging.info(f'  NaN ratio in last {last_n} samples: {nan_ratio_end:.2%}')
            if nan_ratio_end < 0.3 and horizon > 0:
                msg = f"WARNING: Expected more NaN values at end for horizon={horizon} (got {nan_ratio_end:.1%})"
                results['warnings'].append(msg)
                logging.warning(f'    {msg}')
            else:
                logging.info(f'   PASSED: Appropriate NaN pattern at end')
            if self.classification:
                unique_vals = target_values.nunique()
                value_counts = target_values.value_counts(normalize=True)
                results['test_results']['unique_values'] = int(unique_vals)
                results['test_results']['class_distribution'] = value_counts.to_dict()
                logging.info(f'  Unique values: {unique_vals}')
                logging.info(f'  Class distribution: {value_counts.to_dict()}')
                min_class_ratio = value_counts.min()
                if min_class_ratio < 0.05:
                    msg = f"WARNING: Severe class imbalance (min class: {min_class_ratio:.1%})"
                    results['warnings'].append(msg)
                    logging.warning(f'    {msg}')
        logging.info(f'[TEST-4] Temporal consistency check...')
        if len(target_values) > 10 and self.classification:
            target_changes = target_values.diff().abs()
            change_rate = (target_changes > 0).sum() / len(target_changes)
            results['test_results']['target_change_rate'] = float(change_rate)
            logging.info(f'  Target change rate: {change_rate:.2%}')
            if change_rate > 0.8:
                msg = f"WARNING: Very high target change rate ({change_rate:.1%}) - may indicate noise"
                results['warnings'].append(msg)
                logging.warning(f'    {msg}')
        logging.info(f'\n{"="*70}')
        if results['passed'] and len(results['critical_issues']) == 0:
            logging.info('[RESULT]  TARGET CAUSALITY VALIDATION PASSED')
            logging.info('  No critical data leakage detected')
            if len(results['warnings']) > 0:
                logging.info(f'  {len(results["warnings"])} warning(s) - review recommended')
        else:
            logging.error('[RESULT]  TARGET CAUSALITY VALIDATION FAILED')
            logging.error(f'  {len(results["critical_issues"])} critical issue(s) found')
            for issue in results['critical_issues']:
                logging.error(f'    - {issue}')
        if len(results['warnings']) > 0:
            logging.info('\n[WARNINGS]')
            for warning in results['warnings']:
                logging.info(f'  - {warning}')
        logging.info(f'{"="*70}\n')
        if not results['passed'] or len(results['critical_issues']) > 0:
            raise ValueError(
                f"Target causality validation FAILED! Found {len(results['critical_issues'])} "
                f"critical issues. Target appears to use future information (look-ahead bias)."
            )
        return results
    def create_labels_with_embargo(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        label_horizon: int = 5,
        label_column: str = 'label'
    ) -> pd.DataFrame:
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found in DataFrame")
        df[label_column] = df[price_column].pct_change(label_horizon).shift(-label_horizon)
        if isinstance(df.index, pd.DatetimeIndex):
            df['label_end_time'] = df.index.shift(-label_horizon)
        else:
            df['label_end_index'] = np.arange(len(df)) + label_horizon
        logging.info(f'[LABEL EMBARGO] Created labels with horizon={label_horizon}')
        logging.info(f'  Label NaNs: {df[label_column].isna().sum()} (expected: {label_horizon} at end)')
        return df
    def detect_lookahead_statistically(self, X: pd.DataFrame, y: pd.Series, n_random_splits: int = 10) -> List[Dict]:
        logging.info("[ISSUE#17] Running statistical lookahead bias detection...")
        suspicious_features = []
        for col in X.columns:
            if X[col].dtype not in [np.float32, np.float64, np.int32, np.int64]:
                continue
            correlations = []
            for seed in range(n_random_splits):
                np.random.seed(self.random_state + seed)
                split_point = np.random.randint(
                    int(0.3 * len(X)),
                    int(0.7 * len(X))
                )
                X_before = X.iloc[:split_point]
                y_before = y.iloc[:split_point]
                if len(X_before) > 10 and X_before[col].std() > 1e-10:
                    try:
                        corr = np.corrcoef(X_before[col].fillna(0), y_before.fillna(0))[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        pass
            if len(correlations) < 3:
                continue
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            cv_corr = std_corr / (mean_corr + 1e-10)
            if mean_corr > 0.90 and std_corr < 0.05:
                suspicion_level = 'HIGH'
                suspicious_features.append({
                    'feature': col,
                    'mean_correlation': float(mean_corr),
                    'std_correlation': float(std_corr),
                    'cv_correlation': float(cv_corr),
                    'suspicion_level': suspicion_level,
                    'reason': 'Extremely high & stable correlation - likely lookahead'
                })
                logging.warning(f"[LOOKAHEAD?] {col}: mean_corr={mean_corr:.4f}, "
                               f"std_corr={std_corr:.4f} (SUSPICIOUS)")
            elif mean_corr > 0.95 and cv_corr < 0.1:
                suspicion_level = 'MEDIUM'
                suspicious_features.append({
                    'feature': col,
                    'mean_correlation': float(mean_corr),
                    'std_correlation': float(std_corr),
                    'cv_correlation': float(cv_corr),
                    'suspicion_level': suspicion_level,
                    'reason': 'Very high correlation with low variance'
                })
                logging.warning(f"[LOOKAHEAD?] {col}: mean_corr={mean_corr:.4f}, "
                               f"cv_corr={cv_corr:.4f} (MEDIUM SUSPICION)")
        if len(suspicious_features) == 0:
            logging.info("[ISSUE#17] [OK] No statistically suspicious features detected")
        else:
            logging.warning(f"[ISSUE#17] [WARNING] Found {len(suspicious_features)} "
                          f"suspicious features (potential lookahead)")
        return suspicious_features
    def detect_feature_windows(self, X: pd.DataFrame) -> int:
        import re
        max_window = 0
        patterns = [
            r'MA_?(\d+)',
            r'EMA_?(\d+)',
            r'SMA_?(\d+)',
            r'ROLLING_?(\d+)',
            r'WINDOW_?(\d+)',
            r'LAG_?(\d+)',
            r'SHIFT_?(\d+)',
            r'DELAY_?(\d+)',
        ]
        for col in X.columns:
            col_upper = col.upper()
            for pattern in patterns:
                matches = re.findall(pattern, col_upper)
                for match in matches:
                    try:
                        window = int(match)
                        max_window = max(max_window, window)
                    except:
                        pass
        if max_window > 0:
            logging.info(f'[C3-EMBARGO] Detected max feature window: {max_window} from feature names')
        return max_window
    def validate_no_leakage_in_preprocess(self) -> bool:
        """
        Validate that fit_preprocess does NOT perform feature selection.

        This test ensures preprocessing only does statistical operations like:
        - Constant feature removal
        - Missing data handling
        - Data type optimization

        And does NOT do:
        - Feature selection based on importance
        - Correlation-based feature removal
        - Variance-based feature selection

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If feature selection leakage is detected in preprocessing
        """
        logging.info(f'\n{"="*70}')
        logging.info('[PREPROCESS-LEAKAGE-CHECK] Validating preprocessing has no feature selection')
        logging.info(f'{"="*70}')

        # Create dummy data with known characteristics
        n_samples = 1000
        n_features = 50

        logging.info(f'[PREPROCESS-LEAKAGE-CHECK] Creating test data: {n_samples} samples, {n_features} features')

        # Generate random features
        X_dummy = pd.DataFrame(
            self.rng.standard_normal((n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_dummy = pd.Series(self.rng.integers(0, 2, n_samples))

        # Add some features with varying correlation to target
        X_dummy['high_corr_feature'] = y_dummy + self.rng.standard_normal(n_samples) * 0.1
        X_dummy['medium_corr_feature'] = y_dummy * 0.5 + self.rng.standard_normal(n_samples) * 0.5
        X_dummy['low_corr_feature'] = self.rng.standard_normal(n_samples)
        X_dummy['zero_corr_feature'] = self.rng.standard_normal(n_samples)

        # Add a constant feature (should be removed)
        X_dummy['constant_feature'] = 42.0

        # Add a high-missing feature (should be removed)
        X_dummy['high_missing_feature'] = self.rng.standard_normal(n_samples)
        X_dummy.loc[self.rng.choice(n_samples, int(n_samples * 0.95), replace=False), 'high_missing_feature'] = np.nan

        cols_before = set(X_dummy.columns.tolist())
        n_features_before = len(cols_before)

        logging.info(f'[PREPROCESS-LEAKAGE-CHECK] Features before preprocessing: {n_features_before}')

        # Run preprocessing
        X_processed, y_processed = self.fit_preprocess(X_dummy.copy(), y_dummy.copy())

        cols_after = set(X_processed.columns.tolist())
        n_features_after = len(cols_after)

        logging.info(f'[PREPROCESS-LEAKAGE-CHECK] Features after preprocessing: {n_features_after}')

        # Check what was removed
        removed_cols = cols_before - cols_after
        logging.info(f'[PREPROCESS-LEAKAGE-CHECK] Removed {len(removed_cols)} features')

        if removed_cols:
            logging.info(f'[PREPROCESS-LEAKAGE-CHECK] Removed features: {list(removed_cols)[:10]}')

        # Validation 1: Check that constant and high-missing features were removed
        expected_removals = {'constant_feature', 'high_missing_feature'}
        actually_removed = removed_cols & expected_removals

        if actually_removed != expected_removals:
            missing_removals = expected_removals - actually_removed
            if missing_removals:
                logging.warning(f'[PREPROCESS-LEAKAGE-CHECK] Expected removals not found: {missing_removals}')

        # Validation 2: Check that correlated features were NOT removed
        important_features = {'high_corr_feature', 'medium_corr_feature', 'low_corr_feature', 'zero_corr_feature'}
        removed_important = removed_cols & important_features

        if removed_important:
            # Check if removal was due to feature selection (correlation/importance) rather than statistical issues
            issues_found = []

            for feat in removed_important:
                # If the feature was removed and it wasn't constant or high-missing, it's leakage
                if feat not in expected_removals:
                    issues_found.append(
                        f"Feature '{feat}' was removed - possible feature selection leakage!"
                    )

            if issues_found:
                logging.error(f'[PREPROCESS-LEAKAGE-CHECK] [CRITICAL] FEATURE SELECTION LEAKAGE DETECTED!')
                for issue in issues_found:
                    logging.error(f'  [X] {issue}')
                raise ValueError(
                    f"Preprocessing performs feature selection! This causes data leakage. "
                    f"Removed {len(removed_important)} non-statistical features: {removed_important}. "
                    f"fit_preprocess should ONLY remove constant/missing features, NOT select by importance."
                )

        # Validation 3: Ensure most valid features remain
        valid_features_before = n_features_before - len(expected_removals)
        features_lost = valid_features_before - n_features_after

        if features_lost > 0:
            loss_pct = (features_lost / valid_features_before) * 100

            if loss_pct > 10:  # If more than 10% of valid features are lost
                logging.error(
                    f'[PREPROCESS-LEAKAGE-CHECK] [CRITICAL] Lost {features_lost} valid features ({loss_pct:.1f}%)'
                )
                raise ValueError(
                    f"Preprocessing removed {features_lost} valid features ({loss_pct:.1f}%). "
                    f"This suggests feature selection is happening in preprocessing - DATA LEAKAGE!"
                )
            elif loss_pct > 5:
                logging.warning(
                    f'[PREPROCESS-LEAKAGE-CHECK] [WARNING] Lost {features_lost} valid features ({loss_pct:.1f}%)'
                )

        # Validation 4: Test determinism - preprocessing should be consistent
        X_processed_2, _ = self.fit_preprocess(X_dummy.copy(), y_dummy.copy())
        cols_after_2 = set(X_processed_2.columns.tolist())

        if cols_after != cols_after_2:
            diff = cols_after.symmetric_difference(cols_after_2)
            logging.error(f'[PREPROCESS-LEAKAGE-CHECK] [WARNING] Preprocessing is NOT deterministic!')
            logging.error(f'[PREPROCESS-LEAKAGE-CHECK] Different features: {diff}')
            raise ValueError(
                f"Preprocessing is not deterministic! Features differ between runs: {diff}. "
                f"This suggests randomness or feature selection based on data."
            )

        logging.info(f'[PREPROCESS-LEAKAGE-CHECK] [OK] PASSED - No feature selection detected')
        logging.info(f'[PREPROCESS-LEAKAGE-CHECK] [OK] Preprocessing only performs statistical operations')
        logging.info(f'{"="*70}\n')

        return True
    def validate_target_calculation(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        price_col: str = 'close'
    ) -> bool:
        """
        Validate that target is NOT calculated using future price information.

        Tests correlation between target and future prices at various lags.
        High correlation with future prices indicates lookahead bias.

        Args:
            df: DataFrame containing target and price columns
            target_col: Name of the target column (default: 'target')
            price_col: Name of the price column (default: 'close')

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If target calculation has lookahead bias (future leakage)
        """
        logging.info(f'\n{"="*70}')
        logging.info('[TARGET-CALC-LEAKAGE-CHECK] Validating target calculation')
        logging.info(f'{"="*70}')

        # Validate inputs
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        if price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame")

        target = df[target_col].replace([np.inf, -np.inf], np.nan).dropna()
        price = df[price_col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(target) < 50:
            logging.warning(f'[TARGET-CALC-LEAKAGE-CHECK] [WARNING] Insufficient data ({len(target)} samples), skipping test')
            return True

        logging.info(f'[TARGET-CALC-LEAKAGE-CHECK] Testing target: "{target_col}" vs price: "{price_col}"')
        logging.info(f'[TARGET-CALC-LEAKAGE-CHECK] Data points: {len(target)}')

        # Test correlation with future prices at different lags
        test_lags = [1, 2, 5, 10, 20]
        correlation_threshold = 0.3

        issues_found = []
        correlations = {}

        for lag in test_lags:
            if lag >= len(price):
                logging.debug(f'[TARGET-CALC-LEAKAGE-CHECK] Skipping lag={lag} (exceeds data length)')
                continue

            # Shift price forward (future prices)
            future_price = price.shift(-lag)

            # Align target and future_price indices
            common_idx = target.index.intersection(future_price.dropna().index)

            if len(common_idx) < 20:
                logging.debug(f'[TARGET-CALC-LEAKAGE-CHECK] Skipping lag={lag} (insufficient overlap)')
                continue

            target_subset = target.loc[common_idx]
            future_price_subset = future_price.loc[common_idx]

            # Calculate correlation
            if target_subset.std() > 1e-10 and future_price_subset.std() > 1e-10:
                corr = np.corrcoef(
                    target_subset.fillna(0).values,
                    future_price_subset.fillna(0).values
                )[0, 1]

                correlations[lag] = corr

                logging.info(f'[TARGET-CALC-LEAKAGE-CHECK] Lag t+{lag:2d}: correlation = {corr:+.4f}')

                # Check if correlation is suspiciously high
                if abs(corr) > correlation_threshold:
                    severity = 'CRITICAL' if abs(corr) > 0.5 else 'HIGH'
                    issues_found.append({
                        'lag': lag,
                        'correlation': corr,
                        'severity': severity,
                        'message': f"Target correlates {corr:+.3f} with future price at t+{lag}"
                    })
                    logging.error(
                        f'[TARGET-CALC-LEAKAGE-CHECK] [{severity}] FUTURE LEAKAGE at t+{lag}: '
                        f'correlation = {corr:+.4f} (threshold = {correlation_threshold})'
                    )

        # Check if any issues were found
        if issues_found:
            logging.error(f'\n[TARGET-CALC-LEAKAGE-CHECK] [CRITICAL] TARGET CALCULATION LEAKAGE DETECTED!')
            logging.error(f'[TARGET-CALC-LEAKAGE-CHECK] Found {len(issues_found)} suspicious correlations:')

            for issue in issues_found:
                logging.error(
                    f"  [X] Lag t+{issue['lag']}: {issue['correlation']:+.3f} "
                    f"(severity: {issue['severity']})"
                )

            logging.error(f'\n[TARGET-CALC-LEAKAGE-CHECK] [EXPLANATION]')
            logging.error(f'  Target should NOT be correlated with FUTURE prices')
            logging.error(f'  High correlation indicates target is using future information')
            logging.error(f'  This will cause overly optimistic backtest results')
            logging.error(f'  But FAIL in real trading (future data not available)')

            # Get the worst offender
            worst = max(issues_found, key=lambda x: abs(x['correlation']))

            raise ValueError(
                f"⚠️ TARGET CALCULATION LEAKAGE DETECTED! "
                f"Target is correlated with future price at t+{worst['lag']} "
                f"(correlation = {worst['correlation']:+.3f}). "
                f"Target should ONLY use past/present data, not future prices. "
                f"Found {len(issues_found)} suspicious lags: {[x['lag'] for x in issues_found]}. "
                f"This indicates the target is calculated using lookahead bias."
            )

        # Test passed
        logging.info(f'\n[TARGET-CALC-LEAKAGE-CHECK] [OK] PASSED - No future correlation detected')
        logging.info(f'[TARGET-CALC-LEAKAGE-CHECK] [OK] Target appears to be calculated from past data only')

        if correlations:
            max_corr = max(abs(c) for c in correlations.values())
            avg_corr = np.mean([abs(c) for c in correlations.values()])
            logging.info(f'[TARGET-CALC-LEAKAGE-CHECK] Max correlation: {max_corr:.4f} (threshold: {correlation_threshold})')
            logging.info(f'[TARGET-CALC-LEAKAGE-CHECK] Avg correlation: {avg_corr:.4f}')

        logging.info(f'{"="*70}\n')

        return True
    def shap_importance_analysis_with_proper_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_sample: Optional[pd.DataFrame] = None,
        n_background_samples: int = 100,
        n_runs: int = 3
    ) -> Dict:
        """
        [CRITICAL-FIX-3] SHAP analysis with proper baseline selection.

        Uses KMeans to select representative background samples from training data.
        This prevents bias that can occur when using random or all training data.

        Based on: SHAP guidelines and Bailey et al. (2014)

        Args:
            X_train: Training features for background selection
            y_train: Training target (used for model training)
            X_sample: Samples to explain (default: use random from X_train)
            n_background_samples: Number of background samples via KMeans (default: 100)
            n_runs: Number of SHAP runs for stability (default: 3)

        Returns:
            Dict with SHAP importance metrics
        """
        logging.info(f'\n{"="*70}')
        logging.info('[C3-SHAP-FIX] SHAP Analysis with Proper Baseline')
        logging.info(f'{"="*70}')

        try:
            import shap
            from sklearn.cluster import KMeans
        except ImportError as e:
            logging.warning(f'[C3-SHAP-FIX] SHAP/sklearn not available: {e}')
            return {
                'shap_mean': np.zeros(X_train.shape[1], dtype=np.float32),
                'shap_std': np.zeros(X_train.shape[1], dtype=np.float32),
                'shap_cv': np.zeros(X_train.shape[1], dtype=np.float32),
                'background_quality': 'skipped'
            }

        logging.info(f'[C3-SHAP-FIX] Training data shape: {X_train.shape}')
        logging.info(f'[C3-SHAP-FIX] Selecting {n_background_samples} representative background samples...')

        # Step 1: Select representative background via KMeans
        if len(X_train) > n_background_samples:
            try:
                kmeans = KMeans(
                    n_clusters=n_background_samples,
                    random_state=self.random_state,
                    n_init=10
                )
                kmeans.fit(X_train)

                # Find closest sample to each cluster center
                distances = np.linalg.norm(
                    X_train.values[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :],
                    axis=2
                )
                closest_indices = np.argmin(distances, axis=0)
                background_indices = np.unique(closest_indices)

                X_background = X_train.iloc[background_indices].reset_index(drop=True)

                logging.info(f'[C3-SHAP-FIX] Selected {len(X_background)} representative samples via KMeans')
                logging.info(f'[C3-SHAP-FIX] Background quality: EXCELLENT (representative via clustering)')
            except Exception as e:
                logging.warning(f'[C3-SHAP-FIX] KMeans selection failed: {e}, using random sample')
                indices = self.rng.choice(len(X_train), min(n_background_samples, len(X_train)), replace=False)
                X_background = X_train.iloc[indices].reset_index(drop=True)
                logging.info(f'[C3-SHAP-FIX] Fallback: Using {len(X_background)} random background samples')
        else:
            X_background = X_train.copy()
            logging.info(f'[C3-SHAP-FIX] Background: Using all {len(X_background)} training samples')

        # Step 2: Select samples to explain
        if X_sample is None:
            n_explain = min(100, len(X_train) // 2)
            sample_indices = self.rng.choice(len(X_train), n_explain, replace=False)
            X_sample = X_train.iloc[sample_indices].reset_index(drop=True)
            logging.info(f'[C3-SHAP-FIX] Explaining {len(X_sample)} random samples from training data')
        else:
            logging.info(f'[C3-SHAP-FIX] Explaining provided samples: {X_sample.shape[0]} samples')

        # Step 3: Train models and compute SHAP values
        shap_runs = []

        for run_idx in range(n_runs):
            try:
                seed = int(self.random_state + run_idx)
                logging.debug(f'[C3-SHAP-FIX] Run {run_idx+1}/{n_runs}')

                # Train model
                if self.classification:
                    model = lgb.LGBMClassifier(
                        **{k: v for k, v in self.base_params.items() if k != 'random_state'},
                        random_state=seed,
                        verbose=-1
                    )
                else:
                    model = lgb.LGBMRegressor(
                        **{k: v for k, v in self.base_params.items() if k != 'random_state'},
                        random_state=seed,
                        verbose=-1
                    )

                model.fit(X_train, y_train)

                # Create SHAP explainer with proper background
                explainer = shap.TreeExplainer(model, data=X_background)

                # Get SHAP values
                try:
                    shap_values = explainer.shap_values(X_sample, check_additivity=False)
                except Exception:
                    shap_values = explainer.shap_values(X_sample, check_additivity=False)

                # Handle list output (binary classification)
                if isinstance(shap_values, list):
                    shap_vals = np.abs(shap_values[1])  # Use positive class
                else:
                    shap_vals = np.abs(shap_values)

                # Average across samples
                shap_mean_run = np.mean(shap_vals, axis=0).astype(np.float32)
                shap_runs.append(shap_mean_run)

                logging.debug(f'[C3-SHAP-FIX] Run {run_idx+1} complete: mean SHAP computed')

            except Exception as e:
                logging.warning(f'[C3-SHAP-FIX] Run {run_idx+1} failed: {e}')
                shap_runs.append(np.zeros(X_train.shape[1], dtype=np.float32))

        # Step 4: Aggregate SHAP values across runs
        if shap_runs:
            shap_values_array = np.vstack(shap_runs)
            shap_mean = np.mean(shap_values_array, axis=0).astype(np.float32)
            shap_std = np.std(shap_values_array, axis=0).astype(np.float32)
            shap_cv = (shap_std / (np.abs(shap_mean) + 1e-10)).astype(np.float32)

            logging.info(f'[C3-SHAP-FIX] Aggregated {n_runs} SHAP runs')
            logging.info(f'[C3-SHAP-FIX] Mean SHAP - Min: {shap_mean.min():.6f}, '
                        f'Max: {shap_mean.max():.6f}, Mean: {shap_mean.mean():.6f}')
            logging.info(f'[C3-SHAP-FIX] Stability (CV) - Mean: {shap_cv.mean():.3f}, '
                        f'Max: {shap_cv.max():.3f}')

            if shap_cv.mean() < 0.2:
                stability = 'EXCELLENT'
            elif shap_cv.mean() < 0.5:
                stability = 'GOOD'
            else:
                stability = 'UNSTABLE'

            logging.info(f'[C3-SHAP-FIX] Stability assessment: {stability}')
        else:
            logging.error('[C3-SHAP-FIX] All SHAP runs failed')
            shap_mean = np.zeros(X_train.shape[1], dtype=np.float32)
            shap_std = np.zeros(X_train.shape[1], dtype=np.float32)
            shap_cv = np.zeros(X_train.shape[1], dtype=np.float32)

        logging.info(f'{"="*70}\n')

        return {
            'shap_mean': shap_mean,
            'shap_std': shap_std,
            'shap_cv': shap_cv,
            'n_background_samples': len(X_background),
            'n_explained_samples': len(X_sample),
            'n_runs': len([r for r in shap_runs if r is not None]),
            'stability': stability if shap_runs else 'failed',
            'background_quality': 'representative_via_kmeans' if len(X_train) > n_background_samples else 'all_training_data'
        }
    def remove_multicollinearity_comprehensive(
        self,
        X: pd.DataFrame,
        threshold_corr: float = 0.85,
        threshold_vif: float = 10.0
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        [CRITICAL-FIX-7] Comprehensive multicollinearity removal.

        Iteratively removes correlated features and those with high VIF.
        Unlike simple single-pass removal, this approach:
        1. Removes high correlations iteratively
        2. Recalculates VIF after each removal
        3. Uses importance to decide which feature to drop

        Based on: Statistical multicollinearity best practices

        Args:
            X: Feature DataFrame
            threshold_corr: Correlation threshold (default: 0.85)
            threshold_vif: VIF threshold (default: 10.0)

        Returns:
            Tuple of (X_reduced, removed_features_list)
        """
        logging.info(f'\n{"="*70}')
        logging.info('[C7-MULTICOLL-FIX] Comprehensive Multicollinearity Removal')
        logging.info(f'{"="*70}')
        logging.info(f'  Correlation threshold: {threshold_corr}')
        logging.info(f'  VIF threshold: {threshold_vif}')

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logging.warning('[C7-MULTICOLL-FIX] statsmodels not available, skipping VIF')
            vif_available = False
        else:
            vif_available = True

        X_reduced = X.copy()
        removed_features = []

        # Phase 1: Correlation-based removal (iterative)
        iteration = 0
        max_iterations = 50  # Safety limit

        while iteration < max_iterations and len(X_reduced.columns) > 1:
            iteration += 1

            # Calculate correlation matrix
            numeric_cols = X_reduced.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                break

            X_numeric = X_reduced[numeric_cols]
            corr_matrix = X_numeric.corr().abs()

            # Find highest correlation pair
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            high_corr_pairs = []
            for col in upper_tri.columns:
                for row in upper_tri.index:
                    if upper_tri.loc[row, col] > threshold_corr:
                        high_corr_pairs.append({
                            'feat1': row,
                            'feat2': col,
                            'corr': upper_tri.loc[row, col]
                        })

            if not high_corr_pairs:
                logging.debug(f'[C7-MULTICOLL-FIX] Iteration {iteration}: No more high correlations detected')
                break

            # Find feature with highest average correlation (most problematic)
            feature_corr_counts = {}
            for pair in high_corr_pairs:
                for feat in [pair['feat1'], pair['feat2']]:
                    if feat not in feature_corr_counts:
                        feature_corr_counts[feat] = 0
                    feature_corr_counts[feat] += 1

            # Remove feature involved in most correlations
            feat_to_remove = max(feature_corr_counts, key=feature_corr_counts.get)

            logging.debug(
                f'[C7-MULTICOLL-FIX] Iteration {iteration}: '
                f'Removing "{feat_to_remove}" '
                f'(involved in {feature_corr_counts[feat_to_remove]} high-correlation pairs)'
            )

            X_reduced = X_reduced.drop(columns=[feat_to_remove])
            removed_features.append(feat_to_remove)

        logging.info(f'[C7-MULTICOLL-FIX] Phase 1 (Correlation): '
                    f'Removed {len(removed_features)} features in {iteration} iterations')

        # Phase 2: VIF-based removal (if available)
        if vif_available and len(X_reduced.columns) > 2:
            logging.info(f'[C7-MULTICOLL-FIX] Phase 2 (VIF): Computing Variance Inflation Factors...')

            vif_iteration = 0
            max_vif_iterations = 50

            while vif_iteration < max_vif_iterations and len(X_reduced.columns) > 2:
                vif_iteration += 1

                numeric_cols = X_reduced.select_dtypes(include=[np.number]).columns.tolist()

                if len(numeric_cols) < 2:
                    break

                X_numeric = X_reduced[numeric_cols]

                # Calculate VIF for each feature
                vif_data = []
                for i, col in enumerate(numeric_cols):
                    try:
                        vif = variance_inflation_factor(X_numeric.values, i)
                        vif_data.append({'feature': col, 'vif': vif})
                    except Exception:
                        vif_data.append({'feature': col, 'vif': np.nan})

                # Find feature with highest VIF
                max_vif_feature = None
                max_vif_value = 0

                for data in vif_data:
                    if not np.isnan(data['vif']) and data['vif'] > max_vif_value:
                        max_vif_value = data['vif']
                        max_vif_feature = data['feature']

                if max_vif_feature is None or max_vif_value <= threshold_vif:
                    logging.debug(
                        f'[C7-MULTICOLL-FIX] VIF iteration {vif_iteration}: '
                        f'All features have acceptable VIF (max: {max_vif_value:.2f})'
                    )
                    break

                logging.debug(
                    f'[C7-MULTICOLL-FIX] VIF iteration {vif_iteration}: '
                    f'Removing "{max_vif_feature}" (VIF={max_vif_value:.2f} > {threshold_vif})'
                )

                X_reduced = X_reduced.drop(columns=[max_vif_feature])
                removed_features.append(max_vif_feature)

            logging.info(f'[C7-MULTICOLL-FIX] Phase 2 (VIF): '
                        f'Removed {vif_iteration} additional features')

        # Summary
        logging.info(f'\n[C7-MULTICOLL-FIX] Summary:')
        logging.info(f'  Features before: {len(X.columns)}')
        logging.info(f'  Features after: {len(X_reduced.columns)}')
        logging.info(f'  Features removed: {len(removed_features)}')
        logging.info(f'  Retention rate: {len(X_reduced.columns)/len(X.columns):.1%}')

        if len(removed_features) > 0:
            logging.info(f'  Removed features (top 10): {removed_features[:10]}')

        logging.info(f'{"="*70}\n')

        return X_reduced, removed_features
    def calculate_universal_embargo_gap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        label_horizon: int = 0
    ) -> int:
        """
        [CRITICAL-FIX-6] Universal embargo gap calculation.

        Calculates embargo gap for ALL train/test splits (not just nested CV).
        Must be used consistently across all splitting methods.

        The embargo gap prevents information leakage when labels have a horizon.
        For example, if label_horizon=10, there should be a 10-sample gap
        between train and test to avoid using same information for both.

        Based on: Bailey et al. (2014) and financial ML best practices

        Args:
            X: Feature DataFrame
            y: Target Series
            label_horizon: Label calculation horizon (default: 0)

        Returns:
            Optimal embargo gap in samples
        """
        logging.info(f'\n{"="*70}')
        logging.info('[C6-EMBARGO-FIX] Universal Embargo Gap Calculation')
        logging.info(f'{"="*70}')

        gap_acf = self.calculate_adaptive_gap(X, y, label_horizon)

        # Method 2: Minimum required gap
        gap_from_label = label_horizon * 3 if label_horizon > 0 else 0
        gap_from_percent = int(0.02 * len(y))  # 2% of dataset
        gap_min = max(gap_from_label, gap_from_percent, 10)

        # Use the larger of ACF-based and minimum required
        embargo_gap = max(gap_acf, gap_min)

        # Limit: Should not exceed 10% of dataset
        max_gap = int(0.1 * len(y))
        embargo_gap = min(embargo_gap, max_gap)

        logging.info(f'[C6-EMBARGO-FIX] ACF-based gap: {gap_acf}')
        logging.info(f'[C6-EMBARGO-FIX] Label horizon contribution: {gap_from_label}')
        logging.info(f'[C6-EMBARGO-FIX] Percentage contribution (2%): {gap_from_percent}')
        logging.info(f'[C6-EMBARGO-FIX] Final gap (maximum): {embargo_gap}')
        logging.info(f'[C6-EMBARGO-FIX] Gap / Dataset ratio: {embargo_gap / len(y):.2%}')
        logging.info(f'{"="*70}\n')

        return int(embargo_gap)
    def check_stationarity_adf(
        self,
        X: pd.DataFrame,
        significance_level: float = 0.05,
        max_features_to_check: int = 100
    ) -> Dict[str, any]:
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logging.warning('[STATIONARITY] statsmodels not available - skipping ADF test')
            return {'skipped': True, 'reason': 'statsmodels not installed'}
        logging.info(f'[STATIONARITY] Running ADF test on features (significance={significance_level})')
        non_stationary_features = []
        stationary_features = []
        failed_tests = []
        features_to_check = X.columns[:max_features_to_check] if len(X.columns) > max_features_to_check else X.columns
        for col in features_to_check:
            try:
                series = X[col].dropna()
                if len(series) < 20:
                    failed_tests.append(col)
                    continue
                if series.std() < 1e-8:
                    failed_tests.append(col)
                    continue
                result = adfuller(series, autolag='AIC')
                adf_statistic = result[0]
                p_value = result[1]
                if p_value < significance_level:
                    stationary_features.append(col)
                else:
                    non_stationary_features.append(col)
            except Exception as e:
                failed_tests.append(col)
                logging.debug(f'[STATIONARITY] ADF test failed for {col}: {e}')
        total_checked = len(stationary_features) + len(non_stationary_features)
        stationary_pct = len(stationary_features) / total_checked * 100 if total_checked > 0 else 0
        logging.info(f'[STATIONARITY] Results:')
        logging.info(f'  - Stationary: {len(stationary_features)}/{total_checked} ({stationary_pct:.1f}%)')
        logging.info(f'  - Non-stationary: {len(non_stationary_features)}/{total_checked}')
        logging.info(f'  - Failed tests: {len(failed_tests)}')
        if len(non_stationary_features) > 0:
            logging.warning(f'[STATIONARITY] WARNING: {len(non_stationary_features)} non-stationary features detected')
            logging.warning(f'[STATIONARITY] Consider: differencing, fractional differentiation, or regime-based modeling')
            if len(non_stationary_features) <= 10:
                logging.warning(f'[STATIONARITY] Non-stationary features: {non_stationary_features[:10]}')
        return {
            'stationary_count': len(stationary_features),
            'non_stationary_count': len(non_stationary_features),
            'failed_count': len(failed_tests),
            'stationary_percentage': stationary_pct,
            'non_stationary_features': non_stationary_features[:20],
            'checked_features': total_checked
        }
    def temporal_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        gap: int = None,
        label_horizon: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        n = len(X)
        train_size = int(n * (1 - self.test_size_ratio))
        detected_window = self.detect_feature_windows(X)
        label_h = label_horizon if label_horizon is not None else (self.label_horizon or 0)
        min_gap = max(
            detected_window * 2,
            label_h * 3 if label_h > 0 else 0,
            int(n * 0.05),
            100
        )
        if gap is None:
            gap = min_gap
            logging.info(f'[GAP] Auto-calculated: {gap} samples ({gap/n*100:.1f}% of {n})')
            logging.info(f'[GAP] Components: window={detected_window}*2, horizon={label_h}*3, 5%={int(n*0.05)}, min=100')
        elif gap < min_gap:
            logging.warning(f'[C3-EMBARGO] Increasing gap: {gap} -> {min_gap} for data integrity')
            gap = min_gap
        X_train = X.iloc[:train_size].copy()
        y_train = y.iloc[:train_size].copy()
        test_start = min(train_size + gap, n)
        X_test = X.iloc[test_start:].copy()
        y_test = y.iloc[test_start:].copy()
        if 'label_end_time' in X.columns or 'label_end_index' in X.columns:
            logging.info(f'[EMBARGO] Detected label metadata - validating no overlap')
            if 'label_end_index' in X.columns:
                leaked_rows = X_train['label_end_index'] >= train_size
                if leaked_rows.any():
                    n_leaked = leaked_rows.sum()
                    logging.warning(f'[EMBARGO] Removing {n_leaked} train rows with labels overlapping test period')
                    X_train = X_train[~leaked_rows]
                    y_train = y_train[~leaked_rows]
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
        max_bin = int(self.base_params.get('max_bin', 255))
        total_bins = n_features * max_bin
        ratio = n_features / max(1, n_samples)
        if n_features > 1000 or ratio > 0.5:
            strategy = 'col_wise'
            reason = 'high-dimensional data (many features)'
        elif n_samples > 50000 and ratio < 0.1:
            strategy = 'row_wise'
            reason = 'large dataset with fewer features'
        elif total_bins > 100000:
            strategy = 'col_wise'
            reason = 'large total bins'
        elif self.num_threads and self.num_threads > 8 and n_features > 100:
            strategy = 'col_wise'
            reason = 'multi-threaded with moderate features'
        else:
            strategy = 'col_wise'
            reason = 'default (safer for reproducibility)'
        config_key = f"{strategy}_{n_samples}_{n_features}"
        if not hasattr(self, '_logged_hist_configs'):
            self._logged_hist_configs = set()
        if config_key not in self._logged_hist_configs:
            logging.debug(f"[HIST] Histogram strategy: {strategy} ({reason}), samples={n_samples}, features={n_features}, ratio={ratio:.4f}")
            self._logged_hist_configs.add(config_key)
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
        logging.debug(f"Optimal min_data_in_leaf: {min_leaf} (samples={n_samples}, leaves={num_leaves})")
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
        logging.debug(f"Feature sampling: tree={configs['feature_fraction']}, node={configs['feature_fraction_bynode']} (features={n_features})")
        return configs
    def _optimize_for_large_feature_sets(self, n_features: int, n_samples: int) -> Dict:
        optimizations = {}
        if n_features < 1000:
            return {
                'shap_sample_size': self.shap_sample_size,
                'null_importance_n_rounds': 50,
                'stability_selection_iterations': self.stability_selection_iterations,
                'n_bootstrap_ci': self.n_bootstrap_ci,
                'use_batch_processing': False
            }
        logging.info(f'\n{"="*70}')
        logging.info('[LARGE-SCALE OPTIMIZATION] Adjusting parameters for {n_features} features')
        logging.info(f'{"="*70}')
        if n_features >= 5000:
            shap_sample = min(500, n_samples // 4)
            optimizations['shap_sample_size'] = shap_sample
            logging.info(f'  SHAP sample size: {self.shap_sample_size} -> {shap_sample} (memory reduction)')
        elif n_features >= 3000:
            shap_sample = min(800, n_samples // 3)
            optimizations['shap_sample_size'] = shap_sample
            logging.info(f'  SHAP sample size: {self.shap_sample_size} -> {shap_sample}')
        elif n_features >= 1500:
            shap_sample = min(1000, n_samples // 2)
            optimizations['shap_sample_size'] = shap_sample
            logging.info(f'  SHAP sample size: {self.shap_sample_size} -> {shap_sample}')
        else:
            optimizations['shap_sample_size'] = self.shap_sample_size
        if n_features >= 5000:
            null_rounds = 30
            optimizations['null_importance_n_rounds'] = null_rounds
            logging.info(f'  Null Importance rounds: 50 -> {null_rounds} (time optimization)')
        elif n_features >= 3000:
            null_rounds = 40
            optimizations['null_importance_n_rounds'] = null_rounds
            logging.info(f'  Null Importance rounds: 50 -> {null_rounds}')
        else:
            optimizations['null_importance_n_rounds'] = 50
        if n_features >= 5000:
            stab_iters = max(20, self.stability_selection_iterations // 2)
            optimizations['stability_selection_iterations'] = stab_iters
            logging.info(f'  Stability Selection iterations: {self.stability_selection_iterations} -> {stab_iters}')
        elif n_features >= 3000:
            stab_iters = max(25, int(self.stability_selection_iterations * 0.7))
            optimizations['stability_selection_iterations'] = stab_iters
            logging.info(f'  Stability Selection iterations: {self.stability_selection_iterations} -> {stab_iters}')
        else:
            optimizations['stability_selection_iterations'] = self.stability_selection_iterations
        if n_features >= 5000:
            ci_bootstrap = max(30, self.n_bootstrap_ci // 2)
            optimizations['n_bootstrap_ci'] = ci_bootstrap
            logging.info(f'  CI bootstrap samples: {self.n_bootstrap_ci} -> {ci_bootstrap}')
        elif n_features >= 3000:
            ci_bootstrap = max(40, int(self.n_bootstrap_ci * 0.8))
            optimizations['n_bootstrap_ci'] = ci_bootstrap
            logging.info(f'  CI bootstrap samples: {self.n_bootstrap_ci} -> {ci_bootstrap}')
        else:
            optimizations['n_bootstrap_ci'] = self.n_bootstrap_ci
        if n_features >= 3000:
            optimizations['use_batch_processing'] = True
            optimizations['batch_size'] = min(1000, n_features // 3)
            logging.info(f'  Batch processing: ENABLED (batch_size={optimizations["batch_size"]})')
        else:
            optimizations['use_batch_processing'] = False
        if n_features >= 3000:
            optimizations['force_float32'] = True
            logging.info(f'  Force float32: ENABLED (memory optimization)')
        else:
            optimizations['force_float32'] = False
        if n_features >= 5000:
            optimizations['early_feature_filter_threshold'] = 0.001
            optimizations['progressive_filtering'] = True
            logging.info(f'  Progressive filtering: ENABLED (removes low-importance features early)')
        elif n_features >= 3000:
            optimizations['early_feature_filter_threshold'] = 0.0005
            optimizations['progressive_filtering'] = True
            logging.info(f'  Progressive filtering: ENABLED')
        else:
            optimizations['progressive_filtering'] = False
        logging.info(f'{"="*70}\n')
        return optimizations
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
                logging.info('Reduced max_bin: 255 -> 127')
            if num_leaves > 31:
                optimizations['num_leaves'] = 31
                logging.info('Reduced num_leaves: 80 -> 31')
            pool_size = self.base_params.get('histogram_pool_size', 1024)
            if pool_size > 512:
                optimizations['histogram_pool_size'] = 512
                logging.info('Reduced histogram_pool_size -> 512MB')
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
    def _fit_with_categorical(self, model, X, y, sample_weight=None, **kwargs):
        cat_cols = []
        if self.use_categorical:
            try:
                cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            except Exception:
                cat_cols = []
        try:
            if cat_cols:
                X_local = X.copy()
                for c in cat_cols:
                    try:
                        X_local[c] = X_local[c].astype('category')
                    except Exception:
                        pass
                model.fit(X_local, y, sample_weight=sample_weight, categorical_feature=cat_cols, **kwargs)
            else:
                model.fit(X, y, sample_weight=sample_weight, **kwargs)
        except TypeError:
            for c in cat_cols:
                try:
                    X[c] = X[c].astype('category')
                except Exception:
                    pass
            model.fit(X, y, sample_weight=sample_weight, **kwargs)
        return model
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
        logging.debug('Final model saving disabled')
        return None
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
        logging.debug('Smoke reproducibility test: %s', 'PASS' if same else 'FAIL')
        return bool(same)
    def compute_time_weighted_samples(
        self,
        y: pd.Series,
        label_horizon: int = None,
        event_times: pd.Series = None,
        use_overlap_weighting: bool = True
    ) -> np.ndarray:
        n = len(y)
        if label_horizon is None:
            label_horizon = self.label_horizon or 1
        overlap_weights = np.ones(n, dtype=np.float32)
        if use_overlap_weighting and label_horizon > 1:
            logging.debug(f'[CRITICAL-FIX-3] Computing overlap weights with horizon={label_horizon}')
            for i in range(n):
                overlap_count = 0
                for j in range(n):
                    if i != j:
                        distance = abs(i - j)
                        if distance < label_horizon:
                            overlap_count += 1
                overlap_weights[i] = 1.0 / (1.0 + overlap_count / label_horizon)
            logging.debug(f'[CRITICAL-FIX-3] Overlap weights: min={overlap_weights.min():.3f}, '
                         f'mean={overlap_weights.mean():.3f}, max={overlap_weights.max():.3f}')
        class_weights = np.ones(n, dtype=np.float32)
        if self.classification:
            try:
                class_weights = compute_sample_weight('balanced', y=y)
                logging.debug(f'[CRITICAL-FIX-3] Class weights computed')
            except Exception as e:
                logging.warning(f'[CRITICAL-FIX-3] Class weight computation failed: {e}')
        temporal_decay = self.temporal_decay if hasattr(self, 'temporal_decay') and self.temporal_decay else None
        half_life = self.sample_weight_half_life if hasattr(self, 'sample_weight_half_life') else None
        if half_life is not None and half_life > 0:
            temporal_decay = np.exp(-np.log(2) / half_life)
        if temporal_decay is not None and temporal_decay < 1.0:
            time_weights = np.power(temporal_decay, np.arange(n-1, -1, -1))
            logging.debug(f'[CRITICAL-FIX-3] Temporal decay: {temporal_decay:.4f}')
        else:
            time_weights = np.ones(n, dtype=np.float32)
        final_weights = overlap_weights * class_weights * time_weights
        final_weights = final_weights / final_weights.mean()
        final_weights = final_weights.astype(np.float32)
        logging.debug(f'[CRITICAL-FIX-3] Final weights: min={final_weights.min():.3f}, '
                     f'mean={final_weights.mean():.3f}, max={final_weights.max():.3f}')
        return final_weights
    def compute_sample_weights(self, y: pd.Series, temporal_decay: float = None, half_life: int = None) -> np.ndarray:
        """
        [C5-FIX] Improved sample weights with temporal decay, NO label_horizon leakage
        Lopez de Prado (2018): Sample weights based on recency, not future info
        """
        if half_life is None and hasattr(self, 'sample_weight_half_life'):
            half_life = self.sample_weight_half_life
        if temporal_decay is None and hasattr(self, 'temporal_decay'):
            temporal_decay = self.temporal_decay
        if half_life is not None and half_life > 0:
            temporal_decay = np.exp(-np.log(2) / half_life)
            logging.debug(f"[ISSUE#13-FIX] Using half_life={half_life} → decay={temporal_decay:.6f}")
        elif temporal_decay is None:
            temporal_decay = 0.95
        if not self.classification:
            n = len(y)
            time_weights = np.power(temporal_decay, np.arange(n-1, -1, -1))
            return (time_weights / time_weights.sum() * n).astype(np.float32)
        try:
            class_weights = compute_sample_weight('balanced', y=y)
            n = len(y)
            time_weights = np.power(temporal_decay, np.arange(n-1, -1, -1))
            combined = class_weights * time_weights
            combined = (combined / combined.sum() * n).astype(np.float32)
            logging.debug(f"[C5-FIX] Sample weights computed: class-balanced + temporal decay (NO label_horizon)")
            return combined
        except Exception as e:
            logging.warning(f'[C5-FIX] Sample weight computation failed: {str(e)} - using uniform weights')
            return np.ones(len(y), dtype=np.float32)

    def compute_sample_weights_by_uniqueness(
        self,
        y: pd.Series,
        label_times: Optional[pd.DataFrame] = None,
        temporal_decay: float = 0.95
    ) -> np.ndarray:
        """
        [C5-ADVANCED] Sample weights based on label uniqueness - Lopez de Prado (2018)

        Samples with fewer concurrent labels → higher weight
        This avoids using future information (label_horizon) which causes leakage.

        Args:
            y: Target series
            label_times: DataFrame with columns 't_start' and 't_end' for label concurrency
            temporal_decay: Exponential decay factor for recency weighting

        Returns:
            Sample weights array, normalized to mean=1.0
        """
        n = len(y)

        # Option 1: Simple overlap-based uniqueness (default)
        if label_times is None:
            logging.info("[C5-FIX] Using simple overlap-based uniqueness weighting")
            # Estimate overlap from temporal proximity
            overlap_weights = np.ones(n, dtype=np.float32)

            # Simple heuristic: penalize samples near duplicates
            for i in range(n):
                nearby_count = 0
                window = max(5, int(0.05 * n))  # 5% window
                for j in range(max(0, i-window), min(n, i+window)):
                    if i != j:
                        nearby_count += 1
                overlap_weights[i] = 1.0 / (1.0 + nearby_count / window)
        else:
            # Option 2: Exact overlap-based uniqueness using label_times
            logging.info("[C5-FIX] Using exact label uniqueness weighting")
            overlap_weights = np.ones(n, dtype=np.float32)

            for i in range(n):
                t_start_i = label_times.iloc[i]['t_start']
                t_end_i = label_times.iloc[i]['t_end']

                # Count overlapping labels
                overlaps = (
                    (label_times['t_start'] <= t_end_i) &
                    (label_times['t_end'] >= t_start_i)
                ).sum() - 1  # Exclude self

                overlap_weights[i] = 1.0 / (1.0 + overlaps)

        # Apply temporal decay (recent samples get higher weight)
        time_weights = np.power(temporal_decay, np.arange(n-1, -1, -1))

        # Combine uniqueness and temporal decay
        final_weights = overlap_weights * time_weights
        final_weights = final_weights / final_weights.mean()

        # Apply class balancing if classification
        if self.classification:
            try:
                class_weights = compute_sample_weight('balanced', y=y)
                final_weights = final_weights * class_weights
                final_weights = final_weights / final_weights.mean()
            except Exception as e:
                logging.warning(f"[C5-FIX] Class weight computation failed: {e}")

        final_weights = final_weights.astype(np.float32)
        logging.debug(f"[C5-FIX] Uniqueness weights: min={final_weights.min():.3f}, "
                     f"mean={final_weights.mean():.3f}, max={final_weights.max():.3f}")

        return final_weights

    def _calculate_optimal_shap_sample_size(self, n_features: int, n_samples: int) -> int:
            try:
                recommended = int(20 * np.sqrt(max(1, n_features)))
            except Exception:
                recommended = 1000
            max_allowed = max(1, int(0.3 * n_samples))
            min_required = min(int(0.5 * n_samples), 500)
            opt = max(min_required, recommended)
            opt = min(opt, max_allowed)
            logging.debug(f"SHAP sample size set to {opt} (recommended={recommended}, max_allowed={max_allowed})")
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
        if not hasattr(self, '_ts_cv_logged'):
            logging.info(f"Optimal TS-CV: n_splits={n_splits}, gap={gap}, test_size={test_size} (n_samples={n_samples})")
            self._ts_cv_logged = True
        return {'n_splits': n_splits, 'gap': gap, 'test_size': test_size}
    def _shap_interaction_for_top_features(self, X: pd.DataFrame, y: pd.Series, shap_mean: np.ndarray, n_top_features: int = 50) -> np.ndarray:
            import shap
            n_top = min(n_top_features, X.shape[1])
            top_idx = np.argsort(shap_mean)[-n_top:][::-1]
            top_feats = X.columns[top_idx]
            sample_size = min(300, len(X))
            start_idx = max(0, len(X) - sample_size)
            X_sample = X.iloc[start_idx: start_idx + sample_size][top_feats]
            y_sample = y.iloc[start_idx: start_idx + sample_size]
            if self.classification:
                model = lgb.LGBMClassifier(**self.base_params)
            else:
                model = lgb.LGBMRegressor(**self.base_params)
            self._fit_with_categorical(model, X_sample, y_sample)
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
        logging.debug('Detecting multicollinearity...')
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
                    if eigenvalues_min[0] < 1e-10:
                        logging.debug(f'Matrix near-singular (min eigenvalue={eigenvalues_min[0]:.2e})')
                        condition_index = np.inf
                    else:
                        condition_index = np.sqrt(np.float64(eigenvalues_max[0]) / np.float64(eigenvalues_min[0]))
                except Exception as e:
                    logging.debug(f'Sparse eigenvalue computation failed: {e}, falling back to full')
                    from scipy import linalg as scipy_linalg
                    eigenvalues = scipy_linalg.eigvals(corr_matrix)
                    eigenvalues = np.real(eigenvalues)
                    eigenvalues = eigenvalues[eigenvalues > 1e-10]
                    if len(eigenvalues) > 0:
                        min_eig = eigenvalues.min()
                        if min_eig < 1e-10:
                            condition_index = np.inf
                        else:
                            condition_index = np.sqrt(np.float64(eigenvalues.max()) / np.float64(min_eig))
                    else:
                        condition_index = np.inf
            else:
                eigenvalues = np.linalg.eigvals(corr_matrix)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                if len(eigenvalues) > 0:
                    min_eig = eigenvalues.min()
                    if min_eig < 1e-10:
                        condition_index = np.inf
                    else:
                        condition_index = np.sqrt(np.float64(eigenvalues.max()) / np.float64(min_eig))
                else:
                    condition_index = np.inf
            if np.isnan(condition_index):
                logging.warning('Condition index is NaN (matrix may be singular)')
                condition_index = np.inf
            elif np.isinf(condition_index):
                logging.debug('Matrix is singular or near-singular (condition index = inf)')
        except Exception as e:
            logging.warning(f'Condition index calculation failed: {str(e)}')
            condition_index = np.inf
        if np.isinf(condition_index):
            ci_display = "inf (singular/near-singular)"
        elif np.isnan(condition_index):
            ci_display = "nan (error)"
        else:
            ci_display = f"{condition_index:.2f}"
        logging.info(f'High correlation pairs: {len(high_corr_pairs)}, Condition Index: {ci_display}')
        logging.info(f'\n{"-"*70}')
        logging.info('[MULTICOLL] Detailed Analysis:')
        n_pairs_90 = int(np.sum(corr_matrix > 0.9)) // 2
        n_pairs_85 = int(np.sum(corr_matrix > 0.85)) // 2
        n_pairs_70 = int(np.sum(corr_matrix > 0.7)) // 2
        logging.info(f'  Correlation pairs:')
        logging.info(f'    Very high (>0.9): {n_pairs_90} pairs')
        logging.info(f'    High (>0.85): {n_pairs_85} pairs')
        logging.info(f'    Medium (>0.7): {n_pairs_70} pairs')
        if len(high_corr_pairs) > 0:
            logging.info(f'  Top correlated pairs:')
            for idx in range(min(5, len(high_corr_pairs))):
                pair_dict = high_corr_pairs[idx]
                feat1_name = str(pair_dict['feature1'])[:20]
                feat2_name = str(pair_dict['feature2'])[:20]
                corr_val = float(pair_dict['correlation'])
                logging.info(f'    {idx+1}. {feat1_name} <-> {feat2_name}: {corr_val:.3f}')
        if np.isinf(condition_index) or np.isnan(condition_index):
            logging.warning(f'  WARNING: Matrix singular/near-singular - SEVERE multicollinearity!')
        elif condition_index > 30:
            logging.warning(f'  WARNING: High condition index ({condition_index:.1f}) - Severe multicollinearity!')
        elif condition_index > 15:
            logging.warning(f'  Note: Moderate condition index ({condition_index:.1f}) - Some multicollinearity present')
        else:
            logging.info(f'  Status: Condition index acceptable - GOOD')
        n_features = len(feature_names)
        if len(high_corr_pairs) > n_features * 0.5:
            logging.warning(f'  Recommendation: Consider PCA or removing redundant features')
        elif len(high_corr_pairs) > n_features * 0.3:
            logging.warning(f'  Recommendation: Consider automatic redundant feature removal')
        logging.info(f'\n[HIGH-PRIORITY-FIX-4] Computing VIF (Variance Inflation Factor)...')
        vif_data = []
        high_vif_features = []
        max_features_for_vif = min(100, len(feature_names))
        if len(feature_names) > max_features_for_vif:
            variances = np.var(Xvalues, axis=0)
            top_var_indices = np.argsort(-variances)[:max_features_for_vif]
            X_for_vif = Xvalues[:, top_var_indices]
            features_for_vif = [feature_names[i] for i in top_var_indices]
            logging.info(f'  Computing VIF for top {max_features_for_vif} features (by variance)')
        else:
            X_for_vif = Xvalues
            features_for_vif = feature_names
        vif_threshold = self.vif_threshold if hasattr(self, 'vif_threshold') else 10.0
        for i, col_name in enumerate(features_for_vif):
            try:
                X_others = np.delete(X_for_vif, i, axis=1)
                y_target = X_for_vif[:, i]
                if np.std(y_target) < 1e-10:
                    vif = 1.0
                else:
                    X_with_intercept = np.column_stack([np.ones(len(X_others)), X_others])
                    try:
                        XtX = X_with_intercept.T @ X_with_intercept
                        Xty = X_with_intercept.T @ y_target
                        XtX += np.eye(XtX.shape[0]) * 1e-6
                        beta = np.linalg.solve(XtX, Xty)
                        y_pred = X_with_intercept @ beta
                        ss_res = np.sum((y_target - y_pred) ** 2)
                        ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
                        if ss_tot < 1e-10:
                            r_squared = 0.0
                        else:
                            r_squared = 1 - (ss_res / ss_tot)
                        if r_squared >= 0.9999:
                            vif = 1000.0
                        else:
                            vif = 1.0 / (1.0 - r_squared)
                    except np.linalg.LinAlgError:
                        vif = 1000.0
                vif_data.append({
                    'feature': col_name,
                    'vif': float(vif)
                })
                if vif > vif_threshold:
                    high_vif_features.append(col_name)
            except Exception as e:
                logging.debug(f'  VIF computation failed for {col_name}: {e}')
                vif_data.append({
                    'feature': col_name,
                    'vif': np.nan
                })
        vif_data_sorted = sorted(vif_data, key=lambda x: x['vif'] if not np.isnan(x['vif']) else 0, reverse=True)
        logging.info(f'  VIF Analysis (threshold={vif_threshold}):')
        logging.info(f'    High VIF features (>{vif_threshold}): {len(high_vif_features)}')
        if len(vif_data_sorted) > 0:
            logging.info(f'  Top VIF features:')
            for idx in range(min(10, len(vif_data_sorted))):
                feat = vif_data_sorted[idx]
                feat_name = str(feat['feature'])[:40]
                vif_val = feat['vif']
                if not np.isnan(vif_val):
                    if vif_val > vif_threshold:
                        logging.warning(f'    {idx+1}. {feat_name}: VIF={vif_val:.2f} [HIGH]')
                    else:
                        logging.info(f'    {idx+1}. {feat_name}: VIF={vif_val:.2f}')
        n_severe_vif = sum(1 for v in vif_data if not np.isnan(v['vif']) and v['vif'] > 10)
        n_moderate_vif = sum(1 for v in vif_data if not np.isnan(v['vif']) and 5 < v['vif'] <= 10)
        if n_severe_vif > 0:
            logging.warning(f'  WARNING: {n_severe_vif} features with severe multicollinearity (VIF>10)')
            logging.warning(f'  These features are highly redundant - consider removing them')
        if n_moderate_vif > 0:
            logging.info(f'  Note: {n_moderate_vif} features with moderate multicollinearity (5<VIF<=10)')
        logging.info(f'{"-"*70}')
        return {
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'high_corr_features': list(high_corr_features),
            'condition_index': float(condition_index),
            'vif_data': vif_data_sorted,
            'high_vif_features': high_vif_features,
            'n_severe_vif': n_severe_vif,
            'n_moderate_vif': n_moderate_vif
        }
    def remove_redundant_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        correlation_threshold: float = 0.95,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        logging.info(f'\n{"="*70}')
        logging.info('[REDUNDANT REMOVAL] Removing highly correlated features')
        logging.info(f'{"="*70}')
        logging.info(f'  Threshold: correlation > {correlation_threshold}')
        numeric_X = X.select_dtypes(include=[np.number])
        corr_matrix = np.abs(np.corrcoef(numeric_X.values.T).astype(np.float32))
        params = self._get_adaptive_params(X, y)
        params['n_estimators'] = 100
        sample_weights = self.compute_sample_weights(y)
        train_data = self._create_dataset(X, y, weight=sample_weights)
        model = self._train_with_fallback(
            params,
            train_data,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        feature_importance = model.feature_importance(importance_type=importance_type)
        n_features = len(X.columns)
        features_to_remove = set()
        for i in range(n_features):
            if X.columns[i] in features_to_remove:
                continue
            for j in range(i + 1, n_features):
                if X.columns[j] in features_to_remove:
                    continue
                if corr_matrix[i, j] > correlation_threshold:
                    if feature_importance[i] >= feature_importance[j]:
                        features_to_remove.add(X.columns[j])
                        logging.debug(f'  Removing {X.columns[j]} (corr={corr_matrix[i,j]:.3f} with {X.columns[i]}, '
                                     f'imp={feature_importance[j]:.1f} < {feature_importance[i]:.1f})')
                    else:
                        features_to_remove.add(X.columns[i])
                        logging.debug(f'  Removing {X.columns[i]} (corr={corr_matrix[i,j]:.3f} with {X.columns[j]}, '
                                     f'imp={feature_importance[i]:.1f} < {feature_importance[j]:.1f})')
                        break
        features_to_keep = [col for col in X.columns if col not in features_to_remove]
        X_filtered = X[features_to_keep]
        logging.info(f'  Removed {len(features_to_remove)} redundant features')
        logging.info(f'  Features: {len(X.columns)} -> {len(X_filtered.columns)} ({len(X_filtered.columns)/len(X.columns):.1%} retained)')
        if len(features_to_remove) > 0:
            logging.info(f'  Removed features (top 10): {list(features_to_remove)[:10]}')
        logging.info(f'{"="*70}\n')
        return X_filtered
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
                start_idx = max(0, len(X) - min(sample_size, len(X)))
                X_sample = X.iloc[start_idx: start_idx + min(sample_size, len(X))]
                y_sample = y.iloc[start_idx: start_idx + min(sample_size, len(X))]
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
        if self.shap_feature_perturbation != 'interventional':
            logging.warning(f"[LEAKAGE-FIX-6] Forcing SHAP to 'interventional' mode (was: {self.shap_feature_perturbation})")
            logging.warning(f"[LEAKAGE-FIX-6] tree_path_dependent is biased with correlated features (FastPD 2024)")
            self.shap_feature_perturbation = 'interventional'
        recommended = self._calculate_optimal_shap_sample_size(len(X.columns), len(X))
        sample_size = min(self.shap_sample_size or recommended, recommended)
        n_runs = max(1, int(n_runs))
        n_features = len(X.columns)
        try:
            from joblib import Parallel, delayed
            use_joblib = True
        except Exception:
            use_joblib = False
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        X_train_shap = X.iloc[:split_idx].copy()
        y_train_shap = y.iloc[:split_idx].copy()
        X_holdout_shap = X.iloc[split_idx:].copy()
        y_holdout_shap = y.iloc[split_idx:].copy()
        logging.info(f"[SHAP-FIX-ISSUE#6] Using holdout set: train={len(X_train_shap)}, holdout={len(X_holdout_shap)}")
        base_explainer = None
        if cache_explainer:
            try:
                logging.debug('Building SHAP base model on train subset (NOT full data)')
                base_sample_weights = self.compute_sample_weights(y_train_shap) if use_sample_weights else None
                if self.classification:
                    base_model = lgb.LGBMClassifier(**self.base_params)
                else:
                    base_model = lgb.LGBMRegressor(**self.base_params)
                self._fit_with_categorical(base_model, X_train_shap, y_train_shap, sample_weight=base_sample_weights)
                model_output_to_use = model_output
                if model_output_to_use is None:
                    if self.shap_feature_perturbation == 'tree_path_dependent':
                        model_output_to_use = 'raw'
                    else:
                        model_output_to_use = 'probability' if self.classification else 'raw'
                if self.shap_feature_perturbation == 'interventional':
                    rng_base = np.random.default_rng(self.random_state)
                    background_size = min(100, len(X_train_shap))
                    background_data = X_train_shap.iloc[-background_size:].copy()
                    base_explainer = shap.TreeExplainer(
                        base_model,
                        data=background_data,
                        feature_perturbation=self.shap_feature_perturbation,
                        model_output=model_output_to_use,
                        feature_names=X.columns.tolist() if add_feature_names else None
                    )
                else:
                    base_explainer = shap.TreeExplainer(
                        base_model,
                        feature_perturbation=self.shap_feature_perturbation,
                        model_output=model_output_to_use,
                        feature_names=X.columns.tolist() if add_feature_names else None
                    )
                expected_val = base_explainer.expected_value
                if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
                    logging.debug(f'SHAP expected_value (multiclass): {[float(v) for v in expected_val[:3]]}...')
                else:
                    ev = float(expected_val[0]) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val)
                    logging.debug(f'SHAP expected_value: {ev:.6f}')
            except Exception as e:
                logging.warning(f'Failed to build cached explainer: {e}')
                base_explainer = None
        def _single_shap_run(run_idx: int):
            seed = int(self.random_state + run_idx)
            rng_run = np.random.default_rng(seed)
            start_idx = max(0, len(X_holdout_shap) - min(sample_size, len(X_holdout_shap)))
            X_sample = X_holdout_shap.iloc[start_idx: start_idx + min(sample_size, len(X_holdout_shap))]
            y_sample = y_holdout_shap.iloc[start_idx: start_idx + min(sample_size, len(X_holdout_shap))]
            weights_sample = self.compute_sample_weights(y_sample) if use_sample_weights else None
            if base_explainer is None:
                model_params = self.base_params.copy()
                model_params.pop('random_state', None)
                if self.classification:
                    model = lgb.LGBMClassifier(**model_params, random_state=seed)
                else:
                    model = lgb.LGBMRegressor(**model_params, random_state=seed)
                self._fit_with_categorical(model, X_train_shap, y_train_shap, sample_weight=self.compute_sample_weights(y_train_shap) if use_sample_weights else None)
                model_output_to_use = model_output if model_output is not None else ('probability' if self.classification else 'raw')
                if self.shap_feature_perturbation == 'tree_path_dependent':
                    model_output_to_use = 'raw'
                if self.shap_feature_perturbation == 'interventional':
                    background_size = min(100, len(X_train_shap))
                    background_data = X_train_shap.iloc[-background_size:].copy()
                    explainer = shap.TreeExplainer(
                        model,
                        data=background_data,
                        feature_perturbation=self.shap_feature_perturbation,
                        model_output=model_output_to_use,
                        feature_names=X.columns.tolist() if add_feature_names else None
                    )
                else:
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
        logging.debug(f'SHAP stability - Median CV: {np.median(shap_cv):.6f}, Max CV: {np.max(shap_cv):.6f}')
        logging.info(f'\n{"-"*70}')
        logging.info('[SHAP] Detailed Analysis:')
        logging.info(f'  Value range: Min={np.min(shap_mean):.6f}, Max={np.max(shap_mean):.6f}, Ratio={np.max(shap_mean)/(np.min(shap_mean)+1e-10):.1f}x')
        n_unstable_20 = int(np.sum(shap_cv > 0.2))
        n_unstable_30 = int(np.sum(shap_cv > 0.3))
        logging.info(f'  Unstable features: CV>20%={n_unstable_20}, CV>30%={n_unstable_30}')
        median_cv = np.median(shap_cv)
        max_cv = np.max(shap_cv)
        logging.info(f'  CV distribution: Median={median_cv:.4f}, Max={max_cv:.4f}')
        if n_unstable_20 > n_features * 0.3:
            logging.warning(f'  WARNING: {n_unstable_20}/{n_features} features ({n_unstable_20/n_features:.1%}) have CV>20% - SHAP values unreliable!')
        elif n_unstable_20 > 0:
            logging.info(f'  Note: {n_unstable_20} features with moderate instability (CV>20%)')
        else:
            logging.info(f'  Status: All features stable - EXCELLENT')
        if max_cv > 0.5:
            logging.warning(f'  WARNING: Max CV={max_cv:.2f} detected - Some features extremely unstable!')
        logging.info(f'{"-"*70}')
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
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_actual: int = 15,
        n_null: int = 75,
        train_only: bool = True
    ) -> Dict:
        logging.info('Null Importance - TRAINING DATA ONLY (no leakage)')
        n_features = len(X_train.columns)
        actual_gain = np.zeros((n_actual, n_features), dtype=np.float32, order='F')
        actual_split = np.zeros((n_actual, n_features), dtype=np.float32, order='F')
        actual_cover = np.zeros((n_actual, n_features), dtype=np.float32, order='F')
        null_gain = np.zeros((n_null, n_features), dtype=np.float32, order='F')
        null_split = np.zeros((n_null, n_features), dtype=np.float32, order='F')
        null_cover = np.zeros((n_null, n_features), dtype=np.float32, order='F')
        params = self._get_adaptive_params(X_train, y_train)
        params['n_estimators'] = 500
        n_samples = len(X_train)
        val_size = int(0.15 * n_samples)
        gap_size = self._calculate_early_stopping_gap(n_samples)
        train_end = n_samples - val_size - gap_size
        X_tr = X_train.iloc[:train_end]
        y_tr = y_train.iloc[:train_end]
        val_start = train_end + gap_size
        X_val = X_train.iloc[val_start:]
        y_val = y_train.iloc[val_start:]
        w_train = self.compute_sample_weights(y_tr)
        w_val = self.compute_sample_weights(y_val)
        logging.debug(f'[NULL-IMP] Train/Val split with gap: train={len(X_tr)}, gap={gap_size}, val={len(X_val)}')
        try:
            ref_train = self._get_binned_reference(X_tr, cache_key=f'null_bins_train_{len(X_tr)}_{len(X_tr.columns)}')
            ref_val = self._get_binned_reference(X_val, cache_key=f'null_bins_val_{len(X_val)}_{len(X_val.columns)}')
        except Exception:
            ref_train = None
            ref_val = None
        for run in range(n_actual):
            train_data = self._create_dataset(X_tr, y_tr, weight=w_train, reference=ref_train)
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
            rng_null = np.random.default_rng(self.random_state + n_actual + run)
            block_size = max(1, int(np.sqrt(len(y_tr))))
            shift_amount = rng_null.integers(block_size, len(y_tr) - block_size)
            y_shuffled = np.roll(y_tr.values, shift_amount)
            X_tr_shuffled = X_tr.copy()
            for col in X_tr_shuffled.columns:
                col_shift = rng_null.integers(1, len(X_tr_shuffled))
                X_tr_shuffled[col] = np.roll(X_tr_shuffled[col].values, col_shift)
            train_data = self._create_dataset(X_tr_shuffled, y_shuffled, weight=w_train, reference=ref_train)
            val_shift = rng_null.integers(block_size, max(block_size + 1, len(y_val) - block_size))
            val_shuffled = np.roll(y_val.values, val_shift)
            X_val_shuffled = X_val.copy()
            for col in X_val_shuffled.columns:
                col_shift = rng_null.integers(1, len(X_val_shuffled))
                X_val_shuffled[col] = np.roll(X_val_shuffled[col].values, col_shift)
            val_data = self._create_dataset(X_val_shuffled, val_shuffled, weight=w_val, reference=ref_train)
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
            if n_features > 1000:
                alpha_adjusted = 0.01
            elif n_features > 500:
                alpha_adjusted = 0.02
            else:
                alpha_adjusted = 0.05
            logging.info(f'[NULL-IMP] Multiple testing correction: {n_features} features -> alpha={alpha_adjusted}')
            if HAS_STATSMODELS:
                rejected_gain_fdr, p_values_gain_fdr, _, _ = multipletests(
                    p_values_gain_raw, alpha=alpha_adjusted, method='fdr_bh', is_sorted=False
                )
                rejected_split_fdr, p_values_split_fdr, _, _ = multipletests(
                    p_values_split_raw, alpha=alpha_adjusted, method='fdr_bh', is_sorted=False
                )
                rejected_cover_fdr, p_values_cover_fdr, _, _ = multipletests(
                    p_values_cover_raw, alpha=alpha_adjusted, method='fdr_bh', is_sorted=False
                )
                p_values_gain = p_values_gain_fdr.astype(np.float32)
                p_values_split = p_values_split_fdr.astype(np.float32)
                p_values_cover = p_values_cover_fdr.astype(np.float32)
                logging.info(f'FDR Control (Benjamini-Hochberg): Applied with alpha={alpha_adjusted} - more powerful than Bonferroni')
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
        significance_threshold = alpha_adjusted if 'alpha_adjusted' in locals() else 0.05
        significant_gain = p_values_gain < significance_threshold
        significant_split = p_values_split < significance_threshold
        significant_cover = p_values_cover < significance_threshold
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
        logging.info(f'\n{"-"*70}')
        logging.info('[NULL-IMP] Detailed Analysis:')
        min_p = np.min(p_values_gain)
        median_p = np.median(p_values_gain)
        max_p = np.max(p_values_gain)
        logging.info(f'  P-value distribution: Min={min_p:.6f}, Median={median_p:.4f}, Max={max_p:.4f}')
        mean_z = np.mean(gain_z_score)
        max_z = np.max(gain_z_score)
        logging.info(f'  Z-score distribution: Mean={mean_z:.2f}, Max={max_z:.2f}')
        logging.info(f'  Significance levels: 90%={np.sum(above_90_gain)}, 95%={np.sum(above_95_gain)}, 99%={np.sum(above_99_gain)} features')
        sig_rate = np.sum(significant_gain) / n_features
        if sig_rate < 0.05:
            logging.warning(f'  WARNING: Very low significance rate ({sig_rate:.1%}) - Model may be weak or features uninformative!')
        elif sig_rate > 0.5:
            logging.warning(f'  WARNING: High significance rate ({sig_rate:.1%}) - Possible overfitting or data leakage!')
        else:
            logging.info(f'  Status: Significance rate {sig_rate:.1%} - NORMAL')
        logging.info(f'{"-"*70}')
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
        logging.debug('Boosting ensemble with optimized parameters')
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
        logging.debug('Feature fraction analysis')
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
    def per_feature_null_importance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_rounds: int = 30,
        importance_type: str = 'gain'
    ) -> Dict:
        logging.info(f'Per-Feature Null: n_rounds={n_rounds}')
        n_features = X_train.shape[1]
        params = self._get_adaptive_params(X_train, y_train)
        params['n_estimators'] = 150
        sw = self.compute_sample_weights(y_train)
        n_samples = len(X_train)
        val_size = int(0.15 * n_samples)
        gap_size = self._calculate_early_stopping_gap(n_samples)
        train_end = n_samples - val_size - gap_size
        X_tr = X_train.iloc[:train_end]
        y_tr = y_train.iloc[:train_end]
        val_start = train_end + gap_size
        X_val = X_train.iloc[val_start:]
        y_val = y_train.iloc[val_start:]
        w_tr = sw[:train_end] if sw is not None else None
        w_val = sw[val_start:] if sw is not None else None
        train_data = self._create_dataset(X_tr, y_tr, weight=w_tr)
        val_data = self._create_dataset(X_val, y_val, weight=w_val, reference=train_data)
        model = self._train_with_fallback(
            params, train_data, num_boost_round=150,
            valid_sets=[val_data],
            callbacks=[
                self._adaptive_early_stopping(n_estimators=150, context='per_feature_null'),
                lgb.log_evaluation(0)
            ]
        )
        actual_imp = model.feature_importance(importance_type=importance_type)
        null_imps = np.zeros((n_features, n_rounds), dtype=np.float32)
        for fidx in range(n_features):
            for ridx in range(n_rounds):
                Xs_tr = X_tr.copy()
                rng = np.random.default_rng(self.random_state + fidx * 1000 + ridx)
                Xs_tr.iloc[:, fidx] = rng.permutation(Xs_tr.iloc[:, fidx].values)
                td = self._create_dataset(Xs_tr, y_tr, weight=w_tr)
                vd = self._create_dataset(X_val, y_val, weight=w_val, reference=td)
                m = self._train_with_fallback(
                    params, td, num_boost_round=150,
                    valid_sets=[vd],
                    callbacks=[
                        self._adaptive_early_stopping(n_estimators=150, context='per_feature_null'),
                        lgb.log_evaluation(0)
                    ]
                )
                null_imps[fidx, ridx] = m.feature_importance(importance_type=importance_type)[fidx]
            if (fidx + 1) % 10 == 0:
                logging.debug(f'  {fidx + 1}/{n_features}')
        null_mean = null_imps.mean(axis=1)
        null_std = null_imps.std(axis=1)
        p_vals = np.array([(null_imps[i] >= actual_imp[i]).mean() for i in range(n_features)])
        z = (actual_imp - null_mean) / (null_std + 1e-10)
        if HAS_STATSMODELS:
            from statsmodels.stats.multitest import multipletests
            rej, padj, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
        else:
            rej = p_vals < (0.05 / n_features)
            padj = p_vals * n_features
        logging.info(f'Per-Feature: {np.sum(rej)}/{n_features} significant')
        return {
            'actual': actual_imp,
            'null_mean': null_mean,
            'p_values': p_vals,
            'p_adjusted': padj,
            'z_scores': z,
            'significant': rej,
            'n_significant': int(np.sum(rej))
        }
    def create_preprocessing_pipeline(
        self,
        impute_strategy: str = 'median',
        n_neighbors: int = 5,
        add_scaler: bool = True
    ) -> Pipeline:
        from sklearn.impute import SimpleImputer, KNNImputer
        steps = []
        if impute_strategy in ['mean', 'median', 'most_frequent']:
            steps.append(('imputer', SimpleImputer(strategy=impute_strategy)))
        elif impute_strategy == 'knn':
            steps.append(('knn_imputer', KNNImputer(n_neighbors=n_neighbors)))
        else:
            raise ValueError(f"Unknown: {impute_strategy}")
        if add_scaler:
            steps.append(('scaler', StandardScaler()))
        pipe = Pipeline(steps)
        logging.info(f'Pipeline: {impute_strategy} + {"scaler" if add_scaler else "no scaler"}')
        return pipe
    def final_test_evaluation(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        selected_features: List[str],
        model,
        classification: bool = True
    ) -> Dict:
        logging.info("="*70)
        logging.info("FINAL TEST SET EVALUATION (UNBIASED)")
        logging.info("="*70)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        if classification:
            from sklearn.metrics import classification_report
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            metrics = {
                'auc': roc_auc_score(y_test, y_pred_proba),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            logging.info(f"  AUC:       {metrics['auc']:.4f}")
            logging.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logging.info(f"  Precision: {metrics['precision']:.4f}")
            logging.info(f"  Recall:    {metrics['recall']:.4f}")
            logging.info(f"  F1:        {metrics['f1']:.4f}")
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
        else:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            logging.info(f"  MSE:  {metrics['mse']:.4f}")
            logging.info(f"  RMSE: {metrics['rmse']:.4f}")
            logging.info(f"  MAE:  {metrics['mae']:.4f}")
            logging.info(f"  R²:   {metrics['r2']:.4f}")
        logging.info("="*70)
        return metrics
    def train_with_monotone_constraints(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        monotone_features: Dict[str, int],
        params: Optional[Dict] = None,
        classification: bool = True
    ):
        constraints = []
        for col in X_train.columns:
            if col in monotone_features:
                constraints.append(monotone_features[col])
            else:
                constraints.append(0)
        if params is None:
            params = {}
        params.update({
            'monotone_constraints': constraints,
            'monotone_constraints_method': 'advanced',
            'monotone_penalty': 0.0
        })
        if classification:
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        self._fit_with_categorical(model, X_train, y_train)
        logging.info(f"Trained with {len([c for c in constraints if c != 0])} monotone constraints")
        return model
    def adversarial_validation(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        n_splits: int = 3
    ) -> Dict:
        logging.info('Adversarial validation: train vs TEST distribution (C3-2 FIX)')
        assert isinstance(X_train, pd.DataFrame), "X_train must be DataFrame"
        if X_test is None:
            raise ValueError(
                "X_test MUST be provided! Cannot split X_train - creates false negative shift signal. "
                "Pass real test data to compare distributions properly."
            )
        assert isinstance(X_test, pd.DataFrame), "X_test must be DataFrame"
        logging.info(f'Adversarial validation: train ({len(X_train)}) vs test ({len(X_test)}) samples')
        X_early = X_train.copy()
        X_late = X_test.copy()
        X_early['is_test'] = 0
        X_late['is_test'] = 1
        X_combined = pd.concat([X_early, X_late], axis=0, ignore_index=True)
        y_combined = X_combined['is_test'].values
        X_combined = X_combined.drop('is_test', axis=1)
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
        cv = TimeSeriesSplit(n_splits=n_splits)
        params = self.base_params.copy()
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['n_estimators'] = 200
        auc_scores = []
        model = None
        for train_idx, val_idx in cv.split(X_combined):
            X_cv_train = X_combined.iloc[train_idx]
            X_cv_val = X_combined.iloc[val_idx]
            y_cv_train = y_combined[train_idx]
            y_cv_val = y_combined[val_idx]
            if len(np.unique(y_cv_val)) < 2:
                continue
            try:
                model_skl = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k in lgb.LGBMClassifier().get_params().keys()})
                self._fit_with_categorical(model_skl, X_cv_train, y_cv_train)
                y_pred = model_skl.predict_proba(X_cv_val)[:, 1]
                auc_val = roc_auc_score(y_cv_val, y_pred)
                auc_scores.append(auc_val)
            except Exception as e:
                logging.debug(f'Adversarial CV fold failed: {e}')
                continue
        try:
            model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k in lgb.LGBMClassifier().get_params().keys()})
            self._fit_with_categorical(model, X_combined, y_combined)
        except Exception:
            model = None
        if model is None:
            adv_importance = np.zeros(X_combined.shape[1], dtype=np.float32)
        elif hasattr(model, 'feature_importances_'):
            adv_importance = np.array(model.feature_importances_, dtype=np.float32)
        elif hasattr(model, 'booster_'):
            try:
                adv_importance = model.booster_.feature_importance(importance_type='gain').astype(np.float32)
            except Exception:
                adv_importance = np.array(getattr(model, 'feature_importance', lambda importance_type=None: np.zeros(X_combined.shape[1]))(importance_type='gain'), dtype=np.float32)
        elif hasattr(model, 'feature_importance'):
            adv_importance = np.array(model.feature_importance(importance_type='gain'), dtype=np.float32)
        else:
            adv_importance = np.zeros(X_combined.shape[1], dtype=np.float32)
        adv_importance_normalized = adv_importance / (adv_importance.sum() + 1e-10)
        shift_analysis = {}
        high_shift_features = []
        for i, col in enumerate(X_combined.columns):
            train_data = X_train[col].dropna()
            test_data = X_test[col].dropna()
            mean_shift = abs(train_data.mean() - test_data.mean()) / (train_data.std() + 1e-8)
            var_shift = abs(train_data.std() - test_data.std()) / (train_data.std() + 1e-8)
            shift_analysis[col] = {
                'mean_shift': float(mean_shift),
                'var_shift': float(var_shift),
                'importance': float(adv_importance_normalized[i])
            }
            if adv_importance_normalized[i] > 0.05 and mean_shift > 2.0:
                high_shift_features.append(col)
        logging.info(f'High shift features (per-feature analysis): {len(high_shift_features)}/{len(X_combined.columns)}')
        from sklearn.metrics import roc_auc_score
        y_test_adv = y_combined[-int(len(X_combined) * (1 - 0.8)):] if len(auc_scores) > 0 else y_combined
        X_test_adv = X_combined.iloc[-int(len(X_combined) * (1 - 0.8)):] if len(auc_scores) > 0 else X_combined
        unique_classes = np.unique(y_test_adv)
        if len(unique_classes) < 2:
            logging.warning(f'  AUC cannot be computed: only {unique_classes} class(es) in validation set')
            auc_score = np.nan
        else:
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_adv)[:, 1]
                elif hasattr(model, 'predict'):
                    y_pred_proba = model.predict(X_test_adv)
                elif hasattr(model, 'booster_'):
                    y_pred_proba = model.booster_.predict(X_test_adv)
                else:
                    y_pred_proba = model.predict(X_test_adv)
                if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                    logging.warning('  NaN/Inf detected in predictions, AUC cannot be computed reliably')
                    auc_score = np.nan
                else:
                    if y_pred_proba.ndim == 1 and set(np.unique(y_pred_proba)) <= {0, 1}:
                        logging.warning('  Predictions appear to be class labels; using them as scores for AUC')
                        y_pred_proba = y_pred_proba.astype(float)
                    auc_score = roc_auc_score(y_test_adv, y_pred_proba)
            except Exception as e:
                logging.warning(f'  AUC computation failed: {e}')
                auc_score = np.nan
        logging.info(f'\n{"-"*70}')
        logging.info('[ADV-VAL] Detailed Analysis:')
        if np.isnan(auc_score):
            logging.warning(f'  Model AUC: NaN (cannot compute - check class balance)')
            logging.info(f'  Validation set classes: {unique_classes}')
        else:
            logging.info(f'  Model AUC: {auc_score:.4f}')
        logging.info(f'  Per-Feature Shift Report:')
        for col in high_shift_features[:10]:
            analysis = shift_analysis[col]
            logging.info(f'    {col[:40]}: mean_shift={analysis["mean_shift"]:.2f}, var_shift={analysis["var_shift"]:.2f}, importance={analysis["importance"]:.4f}')
        if np.isnan(auc_score) or auc_score < 0.55:
            logging.info(f'  Status: [OK] EXCELLENT - No distribution shift (AUC < 0.55)')
            logging.info(f'  -> AUC ~0.5 means train/test indistinguishable (ideal)')
        elif auc_score < 0.65:
            logging.info(f'  Status: [OK] GOOD - Minor shift (0.55 <= AUC < 0.65)')
            logging.info(f'  -> Shift is minor, unlikely to affect performance')
        elif auc_score < 0.75:
            logging.warning(f'  Status: [WARNING] MODERATE shift (0.65 <= AUC < 0.75)')
            logging.warning(f'  -> Consider: temporal validation, shift-aware weighting, or remove high-shift features')
        elif auc_score < 0.85:
            logging.error(f'  Status: [CRITICAL] HIGH SHIFT - Model can distinguish train/test (0.75 <= AUC < 0.85)')
            logging.error(f'  -> RISK: Performance may degrade. Apply domain adaptation or remove problematic features')
        else:
            logging.error(f'  Status: [CRITICAL] SEVERE SHIFT - Severe mismatch (AUC >= 0.85)')
            logging.error(f'  -> DO NOT DEPLOY: Model will likely fail. Investigate data collection process')
        n_shift = int(len(high_shift_features))
        if n_shift > 0:
            top_shift_idx = np.argsort(adv_importance_normalized)[-min(5, len(X_combined.columns)):][::-1]
            logging.info(f'  Top importance features:')
            for i, idx in enumerate(top_shift_idx[:5], 1):
                feat_name = X_combined.columns[idx]
                imp_val = adv_importance_normalized[idx]
                logging.info(f'    {i}. {feat_name[:45]}: {imp_val:.4f}')
        if auc_score >= 0.75:
            logging.warning(f'  Recommendation: Remove {len(high_shift_features)} high-shift features or apply domain adaptation')
        logging.info(f'{"-"*70}')
        high_shift = np.array([col in high_shift_features for col in X_combined.columns], dtype=bool)
        del X_early, X_late, X_combined
        gc.collect()
        return {
            'adv_importance': adv_importance_normalized,
            'high_shift': high_shift,
            'high_shift_features': high_shift_features,
            'auc_score': auc_score
        }
    def mitigate_distribution_shift(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        adv_results: Dict,
        auc_threshold: float = 0.75,
        remove_high_shift: bool = False,
        shift_importance_threshold: float = 0.15
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        auc_score = adv_results.get('auc_score', 0.5)
        if np.isnan(auc_score):
            logging.info(f'[SHIFT-MITIGATE] AUC is NaN - skipping mitigation (likely class imbalance in validation)')
            return X, self.compute_sample_weights(y)
        if auc_score < auc_threshold:
            logging.info(f'[SHIFT-MITIGATE] No severe shift detected (AUC={auc_score:.4f} < {auc_threshold}), skipping mitigation')
            return X, self.compute_sample_weights(y)
        logging.info(f'\n{"="*70}')
        logging.info('[SHIFT-MITIGATE] Applying Distribution Shift Mitigation')
        logging.info(f'{"="*70}')
        logging.info(f'  Detected AUC: {auc_score:.4f} (>= {auc_threshold} SEVERE)')
        adv_importance = adv_results['adv_importance']
        high_shift = adv_results['high_shift']
        if len(adv_importance) != len(X.columns):
            logging.warning(f'  WARNING: adv_importance length ({len(adv_importance)}) != X.columns ({len(X.columns)})')
            logging.warning(f'  Truncating/padding adv_importance to match X dimensions')
            if len(adv_importance) > len(X.columns):
                adv_importance = adv_importance[:len(X.columns)]
                high_shift = high_shift[:len(X.columns)]
            else:
                adv_importance = np.pad(adv_importance, (0, len(X.columns) - len(adv_importance)), 'constant', constant_values=0)
                high_shift = np.pad(high_shift, (0, len(X.columns) - len(high_shift)), 'constant', constant_values=False)
        critical_shift_features = adv_importance > shift_importance_threshold
        n_critical = np.sum(critical_shift_features)
        if n_critical > 0:
            critical_idx = np.where(critical_shift_features)[0]
            logging.info(f'  Critical shifting features: {n_critical}')
            for i, idx in enumerate(critical_idx[:5], 1):
                if idx < len(X.columns):
                    feat_name = X.columns[idx]
                    imp_val = adv_importance[idx]
                    logging.info(f'    {i}. {feat_name[:40]}: importance={imp_val:.4f}')
                else:
                    logging.warning(f'    {i}. Index {idx} out of bounds, skipping')
            if remove_high_shift:
                keep_mask = ~critical_shift_features
                X_mitigated = X.iloc[:, keep_mask]
                logging.info(f'  Strategy: REMOVED {n_critical} high-shift features')
                logging.info(f'  Features: {len(X.columns)} -> {len(X_mitigated.columns)}')
            else:
                X_mitigated = X.copy()
                logging.info(f'  Strategy: KEEPING features, using sample reweighting')
        else:
            X_mitigated = X.copy()
            logging.info(f'  No critical features above threshold {shift_importance_threshold}')
        n = len(X)
        train_size = int(n * 0.5)
        from sklearn.model_selection import TimeSeriesSplit
        X_early = X.iloc[:train_size]
        X_late = X.iloc[train_size:]
        y_adv = np.concatenate([np.zeros(len(X_early)), np.ones(len(X_late))])
        X_combined = pd.concat([X_early, X_late], ignore_index=True)
        est_params = self.base_params.copy()
        est_params.update({'n_estimators': 100, 'num_threads': self.num_threads})
        if self.classification:
            adv_model = lgb.LGBMClassifier(**est_params)
        else:
            adv_model = lgb.LGBMClassifier(**est_params)
        self._fit_with_categorical(adv_model, X_combined, y_adv)
        adv_proba = adv_model.predict_proba(X_mitigated)[:, 1]
        shift_scores = np.abs(adv_proba - 0.5) * 2
        sample_weights = 1.0 / (1.0 + shift_scores * 3)
        sample_weights = sample_weights / np.mean(sample_weights)
        base_weights = self.compute_sample_weights(y)
        final_weights = sample_weights * base_weights
        final_weights = final_weights / np.mean(final_weights)
        logging.info(f'  Sample weights: Min={np.min(final_weights):.3f}, '
                    f'Mean={np.mean(final_weights):.3f}, Max={np.max(final_weights):.3f}')
        logging.info(f'  Mitigation complete')
        logging.info(f'{"="*70}\n')
        del X_combined, adv_model
        gc.collect()
        return X_mitigated, final_weights
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
        if self.enable_metadata_routing:
            try:
                base_estimator.set_fit_request(sample_weight=True)
            except AttributeError:
                logging.debug('Estimator does not support metadata routing (sklearn < 1.6)')
        gap_calc = self.calculate_adaptive_gap(X, y, label_horizon=self.label_horizon if hasattr(self, 'label_horizon') else 0)
        logging.info(f"[ISSUE#7-FIX] RFE gap calculated with label_horizon={self.label_horizon if hasattr(self, 'label_horizon') else 0}: gap={gap_calc}")
        cv_splitter = TimeSeriesSplit(n_splits=3, gap=gap_calc)
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
                logging.info(f'Phase 1: {n_total} -> {np.sum(rfecv_phase1.support_)} features')
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
                logging.info(f'Phase 2: {len(X_phase2.columns)} -> {np.sum(rfecv_phase2.support_)} features')
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
        try:
            adaptive_gap = self.calculate_adaptive_gap(X, y, label_horizon=self.label_horizon if hasattr(self, 'label_horizon') else 0)
            gap = adaptive_gap
            logging.info(f"[ISSUE#7-FIX] CV gap calculated with label_horizon={self.label_horizon if hasattr(self, 'label_horizon') else 0}: gap={gap}")
        except Exception as e:
            logging.warning(f"[ISSUE#7-FIX] Gap calculation failed: {e}, using default")
            pass
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
        cv_n_jobs, model_n_threads = self._parallel_cv_strategy(n_splits=n_splits, n_threads_available=self.num_threads)
        n_features = len(X.columns)
        gain_importances = np.zeros((n_splits, n_features), dtype=np.float32, order='F')
        split_importances = np.zeros((n_splits, n_features), dtype=np.float32, order='F')
        perm_importances = np.zeros((n_splits, n_features), dtype=np.float32, order='F')
        cv_scores = np.zeros(n_splits, dtype=np.float32)
        params = self._get_adaptive_params(X, y)
        if cv_n_jobs > 1:
            from joblib import Parallel, delayed
            def _run_fold(fold, train_idx, val_idx):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]
                weights_train = self.compute_sample_weights(y_train)
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
                weights_train = self.compute_sample_weights(y_train)
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
    def normalize_with_stability(self, scores, epsilon=1e-10, min_threshold=1e-6):
        if isinstance(scores, np.ndarray):
            if scores.dtype != np.float32:
                scores_array = scores.astype(np.float32)
            else:
                scores_array = scores.copy()
        else:
            scores_array = np.asarray(scores, dtype=np.float32)
        np.abs(scores_array, out=scores_array)
        total = scores_array.sum()
        if total < min_threshold:
            logging.warning(f"[NORMALIZE] All importances summed to {total:.2e} < threshold {min_threshold} - returning zeros")
            return np.zeros_like(scores_array, dtype=np.float32)
        normalized = scores_array / (total + epsilon)
        noise_floor = min_threshold / len(scores_array)
        normalized[normalized < noise_floor] = 0.0
        normalized = normalized / (normalized.sum() + epsilon)
        return normalized.astype(np.float32)
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
        block_size = max(1, int(np.sqrt(n_samples)))
        logging.info(f"[H8-FIX] Using block bootstrap with block_size={block_size} (√n_samples)")
        for b in range(n_bootstrap):
            rng_b = np.random.default_rng(self.random_state + b * 1000)
            n_blocks = int(np.ceil(n_samples / block_size))
            block_starts = rng_b.choice(max(1, n_samples - block_size), size=n_blocks, replace=True)
            boot_idx = []
            for start in block_starts:
                end = min(start + block_size, n_samples)
                boot_idx.extend(range(start, end))
            boot_idx = boot_idx[:n_samples]
            X_boot = X.iloc[boot_idx].reset_index(drop=True)
            y_boot = y.iloc[boot_idx].reset_index(drop=True)
            w_boot = sample_weights[boot_idx] if sample_weights is not None else None
            val_size = int(0.15 * n_samples)
            gap_size = self._calculate_early_stopping_gap(n_samples)
            train_end = n_samples - val_size - gap_size
            X_train_b = X_boot.iloc[:train_end]
            y_train_b = y_boot.iloc[:train_end]
            val_start = train_end + gap_size
            X_val_b = X_boot.iloc[val_start:]
            y_val_b = y_boot.iloc[val_start:]
            w_train_b = w_boot[:train_end] if w_boot is not None else None
            w_val_b = w_boot[val_start:] if w_boot is not None else None
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
        logging.info(f'[H8-FIX] CI computed with block bootstrap: Mean width={np.mean(ci_width):.4f}, Significant features: {np.sum(significant_features)}/{n_features}')
        logging.info(f'\n{"-"*70}')
        logging.info('[CONFIDENCE-INT] Detailed Analysis:')
        n_narrow = int(np.sum(ci_width < 100))
        n_wide = int(np.sum(ci_width > 500))
        logging.info(f'  CI Width distribution:')
        logging.info(f'    Narrow (<100): {n_narrow} features (high confidence)')
        logging.info(f'    Wide (>500): {n_wide} features (low confidence)')
        zero_crossing = (ci_lower < 0) & (ci_upper > 0)
        n_zero_cross = int(np.sum(zero_crossing))
        logging.info(f'  Zero-crossing features: {n_zero_cross} (CI includes zero -> not significant)')
        if n_wide > n_features * 0.3:
            logging.warning(f'  WARNING: {n_wide}/{n_features} features ({n_wide/n_features:.1%}) have wide CIs - High uncertainty!')
        elif n_wide > 0:
            logging.info(f'  Note: {n_wide} features with moderate uncertainty')
        else:
            logging.info(f'  Status: All features have narrow CIs - EXCELLENT')
        logging.info(f'{"-"*70}')
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
        logging.debug('Computing Mutual Information scores...')
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
        logging.info(f'\n{"-"*70}')
        logging.info('[MI] Detailed Analysis:')
        n_high = int(np.sum(mi_scores > 0.005))
        n_low = int(np.sum(mi_scores < 0.001))
        logging.info(f'  MI distribution:')
        logging.info(f'    High (>0.005): {n_high} features (strong non-linear relationship)')
        logging.info(f'    Low (<0.001): {n_low} features (weak relationship)')
        if n_high > 0:
            top_mi_idx = np.argsort(mi_scores)[-min(5, n_high):][::-1]
            logging.info(f'  Top MI features:')
            for i, idx in enumerate(top_mi_idx, 1):
                feat_name = X.columns[idx]
                mi_val = mi_scores[idx]
                logging.info(f'    {i}. {feat_name[:45]}: {mi_val:.6f}')
        if n_high < n_features * 0.1:
            logging.warning(f'  Note: Few high-MI features - Relationships may be mostly linear')
        logging.info(f'{"-"*70}')
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
        threshold: Optional[float] = None
    ) -> Dict:
        logging.info(f'\n{"="*70}')
        logging.info('[CRITICAL-FIX-2] STABILITY SELECTION - TEMPORAL BLOCKING')
        logging.info(f'{"="*70}')
        logging.info(f'Iterations: {n_iterations}, Sample fraction: {sample_fraction}')
        n_features = len(X.columns)
        n_samples = len(X)
        subsample_size = int(n_samples * sample_fraction)
        selection_frequency = np.zeros(n_features, dtype=np.float32)
        params = self._get_adaptive_params(X, y)
        params['n_estimators'] = 150
        top_k = min(max(int(np.sqrt(n_features)), 10), n_features // 2)
        adaptive_gap = self.calculate_adaptive_gap(X, y, self.label_horizon or 0)
        block_size = max(adaptive_gap * 2, n_samples // 20)
        logging.info(f'[CRITICAL-FIX-2] Adaptive gap: {adaptive_gap}, Block size: {block_size}')
        logging.info(f'[CRITICAL-FIX-2] Using TEMPORAL blocks (not random sampling)')
        for iteration in range(n_iterations):
            rng_iter = np.random.default_rng(self.random_state + iteration * 100)
            n_blocks = max(2, n_samples // block_size)
            n_blocks_to_select = max(1, int(n_blocks * sample_fraction))
            selected_block_ids = rng_iter.choice(n_blocks, size=n_blocks_to_select, replace=False)
            blocks = []
            for block_id in sorted(selected_block_ids):
                start_idx = block_id * block_size
                end_idx = min(start_idx + block_size, n_samples)
                block_indices = np.arange(start_idx, end_idx)
                blocks.append(block_indices)
            sub_idx = np.concatenate(blocks)
            sub_idx = np.unique(sub_idx)
            if len(sub_idx) < subsample_size * 0.9:
                available_blocks = np.setdiff1d(np.arange(n_blocks), selected_block_ids)
                if len(available_blocks) > 0:
                    extra_blocks_needed = (subsample_size - len(sub_idx)) // block_size + 1
                    extra_blocks = rng_iter.choice(
                        available_blocks,
                        size=min(extra_blocks_needed, len(available_blocks)),
                        replace=False
                    )
                    for block_id in sorted(extra_blocks):
                        start_idx = block_id * block_size
                        end_idx = min(start_idx + block_size, n_samples)
                        blocks.append(np.arange(start_idx, end_idx))
                    sub_idx = np.concatenate(blocks)
                    sub_idx = np.unique(sub_idx)
            sub_idx = np.sort(sub_idx)
            X_sub = X.iloc[sub_idx].reset_index(drop=True)
            y_sub = y.iloc[sub_idx].reset_index(drop=True)
            w_sub = self.compute_sample_weights(y_sub)
            val_size = int(0.15 * len(X_sub))
            gap_size = self._calculate_early_stopping_gap(len(X_sub))
            train_end = len(X_sub) - val_size - gap_size
            X_train_sub = X_sub.iloc[:train_end]
            y_train_sub = y_sub.iloc[:train_end]
            val_start = train_end + gap_size
            X_val_sub = X_sub.iloc[val_start:]
            y_val_sub = y_sub.iloc[val_start:]
            w_train_sub = w_sub[:train_end] if w_sub is not None else None
            w_val_sub = w_sub[val_start:] if w_sub is not None else None
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
            if (iteration + 1) % 10 == 0:
                gc.collect()
        selection_probability = selection_frequency / n_iterations
        if threshold is None:
            threshold = self._adaptive_stability_threshold(n_features, n_iterations=n_iterations)
        stable_features_mask = selection_probability >= threshold
        n_stable = np.sum(stable_features_mask)
        expected_false_discoveries = ((1 - threshold) / threshold) * (n_features - n_stable) if n_stable > 0 else 0
        logging.info(f'Stability Selection (temporal blocks): Stable features={n_stable}/{n_features} (threshold>={threshold:.3f}), Expected FD<={expected_false_discoveries:.2f}')
        logging.info(f'\n{"-"*70}')
        logging.info('[STAB-SEL] Detailed Analysis:')
        n_always = int(np.sum(selection_probability == 1.0))
        n_never = int(np.sum(selection_probability == 0.0))
        n_border = int(np.sum((selection_probability > 0.5) & (selection_probability < 0.7)))
        logging.info(f'  Selection probability distribution:')
        logging.info(f'    Always selected (p=1.0): {n_always}')
        logging.info(f'    Never selected (p=0.0): {n_never}')
        logging.info(f'    Borderline (0.5<p<0.7): {n_border}')
        logging.info(f'  Expected false discoveries: {expected_false_discoveries:.1f} features')
        actual_vs_expected = n_stable / max(expected_false_discoveries, 1)
        logging.info(f'  Actual/Expected ratio: {actual_vs_expected:.2f}')
        if n_stable > 0 and expected_false_discoveries > n_stable * 0.3:
            logging.warning(f'  WARNING: High expected FD rate ({expected_false_discoveries/n_stable:.1%}) - Low stability!')
        else:
            logging.info(f'  Status: FD rate acceptable - GOOD (temporal blocks)')
        logging.info(f'{"-"*70}')
        return {
            'selection_probability': selection_probability,
            'stable_features': stable_features_mask,
            'n_stable': int(n_stable),
            'threshold': threshold,
            'n_iterations': n_iterations,
            'sample_fraction': sample_fraction,
            'expected_false_discoveries': float(expected_false_discoveries)
        }
    def calculate_adaptive_gap(self, X: pd.DataFrame, y: pd.Series, label_horizon: int = 0) -> int:
        try:
            from statsmodels.tsa.stattools import acf
            acf_data = None
            if 'close' in X.columns:
                returns = X['close'].pct_change().dropna()
                acf_data = returns
                logging.debug("[H7-FIX] Using 'close' returns for ACF calculation")
            elif 'price' in X.columns:
                returns = X['price'].pct_change().dropna()
                acf_data = returns
                logging.debug("[H7-FIX] Using 'price' returns for ACF calculation")
            elif 'open' in X.columns:
                returns = X['open'].pct_change().dropna()
                acf_data = returns
                logging.debug("[H7-FIX] Using 'open' returns for ACF calculation")
            else:
                acf_data = y
                logging.warning("[H7-FIX] No price column found, using target for ACF (suboptimal)")
            if len(acf_data) < 10:
                logging.warning("[H7-FIX] Insufficient data for ACF, using heuristic gap")
                return max(50, int(0.05 * len(y)))
            max_lag_check = min(200, len(acf_data) // 2 - 1)
            if max_lag_check <= 1:
                return max(50, int(0.05 * len(y)))
            autocorr = acf(acf_data, nlags=max_lag_check, fft=True)
            significant_threshold = 0.05
            significant_lags = np.where(np.abs(autocorr[1:]) > significant_threshold)[0]
            if len(significant_lags) == 0:
                optimal_gap = max(10, int(np.sqrt(len(y))))
            else:
                last_significant = significant_lags[-1] + 1
                optimal_gap = last_significant * 2
            if label_horizon > 0:
                optimal_gap = max(optimal_gap, label_horizon * 2)
            min_gap = max(5, int(0.01 * len(y)))
            max_reasonable_gap = len(y) // 10
            optimal_gap = np.clip(optimal_gap, min_gap, max_reasonable_gap)
            logging.info(f"[H7-FIX] ACF-based gap: last_sig_lag={last_significant if len(significant_lags) > 0 else 'N/A'}, gap={optimal_gap}, label_horizon={label_horizon}")
            return int(optimal_gap)
        except ImportError:
            logging.warning("[H7-FIX] statsmodels not available, using heuristic gap")
            heuristic_gap = max(50, int(0.05 * len(y)))
            return heuristic_gap
        except Exception as e:
            logging.warning(f"[H7-FIX] ACF calculation failed: {e}, using heuristic gap")
            heuristic_gap = max(50, int(0.05 * len(y)))
            return heuristic_gap
    def _calculate_early_stopping_gap(self, n_samples: int) -> int:
        label_h = self.label_horizon if hasattr(self, 'label_horizon') and self.label_horizon is not None else 0
        gap_from_label = label_h * 3 if label_h and label_h > 0 else 0
        gap_from_percent = int(0.05 * n_samples)
        gap = max(
            10,
            gap_from_label,
            gap_from_percent,
            24
        )
        logging.debug(f"[ISSUE#9-FIX] Early stopping gap: label_h={label_h}, "
                     f"gap_from_label={gap_from_label}, gap_from_percent={gap_from_percent}, final_gap={gap}")
        return gap
    def _adaptive_stability_threshold(self, n_features: int, n_iterations: int = 100, target_fdr: float = 0.05) -> float:
        try:
            thr = 0.6 + 0.4 * np.sqrt(max(0.0, 1.0 - float(target_fdr)))
        except Exception:
            thr = 0.6
        thr = float(np.clip(thr, 0.6, 0.95))
        logging.info(f'Adaptive stability threshold computed: {thr:.3f} (target_fdr={target_fdr})')
        return thr
    def calculate_sharpe_from_predictions(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        returns_per_signal: float = 0.01,
        annual_factor: int = 252
    ) -> float:
        """
        [DEPRECATED - Use calculate_sharpe_with_costs_and_dsr instead]
        This method uses unrealistic constant returns. It's kept for backward compatibility.
        """
        positions = np.where(y_pred_proba > 0.5, 1, -1)
        actual_returns = np.where(y_true == 1, returns_per_signal, -returns_per_signal)
        strategy_returns = positions * actual_returns
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        if std_return < 1e-10:
            logging.warning("[C4-WARN] Zero volatility in strategy returns, returning 0 - DEPRECATED METHOD")
            return 0.0
        sharpe = (mean_return / std_return) * np.sqrt(annual_factor)
        logging.debug(f"[C4-WARN] DEPRECATED - Mean return: {mean_return:.6f}, Std: {std_return:.6f}, Sharpe: {sharpe:.4f}")
        return float(sharpe)

    def calculate_sharpe_with_costs_and_dsr(
        self,
        signals: np.ndarray,
        actual_price_returns: np.ndarray,
        transaction_cost: float = 0.0002,
        slippage: float = 0.0001,
        n_trials: int = 100,
        annual_factor: int = 252
    ) -> Dict:
        """
        [C4-FIX] Calculate realistic Sharpe Ratio with costs and Deflated Sharpe Ratio

        Based on: Bailey & Lopez de Prado (2014) - "The Deflated Sharpe Ratio"

        Args:
            signals: Array of signals (+1, -1, or 0)
            actual_price_returns: Actual returns from price data (not assumed constant)
            transaction_cost: Typical 0.0002 (2 pips for FX)
            slippage: Typical 0.0001 (1 pip for FX)
            n_trials: Number of strategies tested (for DSR adjustment)
            annual_factor: 252 for daily, 52 for weekly, 12 for monthly

        Returns:
            Dict with sharpe, dsr, psr, and other metrics
        """
        try:
            from scipy.stats import norm, skew, kurtosis
        except ImportError:
            logging.error("[C4-FIX] scipy.stats required for DSR calculation")
            return {
                'sharpe': 0.0,
                'dsr': 0.0,
                'psr': 0.0,
                'error': 'scipy required'
            }

        if len(signals) < 2:
            return {'sharpe': 0.0, 'dsr': 0.0, 'psr': 0.0, 'error': 'insufficient data'}

        # 1. Calculate position changes and costs
        position_changes = np.abs(np.diff(signals))
        costs = position_changes * (transaction_cost + slippage)

        # 2. Calculate strategy returns (signal * next return)
        strategy_returns = signals[:-1] * actual_price_returns[1:]

        # 3. Subtract costs to get net returns
        net_returns = strategy_returns - costs

        # 4. Calculate basic Sharpe Ratio
        mean_ret = np.mean(net_returns)
        std_ret = np.std(net_returns)

        if std_ret < 1e-10:
            logging.warning("[C4-FIX] Zero volatility in net returns")
            return {
                'sharpe': 0.0,
                'dsr': 0.0,
                'psr': 0.0,
                'mean_return': 0.0,
                'volatility': 0.0,
                'total_costs': float(np.sum(costs)),
                'n_trades': int(position_changes.sum()),
                'error': 'zero volatility'
            }

        sharpe = (mean_ret / std_ret) * np.sqrt(annual_factor)

        # 5. Calculate moments for DSR
        skewness = skew(net_returns)
        kurt = kurtosis(net_returns)

        # 6. Variance of Sharpe (Bailey & Lopez de Prado formula)
        T = len(net_returns)
        var_sr = (1 / T) * (
            1 + 0.5 * sharpe**2
            - skewness * sharpe
            + (kurt / 4) * sharpe**2
        )
        var_sr = max(1e-8, var_sr)  # Avoid division by zero

        # 7. Expected Maximum SR under null hypothesis (no skill)
        euler = 0.5772156649  # Euler-Mascheroni constant
        sr_threshold = np.sqrt(var_sr) * (
            (1 - euler) * norm.ppf(1 - 1/n_trials) +
            euler * norm.ppf(1 - 1/(n_trials * np.e))
        )

        # 8. Deflated Sharpe Ratio
        dsr = (sharpe - sr_threshold) / np.sqrt(var_sr) if np.sqrt(var_sr) > 1e-10 else 0.0

        # 9. Probabilistic Sharpe Ratio
        psr = norm.cdf(dsr)

        # 10. Interpretation
        if psr >= 0.95:
            interpretation = "EXCELLENT - Strategy has strong skill (95%+ confidence)"
        elif psr >= 0.90:
            interpretation = "GOOD - Strategy likely has skill (90%+ confidence)"
        elif psr >= 0.75:
            interpretation = "MODERATE - Some evidence of skill"
        elif psr >= 0.50:
            interpretation = "WEAK - Likely due to luck, not skill"
        else:
            interpretation = "POOR - No evidence of skill"

        result = {
            'sharpe': float(sharpe),
            'deflated_sharpe': float(dsr),
            'probabilistic_sharpe': float(psr),
            'sharpe_threshold': float(sr_threshold),
            'mean_return': float(mean_ret * annual_factor),  # annualized
            'volatility': float(std_ret * np.sqrt(annual_factor)),  # annualized
            'total_costs': float(np.sum(costs)),
            'n_trades': int(position_changes.sum()),
            'skewness': float(skewness),
            'kurtosis': float(kurt),
            'var_sharpe': float(var_sr),
            'n_trials': n_trials,
            'n_observations': T,
            'interpretation': interpretation,
            'psr_level': float(psr)
        }

        logging.info(f"[C4-FIX] Sharpe={sharpe:.4f}, DSR={dsr:.4f}, PSR={psr:.4f}")
        logging.info(f"[C4-FIX] {interpretation}")

        return result

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
        max_paths = 15
        n_paths = int(np.math.comb(outer_n_splits, 2)) if outer_n_splits >= 2 else 0
        if n_paths > max_paths:
            logging.warning(f"[ISSUE#8] CPCV would create {n_paths} paths (>{max_paths}). "
                          f"Using TimeSeriesSplit instead to save memory.")
            outer_cv = TimeSeriesSplit(n_splits=outer_n_splits, gap=outer_gap, test_size=outer_test_size)
        else:
            try:
                outer_cv = CombinatorialPurgedCV(
                    n_splits=outer_n_splits,
                    n_test_groups=2,
                    pct_embargo=self.pct_embargo if hasattr(self, 'pct_embargo') else 0.01,
                    label_horizon=self.label_horizon if hasattr(self, 'label_horizon') else 0,
                    embargo_safety_factor=2.0
                )
                logging.info(f"[ISSUE#8] Using CPCV (n_splits={outer_n_splits}, paths={n_paths})")
            except Exception as e:
                logging.warning(f"CPCV initialization failed: {e}, falling back to TimeSeriesSplit")
                outer_cv = TimeSeriesSplit(n_splits=outer_n_splits, gap=outer_gap, test_size=outer_test_size)
        outer_scores = []
        feature_importances_outer = []
        models_per_fold = []
        stability_results_per_fold = []
        for fold_idx, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X)):
            X_train_outer = X.iloc[train_outer_idx].reset_index(drop=True)
            y_train_outer = y.iloc[train_outer_idx].reset_index(drop=True)
            X_test_outer = X.iloc[test_outer_idx].reset_index(drop=True)
            y_test_outer = y.iloc[test_outer_idx].reset_index(drop=True)
            logging.info(f"[LEAKAGE-FIX-3] Outer fold {fold_idx}: Running prefilter on THIS fold only")
            X_train_outer_filtered, dropped_features_fold = self.quick_prefilter(X_train_outer, y_train_outer)
            X_test_outer_filtered = X_test_outer.drop(columns=dropped_features_fold, errors='ignore')
            common_cols = [c for c in X_train_outer_filtered.columns if c in X_test_outer_filtered.columns]
            X_train_outer_filtered = X_train_outer_filtered[common_cols]
            X_test_outer_filtered = X_test_outer_filtered[common_cols]
            logging.debug(f"[LEAKAGE-FIX-3] Fold {fold_idx}: {X_train_outer.shape[1]} -> {X_train_outer_filtered.shape[1]} features")
            X_train_outer = X_train_outer_filtered
            X_test_outer = X_test_outer_filtered
            w_train_outer = self.compute_sample_weights(y_train_outer) if self.classification else None
            logging.info(f"[C1-FIX] Outer fold {fold_idx}: Running stability selection on {len(X_train_outer)} train samples (not global)")
            fold_stability = self.stability_selection_framework(
                X_train_outer,
                y_train_outer,
                n_iterations=min(self.stability_selection_iterations, 20),
                sample_fraction=0.5,
                threshold=0.6
            )
            stability_results_per_fold.append(fold_stability)
            fold_gap = self.calculate_adaptive_gap(
                X_train_outer,
                y_train_outer,
                label_horizon=self.label_horizon if hasattr(self, 'label_horizon') else 0
            )
            logging.info(f"[LEAKAGE-FIX-5] Outer fold {fold_idx}: Adaptive gap={fold_gap} (calculated from fold data)")
            inner_params = self._calculate_optimal_ts_cv_params(len(X_train_outer))
            inner_n_splits = min(n_inner_splits, inner_params['n_splits'])
            inner_gap = self.calculate_adaptive_gap(
                X_train_outer,
                y_train_outer,
                label_horizon=self.label_horizon if hasattr(self, 'label_horizon') else 0
            )
            logging.info(f"[HIGH-PRIORITY-FIX-5] Inner fold gap={inner_gap} (recalculated from outer fold {fold_idx} data)")
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
                    if self.classification:
                        w_train_inner = self.compute_time_weighted_samples(
                            y_train_inner,
                            label_horizon=self.label_horizon
                        )
                        w_val_inner = self.compute_time_weighted_samples(
                            y_val_inner,
                            label_horizon=self.label_horizon
                        )
                    else:
                        w_train_inner = None
                        w_val_inner = None
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
            gap_size = self._calculate_early_stopping_gap(len(X_train_outer))
            train_end = len(X_train_outer) - val_size_outer - gap_size
            X_train_final = X_train_outer.iloc[:train_end]
            y_train_final = y_train_outer.iloc[:train_end]
            val_start = train_end + gap_size
            X_val_final = X_train_outer.iloc[val_start:]
            y_val_final = y_train_outer.iloc[val_start:]
            logging.debug(f"Early stopping validation: train={len(X_train_final)}, gap={gap_size}, val={len(X_val_final)}")
            w_train_final = self.compute_sample_weights(y_train_final) if self.classification else None
            w_val_final = self.compute_sample_weights(y_val_final) if self.classification else None
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
        logging.info(f'\n{"-"*70}')
        logging.info('[NESTED-CV] Detailed Analysis:')
        logging.info(f'  Per-fold scores: {[f"{s:.4f}" for s in outer_scores]}')
        score_variance = np.var(outer_scores)
        logging.info(f'  Score variance: {score_variance:.6f}')
        if score_variance < 0.001:
            logging.info(f'  Status: Very stable - EXCELLENT')
        elif score_variance < 0.005:
            logging.info(f'  Status: Stable - GOOD')
        else:
            logging.warning(f'  WARNING: High variance detected - Model may be unstable or overfitting!')
        n_stable_imp = int(np.sum(feature_cv < 0.1))
        n_unstable_imp = int(np.sum(feature_cv > 0.3))
        n_features = len(feature_cv)
        logging.info(f'  Feature importance stability: CV<10%={n_stable_imp}, CV>30%={n_unstable_imp}')
        if n_unstable_imp > n_features * 0.2:
            logging.warning(f'  WARNING: {n_unstable_imp} features ({n_unstable_imp/n_features:.1%}) highly inconsistent - Consider removing!')
        logging.info(f'{"-"*70}')
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
    def calculate_real_sharpe_from_backtest(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        transaction_cost: float = 0.0001,
        annual_factor: int = 252
    ) -> Dict:
        logging.info("[SHARPE-BACKTEST] Computing REAL Sharpe from backtest signals")
        try:
            signals = np.asarray(signals, dtype=np.float32)
            returns = np.asarray(returns, dtype=np.float32)
            if len(signals) != len(returns):
                raise ValueError(f"signals ({len(signals)}) and returns ({len(returns)}) length mismatch")
            strategy_returns = signals[:-1] * returns[1:]
            position_changes = np.abs(np.diff(signals))
            costs = position_changes[:-1] * transaction_cost
            net_returns = strategy_returns - costs
            mean_return = np.mean(net_returns)
            std_return = np.std(net_returns)
            if std_return < 1e-10:
                sharpe = 0.0
            else:
                sharpe = (mean_return / std_return) * np.sqrt(annual_factor)
            win_rate = np.mean(net_returns > 0)
            gross_profit = np.sum(net_returns[net_returns > 0]) if np.any(net_returns > 0) else 0.0
            gross_loss = np.abs(np.sum(net_returns[net_returns < 0])) if np.any(net_returns < 0) else 1.0
            profit_factor = gross_profit / max(gross_loss, 1e-10)
            cumulative_returns = np.cumsum(net_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 1.0
            total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0
            calmar_ratio = total_return / max_drawdown if max_drawdown > 1e-10 else 0.0
            logging.info(f"[SHARPE-BACKTEST] Results:")
            logging.info(f"  Sharpe Ratio: {sharpe:.4f}")
            logging.info(f"  Mean Return: {mean_return:.6f}")
            logging.info(f"  Std Return: {std_return:.6f}")
            logging.info(f"  Win Rate: {win_rate:.1%}")
            logging.info(f"  Profit Factor: {profit_factor:.2f}")
            logging.info(f"  Max Drawdown: {max_drawdown:.4f}")
            logging.info(f"  Calmar Ratio: {calmar_ratio:.4f}")
            return {
                'sharpe_ratio': float(sharpe),
                'mean_return': float(mean_return),
                'std_return': float(std_return),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'max_drawdown': float(max_drawdown),
                'total_return': float(total_return),
                'calmar_ratio': float(calmar_ratio),
                'n_periods': len(net_returns)
            }
        except Exception as e:
            logging.error(f"[SHARPE-BACKTEST] Failed: {e}")
            return {
                'sharpe_ratio': 0.0,
                'mean_return': 0.0,
                'std_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'max_drawdown': 1.0,
                'total_return': 0.0,
                'calmar_ratio': 0.0,
                'n_periods': 0,
                'error': str(e)
            }
    def calculate_sharpe_ratio_from_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        position_size: float = 1.0,
        trading_cost: float = 0.0001
    ) -> Dict:
        logging.debug(f"[SHARPE-CALC] Computing Sharpe from predictions (n={len(y_true)})")
        try:
            if self.classification:
                positions = np.where(y_pred > 0.5, 1, -1)
                actual_direction = np.where(y_true >= 0.5, 1, -1)
                pnl = positions * actual_direction * position_size
            else:
                positions = np.sign(y_pred)
                pnl = positions * y_true * position_size
            position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
            costs = position_changes * trading_cost
            pnl_net = pnl - costs
            mean_return = np.mean(pnl_net)
            std_return = np.std(pnl_net)
            if std_return < 1e-10:
                sharpe = 0.0
            else:
                sharpe = (mean_return / std_return) * np.sqrt(252)
            win_rate = np.mean(pnl_net > 0) if len(pnl_net) > 0 else 0.0
            gross_profit = np.sum(pnl_net[pnl_net > 0]) if np.any(pnl_net > 0) else 0.0
            gross_loss = np.abs(np.sum(pnl_net[pnl_net < 0])) if np.any(pnl_net < 0) else 1.0
            profit_factor = gross_profit / max(gross_loss, 1e-10)
            logging.debug(f"[SHARPE-CALC] Sharpe={sharpe:.3f}, WinRate={win_rate:.1%}, ProfitFactor={profit_factor:.2f}")
            return {
                'sharpe_ratio': float(sharpe),
                'mean_return': float(mean_return),
                'std_return': float(std_return),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'gross_profit': float(gross_profit),
                'gross_loss': float(gross_loss)
            }
        except Exception as e:
            logging.warning(f"[SHARPE-CALC] Failed to compute Sharpe: {e}")
            return {
                'sharpe_ratio': 0.0,
                'mean_return': 0.0,
                'std_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0
            }
    def calculate_pbo_with_cscv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_scenarios: int = 16,
        n_strategies_per_scenario: int = 50,
        random_state: int = None
    ) -> Dict:
        """
        [C3-FIX] Probability of Backtest Overfitting using CSCV

        Based on: Bailey et al. (2014) - "The Probability of Backtest Overfitting"

        CSCV = Combinatorially Symmetric Cross-Validation
        Tests multiple train/test scenarios to detect overfitting.

        Args:
            X: Features DataFrame
            y: Target Series
            n_scenarios: Number of CSCV train/test combinations (typically 16 or C(6,2)=15)
            n_strategies_per_scenario: Different strategies to test per scenario
            random_state: Random seed

        Returns:
            PBO metric and interpretation
        """
        from itertools import combinations
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score

        if random_state is None:
            random_state = self.random_state

        logging.info("="*70)
        logging.info("[C3-FIX] PBO - BAILEY ET AL. (2014) CSCV IMPLEMENTATION")
        logging.info(f"Creating {n_scenarios} CSCV scenarios with {n_strategies_per_scenario} strategies each")
        logging.info("="*70)

        n = len(X)
        n_splits = 6  # CSCV typically uses 6 groups
        group_size = n // n_splits

        # Create fold indices
        folds = [np.arange(i*group_size, min((i+1)*group_size, n)) for i in range(n_splits)]

        # Generate all C(6,2) = 15 combinations for test set
        test_combinations = list(combinations(range(n_splits), 2))
        if len(test_combinations) > n_scenarios:
            rng = np.random.default_rng(random_state)
            test_combinations = list(rng.choice(range(len(test_combinations)), n_scenarios, replace=False).astype(int))
            test_combinations = [test_combinations[i] for i in range(n_scenarios)]

        logging.info(f"[C3-FIX] Using {len(test_combinations)} scenarios (out of C(6,2)={15})")

        is_performance = []  # In-sample performance per scenario, per strategy
        oos_performance = []  # Out-of-sample performance per scenario, per strategy

        for scenario_idx, (test_fold_1, test_fold_2) in enumerate(test_combinations):
            logging.debug(f"[C3-FIX] Scenario {scenario_idx+1}: Test folds {test_fold_1}, {test_fold_2}")

            # Get train and test indices for this scenario
            test_idx = np.concatenate([folds[test_fold_1], folds[test_fold_2]])
            train_idx = np.concatenate([folds[i] for i in range(n_splits) if i not in [test_fold_1, test_fold_2]])

            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)

            scenario_is_perf = []
            scenario_oos_perf = []

            rng = np.random.default_rng(random_state + scenario_idx)

            # Test multiple strategies within this scenario
            for strategy_id in range(n_strategies_per_scenario):
                # Random feature selection
                n_features = rng.integers(
                    max(1, X_train.shape[1] // 4),
                    max(2, X_train.shape[1] // 2) + 1
                )
                selected_features = list(rng.choice(
                    range(X_train.shape[1]),
                    size=min(n_features, X_train.shape[1]),
                    replace=False
                ))

                try:
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        random_state=int(random_state + scenario_idx + strategy_id),
                        verbose=-1,
                        **{k: v for k, v in self.base_params.items() if k != 'n_estimators'}
                    )

                    model.fit(X_train.iloc[:, selected_features], y_train)

                    # In-sample performance
                    y_pred_is = (model.predict_proba(X_train.iloc[:, selected_features])[:, 1] > 0.5).astype(int)
                    is_perf = accuracy_score(y_train, y_pred_is)
                    scenario_is_perf.append(is_perf)

                    # Out-of-sample performance
                    y_pred_oos = (model.predict_proba(X_test.iloc[:, selected_features])[:, 1] > 0.5).astype(int)
                    oos_perf = accuracy_score(y_test, y_pred_oos)
                    scenario_oos_perf.append(oos_perf)

                except Exception as e:
                    logging.debug(f"[C3-FIX] Scenario {scenario_idx}, Strategy {strategy_id} failed: {e}")
                    continue

            is_performance.append(scenario_is_perf)
            oos_performance.append(scenario_oos_perf)

        # Calculate PBO
        if not is_performance or not oos_performance:
            logging.warning("[C3-FIX] Insufficient scenarios/strategies completed")
            return {
                'pbo': np.nan,
                'method': 'cscv',
                'interpretation': 'Insufficient data for PBO calculation',
                'n_scenarios_completed': 0
            }

        is_performance = np.array(is_performance)  # (n_scenarios, n_strategies)
        oos_performance = np.array(oos_performance)  # (n_scenarios, n_strategies)

        # For each scenario, find the strategy with best IS performance
        # and check its OOS rank
        pbo_values = []

        for scenario_idx in range(len(is_performance)):
            is_perf = is_performance[scenario_idx]
            oos_perf = oos_performance[scenario_idx]

            if len(is_perf) < 2:
                continue

            best_is_idx = np.argmax(is_perf)
            best_oos_perf = oos_perf[best_is_idx]

            # Rank in OOS (how many strategies beat this one in OOS?)
            better_oos = np.sum(oos_perf > best_oos_perf)
            oos_rank = better_oos + 1  # Rank (1-indexed)

            pbo_scenario = oos_rank / len(is_perf)
            pbo_values.append(pbo_scenario)

        if not pbo_values:
            return {
                'pbo': np.nan,
                'method': 'cscv',
                'interpretation': 'No valid scenarios',
                'n_scenarios_completed': 0
            }

        pbo_values = np.array(pbo_values)
        pbo_mean = np.mean(pbo_values)
        pbo_std = np.std(pbo_values)

        # Interpretation
        if pbo_mean < 0.5:
            status = "GOOD - Low overfitting risk, strategy appears robust"
        elif pbo_mean < 0.7:
            status = "MODERATE - Some overfitting detected"
        else:
            status = "HIGH RISK - Likely overfitted, results may not generalize"

        logging.info(f"[C3-FIX] PBO Results:")
        logging.info(f"  Mean PBO: {pbo_mean:.4f} ± {pbo_std:.4f}")
        logging.info(f"  Min/Max PBO: {np.min(pbo_values):.4f} / {np.max(pbo_values):.4f}")
        logging.info(f"  {status}")
        logging.info("="*70)

        return {
            'pbo': float(pbo_mean),
            'pbo_std': float(pbo_std),
            'pbo_min': float(np.min(pbo_values)),
            'pbo_max': float(np.max(pbo_values)),
            'method': 'cscv',
            'interpretation': status,
            'n_scenarios': len(is_performance),
            'n_strategies_per_scenario': n_strategies_per_scenario,
            'is_overfitted': pbo_mean > 0.5
        }

    def calculate_pbo_with_multiple_strategies(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_strategies: int = 50,
        random_state: int = None,
        use_diverse_methods: bool = False
    ) -> Dict:
        """
        [LEGACY] Simple PBO with single train/test split
        For new code, use calculate_pbo_with_cscv() instead
        """
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score
        if random_state is None:
            random_state = self.random_state
        logging.info("="*70)
        logging.warning("[C3-NOTE] Using legacy single-split PBO. Consider calculate_pbo_with_cscv() for better results.")
        if use_diverse_methods:
            logging.info("[HIGH-PRIORITY-FIX-6] PBO - DIVERSE METHODS IMPLEMENTATION")
            logging.info(f"Testing {n_strategies} different feature selection methods")
            logging.warning("[IMPROVEMENT] Using multiple FS methods (not just random subsets)")
        else:
            logging.info("PBO - BAILEY ET AL. (2014) IMPLEMENTATION")
            logging.info(f"Testing {n_strategies} different feature subsets")
            logging.warning("[FIXED] Now using proper temporal split with embargo gap")
        logging.info("="*70)
        gap = self.calculate_adaptive_gap(X, y, self.label_horizon or 0)
        n = len(X)
        is_end = n // 2
        oos_start = is_end + gap
        if oos_start >= n:
            logging.warning(f"[PBO] Gap {gap} too large for dataset size {n}. Reducing gap.")
            gap = max(10, (n // 2) - 50)
            oos_start = is_end + gap
        X_is = X.iloc[:is_end]
        y_is = y.iloc[:is_end]
        X_oos = X.iloc[oos_start:]
        y_oos = y.iloc[oos_start:]
        assert is_end + gap <= oos_start, f"Overlap detected! is_end={is_end}, gap={gap}, oos_start={oos_start}"
        logging.info(f"[FIXED-PBO-ISSUE#4] IS: 0-{is_end} ({len(X_is)} samples)")
        logging.info(f"[FIXED-PBO-ISSUE#4] Gap: {is_end}-{oos_start} ({gap} samples)")
        logging.info(f"[FIXED-PBO-ISSUE#4] OOS: {oos_start}-{n} ({len(X_oos)} samples)")
        is_scores = []
        oos_scores = []
        rng = np.random.default_rng(random_state)
        for strategy_id in range(n_strategies):
            n_features_to_select = rng.integers(
                max(5, X.shape[1] // 4),
                max(10, X.shape[1] // 2)
            )
            selected_features = rng.choice(
                X.columns,
                size=min(n_features_to_select, len(X.columns)),
                replace=False
            )
            try:
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=int(random_state + strategy_id),
                    verbose=-1,
                    **{k: v for k, v in self.base_params.items() if k != 'n_estimators'}
                )
                model.fit(X_is[selected_features], y_is)
                y_pred_is = (model.predict_proba(X_is[selected_features])[:, 1] > 0.5).astype(int)
                is_score = accuracy_score(y_is, y_pred_is)
                y_pred_oos = (model.predict_proba(X_oos[selected_features])[:, 1] > 0.5).astype(int)
                oos_score = accuracy_score(y_oos, y_pred_oos)
                is_scores.append(is_score)
                oos_scores.append(oos_score)
                if strategy_id % 10 == 0:
                    logging.debug(f"[PBO] Strategy {strategy_id}/{n_strategies}: IS={is_score:.4f}, OOS={oos_score:.4f}")
            except Exception as e:
                logging.debug(f"[PBO] Strategy {strategy_id} failed: {e}")
                continue
        if len(is_scores) < 2:
            logging.warning("[PBO] Insufficient strategies tested")
            return {
                'pbo': np.nan,
                'n_strategies': len(is_scores),
                'interpretation': 'Insufficient strategies tested',
                'is_overfitted': False
            }
        is_scores = np.array(is_scores)
        oos_scores = np.array(oos_scores)
        best_is_idx = np.argmax(is_scores)
        best_is_score = is_scores[best_is_idx]
        best_oos_score = oos_scores[best_is_idx]
        oos_sorted = np.argsort(oos_scores)[::-1]
        oos_rank = np.where(oos_sorted == best_is_idx)[0][0] + 1
        pbo = oos_rank / len(is_scores)
        if pbo > 0.5:
            status = "CRITICAL: HIGH OVERFITTING RISK"
            recommendation = "DO NOT USE - Results likely due to LUCK, not SKILL"
        elif pbo > 0.3:
            status = "WARNING: Moderate overfitting risk"
            recommendation = "Use with caution - Consider more robust validation"
        else:
            status = "LOW: Overfitting risk acceptable"
            recommendation = "Results appear robust"
        logging.info(f"[LEAKAGE-FIX-8] Best IS score: {best_is_score:.4f}, OOS score: {best_oos_score:.4f}")
        logging.info(f"[LEAKAGE-FIX-8] OOS rank: {oos_rank}/{len(is_scores)}, PBO: {pbo:.4f}")
        logging.info(f"[LEAKAGE-FIX-8] {status}")
        logging.info(f"[LEAKAGE-FIX-8] {recommendation}")
        logging.info("="*70)
        return {
            'pbo': float(pbo),
            'n_strategies': len(is_scores),
            'best_is_score': float(best_is_score),
            'best_oos_score': float(best_oos_score),
            'oos_rank': int(oos_rank),
            'interpretation': status,
            'recommendation': recommendation,
            'is_overfitted': pbo > 0.5,
            'method': 'bailey_2014_correct'
        }
    def calculate_probability_of_backtest_overfitting(
        self,
        is_performance: List[float],
        oos_performance: List[float],
        n_samples: int = None
    ) -> Dict:
        logging.info("="*70)
        logging.info("PROBABILITY OF BACKTEST OVERFITTING (PBO) - LEGACY METHOD")
        logging.warning("For CORRECT PBO, use calculate_pbo_with_multiple_strategies()")
        logging.info("="*70)
        IS_perf = np.array(is_performance, dtype=float)
        OOS_perf = np.array(oos_performance, dtype=float)
        if len(IS_perf) != len(OOS_perf):
            logging.warning(f"IS and OOS have different lengths: {len(IS_perf)} vs {len(OOS_perf)}")
            min_len = min(len(IS_perf), len(OOS_perf))
            IS_perf = IS_perf[:min_len]
            OOS_perf = OOS_perf[:min_len]
        N = len(IS_perf)
        if N < 2:
            logging.warning(f"Not enough strategies for PBO: {N} < 2")
            return {
                'pbo': np.nan,
                'n_strategies': N,
                'best_is_idx': 0,
                'best_is_score': float(IS_perf[0]) if len(IS_perf) > 0 else np.nan,
                'best_oos_score': float(OOS_perf[0]) if len(OOS_perf) > 0 else np.nan,
                'median_oos': float(np.median(OOS_perf)) if len(OOS_perf) > 0 else np.nan,
                'interpretation': 'Insufficient data',
                'method': 'bailey_2014'
            }
        best_is_idx = np.argmax(IS_perf)
        best_is_score = IS_perf[best_is_idx]
        best_oos_score = OOS_perf[best_is_idx]
        oos_sorted_indices = np.argsort(OOS_perf)[::-1]
        oos_rank_of_best_is = np.where(oos_sorted_indices == best_is_idx)[0][0] + 1
        median_rank = (N + 1) / 2.0
        pbo = oos_rank_of_best_is / N
        if pbo > 0.5:
            status = "CRITICAL: HIGH OVERFITTING RISK"
            recommendation = "DO NOT USE - Results likely due to LUCK, not SKILL"
        elif pbo > 0.3:
            status = "WARNING: Moderate overfitting risk"
            recommendation = "Use with caution - Consider more robust validation"
        else:
            status = "LOW: Overfitting risk acceptable"
            recommendation = "Results appear robust"
        logging.info(f"[H9-FIX] Bailey et al. (2014) Methodology:")
        logging.info(f"Number of strategies tested: {N}")
        logging.info(f"Best IS performance: {best_is_score:.4f} (at index {best_is_idx})")
        logging.info(f"Best IS OOS performance: {best_oos_score:.4f}")
        logging.info(f"OOS rank of best IS strategy: {oos_rank_of_best_is}/{N}")
        logging.info(f"\nProbability of Backtest Overfitting (PBO): {pbo:.4f}")
        logging.info(f"Status: {status}")
        logging.info(f"Recommendation: {recommendation}")
        logging.info("="*70)
        logging.info(f"Recommendation: {recommendation}")
        if n_samples is not None and best_oos_score > 0:
            try:
                from scipy.stats import norm
                estimated_sr = best_oos_score / max(np.std(OOS_perf), 1e-6)
                var_sr = (1 + 0.5 * estimated_sr**2) / n_samples
                euler_mascheroni = 0.5772156649
                sr_star = np.sqrt(var_sr) * (
                    (1 - euler_mascheroni) * norm.ppf(1 - 1/N) +
                    euler_mascheroni * norm.ppf(1 - 1/(N * np.e))
                )
                deflated_sr = (estimated_sr - sr_star) / np.sqrt(var_sr)
                psr = norm.cdf(deflated_sr)
                logging.info(f"\n[DEFLATED SHARPE RATIO ADJUSTMENT]")
                logging.info(f"  Estimated SR: {estimated_sr:.4f}")
                logging.info(f"  SR threshold (adjusted for {N} trials): {sr_star:.4f}")
                logging.info(f"  Deflated SR: {deflated_sr:.4f}")
                logging.info(f"  Probabilistic SR: {psr:.2%}")
                if psr < 0.95:
                    logging.warning(f"   PSR < 95%: Results may not be statistically significant!")
            except Exception as e:
                logging.debug(f"Deflated Sharpe calculation skipped: {e}")
        logging.info(f"{'-'*70}")
        return {
            'pbo': float(pbo),
            'n_strategies': N,
            'best_is_idx': int(best_is_idx),
            'best_is_score': float(best_is_score),
            'best_oos_score': float(best_oos_score),
            'median_oos': float(median_oos),
            'performance_gap': float(best_is_score - best_oos_score),
            'interpretation': status,
            'recommendation': recommendation,
            'is_overfitted': pbo > 0.5
        }
    def calculate_minimum_track_record_length(
        self,
        estimated_sharpe_ratio: float,
        n_samples: int,
        target_sharpe_ratio: float = 0.0,
        confidence_level: float = 0.95,
        skewness: float = 0.0,
        excess_kurtosis: float = 3.0
    ) -> Dict:
        logging.info("="*70)
        logging.info("MINIMUM TRACK RECORD LENGTH (MinTRL) ANALYSIS")
        logging.warning("[CRITICAL] Using Sharpe Ratio from BACKTEST RETURNS (not from AUC or metrics)!")
        logging.info("="*70)
        from scipy.stats import norm
        var_sr = (
            1 + 0.5 * estimated_sharpe_ratio**2 -
            skewness * estimated_sharpe_ratio +
            (excess_kurtosis - 3) / 4 * estimated_sharpe_ratio**2
        )
        z_score = norm.ppf(confidence_level)
        numerator = var_sr * (z_score ** 2)
        denominator = (estimated_sharpe_ratio - target_sharpe_ratio) ** 2
        if denominator <= 0:
            min_trl = np.inf
            logging.warning(f" Estimated SR <= target SR: Cannot achieve goal!")
        else:
            min_trl = numerator / denominator
        if n_samples >= min_trl:
            status = f" SUFFICIENT: {n_samples:.0f} >= {min_trl:.0f} (MinTRL)"
            recommendation = "Track record is adequate for statistical confidence"
        else:
            deficit = min_trl - n_samples
            status = f" INSUFFICIENT: {n_samples:.0f} < {min_trl:.0f} (MinTRL)"
            recommendation = f"Need {deficit:.0f} more samples ({deficit/252:.1f} years)"
        if n_samples > 0:
            confidence_achieved = norm.cdf(
                estimated_sharpe_ratio / np.sqrt(var_sr / n_samples)
            )
        else:
            confidence_achieved = 0.0
        logging.info(f"Estimated Sharpe Ratio: {estimated_sharpe_ratio:.4f}")
        logging.info(f"Target Sharpe Ratio: {target_sharpe_ratio:.4f}")
        logging.info(f"Confidence Level: {confidence_level:.1%}")
        logging.info(f"Return Distribution:")
        logging.info(f"  Skewness: {skewness:.4f}")
        logging.info(f"  Excess Kurtosis: {excess_kurtosis:.4f}")
        logging.info(f"\nMinimum Track Record Length: {min_trl:.0f} samples")
        logging.info(f"Available samples: {n_samples:.0f}")
        logging.info(f"Status: {status}")
        logging.info(f"Recommendation: {recommendation}")
        logging.info(f"Confidence achieved: {confidence_achieved:.2%}")
        min_trl_years = min_trl / 252
        available_years = n_samples / 252
        logging.info(f"\nTime equivalents (252 trading days/year):")
        logging.info(f"  MinTRL: {min_trl_years:.1f} years")
        logging.info(f"  Available: {available_years:.1f} years")
        if available_years < 1.0:
            logging.warning(f" Less than 1 year of data - results may not be reliable!")
        elif available_years < 3.0:
            logging.warning(f" Less than 3 years - limited for strategy validation")
        logging.info(f"{'-'*70}")
        return {
            'min_trl': float(min_trl),
            'n_samples_available': int(n_samples),
            'deficit': float(max(0, min_trl - n_samples)),
            'var_sr': float(var_sr),
            'confidence_achieved': float(confidence_achieved),
            'min_trl_years': float(min_trl_years),
            'available_years': float(available_years),
            'is_sufficient': n_samples >= min_trl,
            'status': status,
            'recommendation': recommendation
        }
    def walk_forward_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 10,
        retrain_frequency: int = 1,
        min_train_size: int = None,
        embargo_pct: float = 0.01
    ) -> Dict:
        logging.info(f'\n{"="*70}')
        logging.info('[H1-FIX] WALK-FORWARD ANALYSIS WITH EMBARGO')
        logging.info(f'{"="*70}')
        logging.info(f'  Splits: {n_splits}, Retrain frequency: {retrain_frequency}')
        logging.info(f'  Embargo: {embargo_pct*100:.1f}% of dataset')
        n_samples = len(X)
        if min_train_size is None:
            min_train_size = n_samples // 2
        embargo_size = int(n_samples * embargo_pct)
        test_size = (n_samples - min_train_size - embargo_size) // n_splits
        if test_size < 10:
            logging.warning(f'Test size too small ({test_size}), reducing n_splits')
            n_splits = max(2, (n_samples - min_train_size - embargo_size) // 10)
            test_size = (n_samples - min_train_size - embargo_size) // n_splits
        logging.info(f'  Min train size: {min_train_size}, Embargo size: {embargo_size}, Test size per fold: {test_size}')
        results = []
        model = None
        train_end = min_train_size
        for fold in range(n_splits):
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            embargo_start = train_end
            embargo_end = train_end + embargo_size
            test_start = embargo_end
            test_end = min(test_start + test_size, n_samples)
            if test_start >= n_samples:
                logging.warning(f'Fold {fold}: No more test data available')
                break
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            should_retrain = (fold % retrain_frequency == 0) or (model is None)
            if should_retrain:
                logging.debug(f'Fold {fold}: Retraining model (train_size={len(X_train)})')
                if self.classification:
                    w_train = self.compute_time_weighted_samples(
                        y_train,
                        label_horizon=self.label_horizon
                    )
                else:
                    w_train = None
                params = self._get_adaptive_params(X_train, y_train)
                params['n_estimators'] = 200
                train_data = self._create_dataset(X_train, y_train, weight=w_train)
                model = self._train_with_fallback(
                    params,
                    train_data,
                    num_boost_round=200,
                    callbacks=[lgb.log_evaluation(period=0)]
                )
            if self.classification:
                y_pred = model.predict(X_test)
                y_pred_binary = (y_pred > 0.5).astype(int)
                score = accuracy_score(y_test, y_pred_binary)
                metric_name = 'accuracy'
            else:
                y_pred = model.predict(X_test)
                score = -mean_squared_error(y_test, y_pred)
                metric_name = 'neg_mse'
            results.append({
                'fold': fold,
                'train_start': 0,
                'train_end': train_end,
                'embargo_start': embargo_start,
                'embargo_end': embargo_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'score': float(score),
                'retrained': should_retrain
            })
            logging.info(f'  Fold {fold}: Train[0:{train_end}], Embargo[{embargo_start}:{embargo_end}], Test[{test_start}:{test_end}], '
                        f'{metric_name}={score:.4f}{"*" if should_retrain else ""}')
            train_end = test_end + embargo_size
        scores = [r['score'] for r in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        if len(scores) >= 3:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            degradation = np.mean(first_half) - np.mean(second_half)
        else:
            degradation = 0.0
        logging.info(f'\n[WF-SUMMARY]')
        logging.info(f'  Mean {metric_name}: {mean_score:.4f} ± {std_score:.4f}')
        logging.info(f'  Range: [{min_score:.4f}, {max_score:.4f}]')
        logging.info(f'  Performance degradation: {degradation:.4f}')
        if abs(degradation) > 0.05:
            logging.warning(f'  WARNING: Significant performance degradation over time!')
            logging.warning(f'  Consider more frequent retraining or feature engineering')
        else:
            logging.info(f'  Status: Performance stable over time - GOOD')
        logging.info(f'{"="*70}\n')
        return {
            'results_per_fold': results,
            'scores': scores,
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'min_score': float(min_score),
            'max_score': float(max_score),
            'degradation': float(degradation),
            'n_folds': len(results),
            'retrain_frequency': retrain_frequency
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
    def identify_unstable_features(
        self,
        feature_names: List[str],
        shap_importance: Optional[Dict] = None,
        stability_selection: Optional[Dict] = None,
        confidence_intervals: Optional[Dict] = None,
        nested_cv_results: Optional[Dict] = None,
        shap_cv_threshold: float = 0.25,
        stability_prob_threshold: float = 0.4,
        ci_width_threshold: float = 0.8,
        cv_importance_cv_threshold: float = 0.35
    ) -> Dict:
        n_features = len(feature_names)
        instability_flags = np.zeros(n_features, dtype=int)
        instability_reasons = [[] for _ in range(n_features)]
        stability_scores = np.ones(n_features, dtype=np.float32)
        if shap_importance is not None and 'shap_cv' in shap_importance:
            shap_cv = shap_importance['shap_cv']
            high_shap_cv = shap_cv > shap_cv_threshold
            for i in np.where(high_shap_cv)[0]:
                instability_flags[i] += 1
                instability_reasons[i].append(f'SHAP_CV={shap_cv[i]:.3f}>{shap_cv_threshold}')
                stability_scores[i] *= (1 - min(shap_cv[i], 1.0))
        if stability_selection is not None and 'selection_probability' in stability_selection:
            sel_prob = stability_selection['selection_probability']
            low_prob = sel_prob < stability_prob_threshold
            for i in np.where(low_prob)[0]:
                instability_flags[i] += 1
                instability_reasons[i].append(f'StabSel_prob={sel_prob[i]:.3f}<{stability_prob_threshold}')
                stability_scores[i] *= sel_prob[i]
        if confidence_intervals is not None:
            ci_lower = confidence_intervals.get('ci_lower', np.zeros(n_features))
            ci_upper = confidence_intervals.get('ci_upper', np.ones(n_features))
            ci_mean = confidence_intervals.get('mean_importance', np.ones(n_features))
            ci_width_rel = np.where(
                ci_mean > 1e-6,
                (ci_upper - ci_lower) / (ci_mean + 1e-6),
                0
            )
            wide_ci = ci_width_rel > ci_width_threshold
            for i in np.where(wide_ci)[0]:
                instability_flags[i] += 1
                instability_reasons[i].append(f'CI_width_rel={ci_width_rel[i]:.3f}>{ci_width_threshold}')
                stability_scores[i] *= (1 - min(ci_width_rel[i] / 2, 0.8))
        if nested_cv_results is not None and 'feature_importance_cv' in nested_cv_results:
            cv_imp_cv = nested_cv_results['feature_importance_cv']
            high_cv_imp_cv = cv_imp_cv > cv_importance_cv_threshold
            for i in np.where(high_cv_imp_cv)[0]:
                instability_flags[i] += 1
                instability_reasons[i].append(f'NestedCV_CV={cv_imp_cv[i]:.3f}>{cv_importance_cv_threshold}')
                stability_scores[i] *= (1 - min(cv_imp_cv[i], 1.0))
        unstable_mask = instability_flags >= 2
        n_unstable = np.sum(unstable_mask)
        logging.info(f'\n{"="*70}')
        logging.info('[STABILITY FILTER] Unstable Feature Detection')
        logging.info(f'{"="*70}')
        logging.info(f'  Unstable features: {n_unstable}/{n_features} ({n_unstable/n_features:.1%})')
        logging.info(f'  Criteria: SHAP_CV>{shap_cv_threshold}, StabSel<{stability_prob_threshold}, '
                    f'CI_width>{ci_width_threshold}, CV>{cv_importance_cv_threshold}')
        if n_unstable > 0:
            instability_scores = -stability_scores
            top_unstable_idx = np.argsort(instability_scores)[:min(10, n_unstable)]
            logging.info(f'  Top unstable features:')
            for rank, idx in enumerate(top_unstable_idx, 1):
                if unstable_mask[idx]:
                    feat_name = feature_names[idx][:40]
                    reasons = ', '.join(instability_reasons[idx])
                    score = stability_scores[idx]
                    logging.info(f'    {rank}. {feat_name}: score={score:.3f}, reasons=[{reasons}]')
        logging.info(f'{"="*70}\n')
        return {
            'unstable_mask': unstable_mask,
            'instability_reasons': instability_reasons,
            'stability_scores': stability_scores,
            'n_unstable': int(n_unstable),
            'criteria_thresholds': {
                'shap_cv': shap_cv_threshold,
                'stability_prob': stability_prob_threshold,
                'ci_width': ci_width_threshold,
                'cv_importance_cv': cv_importance_cv_threshold
            }
        }
    def compute_adaptive_ensemble_weights(
        self,
        df: pd.DataFrame,
        default_fallback: bool = False
    ) -> Dict[str, float]:
        fixed_weights = {
            'null_importance': 0.20,
            'boosting_ensemble': 0.15,
            'feature_fraction': 0.10,
            'adversarial': 0.10,
            'rfe': 0.15,
            'cv_metrics': 0.15,
            'stability': 0.10,
            'shap_importance': 0.05,
        }
        logging.info(f'[ENSEMBLE-WEIGHTS] Using fixed weights to prevent data leakage')
        logging.debug(f'Fixed weights: {fixed_weights}')
        return fixed_weights
    @staticmethod
    def normalize_weights(weights_dict: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights_dict.values())
        if total <= 0:
            raise ValueError(f"Sum of weights must be positive, got {total}")
        normalized = {k: v / total for k, v in weights_dict.items()}
        sum_weights = sum(normalized.values())
        if abs(sum_weights - 1.0) > 1e-6:
            logging.warning(f"Weights don't sum exactly to 1.0: {sum_weights:.10f}")
        logging.debug(f"Normalized {len(weights_dict)} weights: sum={sum_weights:.10f}")
        return normalized
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
        logging.debug('Ensemble ranking with weighted aggregation and multicollinearity penalty')
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
        try:
            weights = self.compute_adaptive_ensemble_weights(df, default_fallback=True)
            logging.info("Using data-driven adaptive ensemble weights")
        except Exception as e:
            logging.warning(f"Adaptive weight computation failed: {e}, using default weights")
            weights_raw = {
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
            weights = self.normalize_weights(weights_raw)
            logging.warning("Using hardcoded default weights (not adaptive)")
        logging.info(f"Ensemble weights normalized: sum={sum(weights.values()):.10f}")
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
        logging.debug(f'Categorized: {len(strong)} strong, {len(medium)} medium, {len(weak)} weak')
        return {
            'strong': strong.tolist(),
            'medium': medium.tolist(),
            'weak': weak.tolist(),
            'strong_count': len(strong),
            'medium_count': len(medium),
            'weak_count': len(weak)
        }
    def quick_prefilter(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100
    ) -> tuple:
        logging.info('[PREFILTER] Using X_train ONLY (C5-3 FIX)')
        dropped_features = []
        constant_features = []
        for col in X_train.columns:
            if X_train[col].var() < 1e-8:
                constant_features.append(col)
                dropped_features.append(col)
        high_missing_features = []
        missing_ratio = X_train.isnull().sum() / len(X_train)
        for col in X_train.columns:
            if col not in dropped_features and missing_ratio[col] > 0.9:
                high_missing_features.append(col)
                dropped_features.append(col)
        X_temp = X_train.drop(columns=dropped_features, errors='ignore')
        if False and X_temp.shape[1] > 1000:
            logging.warning('[C8-LEAKAGE] Model-based prefiltering DISABLED - would cause feature selection leakage')
            logging.warning('[C8-LEAKAGE] Use nested CV for proper feature selection instead')
        X_train_filtered = X_train.drop(columns=dropped_features, errors='ignore')
        if constant_features:
            logging.info(f'[FILTER] Dropped {len(constant_features)} constant/quasi-constant features')
        if high_missing_features:
            logging.info(f'[FILTER] Dropped {len(high_missing_features)} high-missing features (>90%)')
        logging.info(f'[OK] Quick pre-filter: {X_train.shape[1]} -> {X_train_filtered.shape[1]} features (-{len(dropped_features)})')
        logging.info(f'[C3-FIX-VERIFIED] Feature leakage check: Pre-filter uses ONLY statistical filters (constant/missing). No model training = NO feature leakage!')
        return X_train_filtered, dropped_features
    def process_batch(
        self,
        features_df: pd.DataFrame,
        batch_id: int,
        output_dir: str = 'feature_selection_results',
        save_final_model: bool = False
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
            logging.warning('psutil not installed - memory stats not available')
            process = None
            memory_start = 0.0
        logging.debug(f'Batch {batch_id} starting - Memory: {memory_start:.2f} MB')
        if self.ensure_reproducible:
            try:
                is_repro = self.smoke_test_reproducibility(n_samples=min(100, len(features_df)), n_features=min(10, len(features_df.columns)-1))
                if not is_repro:
                    logging.warning('Reproducibility smoke test FAILED - results may vary between runs')
                else:
                    logging.debug('Reproducibility smoke test PASSED')
            except Exception as e:
                logging.warning(f'Reproducibility smoke test error: {e}')
        X = features_df.drop(columns=[self.target_column])
        y = features_df[self.target_column]
        if self.check_lookahead_bias:
            try:
                self.validate_no_lookahead_bias(X)
                self.validate_target_no_lookahead(features_df, target_col=self.target_column)
                price_cols = ['close', 'Close', 'price', 'Price']
                price_col = None
                for col in price_cols:
                    if col in features_df.columns:
                        price_col = col
                        break
                if price_col:
                    self.validate_target_causality(
                        features_df,
                        target_col=self.target_column,
                        price_col=price_col,
                        horizon=self.label_horizon
                    )
                else:
                    logging.warning('[CRITICAL-FIX-1] No price column found - skipping advanced causality tests')
            except ValueError as e:
                logging.error(f'Feature validation failed: {e}')
                raise
        if self.check_stationarity:
            try:
                stationarity_results = self.check_stationarity_adf(X)
                if not stationarity_results.get('skipped', False):
                    non_stat_pct = (stationarity_results['non_stationary_count'] /
                                   stationarity_results['checked_features'] * 100
                                   if stationarity_results['checked_features'] > 0 else 0)
                    if non_stat_pct > 50:
                        logging.warning(
                            f'[STATIONARITY] CRITICAL: {non_stat_pct:.1f}% features are non-stationary. '
                            'Consider fractional differentiation or regime-based modeling.'
                        )
            except Exception as e:
                logging.warning(f'[STATIONARITY] Stationarity check failed: {e}')
        logging.info('Performing temporal split BEFORE preprocessing (data leakage prevention)')
        X_train_raw, X_test_raw, y_train, y_test = self.temporal_split(X, y, label_horizon=self.label_horizon)
        assert len(X_train_raw) + len(X_test_raw) <= len(X), "Train+Test exceeds total samples"
        assert len(X_train_raw) > 0 and len(X_test_raw) > 0, "Empty train or test set"
        logging.info(" FEATURE SELECTION STRATEGY: Two-stage approach")
        logging.info("   Stage 1: Quick prefilter on full train (removes obvious bad features)")
        logging.info("   Stage 2: Nested CV will validate on per-fold selection")
        logging.info("   Reference: FSZ6.md Issue #3 - Feature Selection Leakage")
        X_train_filtered, dropped_features = self.quick_prefilter(X_train_raw, y_train)
        X_test_filtered = X_test_raw.drop(columns=dropped_features, errors='ignore')
        common_cols = [c for c in X_train_filtered.columns if c in X_test_filtered.columns]
        X_train_filtered = X_train_filtered[common_cols]
        X_test_filtered = X_test_filtered[common_cols]
        assert set(X_test_filtered.columns) == set(X_train_filtered.columns), \
            "Test features differ from train features - potential leakage!"
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
        X_train, y_train = self.fit_preprocess(X_train_filtered, y_train)
        X_test = self.transform_preprocess(X_test_filtered)
        logging.info(f'Features: {X_train.shape[1]}, Train Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}')
        logging.info(f' DATA LEAKAGE PREVENTION: fit_transform on train, transform on test (scikit-learn style)')
        large_scale_opts = self._optimize_for_large_feature_sets(
            n_features=X_train.shape[1],
            n_samples=X_train.shape[0]
        )
        if 'shap_sample_size' in large_scale_opts:
            self.shap_sample_size = large_scale_opts['shap_sample_size']
        if 'n_bootstrap_ci' in large_scale_opts:
            original_n_bootstrap_ci = self.n_bootstrap_ci
            self.n_bootstrap_ci = large_scale_opts['n_bootstrap_ci']
        if 'stability_selection_iterations' in large_scale_opts:
            original_stab_iters = self.stability_selection_iterations
            self.stability_selection_iterations = large_scale_opts['stability_selection_iterations']
        multicollinearity = None
        if self.should_detect_multicollinearity:
            multicollinearity = self.detect_multicollinearity(X_train)
            if multicollinearity is not None:
                n_high_corr_pairs = len(multicollinearity.get('high_corr_pairs', []))
                n_features = len(X_train.columns)
                if n_high_corr_pairs > n_features * 0.5:
                    logging.warning(f'EXCESSIVE REDUNDANCY: {n_high_corr_pairs} pairs (>{n_features*0.5:.0f}) with correlation >0.9')
                    logging.warning(f'AUTO-REMOVING redundant features with correlation >0.95')
                    features_before = set(X_train.columns)
                    X_train = self.remove_redundant_features(X_train, y_train, correlation_threshold=0.95)
                    features_after = set(X_train.columns)
                    removed_features = features_before - features_after
                    common_removed = [f for f in removed_features if f in X_test.columns]
                    if common_removed:
                        X_test = X_test.drop(columns=common_removed, errors='ignore')
                        logging.info(f'Removed {len(common_removed)} features from both train and test')
                    common_cols = list(set(X_train.columns) & set(X_test.columns))
                    X_train = X_train[common_cols]
                    X_test = X_test[common_cols]
                    logging.info(f'Updated X_test to match reduced feature set: {len(X_test.columns)} features')
                    multicollinearity = self.detect_multicollinearity(X_train)
        null_rounds = large_scale_opts.get('null_importance_n_rounds', 50)
        n_actual_null = max(10, int(null_rounds * 0.2))
        n_null_rounds = null_rounds - n_actual_null
        null_importance = self.null_importance_ultimate(
            X_train, y_train,
            n_actual=n_actual_null,
            n_null=n_null_rounds
        )
        boosting_ensemble = self.boosting_ensemble_complete(X_train, y_train)
        feature_fraction = self.feature_fraction_analysis(X_train, y_train)
        adversarial = self.adversarial_validation(X_train, X_test)
        X_train_mitigated, shift_aware_weights = self.mitigate_distribution_shift(
            X_train, y_train, adversarial,
            auc_threshold=0.75,
            remove_high_shift=False
        )
        rfe = self.rfe_selection(X_train_mitigated, y_train)
        cv_metrics = self.cv_multi_metric(X_train_mitigated, y_train)
        stability = self.stability_bootstrap(X_train_mitigated, y_train)
        shap_importance = None
        if self.use_shap:
            shap_importance = self.shap_importance_analysis_round2(
                X_train_mitigated, y_train,
                n_runs=5,
                use_sample_weights=True,
                cache_explainer=True
            )
        mutual_info = None
        if self.use_mutual_information:
            try:
                mutual_info = self.mutual_information_scores(X_train_mitigated, y_train)
            except Exception as e:
                logging.warning(f'Mutual Information calculation failed: {e}')
        stability_selection = None
        if self.use_stability_selection:
            try:
                stability_selection = self.stability_selection_framework(
                    X_train_mitigated, y_train,
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
                    X_train_mitigated, y_train,
                    n_outer_splits=3,
                    n_inner_splits=2
                )
            except Exception as e:
                logging.warning(f'Nested CV failed: {e}')
        test_set_validation = None
        if X_test is not None and len(X_test) > 0:
            try:
                logging.info("="*70)
                logging.info("FINAL TEST SET VALIDATION (UNBIASED)")
                logging.info("="*70)
                X_test_mitigated = X_test[X_train_mitigated.columns]
                train_data = lgb.Dataset(X_train_mitigated, label=y_train)
                model = lgb.train(
                    self.base_params,
                    train_data,
                    num_boost_round=500
                )
                y_pred = model.predict(X_test_mitigated)
                y_pred_binary = (y_pred >= 0.5).astype(int) if self.classification else y_pred
                if self.classification:
                    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
                    test_set_validation = {
                        'auc': roc_auc_score(y_test, y_pred),
                        'accuracy': accuracy_score(y_test, y_pred_binary),
                        'precision': precision_score(y_test, y_pred_binary, zero_division=0),
                        'recall': recall_score(y_test, y_pred_binary, zero_division=0),
                        'f1': f1_score(y_test, y_pred_binary, zero_division=0)
                    }
                    logging.info(f"  AUC:       {test_set_validation['auc']:.4f}")
                    logging.info(f"  Accuracy:  {test_set_validation['accuracy']:.4f}")
                    logging.info(f"  Precision: {test_set_validation['precision']:.4f}")
                    logging.info(f"  Recall:    {test_set_validation['recall']:.4f}")
                    logging.info(f"  F1:        {test_set_validation['f1']:.4f}")
                    if nested_cv_results and 'mean_auc' in nested_cv_results:
                        cv_auc = nested_cv_results.get('mean_auc', 0)
                        overfitting_gap = cv_auc - test_set_validation['auc']
                        logging.info(f"\n  [OVERFITTING CHECK]")
                        logging.info(f"    CV AUC:      {cv_auc:.4f}")
                        logging.info(f"    Test AUC:    {test_set_validation['auc']:.4f}")
                        logging.info(f"    Gap:         {overfitting_gap:.4f} {' OVERFITTING DETECTED' if overfitting_gap > 0.05 else ' OK'}")
                else:
                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                    test_set_validation = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                    logging.info(f"  MSE:       {test_set_validation['mse']:.4f}")
                    logging.info(f"  RMSE:      {test_set_validation['rmse']:.4f}")
                    logging.info(f"  MAE:       {test_set_validation['mae']:.4f}")
                    logging.info(f"  R²:        {test_set_validation['r2']:.4f}")
            except Exception as e:
                logging.warning(f'Test Set Validation failed: {e}')
                import traceback
                traceback.print_exc()
        confidence_intervals = None
        if self.use_confidence_intervals:
            try:
                confidence_intervals = self.compute_confidence_intervals(
                    X_train_mitigated, y_train,
                    n_bootstrap=self.n_bootstrap_ci,
                    confidence_level=0.95,
                    importance_type='gain'
                )
            except Exception as e:
                logging.warning(f'Confidence Intervals calculation failed: {e}')
        unstable_features = self.identify_unstable_features(
            feature_names=X_train_mitigated.columns.tolist(),
            shap_importance=shap_importance,
            stability_selection=stability_selection,
            confidence_intervals=confidence_intervals,
            nested_cv_results=nested_cv_results,
            shap_cv_threshold=0.25,
            stability_prob_threshold=0.4,
            ci_width_threshold=0.8,
            cv_importance_cv_threshold=0.35
        )
        df_ranking = self.ensemble_ranking(
            feature_names=X_train_mitigated.columns.tolist(),
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
            logging.debug(f'Added {len(dropped_features)} dropped features to ranking with score=0')
        categorized = self.categorize(df_ranking)
        try:
            logging.info("\n" + "="*70)
            logging.info("BACKTEST ROBUSTNESS ASSESSMENT")
            logging.info("="*70)
            if nested_cv_results and test_set_validation:
                is_performance = nested_cv_results.get('outer_scores', [])
                if self.classification and 'auc' in test_set_validation:
                    oos_performance = [test_set_validation['auc']] * len(is_performance)
                else:
                    oos_performance = [test_set_validation.get('r2', test_set_validation.get('auc', 0))] * len(is_performance)
                pbo_result = self.calculate_probability_of_backtest_overfitting(
                    is_performance=is_performance,
                    oos_performance=oos_performance,
                    n_samples=len(X_train_mitigated)
                )
                if pbo_result['is_overfitted']:
                    logging.error("  HIGH OVERFITTING RISK DETECTED!")
                    logging.error("     Features may be overfitted to this dataset")
                    logging.error("     Consider: retraining with more data or different features")
            if test_set_validation:
                if self.classification:
                    estimated_sr = 0.0
                    if 'sharpe' in test_set_validation:
                        estimated_sr = test_set_validation.get('sharpe', 0.0)
                        logging.info(f"[MinTRL] Using Sharpe from CV metrics: {estimated_sr:.4f}")
                    else:
                        logging.warning(f"[MinTRL] No Sharpe Ratio available - MinTRL not computed")
                        logging.warning(f"[MinTRL] To compute MinTRL, calculate Sharpe from BACKTEST using calculate_real_sharpe_from_backtest()")
                    if estimated_sr > 0:
                        mintrl_result = self.calculate_minimum_track_record_length(
                            estimated_sharpe_ratio=max(0.1, estimated_sr),
                            n_samples=len(X_train_mitigated),
                            confidence_level=0.95
                        )
                        if not mintrl_result['is_sufficient']:
                            years_needed = mintrl_result['deficit'] / 252
                            logging.warning(f" INSUFFICIENT DATA: Need {years_needed:.1f} more years of data")
        except Exception as e:
            logging.warning(f"PBO/MinTRL assessment failed: {e}")
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
        logging.info(f'\n{"="*70}')
        logging.info(f'[COMPLETED] FEATURE SELECTION - Batch {batch_id}')
        logging.info(f'{"="*70}')
        logging.info(f'[TIME] Execution Time: {execution_time:.1f}s (~{execution_time/60:.1f} min)')
        logging.info(f'[MEM] Memory Delta: {memory_delta:+.2f} MB')
        logging.info(f'{"-"*70}')
        logging.info(f'[FEATURES] {len(df_ranking)} total ({categorized["strong_count"]} strong, {categorized["medium_count"]} medium, {categorized["weak_count"]} weak)')
        if nested_cv_results:
            logging.info(f'[MODEL] Performance (Nested CV): {nested_cv_results["mean_score"]:.4f} +/- {nested_cv_results["std_score"]:.4f}')
        if shap_importance:
            stable_shap = np.sum(shap_importance['shap_cv'] < 0.1)
            logging.info(f'[SHAP] {stable_shap}/{len(df_ranking)} stable features (CV < 10%)')
        if null_importance:
            sig_features = np.sum(null_importance.get('significant_gain', []))
            logging.info(f'[NULL-IMP] {sig_features} statistically significant features')
        logging.info(f'{"-"*70}')
        logging.info(f'[TOP-10] Best Features by Score:')
        valid_features = df_ranking[~df_ranking['category'].isin(['dropped', 'dropped_prefilter'])]
        if len(valid_features) == 0:
            logging.warning('   No valid features remaining after filtering!')
            top_10 = df_ranking.nlargest(10, 'score')[['feature', 'score', 'category']]
        else:
            top_10 = valid_features.nlargest(10, 'score')[['feature', 'score', 'category']]
        for idx, row in top_10.iterrows():
            category = row["category"]
            if pd.isna(category):
                category_str = "UNKNOW"
            elif not isinstance(category, str):
                category_str = str(category)[:6]
            else:
                category_str = category[:6]
            score = row["score"]
            if pd.isna(score):
                score = 0.0
            feature = row["feature"]
            if pd.isna(feature):
                feature_str = "UNKNOWN"
            elif not isinstance(feature, str):
                feature_str = str(feature)[:50]
            else:
                feature_str = feature[:50]
            logging.info(f'   {category_str:6s} | {score:7.4f} | {feature_str}')
        logging.info(f'{"="*70}')
        logging.info(f'\n{"="*70}')
        logging.info('[QUALITY ASSESSMENT]')
        logging.info(f'{"="*70}')
        warnings_found = []
        recommendations = []
        if null_importance:
            sig_rate = np.sum(null_importance.get('significant_gain', [])) / len(df_ranking)
            if sig_rate < 0.05:
                warnings_found.append(f'Very low significance rate ({sig_rate:.1%}) - Weak predictive features')
                recommendations.append('Consider feature engineering or gathering more relevant data')
            elif sig_rate > 0.5:
                warnings_found.append(f'High significance rate ({sig_rate:.1%}) - Possible overfitting')
                recommendations.append('Review for data leakage or reduce feature complexity')
        if shap_importance:
            unstable_rate = np.sum(shap_importance['shap_cv'] > 0.2) / len(df_ranking)
            if unstable_rate > 0.3:
                warnings_found.append(f'High SHAP instability ({unstable_rate:.1%} features CV>20%)')
                recommendations.append('SHAP values unreliable - Consider increasing sample size')
        if nested_cv_results:
            mean_score = nested_cv_results['mean_score']
            score_var = nested_cv_results['std_score'] ** 2
            if self.classification and mean_score < 0.55:
                warnings_found.append(f'Model performance near random ({mean_score:.4f} ≈ 0.5)')
                recommendations.append('Critical: Features lack predictive power - Review data quality, feature engineering, or problem formulation')
            elif not self.classification and mean_score < -1e6:
                warnings_found.append(f'Poor regression performance (very high error)')
                recommendations.append('Review feature relevance and data quality')
            if score_var > 0.005:
                warnings_found.append(f'High model variance (var={score_var:.6f})')
                recommendations.append('Model unstable - Review hyperparameters or feature quality')
        if multicollinearity and multicollinearity.get('condition_index', 0) > 30:
            warnings_found.append(f'Severe multicollinearity detected')
            recommendations.append('Apply PCA or remove highly correlated features')
        if warnings_found:
            logging.warning(f'\n  WARNINGS DETECTED ({len(warnings_found)}):')
            for i, warning in enumerate(warnings_found, 1):
                logging.warning(f'    {i}. {warning}')
        else:
            logging.info(f'\n  STATUS: All quality checks PASSED - EXCELLENT')
        if recommendations:
            logging.info(f'\n  RECOMMENDATIONS ({len(recommendations)}):')
            for i, rec in enumerate(recommendations, 1):
                logging.info(f'    {i}. {rec}')
        logging.info(f'{"="*70}\n')
        try:
            if self._dataset_cache is not None:
                self._clear_dataset_cache()
        except Exception:
            pass
    def _save(self, df_ranking: pd.DataFrame, categorized: Dict, cv_metrics: Dict, batch_id: int, output_dir: str, execution_time: float = 0, memory_used: float = 0, memory_start: Optional[float] = None, memory_end: Optional[float] = None, process_pid: Optional[int] = None):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        pd.DataFrame({'feature': categorized['strong']}).to_parquet(
            output_path / 'FSZ_strong.parquet',
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        pd.DataFrame({'feature': categorized['medium']}).to_parquet(
            output_path / 'FSZ_medium.parquet',
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        pd.DataFrame({'feature': categorized['weak']}).to_parquet(
            output_path / 'FSZ_weak.parquet',
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logging.info(f'Saved 3 parquet files: FSZ_strong.parquet, FSZ_medium.parquet, FSZ_weak.parquet')
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
        pd.DataFrame({'feature': categorized['strong']}).to_parquet(
            output_path / 'FSZ_strong.parquet',
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        pd.DataFrame({'feature': categorized['medium']}).to_parquet(
            output_path / 'FSZ_medium.parquet',
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        pd.DataFrame({'feature': categorized['weak']}).to_parquet(
            output_path / 'FSZ_weak.parquet',
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logging.info(f'Saved 3 parquet files: FSZ_strong.parquet, FSZ_medium.parquet, FSZ_weak.parquet')
def main():
    logging.debug('Loading F.parquet...')
    df = pd.read_parquet('F.parquet')
    logging.debug(f'F.parquet loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    if 'target' not in df.columns:
        logging.debug('No target column in F.parquet, trying to load from F2.parquet...')
        try:
            target_df = pd.read_parquet('F2.parquet', columns=['target'])
            if len(target_df) == len(df):
                df['target'] = target_df['target'].values
                logging.debug(f'Target loaded from F2.parquet successfully!')
            else:
                logging.error(f'Row count mismatch: F.parquet={len(df)}, F2.parquet={len(target_df)}')
                return
        except Exception as e:
            logging.error(f'Failed to load target from F2.parquet: {e}')
            return
    logging.info(f'Dataset: {df.shape[0]} samples, {df.shape[1]-1} features')
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
    fs.process_batch(
        features_df=df,
        batch_id=0,
        output_dir='feature_selection_results',
        save_final_model=False
    )
    logging.info('Feature Selection Pipeline completed!')
if __name__ == '__main__':
    main()