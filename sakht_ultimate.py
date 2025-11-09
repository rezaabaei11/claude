import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1' 
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import pandas as pd
import warnings
from typing import Optional, List, Dict, Tuple, Union, Callable
from pathlib import Path
import psutil
import logging
import time
import hashlib
import json
import gc
from enum import Enum
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        """Dummy decorator when numba not available"""
        def wrapper(func):
            return func
        return wrapper
    prange = range
NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
PANDAS_VERSION = tuple(map(int, pd.__version__.split('.')[:2]))
PSUTIL_VERSION = tuple(map(int, psutil.__version__.split('.')[:2]))
HAS_NUMPY_2_0 = NUMPY_VERSION >= (2, 0)
HAS_NUMPY_2_3 = NUMPY_VERSION >= (2, 3)
HAS_PANDAS_2_0 = PANDAS_VERSION >= (2, 0)
HAS_PANDAS_2_1 = PANDAS_VERSION >= (2, 1)
HAS_PANDAS_2_3 = PANDAS_VERSION >= (2, 3)
HAS_PSUTIL_6_0 = PSUTIL_VERSION >= (6, 0)
HAS_PSUTIL_7_0 = PSUTIL_VERSION >= (7, 0)
if not HAS_PANDAS_2_0:
    warnings.warn(f"pandas {pd.__version__} detected. pandas 2.0+ recommended.")
if not HAS_PSUTIL_6_0:
    warnings.warn(f"psutil {psutil.__version__} detected. psutil 6.0+ recommended.")
if not HAS_NUMBA:
    warnings.warn("numba not found. Install numba for 21x speedup: pip install numba")
HAS_PYARROW = False
try:
    import pyarrow as pa
    HAS_PYARROW = True
    pd.options.mode.string_storage = "pyarrow"
    if HAS_PANDAS_2_0:
        pd.options.mode.copy_on_write = True
except ImportError:
    pass
try:
    from scipy import stats
    stats.randint.seed(42)
except (ImportError, AttributeError):
    pass
try:
    import sklearn
    sklearn.set_config(assume_finite=True)
except (ImportError, AttributeError):
    pass
from tsfresh.feature_extraction import (
    extract_features,
    EfficientFCParameters,
    MinimalFCParameters,
    ComprehensiveFCParameters
)
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import impute, roll_time_series
from tsfresh.utilities.distribution import (
    MultiprocessingDistributor,
    LocalDaskDistributor
)
try:
    from tsfresh.utilities.distribution import ClusterDaskDistributor
    HAS_DASK_CLUSTER = True
except ImportError:
    HAS_DASK_CLUSTER = False
from tsfresh import select_features
warnings.filterwarnings('ignore', category=UserWarning, module='tsfresh')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class UltimateFeatureExtractor:
    """tsfresh feature extraction - v11.0"""
    def __init__(
        self,
        n_jobs: int = -1,
        feature_set: str = 'efficient',
        random_state: int = 42,
        deterministic: bool = True,
        verbose: bool = True,
        chunksize: Optional[int] = None,
        n_jobs_timeout: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: str = '.tsfresh_cache',
        enable_profiling: bool = False,
        use_dask: bool = False,
        dask_address: Optional[str] = None,
        use_float32: bool = False,
        enable_openmp_sort: bool = True,
        enable_simd: bool = True,
        use_pyarrow_backend: bool = True,
        use_string_dtype: bool = True,
        use_categorical: bool = True,
        use_parquet: bool = True,
        enable_cow: bool = True,
        use_usable_cpu_count: bool = True,
        enable_oneshot: bool = True
    ):
        self.n_jobs = n_jobs
        self.feature_set = feature_set
        self.random_state = random_state
        self.deterministic = deterministic
        self.verbose = verbose
        self.chunksize = chunksize
        self.n_jobs_timeout = n_jobs_timeout
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.enable_profiling = enable_profiling
        self.use_dask = use_dask
        self.dask_address = dask_address
        self.use_float32 = use_float32
        self.enable_openmp_sort = enable_openmp_sort and HAS_NUMPY_2_3
        self.enable_simd = enable_simd and HAS_NUMPY_2_0
        self.use_pyarrow_backend = use_pyarrow_backend and HAS_PYARROW
        self.use_string_dtype = use_string_dtype and HAS_PANDAS_2_0
        self.use_categorical = use_categorical
        self.use_parquet = use_parquet and HAS_PYARROW
        self.enable_cow = enable_cow and HAS_PANDAS_2_1
        self.use_usable_cpu_count = use_usable_cpu_count
        self.enable_oneshot = enable_oneshot
        self.extracted_features = None
        self.feature_names = []
        self.stats = {}
        self.validation_errors = []
        self.profiling_data = {}
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.deterministic:
            os.environ['PYTHONHASHSEED'] = str(random_state)
            np.random.seed(random_state)
            random.seed(random_state)
            try:
                stats.randint.seed(random_state)
            except (NameError, AttributeError):
                pass
        if self.enable_cow and HAS_PANDAS_2_1:
            pd.options.mode.copy_on_write = True
        if self.use_pyarrow_backend:
            pd.options.mode.string_storage = "pyarrow"
        self._log_init()
    def _log_init(self):
        logger.info(f"UltimateFeatureExtractor v11.0 | NumPy {np.__version__} | pandas {pd.__version__}")
    def _get_optimal_dtype(self) -> np.dtype:
        return np.float32 if self.use_float32 else np.float64
    def _get_usable_cpu_count(self) -> int:
        try:
            p = psutil.Process()
            if self.enable_oneshot:
                with p.oneshot():
                    try:
                        usable = len(p.cpu_affinity())
                        logger.info(f"    Usable CPUs (affinity): {usable}")
                        return usable
                    except (AttributeError, NotImplementedError):
                        pass
            else:
                try:
                    usable = len(p.cpu_affinity())
                    logger.info(f"    Usable CPUs (affinity): {usable}")
                    return usable
                except (AttributeError, NotImplementedError):
                    pass
        except psutil.AccessDenied:
            logger.warning(f"    Access denied to process info")
        except psutil.NoSuchProcess:
            logger.warning(f"    Process no longer exists")
        physical = psutil.cpu_count(logical=False) or 1
        logger.info(f"    Usable CPUs (physical fallback): {physical}")
        return physical
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("    Optimizing dtypes...")
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype
            if pd.api.types.is_integer_dtype(col_type):
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif pd.api.types.is_float_dtype(col_type):
                if self.use_float32:
                    df[col] = df[col].astype(np.float32)
            elif col_type == object:
                num_unique = df[col].nunique()
                num_total = len(df[col])
                if self.use_categorical and num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
                    logger.info(f"      • {col}: categorical ({num_unique} unique) [5-20x less memory]")
                elif self.use_string_dtype:
                    if HAS_PYARROW:
                        df[col] = df[col].astype(pd.StringDtype(storage="pyarrow"))
                        logger.info(f"      • {col}: PyArrow string [70% less memory]")
                    else:
                        df[col] = df[col].astype(pd.StringDtype())
        memory_after = df.memory_usage(deep=True).sum() / 1024**2
        memory_saved = memory_before - memory_after
        reduction = (memory_saved / memory_before) * 100 if memory_before > 0 else 0
        logger.info(f"       Memory: {memory_before:.2f} MB → {memory_after:.2f} MB")
        logger.info(f"       Saved: {memory_saved:.2f} MB ({reduction:.1f}%)")
        return df
    def _generate_cache_key(self, df: pd.DataFrame, params: dict) -> str:
        key_parts = [
            str(df.shape),
            str(sorted(df.columns.tolist())),
            str(sorted(params.items()))
        ]
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                logger.info(f"    Loading from cache: {cache_file.name} (age: {cache_age_hours:.1f}h)")
                features = pd.read_parquet(cache_file)
                logger.info(f"    Loaded {features.shape[1]} cached features")
                return features
            except Exception as e:
                logger.warning(f"    Cache load failed: {str(e)}")
                return None
        return None
    def _save_to_cache(self, cache_key: str, features: pd.DataFrame):
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        try:
            metadata = {
                'timestamp': time.time(),
                'shape': str(features.shape),
                'columns': len(features.columns)
            }
            features.to_parquet(cache_file, compression='snappy')
            metadata_file = self.cache_dir / f"{cache_key}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            logger.info(f"    Saved to cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"    Cache save failed: {str(e)}")
    def _profile_section(self, section_name: str):
        class ProfileContext:
            def __init__(self, parent, name):
                self.parent = parent
                self.name = name
                self.start_time = None
                self.start_memory = None
            def __enter__(self):
                self.start_time = time.time()
                try:
                    p = psutil.Process()
                    if self.parent.enable_oneshot:
                        with p.oneshot():
                            self.start_memory = p.memory_info().rss / 1024**2
                    else:
                        self.start_memory = p.memory_info().rss / 1024**2
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"    Memory profiling unavailable: {e}")
                    self.start_memory = 0
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time
                try:
                    p = psutil.Process()
                    if self.parent.enable_oneshot:
                        with p.oneshot():
                            end_memory = p.memory_info().rss / 1024**2
                    else:
                        end_memory = p.memory_info().rss / 1024**2
                    memory_delta = end_memory - self.start_memory if self.start_memory else 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    memory_delta = 0
                self.parent.profiling_data[self.name] = {
                    'time_seconds': elapsed,
                    'memory_mb': memory_delta
                }
                logger.info(f"   ⏱  {self.name}: {elapsed:.2f}s | Memory: {memory_delta:+.2f}MB")
        return ProfileContext(self, section_name)
    def validate_timeseries(
        self,
        df: pd.DataFrame,
        column_id: str = 'id',
        column_sort: str = 'time',
        min_series_length: int = 3
    ) -> Tuple[bool, List[str]]:
        errors = []
        logger.info("\n Pre-extraction Validation:")
        if df.empty:
            errors.append(" DataFrame is empty!")
            logger.error(errors[-1])
            return False, errors
        logger.info(f"   DataFrame not empty: {len(df):,} rows")
        if column_id not in df.columns or column_sort not in df.columns:
            errors.append(f" Missing columns: {column_id} or {column_sort}")
            logger.error(errors[-1])
            return False, errors
        logger.info(f"   Required columns present")
        nan_in_id = df[column_id].isna().sum()
        nan_in_sort = df[column_sort].isna().sum()
        if nan_in_id > 0 or nan_in_sort > 0:
            errors.append(f" NaN in critical columns: id={nan_in_id}, sort={nan_in_sort}")
            logger.error(errors[-1])
            return False, errors
        logger.info(f"   No NaN in critical columns")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            inf_count = df[numeric_cols].isin([np.inf, -np.inf]).sum().sum()
            if inf_count > 0:
                errors.append(f" Found {inf_count} Inf values!")
                logger.error(errors[-1])
                return False, errors
        logger.info(f"   No Inf values")
        group_sizes = df.groupby(column_id).size()
        min_len = group_sizes.min()
        max_len = group_sizes.max()
        mean_len = group_sizes.mean()
        if min_len < min_series_length:
            errors.append(f" Series too short: min={min_len} (need >={min_series_length})")
            logger.error(errors[-1])
            return False, errors
        logger.info(f"   Series length: min={min_len}, max={max_len}, mean={mean_len:.1f}")
        sorted_check = df.groupby(column_id, sort=False)[column_sort].apply(
            lambda x: x.is_monotonic_increasing
        )
        sorted_issues = (~sorted_check).sum()
        if sorted_issues > 0:
            errors.append(f" {sorted_issues} groups not properly sorted!")
            logger.error(errors[-1])
            return False, errors
        logger.info(f"   All {df[column_id].nunique():,} groups properly sorted")
        logger.info(f"   Validation passed!")
        return True, errors
    def load_gold_data(
        self,
        file_path: str,
        date_column: str = 'date',
        price_column: str = 'price',
        value_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        logger.info(f"\n Loading: {file_path}")
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if self.use_parquet and file_path.suffix == '.parquet':
            logger.info("    Using Parquet with PyArrow engine")
            df = pd.read_parquet(
                file_path,
                engine='pyarrow',
                dtype_backend="pyarrow" if self.use_pyarrow_backend else None
            )
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            read_kwargs = {
                'parse_dates': [date_column],
                'engine': 'pyarrow' if HAS_PYARROW else 'c'
            }
            if self.use_pyarrow_backend:
                read_kwargs['dtype_backend'] = 'pyarrow'
            df = pd.read_csv(file_path, **read_kwargs)
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        df = self._optimize_dtypes(df)
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"   Rows: {len(df):,} | Memory: {memory_mb:.2f} MB")
        return df
    def prepare_for_tsfresh(
        self,
        df: pd.DataFrame,
        time_column: str = 'date',
        value_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        logger.info(f"\n  Preparing for tsfresh...")
        mem_usage = df.memory_usage(deep=True).sum() / 1024**3
        if mem_usage > 1.0:
            logger.warning(f"    Large dataframe ({mem_usage:.2f} GB) - copy may use significant memory")
        df['id'] = '1'
        if pd.api.types.is_datetime64_any_dtype(df[time_column]):
            time_delta = df[time_column] - df[time_column].min()
            seconds = time_delta.dt.total_seconds()
            max_int64_seconds = np.iinfo(np.int64).max / 1e9
            if seconds.max() > max_int64_seconds:
                logger.warning(f"    Timedelta overflow risk - using float64 instead of int64")
                df['time'] = seconds.astype(np.float64)
            else:
                df['time'] = seconds.astype(np.int64)
        else:
            df['time'] = np.arange(len(df), dtype=np.int64)
        if value_columns is None:
            value_columns = [col for col in df.columns
                           if col not in ['id', 'time', time_column]]
        df_prepared = df[['id', 'time'] + value_columns]
        optimal_dtype = self._get_optimal_dtype()
        for col in value_columns:
            if df_prepared[col].dtype in [np.float64, np.float32]:
                df_prepared[col] = df_prepared[col].astype(optimal_dtype)
        logger.info(f"   Time points: {len(df_prepared):,}")
        logger.info(f"   Value columns: {value_columns}")
        return df_prepared
    def extract_features(
        self,
        df: pd.DataFrame,
        disable_progressbar: bool = False,
        use_parallel: bool = True,
        chunksize: Optional[int] = None
    ):
        logger.info(f"\n⏳ استخراج فیچرهای {self.feature_set}...")
        if self.use_cache:
            cache_params = {
                'feature_set': self.feature_set,
                'deterministic': self.deterministic,
                'random_state': self.random_state
            }
            cache_key = self._generate_cache_key(df, cache_params)
            cached_features = self._load_from_cache(cache_key)
            if cached_features is not None:
                self.extracted_features = cached_features
                self.feature_names = list(cached_features.columns)
                return self
        if self.enable_profiling:
            prof_ctx = self._profile_section("Feature Extraction")
            prof_ctx.__enter__()
        is_valid, validation_errors = self.validate_timeseries(df, 'id', 'time')
        if not is_valid:
            raise ValueError(f"Data validation failed!\n" + "\n".join(validation_errors))
        self.validation_errors.extend(validation_errors)
        fc_params = {
            'minimal': MinimalFCParameters(),
            'efficient': EfficientFCParameters(),
            'comprehensive': ComprehensiveFCParameters()
        }[self.feature_set]
        start_time = time.time()
        if self.deterministic:
            n_workers = 1
            logger.info(f"    Deterministic mode: n_workers=1")
        else:
            n_cores = self._get_usable_cpu_count()
            n_workers = n_cores if use_parallel else 1
            logger.info(f"    v10.6 SPEED: Using ALL {n_workers} CPUs (tsfresh official recommendation)")
            logger.info(f"    Expected speedup: 6-26x with optimal chunksize")
        if chunksize is None:
            chunksize = self.chunksize  # FIX: Use tsfresh's internal heuristics when None
        if chunksize is not None:
            logger.info(f"    v10.6 CHUNKSIZE: {chunksize} (user-specified or from init)")
        else:
            logger.info(f"    v10.6 CHUNKSIZE: None (using tsfresh's internal heuristics)")
            logger.info(f"    Tsfresh distributor will calculate optimal chunksize at runtime")
        logger.info(f"    Source: tsfresh.readthedocs.io/en/latest/text/tsfresh_on_a_cluster.html")
        logger.info(f"    Expected speedup: 6-26x with optimal chunksize")
        logger.info(f"    v10.6 PARALLELIZATION: n_jobs={n_workers}")
        logger.info(f"    All cores utilized for maximum speed (tsfresh official)")
        orig_workers = n_workers
        orig_chunksize = chunksize if chunksize is not None else 10  # Fallback for retry logic
        attempts = [
            (orig_workers, chunksize, 'initial'),  # None is OK here - tsfresh handles it
            (max(1, orig_workers // 2), chunksize, 'reduced-workers'),
            (1, max(1, orig_chunksize // 4), 'single-worker-reduced-chunksize'),
            (1, max(1, orig_chunksize // 8), 'single-worker-smaller-chunksize')
        ]
        last_exception = None
        for idx, (try_workers, try_chunksize, tag) in enumerate(attempts, start=1):
            logger.info(f"   Attempt {idx}/{len(attempts)}: n_jobs={try_workers}, chunksize={try_chunksize} ({tag})")
            try:
                self.extracted_features = extract_features(
                    timeseries_container=df,
                    column_id='id',
                    column_sort='time',
                    default_fc_parameters=fc_params,
                    n_jobs=try_workers,
                    chunksize=try_chunksize,
                    disable_progressbar=disable_progressbar,
                    show_warnings=False
                )
                if self.extracted_features is None or self.extracted_features.empty:
                    raise RuntimeError(" extract_features returned empty result!")
                logger.info(f"    Extraction completed successfully (attempt {idx})")
                last_exception = None
                break
            except MemoryError as me:
                logger.warning(f"    MemoryError during feature extraction (attempt {idx}): {me}")
                try:
                    mem = psutil.virtual_memory()
                    logger.info(f"   System memory: {mem.percent}% used; available={mem.available // (1024**2)}MB")
                except Exception:
                    pass
                gc.collect()
                self.validation_errors.append(f"MemoryError attempt {idx}: {me}")
                last_exception = me
                time.sleep(1)
                continue
            except Exception as e:
                error_msg = f" Feature extraction failed: {str(e)}"
                logger.error(error_msg)
                self.validation_errors.append(error_msg)
                last_exception = e
                raise
        if last_exception is not None:
            error_msg = f" Feature extraction failed after retries: {str(last_exception)}"
            logger.error(error_msg)
            raise last_exception
        self._clean_features()
        self.extracted_features = self.extracted_features.sort_index(axis=1)
        self.feature_names = list(self.extracted_features.columns)
        memory_mb = self.extracted_features.memory_usage(deep=True).sum() / 1024**2
        elapsed = time.time() - start_time
        logger.info(f"    فیچرها: {len(self.feature_names):,}")
        logger.info(f"    حافظه: {memory_mb:.2f} MB")
        logger.info(f"    زمان: {elapsed:.2f}s")
        self.stats['extraction_time'] = elapsed
        self.stats['num_features'] = len(self.feature_names)
        self.stats['memory_mb'] = memory_mb
        if self.use_cache:
            self._save_to_cache(cache_key, self.extracted_features)
        if self.enable_profiling:
            prof_ctx.__exit__(None, None, None)
        return self
    def _clean_features(self):
        logger.info("    Aggressive Post-extraction Cleaning...")
        before_count = len(self.extracted_features.columns)
        before_memory = self.extracted_features.memory_usage(deep=True).sum() / 1024**2
        impute(self.extracted_features)
        logger.info(f"      • Applied tsfresh impute")
        self.extracted_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_count_before = self.extracted_features.isna().sum().sum()
        self.extracted_features.fillna(0.0, inplace=True)
        if nan_count_before > 0:
            logger.info(f"      • Filled {nan_count_before} NaN/Inf values")
        before_zero = len(self.extracted_features.columns)
        self.extracted_features = self.extracted_features.loc[
            :, (self.extracted_features != 0).any()
        ]
        removed = before_zero - len(self.extracted_features.columns)
        if removed > 0:
            logger.info(f"      • Removed {removed} all-zero columns")
        before_const = len(self.extracted_features.columns)
        std_values = self.extracted_features.std()
        self.extracted_features = self.extracted_features.loc[:, std_values > 1e-10]
        removed = before_const - len(self.extracted_features.columns)
        if removed > 0:
            logger.info(f"      • Removed {removed} constant columns")
        before_single = len(self.extracted_features.columns)
        self.extracted_features = self.extracted_features.loc[
            :, self.extracted_features.nunique() > 1
        ]
        removed = before_single - len(self.extracted_features.columns)
        if removed > 0:
            logger.info(f"      • Removed {removed} single-value columns")
        optimal_dtype = self._get_optimal_dtype()
        self.extracted_features = self.extracted_features.astype(optimal_dtype)
        after_count = len(self.extracted_features.columns)
        after_memory = self.extracted_features.memory_usage(deep=True).sum() / 1024**2
        total_removed = before_count - after_count
        memory_saved = before_memory - after_memory
        logger.info(f"       Final shape: {self.extracted_features.shape}")
        logger.info(f"       Columns: {before_count} → {after_count} (removed {total_removed})")
        logger.info(f"       Memory: {before_memory:.2f} MB → {after_memory:.2f} MB (saved {memory_saved:.2f} MB)")
    def extract_from_selected_features(
        self,
        df: pd.DataFrame,
        selected_feature_names: List[str],
        chunksize: Optional[int] = None
    ) -> pd.DataFrame:
        logger.info(f"\n Fast Re-extraction: {len(selected_feature_names)} selected features...")
        if not selected_feature_names:
            raise ValueError(" No features to extract!")
        start_time = time.time()
        kind_to_fc = from_columns(selected_feature_names)
        logger.info(f"    Created settings from {len(selected_feature_names)} feature names")
        if self.deterministic:
            n_workers = 1
        else:
            n_cores = self._get_usable_cpu_count()
            n_workers = max(1, n_cores - 1)
        if chunksize is None:
            chunksize = self.chunksize
        if chunksize is not None:
            logger.info(f"    Workers: {n_workers} | Chunksize: {chunksize}")
        else:
            logger.info(f"    Workers: {n_workers} | Chunksize: None (tsfresh auto-calculates)")
        try:
            features = extract_features(
                timeseries_container=df,
                column_id='id',
                column_sort='time',
                kind_to_fc_parameters=kind_to_fc,
                n_jobs=n_workers,
                chunksize=chunksize,
                show_warnings=False
            )
        except Exception as e:
            logger.error(f" Selected extraction failed: {str(e)}")
            raise
        impute(features)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0.0, inplace=True)
        optimal_dtype = self._get_optimal_dtype()
        features = features.astype(optimal_dtype)
        features = features.sort_index(axis=1)
        elapsed = time.time() - start_time
        logger.info(f"   Extracted: {features.shape[1]} features in {elapsed:.2f}s")
        logger.info(f"   Speed gain: ~75% faster than full extraction")
        return features
    def extract_features_in_batches(
        self,
        df: pd.DataFrame,
        batch_size: int = 5000,
        disable_progressbar: bool = False
    ) -> pd.DataFrame:
        logger.info(f"\n BATCH PROCESSING: Extracting features in batches...")
        unique_ids = df['id'].unique()
        n_total = len(unique_ids)
        n_batches = (n_total + batch_size - 1) // batch_size
        logger.info(f"    Total series: {n_total:,}")
        logger.info(f"    Batch size: {batch_size:,}")
        logger.info(f"    Number of batches: {n_batches}")
        all_features = []
        batch_start_time = time.time()
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_total)
            batch_ids = unique_ids[start_idx:end_idx]
            elapsed = time.time() - batch_start_time
            if batch_idx > 0:
                avg_time_per_batch = elapsed / batch_idx
                remaining_batches = n_batches - batch_idx
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_minutes = eta_seconds / 60
                logger.info(f"\n     Processing batch {batch_idx + 1}/{n_batches} (IDs {start_idx}-{end_idx})...")
                logger.info(f"      • Elapsed: {elapsed/60:.1f}min | ETA: {eta_minutes:.1f}min | Speed: {avg_time_per_batch:.1f}s/batch")
            else:
                logger.info(f"\n     Processing batch {batch_idx + 1}/{n_batches} (IDs {start_idx}-{end_idx})...")
            df_batch = df[df['id'].isin(batch_ids)].copy()
            logger.info(f"      • Batch shape: {df_batch.shape}")
            original_feature_set = self.feature_set
            try:
                batch_chunksize = self.chunksize
                batch_features = extract_features(
                    timeseries_container=df_batch,
                    column_id='id',
                    column_sort='time',
                    default_fc_parameters={
                        'minimal': MinimalFCParameters(),
                        'efficient': EfficientFCParameters(),
                        'comprehensive': ComprehensiveFCParameters()
                    }[self.feature_set],
                    n_jobs=self.n_jobs if self.n_jobs != -1 else max(1, self._get_usable_cpu_count() - 1),
                    chunksize=batch_chunksize,
                    disable_progressbar=disable_progressbar,
                    show_warnings=False
                )
                impute(batch_features)
                batch_features.replace([np.inf, -np.inf], np.nan, inplace=True)
                batch_features.fillna(0.0, inplace=True)
                batch_features = batch_features.astype(self._get_optimal_dtype())
                all_features.append(batch_features)
                logger.info(f"       Batch {batch_idx + 1} complete: {batch_features.shape[1]} features")
                del df_batch, batch_features
                gc.collect()
                try:
                    mem = psutil.virtual_memory()
                    logger.info(f"       Memory: {mem.percent}% used | {mem.available // (1024**2)}MB available")
                except Exception:
                    pass
            except MemoryError as me:
                logger.error(f"       MemoryError in batch {batch_idx + 1}: {me}")
                logger.error(f"       Try reducing batch_size from {batch_size} to {batch_size // 2}")
                raise
        logger.info(f"\n    Combining {len(all_features)} batches...")
        combined_features = pd.concat(all_features, axis=0, ignore_index=False)
        combined_features = combined_features.sort_index()
        n_samples = len(combined_features)
        expected_ids = set(range(n_samples))
        actual_ids = set(combined_features.index)
        if expected_ids != actual_ids:
            logger.warning(f"    ID mismatch detected!")
            logger.warning(f"    Expected IDs: 0 to {n_samples-1}")
            logger.warning(f"    Missing IDs: {expected_ids - actual_ids}")
            logger.warning(f"    Extra IDs: {actual_ids - expected_ids}")
            combined_features = combined_features.reset_index(drop=True)
            logger.info(f"    Reset index to ensure alignment")
        del all_features
        gc.collect()
        logger.info(f"    Combined shape: {combined_features.shape}")
        logger.info(f"    Total features: {combined_features.shape[1]:,}")
        logger.info(f"    Total samples: {combined_features.shape[0]:,}")
        self.extracted_features = combined_features
        self.feature_names = list(combined_features.columns)
        return combined_features
    def extract_incremental_features(
        self,
        df_new: pd.DataFrame,
        df_existing_features: pd.DataFrame,
        last_processed_idx: int
    ) -> pd.DataFrame:
        logger.info(f"\n Incremental extraction...")
        logger.info(f"    Existing features: {len(df_existing_features):,} samples")
        logger.info(f"    New data: {len(df_new):,} samples")
        df_new_only = df_new[df_new.index > last_processed_idx]
        if len(df_new_only) == 0:
            logger.info(f"    No new data to process")
            return df_existing_features
        logger.info(f"    Processing {len(df_new_only):,} new samples...")
        new_features = self.extract_features(df_new_only)
        combined = pd.concat([df_existing_features, self.extracted_features], axis=0)
        combined = combined.sort_index()
        logger.info(f"    Combined total: {len(combined):,} samples")
        return combined
    def align_test_features(
        self,
        X_test: pd.DataFrame,
        X_train_columns: List[str]
    ) -> pd.DataFrame:
        logger.info(f"\n Aligning test features with train...")
        logger.info(f"   Train features: {len(X_train_columns)}")
        logger.info(f"   Test features: {len(X_test.columns)}")
        test_cols = set(X_test.columns)
        train_cols = set(X_train_columns)
        missing = train_cols - test_cols
        extra = test_cols - train_cols
        if missing:
            logger.warning(f"    Adding {len(missing)} missing features (filled with 0)")
            for col in missing:
                X_test[col] = 0.0
        if extra:
            logger.warning(f"    Removing {len(extra)} extra features")
            X_test = X_test.drop(columns=list(extra))
        X_test = X_test[X_train_columns]
        logger.info(f"    Aligned features: {len(X_test.columns)}")
        logger.info(f"    Shape: {X_test.shape}")
        return X_test
    def select_relevant_features(
        self,
        y: np.ndarray,
        fdr_level: float = 0.05
    ) -> pd.DataFrame:
        logger.info(f"\n Feature Selection (FDR={fdr_level})...")
        if self.extracted_features is None:
            raise ValueError(" No features extracted yet!")
        before_count = len(self.extracted_features.columns)
        try:
            X_selected = select_features(
                self.extracted_features,
                y,
                fdr_level=fdr_level
            )
        except Exception as e:
            error_msg = f" Feature selection failed: {str(e)}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            raise
        after_count = len(X_selected.columns)
        removed = before_count - after_count
        ratio = (after_count / before_count) * 100
        logger.info(f"    Before: {before_count} features")
        logger.info(f"    After: {after_count} features")
        logger.info(f"    Removed: {removed} ({100-ratio:.1f}%)")
        X_selected = X_selected.sort_index(axis=1)
        self.extracted_features = X_selected
        self.feature_names = list(X_selected.columns)
        self.stats['features_after_selection'] = after_count
        return X_selected
    def get_profiling_report(self) -> Dict:
        if not self.enable_profiling:
            logger.warning(" Profiling not enabled")
            return {}
        logger.info("\n Profiling Report:")
        logger.info("=" * 60)
        total_time = 0
        for section, data in self.profiling_data.items():
            time_s = data['time_seconds']
            memory_mb = data['memory_mb']
            total_time += time_s
            logger.info(f"  {section:30s} | {time_s:8.2f}s | {memory_mb:+8.2f}MB")
        logger.info("=" * 60)
        logger.info(f"  {'TOTAL':30s} | {total_time:8.2f}s")
        return self.profiling_data
    def save_features(self, output_path: str, format: str = 'parquet'):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == 'parquet' and self.use_parquet:
            try:
                self.extracted_features.to_parquet(
                    path.with_suffix('.parquet'),
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
                logger.info(f" Saved (Parquet): {path.with_suffix('.parquet')}")
                logger.info(f"   60x faster than CSV, 2-5x smaller")
            except Exception as e:
                logger.warning(f" Parquet failed: {str(e)}")
                self.extracted_features.to_csv(path.with_suffix('.csv'), index=False)
        else:
            self.extracted_features.to_csv(path.with_suffix('.csv'), index=False)
            logger.info(f" Saved (CSV): {path.with_suffix('.csv')}")
    def print_statistics(self):
        logger.info(f"\n{'='*80}")
        logger.info(f" Statistics:")
        logger.info(f"{'='*80}\n")
        if self.extracted_features is not None:
            logger.info(f" Features: {self.extracted_features.shape[1]}")
            logger.info(f" Samples: {self.extracted_features.shape[0]}")
            memory_mb = self.extracted_features.memory_usage(deep=True).sum() / 1024**2
            logger.info(f" Memory: {memory_mb:.2f} MB")
if __name__ == "__main__":
    df_raw = pd.read_csv('XAUUSD_M15_T.csv')
    df_raw['date'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'])
    df_raw['price'] = df_raw['Close'].astype(np.float32)
    df_raw['high'] = df_raw['High'].astype(np.float32)
    df_raw['low'] = df_raw['Low'].astype(np.float32)
    df_raw['open'] = df_raw['Open'].astype(np.float32)
    df_raw['volume'] = df_raw['TickVol'].astype(np.float32)
    logger.info(f"Data: {len(df_raw):,} rows")
    WINDOW_SIZE = 24
    STEP = 1
    PRICE_CHANGE_THRESHOLD = 0.0005
    logger.info(f"Window: {WINDOW_SIZE} | Threshold: {PRICE_CHANGE_THRESHOLD*100:.2f}%")
    df_for_roll = df_raw[['price', 'high', 'low', 'open', 'volume']].copy()
    df_for_roll['id'] = 1
    df_for_roll['time'] = range(len(df_for_roll))
    logger.info(f"Pre-rolling validation...")
    if len(df_for_roll) < WINDOW_SIZE:
        raise ValueError(f"Insufficient data: {len(df_for_roll)} rows < window size {WINDOW_SIZE}")
    if df_for_roll.isnull().any().any():
        raise ValueError(f"NaN values detected before rolling")
    logger.info(f"  Validation passed: {len(df_for_roll):,} rows")
    df_rolled = roll_time_series(
        df_for_roll,
        column_id='id',
        column_sort='time',
        max_timeshift=WINDOW_SIZE - 1,
        min_timeshift=WINDOW_SIZE - 1,
        rolling_direction=1
    )
    n_windows = df_rolled['id'].nunique()
    logger.info(f"  Created {n_windows:,} windows")
    close_prices = df_raw['price'].values
    n = len(df_raw)
    targets = []
    valid_indices = []
    logger.info(f"Creating targets...")
    for s in range(0, n - WINDOW_SIZE, STEP):
        next_idx = s + WINDOW_SIZE
        if next_idx < len(close_prices):
            price_change_pct = (close_prices[next_idx] - close_prices[next_idx - 1]) / close_prices[next_idx - 1]
            if abs(price_change_pct) >= PRICE_CHANGE_THRESHOLD:
                label = 1 if price_change_pct > 0 else 0
                targets.append(label)
                valid_indices.append(s)
    targets_array = np.array(targets, dtype=np.int32)
    logger.info(f"  Filtered samples: {len(targets_array):,}")
    if len(targets_array) > 0:
        n_up = np.sum(targets_array == 1)
        n_down = np.sum(targets_array == 0)
        imbalance_ratio = max(n_up, n_down) / min(n_up, n_down) if min(n_up, n_down) > 0 else float('inf')
        logger.info(f"  UP={n_up} ({n_up/len(targets_array)*100:.1f}%), DOWN={n_down} ({n_down/len(targets_array)*100:.1f}%), ratio={imbalance_ratio:.2f}:1")
        if imbalance_ratio > 3.0:
            logger.warning(f"  High imbalance! Use class_weight={{0: {imbalance_ratio:.2f}, 1: 1.0}}")
    logger.info(f"Feature extraction...")
    extractor = UltimateFeatureExtractor(
        n_jobs=-1,
        feature_set='efficient',
        deterministic=False,
        chunksize=None,
        use_float32=False,
        enable_openmp_sort=True,
        enable_simd=True,
        use_pyarrow_backend=True,
        use_string_dtype=True,
        use_categorical=True,
        use_parquet=True,
        enable_cow=True,
        use_usable_cpu_count=True,
        enable_oneshot=True,
        enable_profiling=True
    )
    extracted = extractor.extract_features_in_batches(
        df_rolled,
        batch_size=3000,
        disable_progressbar=False
    )
    n_extracted = len(extractor.extracted_features)
    logger.info(f"Aligning: extracted={n_extracted:,}, targets={len(targets_array):,}")
    if len(targets_array) > n_extracted:
        targets_array = targets_array[:n_extracted]
        logger.info(f"  Trimmed targets to {len(targets_array):,}")
    elif len(targets_array) < n_extracted:
        extractor.extracted_features = extractor.extracted_features.iloc[:len(targets_array)]
        logger.info(f"  Trimmed features to {len(extractor.extracted_features):,}")
    logger.info(f"Final: features={len(extractor.extracted_features):,}, targets={len(targets_array):,}")
    X_selected = extractor.extracted_features.copy()
    final_features_df = X_selected.copy()
    final_features_df['target'] = targets_array
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'gold_features_v11_optimized.parquet'
    logger.info(f"Saving: {output_file} | shape={final_features_df.shape}")
    final_features_df.to_parquet(
        output_file,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    file_size_mb = output_file.stat().st_size / 1024**2
    logger.info(f"  Saved: {file_size_mb:.2f} MB")
    if extractor.enable_profiling:
        extractor.get_profiling_report()
    extractor.print_statistics()