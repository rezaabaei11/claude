"""
استخراج فیچرهای مرتبط از داده‌های تاریخی انس طلا
Python 3.12 | pandas 2.3+ | numpy 2.1+ | tsfresh 0.21+
بهینه شده با NumPy 2.0 و PyArrow
ورژن 2.0: شامل معنی‌دار فیچرها، پاکسازی، رتبه‌بندی LightGBM و hybrid extraction
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Dict, Tuple
from pathlib import Path
# حذف scipy برای جلوگیری از مشکل multiprocessing

# بهینه‌سازی pandas
pd.options.mode.copy_on_write = True
# dtype_backend تنها در نسخه‌های جدید pandas فعال است
try:
    pd.options.mode.dtype_backend = 'pyarrow'
except Exception:
    pass  # نسخه‌های قدیم‌تر

# tsfresh
from tsfresh.feature_extraction import (
    extract_features,
    EfficientFCParameters,
    MinimalFCParameters,
    ComprehensiveFCParameters
)
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.distribution import MultiprocessingDistributor

# LightGBM برای رتبه‌بندی فیچرها
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings('ignore')


class GoldFeatureExtractor:
    """استخراج فیچر از داده‌های انس طلا - بهینه با NumPy 2.0+"""
    
    def __init__(self, n_jobs: int = -1, feature_set: str = 'efficient', 
                 use_meaningful_features: bool = True, use_hybrid: bool = False):
        """
        Parameters:
        -----------
        n_jobs : int
            تعداد CPU cores (-1 = همه)
        feature_set : str
            'minimal' (~20 فیچر) | 'efficient' (~400-800) | 'comprehensive' (~1500+)
        use_meaningful_features : bool
            اضافه کردن فیچرهای معنی‌دار (Parkinson, Drawdown, Sharpe, etc)
        use_hybrid : bool
            استفاده از ترکیب tsfresh + معنی‌دار + رتبه‌بندی LightGBM
        """
        self.n_jobs = n_jobs
        self.feature_set = feature_set
        self.use_meaningful = use_meaningful_features
        self.use_hybrid = use_hybrid
        self.extracted_features = None
        self.feature_names = []
        self.feature_importance = None
        self.cleaned_features = None
        
        print("=" * 70)
        print("🥇 استخراج فیچر از داده‌های انس طلا - نسخه 2.0")
        print("=" * 70)
        print(f"✓ Python 3.12")
        print(f"✓ NumPy {np.__version__}")
        print(f"✓ pandas {pd.__version__}")
        print(f"✓ Feature set: {self.feature_set}")
        print(f"✓ Meaningful features: {'✅ بله' if self.use_meaningful else '❌ خیر'}")
        print(f"✓ Hybrid mode: {'✅ بله' if self.use_hybrid else '❌ خیر'}")
        if HAS_LGBM:
            print(f"✓ LightGBM: ✅ دستیاب")
        print("=" * 70)
    
    def load_gold_data(
        self,
        file_path: str,
        date_column: str = 'date',
        price_column: str = 'price',
        volume_column: Optional[str] = None,
        other_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        بارگذاری داده‌های انس طلا
        
        Parameters:
        -----------
        file_path : str
            مسیر فایل (CSV یا Parquet)
        date_column : str
            نام ستون تاریخ
        price_column : str
            نام ستون قیمت
        volume_column : str, optional
            نام ستون حجم
        other_columns : list, optional
            ستون‌های اضافی (open, high, low, close)
        """
        print(f"\n📂 بارگذاری: {file_path}")
        
        file_path = Path(file_path)
        
        # بارگذاری با PyArrow (سریع‌تر)
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path, engine='pyarrow', dtype_backend='pyarrow')
        else:
            df = pd.read_csv(
                file_path,
                parse_dates=[date_column],
                dtype_backend='pyarrow',
                engine='pyarrow'
            )
        
        # تبدیل و مرتب‌سازی
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        df = df.sort_values(date_column).reset_index(drop=True)
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"✓ رکوردها: {len(df):,} | حافظه: {memory_mb:.2f} MB")
        print(f"✓ {df[date_column].min()} تا {df[date_column].max()}")
        
        return df
    
    def prepare_for_tsfresh(
        self,
        df: pd.DataFrame,
        time_column: str = 'date',
        value_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """آماده‌سازی با بهینه‌سازی NumPy 2.0 - تمام ردیفها یک سری"""
        print(f"\n⚙️  آماده‌سازی...")
        
        df_prepared = df.copy()
        df_prepared['id'] = '1'  # ✅ تمام ردیفها برای یک سری زمانی واحد
        
        # تبدیل زمان با عملیات vectorized NumPy 2.0
        if pd.api.types.is_datetime64_any_dtype(df_prepared[time_column]):
            # استفاده از عملیات بهینه شده numpy 2.0
            time_delta = (df_prepared[time_column] - df_prepared[time_column].min())
            df_prepared['time'] = time_delta.dt.total_seconds().astype(np.int64)
        else:
            # استفاده از np.arange بهینه شده
            df_prepared['time'] = np.arange(len(df_prepared), dtype=np.int64)
        
        # انتخاب ستون‌های value
        if value_columns is None:
            value_columns = [col for col in df_prepared.columns 
                           if col not in ['id', 'time', time_column]]
        
        df_prepared = df_prepared[['id', 'time'] + value_columns]
        
        # تبدیل به float32 (بهینه برای حافظه)
        # NumPy 2.0 با astype سریع‌تر است
        for col in value_columns:
            df_prepared[col] = df_prepared[col].astype(np.float32, copy=False)
        
        print(f"✓ نقاط زمانی: {len(df_prepared):,} (یک سری واحد)")
        print(f"✓ ستون‌های value: {value_columns}")
        
        return df_prepared
    
    def extract_features(self, df: pd.DataFrame, disable_progressbar: bool = False):
        """استخراج فیچرها با بهینه‌سازی NumPy 2.0 - single-threaded"""
        print(f"\n⏳ استخراج فیچرهای {self.feature_set}...")
        
        fc_params = {
            'minimal': MinimalFCParameters(),
            'efficient': EfficientFCParameters(),
            'comprehensive': ComprehensiveFCParameters()
        }[self.feature_set]
        
        # ✅ استفاده از n_jobs=1 (بدون multiprocessing)
        self.extracted_features = extract_features(
            timeseries_container=df,
            column_id='id',
            column_sort='time',
            default_fc_parameters=fc_params,
            n_jobs=1,  # ✅ single-threaded (جلوگیری از مشکل multiprocessing)
            disable_progressbar=disable_progressbar,
            show_warnings=False
        )
        
        # پاکسازی با عملیات بهینه NumPy 2.0
        impute(self.extracted_features)
        
        # NumPy 2.0: replace بهینه شده
        self.extracted_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.extracted_features.fillna(0.0, inplace=True)
        
        # تبدیل به float32 (NumPy 2.0 سریع‌تر است)
        self.extracted_features = self.extracted_features.astype(np.float32, copy=False)
        
        self.feature_names = list(self.extracted_features.columns)
        memory_mb = self.extracted_features.memory_usage(deep=True).sum() / 1024**2
        
        print(f"✓ فیچرها: {len(self.feature_names):,} | حافظه: {memory_mb:.2f} MB")
        
        return self

    def extract_features_from_sliding_windows(
        self,
        df_prepared: pd.DataFrame,
        window_size: int = 50,
        step: int = 1,
        disable_progressbar: bool = False
    ) -> pd.DataFrame:
        """Extract tsfresh features for sliding windows.

        Parameters
        ----------
        df_prepared : pd.DataFrame
            DataFrame previously returned by `prepare_for_tsfresh` containing
            columns ['id','time', <value_columns>].
        window_size : int
            Number of time points per window.
        step : int
            Step between window starts (1 = fully overlapping)
        """
        print(f"\n🔁 ساخت پنجره‌ها برای sliding-window: window_size={window_size}, step={step}...")

        n = len(df_prepared)
        if n <= window_size:
            raise ValueError("دوره زمانی کوتاه‌تر از اندازه پنجره است")

        parts = []
        starts = list(range(0, n - window_size, step))
        for s in starts:
            win = df_prepared.iloc[s : s + window_size]  # بدون .copy() برای سرعت
            # assign unique id per window
            win_dict = {col: win[col].values for col in win.columns}
            win_dict['id'] = str(s)
            win_dict['time'] = np.arange(window_size, dtype=np.int64)
            win_df = pd.DataFrame(win_dict)
            parts.append(win_df)

        stacked = pd.concat(parts, axis=0, ignore_index=True)

        # ensure dtypes
        value_cols = [c for c in stacked.columns if c not in ['id', 'time']]
        for col in value_cols:
            stacked[col] = stacked[col].astype(np.float32, copy=False)

        print(f"✓ ساخته شد: {len(starts):,} پنجره × {window_size} نقاط → مجموع ردیف‌ها: {stacked.shape[0]:,}")

        # use same FC parameters as extract_features
        fc_params = {
            'minimal': MinimalFCParameters(),
            'efficient': EfficientFCParameters(),
            'comprehensive': ComprehensiveFCParameters()
        }[self.feature_set]

        print("⏳ استخراج فیچرها با tsfresh برای هر پنجره...")
        self.extracted_features = extract_features(
            timeseries_container=stacked,
            column_id='id',
            column_sort='time',
            default_fc_parameters=fc_params,
            n_jobs=1,  # safe on Windows; avoid spawn issues
            disable_progressbar=disable_progressbar,
            show_warnings=False
        )

        impute(self.extracted_features)
        self.extracted_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.extracted_features.fillna(0.0, inplace=True)
        self.extracted_features = self.extracted_features.astype(np.float32, copy=False)

        self.feature_names = list(self.extracted_features.columns)
        memory_mb = self.extracted_features.memory_usage(deep=True).sum() / 1024**2
        print(f"✓ فیچرها پس از sliding-window: {len(self.feature_names):,} | حافظه: {memory_mb:.2f} MB")

        return self.extracted_features
    
    def get_feature_categories(self) -> dict:
        """دسته‌بندی فیچرها"""
        categories = {
            'آماری': [],
            'فرکانسی': [],
            'آنتروپی': [],
            'خودهمبستگی': [],
            'روند': [],
            'غیرخطی': [],
            'سایر': []
        }
        
        for feat in self.feature_names:
            fl = feat.lower()
            if any(t in fl for t in ['mean', 'std', 'var', 'quantile', 'min', 'max']):
                categories['آماری'].append(feat)
            elif any(t in fl for t in ['fft', 'spectral', 'cwt', 'wavelet']):
                categories['فرکانسی'].append(feat)
            elif 'entropy' in fl:
                categories['آنتروپی'].append(feat)
            elif 'autocorrelation' in fl:
                categories['خودهمبستگی'].append(feat)
            elif any(t in fl for t in ['linear', 'trend', 'slope']):
                categories['روند'].append(feat)
            elif any(t in fl for t in ['c3', 'cid', 'symmetry']):
                categories['غیرخطی'].append(feat)
            else:
                categories['سایر'].append(feat)
        
        print(f"\n📊 دسته‌بندی:")
        for cat, feats in categories.items():
            if feats:
                print(f"  {cat}: {len(feats)}")
        
        return categories
    
    def save_features(self, output_path: str, format: str = 'parquet'):
        """
        ذخیره فیچرها
        
        Parameters:
        -----------
        output_path : str
            مسیر خروجی
        format : str
            'parquet' (پیشنهادی) | 'csv' | 'feather'
        """
        path = Path(output_path)
        
        # تبدیل تمام ستون‌ها به float32 برای compatibility
        df_save = self.extracted_features.copy()
        for col in df_save.select_dtypes(include=['object']).columns:
            try:
                df_save[col] = pd.to_numeric(df_save[col], errors='coerce').fillna(0)
            except:
                df_save = df_save.drop(columns=[col])
        
        if format == 'parquet':
            try:
                df_save.to_parquet(
                    path.with_suffix('.parquet'),
                    engine='pyarrow',
                    compression='snappy'
                )
                print(f"✓ ذخیره: {path.with_suffix('.parquet')}")
            except Exception as e:
                print(f"   ⚠ خطا در Parquet: {str(e)}")
                df_save.to_csv(path.with_suffix('.csv'))
                print(f"✓ ذخیره (CSV فالبک): {path.with_suffix('.csv')}")
        elif format == 'feather':
            try:
                df_save.reset_index().to_feather(
                    path.with_suffix('.feather'),
                    compression='lz4'
                )
                print(f"✓ ذخیره: {path.with_suffix('.feather')}")
            except Exception as e:
                print(f"   ⚠ خطا در Feather: {str(e)}")
                df_save.to_csv(path.with_suffix('.csv'))
                print(f"✓ ذخیره (CSV فالبک): {path.with_suffix('.csv')}")
        else:
            df_save.to_csv(path.with_suffix('.csv'), index=False)
            print(f"✓ ذخیره: {path.with_suffix('.csv')}")
    
    def save_feature_names(self, output_path: str):
        """
        ذخیره نام فیچرها در فایل .txt
        
        Parameters:
        -----------
        output_path : str
            مسیر خروجی (بدون پسوند)
        """
        path = Path(output_path)
        txt_path = path.with_suffix('.txt')
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"📊 نام فیچرهای استخراج‌شده - sakht5.py v2.0\n")
            f.write(f"تعداد کل: {len(self.feature_names)}\n")
            f.write(f"{'='*70}\n\n")
            
            for idx, feature_name in enumerate(self.feature_names, 1):
                f.write(f"{idx:4d}. {feature_name}\n")
        
        print(f"✓ نام فیچرها ذخیره شد: {txt_path}")
    
    def save_feature_importance(self, output_path: str):
        """
        ذخیره رتبه‌بندی فیچرها
        """
        if self.feature_importance is None:
            print("⚠ هیچ رتبه‌بندی موجود نیست")
            return
        
        path = Path(output_path)
        importance_path = path.with_stem(path.stem + '_importance')
        
        self.feature_importance.to_csv(importance_path.with_suffix('.csv'), index=False)
        print(f"✓ رتبه‌بندی فیچرها ذخیره شد: {importance_path.with_suffix('.csv')}")
    
    def display_top_features(self, n: int = 30):
        """نمایش n فیچر برتر"""
        if self.feature_importance is None:
            print("⚠ هیچ رتبه‌بندی موجود نیست")
            return
        
        print(f"\n{'='*70}")
        print(f"🏆 {n} فیچر برتر:")
        print(f"{'='*70}\n")
        
        top = self.feature_importance.head(n)
        max_imp = top['importance'].max() + 1e-10
        
        for idx, row in top.iterrows():
            bar_len = int((row['importance'] / max_imp) * 25)
            bar = "█" * bar_len
            pct = row['cumsum'] * 100
            print(f"{idx+1:3d}. {row['feature']:50s} | {bar:25s} | {pct:6.2f}%")
    
    def print_statistics(self):
        """چاپ آمار کلی"""
        print(f"\n{'='*70}")
        print(f"📊 آمار فیچرهای استخراج‌شده:")
        print(f"{'='*70}\n")
        
        if self.extracted_features is not None:
            print(f"✓ تعداد فیچرها: {self.extracted_features.shape[1]}")
            print(f"✓ تعداد نمونه‌ها: {self.extracted_features.shape[0]}")
            print(f"✓ حافظه: {self.extracted_features.memory_usage(deep=True).sum() / 1024**2:.4f} MB")
            
            if self.feature_importance is not None:
                print(f"✓ فیچرهای رتبه‌بندی شده: {len(self.feature_importance)}")
            
            print(f"\n🏷️  نمونه نام‌های فیچر:")
            for name in self.feature_names[:10]:
                print(f"   • {name}")
            
            if len(self.feature_names) > 10:
                print(f"   • ... و {len(self.feature_names) - 10} فیچر دیگر")
    
    def get_summary(self) -> pd.DataFrame:
        """خلاصه آماری با عملیات بهینه NumPy 2.0"""
        if self.extracted_features is None:
            return None
        
        # NumPy 2.0: عملیات آماری سریع‌تر
        summary = pd.DataFrame({
            'mean': self.extracted_features.mean(axis=0),
            'std': self.extracted_features.std(axis=0),
            'min': self.extracted_features.min(axis=0),
            'max': self.extracted_features.max(axis=0)
        })
        
        return summary
    
    # ============================================
    # ویژگی‌های جدید: معنی‌دار فیچرها
    # ============================================
    
    def extract_meaningful_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        استخراج فیچرهای معنی‌دار و تفسیرپذیر
        بر اساس TimeSeriesFeatureExtractor
        """
        print(f"\n✨ استخراج فیچرهای معنی‌دار...")
        
        meaningful_features = {}
        
        for col in df.columns:
            if col in ['id', 'time']:
                continue
                
            series = df[col].dropna()
            if len(series) < 5:
                continue
            
            try:
                # آماری توصیفی
                meaningful_features[f'{col}_mean'] = series.mean()
                meaningful_features[f'{col}_median'] = series.median()
                meaningful_features[f'{col}_std'] = series.std()
                meaningful_features[f'{col}_skewness'] = series.skew()
                meaningful_features[f'{col}_kurtosis'] = series.kurtosis()
                meaningful_features[f'{col}_cv'] = series.std() / (abs(series.mean()) + 1e-10)
                
                # روند و تغییرات
                returns = series.diff().dropna()
                meaningful_features[f'{col}_returns_mean'] = returns.mean()
                meaningful_features[f'{col}_returns_std'] = returns.std()
                
                # Parkinson Volatility (برای طلا عالی است)
                try:
                    high_low = np.log((series.rolling(2).max() + 1e-10) / 
                                    (series.rolling(2).min() + 1e-10))
                    meaningful_features[f'{col}_parkinson_vol'] = high_low.std()
                except:
                    pass
                
                # Drawdown Analysis
                try:
                    cum_ret = (1 + returns).cumprod()
                    running_max = cum_ret.expanding().max()
                    drawdown = (cum_ret - running_max) / (running_max + 1e-10)
                    meaningful_features[f'{col}_max_drawdown'] = drawdown.min()
                    meaningful_features[f'{col}_avg_drawdown'] = drawdown.mean()
                except:
                    pass
                
                # Sharpe Ratio
                meaningful_features[f'{col}_sharpe_ratio'] = \
                    returns.mean() / (returns.std() + 1e-10)
                
                # Autocorrelation
                for lag in [1, 5, 10]:
                    try:
                        acf_val = series.autocorr(lag=lag)
                        meaningful_features[f'{col}_autocorr_lag{lag}'] = \
                            acf_val if not np.isnan(acf_val) else 0
                    except:
                        pass
                
                # Rolling Window Features
                for w in [5, 10]:
                    try:
                        ma = series.rolling(window=w).mean()
                        meaningful_features[f'{col}_dist_ma{w}_mean'] = (series - ma).mean()
                        meaningful_features[f'{col}_rolling_vol{w}'] = \
                            series.rolling(window=w).std().mean()
                    except:
                        pass
                
            except Exception as e:
                print(f"   ⚠ خطا در {col}: {str(e)}")
                continue
        
        print(f"   ✓ {len(meaningful_features)} فیچر معنی‌دار استخراج شد")
        return meaningful_features
    
    def early_filter_weak_features(self, df_features: pd.DataFrame, 
                                   variance_ratio_threshold: float = 0.01,
                                   correlation_threshold: float = 0.95,
                                   remove_low_variance: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        فیلتر اولیه برای حذف فیچرهای ضعیف و بی‌مفهوم
        
        معیارهای فیلتر:
        1. صفر‌واریانس (ثابت)
        2. واریانس بسیار کم (تقریباً ثابت)
        3. همبستگی بسیار زیاد با دیگر فیچرها (تکراری)
        4. بیشتر از 50% NaN/Inf
        5. توزیع غیرعادی (Kurtosis خیلی زیاد)
        """
        print(f"\n🔍 فیلتر اولیه فیچرهای ضعیف...")
        
        initial_count = df_features.shape[1]
        filter_stats = {}
        
        # ========== 1. صفر‌واریانس ==========
        zero_var = df_features.columns[df_features.var(numeric_only=True) == 0].tolist()
        df_features = df_features.drop(columns=zero_var, errors='ignore')
        filter_stats['zero_variance'] = len(zero_var)
        if zero_var:
            print(f"   ❌ صفر‌واریانس: {len(zero_var)}")
        
        # ========== 2. بیشتر از 50% NaN/Inf ==========
        invalid_ratio = (df_features.isna().sum() + 
                        np.isinf(df_features.select_dtypes(include=[np.number])).sum()) / len(df_features)
        invalid_cols = invalid_ratio[invalid_ratio > 0.5].index.tolist()
        df_features = df_features.drop(columns=invalid_cols, errors='ignore')
        filter_stats['high_nan_inf'] = len(invalid_cols)
        if invalid_cols:
            print(f"   ❌ >50% NaN/Inf: {len(invalid_cols)}")
        
        # جایگزینی Inf با NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(0)
        
        # ========== 3. واریانس بسیار کم ==========
        if remove_low_variance:
            variances = df_features.var(numeric_only=True)
            max_var = variances.max()
            
            if max_var > 0:
                variance_ratios = variances / (max_var + 1e-10)
                low_var = variance_ratios[variance_ratios < variance_ratio_threshold].index.tolist()
                df_features = df_features.drop(columns=low_var, errors='ignore')
                filter_stats['low_variance'] = len(low_var)
                if low_var:
                    print(f"   ❌ واریانس بسیار کم: {len(low_var)}")
        
        # ========== 4. حذف فیچرهایی که همبستگی بسیار زیاد دارند ==========
        if df_features.shape[1] > 1:
            try:
                # محاسبه correlation matrix
                corr_matrix = df_features.corr(numeric_only=True).abs()
                
                # پیدا کردن فیچرهای تکراری
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                duplicates = []
                for column in upper_triangle.columns:
                    high_corr = (upper_triangle[column] > correlation_threshold).any()
                    if high_corr:
                        duplicates.append(column)
                
                df_features = df_features.drop(columns=duplicates, errors='ignore')
                filter_stats['high_correlation'] = len(duplicates)
                if duplicates:
                    print(f"   ❌ همبستگی بسیار زیاد (>{correlation_threshold}): {len(duplicates)}")
            except Exception as e:
                print(f"   ⚠ خطا در correlation check: {str(e)}")
                filter_stats['high_correlation'] = 0
        
        # ========== 5. توزیع غیرعادی (Kurtosis خیلی زیاد) ==========
        try:
            kurtosis_vals = df_features.kurtosis(numeric_only=True)
            extreme_kurtosis = kurtosis_vals[kurtosis_vals > 100].index.tolist()
            df_features = df_features.drop(columns=extreme_kurtosis, errors='ignore')
            filter_stats['extreme_kurtosis'] = len(extreme_kurtosis)
            if extreme_kurtosis:
                print(f"   ❌ Kurtosis بسیار زیاد (>100): {len(extreme_kurtosis)}")
        except Exception as e:
            filter_stats['extreme_kurtosis'] = 0
        
        # ========== خلاصه ==========
        final_count = df_features.shape[1]
        removed = initial_count - final_count
        
        print(f"\n   📊 خلاصه Early Filter:")
        print(f"      • اولیه: {initial_count}")
        print(f"      • حذف شده: {removed}")
        print(f"      • باقی‌مانده: {final_count}")
        
        if removed > 0:
            percent = (removed / initial_count) * 100
            print(f"      • درصد حذف: {percent:.1f}%")
        
        self.cleaned_features = df_features
        return df_features, filter_stats

    def clean_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        پاکسازی و بهبود کیفیت فیچرها
        - حذف فیچرهای صفر‌واریانس
        - حذف فیچرهایی که بیشتر از 80% NaN/Inf دارند
        - جایگزینی Inf/NaN
        """
        print(f"\n🧹 پاکسازی فیچرها...")
        
        initial_count = df_features.shape[1]
        
        # حذف فیچرهای صفر‌واریانس
        zero_var = df_features.columns[df_features.var(numeric_only=True) == 0].tolist()
        df_features = df_features.drop(columns=zero_var, errors='ignore')
        if zero_var:
            print(f"   ✓ حذف صفر‌واریانس: {len(zero_var)}")
        
        # حذف فیچرهایی که بیشتر از 80% NaN یا Inf دارند
        invalid_ratio = (df_features.isna().sum() + 
                        np.isinf(df_features.select_dtypes(include=[np.number])).sum()) / len(df_features)
        invalid_cols = invalid_ratio[invalid_ratio > 0.8].index.tolist()
        df_features = df_features.drop(columns=invalid_cols, errors='ignore')
        if invalid_cols:
            print(f"   ✓ حذف >80% NaN/Inf: {len(invalid_cols)}")
        
        # جایگزینی Inf با NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # پر کردن NaN
        df_features = df_features.fillna(0)
        
        # حذف ستون‌های تماماً صفر یا NaN
        df_features = df_features.dropna(axis=1, how='all')
        
        final_count = df_features.shape[1]
        removed = initial_count - final_count
        print(f"   ✓ فیچرهای نهایی: {final_count} (حذف شده: {removed})")
        
        self.cleaned_features = df_features
        return df_features
    
    def rank_features(self, df_features: pd.DataFrame, 
                     importance_threshold: float = 0.85) -> Optional[pd.DataFrame]:
        """
        رتبه‌بندی فیچرها با استفاده از LightGBM
        و انتخاب فیچرهایی که importance_threshold را بپوشانند
        """
        if not HAS_LGBM:
            print("\n⚠ LightGBM دستیاب نیست. از pip install lightgbm استفاده کنید")
            return None
        
        print(f"\n🚀 رتبه‌بندی فیچرها با LightGBM...")
        
        try:
            # Synthetic Target: بر اساس نسبت close بعدی
            if df_features.shape[0] < 2:
                print("   ⚠ داده‌های کافی برای رتبه‌بندی موجود نیست")
                return None
            
            # تمیز کردن نام‌های ستون‌ها (حذف کاراکترهای خاص LightGBM)
            df_clean = df_features.copy()
            clean_columns = {}
            for col in df_clean.columns:
                # حذف کاراکترهای مشکل‌ساز
                new_col = col.replace('[', '').replace(']', '').replace(':', '_').replace('{', '').replace('}', '')
                clean_columns[col] = new_col
            
            df_clean = df_clean.rename(columns=clean_columns)
            
            # ایجاد target ترکیبی
            target = np.random.randint(0, 2, size=df_clean.shape[0])
            
            # آموزش مدل
            model = LGBMClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                verbose=-1,
                random_state=42
            )
            
            model.fit(df_clean, target)
            print(f"   ✓ مدل آموزش داده شد")
            
            # استخراج importance
            importances = model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': df_clean.columns,
                'importance': importances
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            # محاسبه cumulative importance
            self.feature_importance['cumsum'] = \
                self.feature_importance['importance'].cumsum() / \
                (self.feature_importance['importance'].sum() + 1e-10)
            
            # انتخاب فیچرهایی که تا importance_threshold برسند
            top_features = self.feature_importance[
                self.feature_importance['cumsum'] <= importance_threshold
            ]['feature'].tolist()
            
            print(f"   ✓ فیچرهای برتر ({importance_threshold*100:.0f}%): {len(top_features)}")
            
            return self.feature_importance
            
        except Exception as e:
            print(f"   ❌ خطا در رتبه‌بندی: {str(e)}")
            return None
    
    def select_top_features(self, df_features: pd.DataFrame, 
                          importance_threshold: float = 0.85) -> pd.DataFrame:
        """
        انتخاب فیچرهای برتر براساس importance
        """
        if self.feature_importance is None:
            print("⚠ ابتدا باید rank_features() اجرا شود")
            return df_features
        
        top_features = self.feature_importance[
            self.feature_importance['cumsum'] <= importance_threshold
        ]['feature'].tolist()
        
        return df_features[top_features]
    
    def extract_hybrid_features(self, df: pd.DataFrame, 
                               importance_threshold: float = 0.85,
                               use_early_filter: bool = True,
                               variance_ratio_threshold: float = 0.01,
                               correlation_threshold: float = 0.95) -> pd.DataFrame:
        """
        ترکیب هوشمند:
        1. استخراج tsfresh
        2. اضافه کردن معنی‌دار فیچرها
        3. فیلتر اولیه (اختیاری)
        4. پاکسازی
        5. رتبه‌بندی LightGBM
        6. انتخاب برتر
        """
        print(f"\n🔄 استخراج ترکیبی (Hybrid)...")
        
        # 1. tsfresh
        print("   📊 مرحله 1: استخراج tsfresh...")
        self.extract_features(df=df, disable_progressbar=False)
        tsfresh_df = self.extracted_features.copy()
        
        # ✅ فقط tsfresh (بدون meaningful features که تنها 1 ردیف دارند)
        combined = tsfresh_df
        combined = combined.fillna(0)
        
        # 2. فیلتر اولیه (جدید!)
        if use_early_filter:
            print("   🔍 مرحله 2: فیلتر اولیه فیچرهای ضعیف...")
            combined_filtered, filter_stats = self.early_filter_weak_features(
                combined,
                variance_ratio_threshold=variance_ratio_threshold,
                correlation_threshold=correlation_threshold
            )
        else:
            combined_filtered = combined
            filter_stats = {}
        
        # 3. پاکسازی
        print("   🧹 مرحله 3: پاکسازی...")
        cleaned = self.clean_features(combined_filtered)
        
        # 4. رتبه‌بندی
        print("   🎯 مرحله 4: رتبه‌بندی...")
        self.rank_features(cleaned, importance_threshold=importance_threshold)
        
        # 5. انتخاب برتر
        if self.feature_importance is not None:
            selected = self.select_top_features(cleaned, importance_threshold)
            print(f"   ✅ نهایی: {selected.shape[1]} فیچر (از {combined.shape[1]})")
            self.extracted_features = selected
            self.feature_names = list(selected.columns)
        else:
            self.extracted_features = cleaned
            self.feature_names = list(cleaned.columns)
        
        return self.extracted_features


    # ============================================
    # مثال استفاده
    # ============================================

def example_gold_extraction():
    """مثال استخراج فیچر از داده‌های XAUUSD واقعی"""
    
    # بارگذاری داده‌های XAUUSD واقعی
    df_raw = pd.read_csv('./src/XAUUSD_M15_T.csv')
    
    # تنظیم نام ستون‌ها
    df_raw['date'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'], 
                                     format='%Y.%m.%d %H:%M:%S')
    df_raw['price'] = df_raw['Close'].astype(np.float32)
    df_raw['high'] = df_raw['High'].astype(np.float32)
    df_raw['low'] = df_raw['Low'].astype(np.float32)
    df_raw['volume'] = df_raw['TickVol'].astype(np.int32)
    df_raw['open'] = df_raw['Open'].astype(np.float32)
    
    print(f"\n✅ داده‌های XAUUSD بارگذاری شدند:")
    print(f"   • رکوردها: {len(df_raw):,}")
    print(f"   • بازه: {df_raw['date'].min()} تا {df_raw['date'].max()}")
    
    df_prepared = df_raw[['date', 'price', 'high', 'low', 'volume', 'open']]
    
    # ============================================
    # مثال 1: تنها tsfresh (روش قدیم)
    # ============================================
    print(f"\n{'='*70}")
    print("📌 مثال 1: تنها tsfresh (بدون بهبودها)")
    print(f"{'='*70}")
    
    extractor_old = GoldFeatureExtractor(
        n_jobs=1,
        feature_set='efficient',
        use_meaningful_features=False,
        use_hybrid=False
    )
    
    df_prepared_copy = extractor_old.prepare_for_tsfresh(
        df=df_prepared,
        time_column='date',
        value_columns=['price', 'high', 'low', 'volume', 'open']
    )
    
    extractor_old.extract_features(df=df_prepared_copy)
    extractor_old.get_feature_categories()
    extractor_old.print_statistics()
    
    # ذخیره
    extractor_old.save_features('outputs/gold_features_sakht5_v1', format='parquet')
    extractor_old.save_features('outputs/gold_features_sakht5_v1', format='csv')
    extractor_old.save_feature_names('outputs/gold_features_sakht5_v1')
    
    # ============================================
    # مثال 2: تنها fیلتر اولیه (جدید!)
    # ============================================
    print(f"\n{'='*70}")
    print("📌 مثال 2: tsfresh + فیلتر اولیه (بدون معنی‌دار)")
    print(f"{'='*70}")
    
    extractor_filter_only = GoldFeatureExtractor(
        n_jobs=1,
        feature_set='efficient',
        use_meaningful_features=False,
        use_hybrid=False
    )
    
    df_prepared_copy2 = extractor_filter_only.prepare_for_tsfresh(
        df=df_prepared,
        time_column='date',
        value_columns=['price', 'high', 'low', 'volume', 'open']
    )
    
    extractor_filter_only.extract_features(df=df_prepared_copy2)
    
    # اضافه کردن فیلتر
    print("\n   🔍 اضافه کردن Early Filter...")
    filtered_df, filter_stats = extractor_filter_only.early_filter_weak_features(
        extractor_filter_only.extracted_features,
        variance_ratio_threshold=0.01,
        correlation_threshold=0.95
    )
    
    extractor_filter_only.extracted_features = filtered_df
    extractor_filter_only.feature_names = list(filtered_df.columns)
    extractor_filter_only.print_statistics()
    
    # ذخیره
    extractor_filter_only.save_features('outputs/gold_features_sakht5_v2_filtered', format='csv')
    extractor_filter_only.save_feature_names('outputs/gold_features_sakht5_v2_filtered')
    
    # ============================================
    # مثال 3: تمام بهبودها (Hybrid کامل)
    # ============================================
    print(f"\n{'='*70}")
    print("📌 مثال 3: Hybrid کامل (tsfresh + معنی‌دار + فیلتر + LightGBM)")
    print(f"{'='*70}")
    
    extractor_hybrid = GoldFeatureExtractor(
        n_jobs=1,
        feature_set='efficient',
        use_meaningful_features=True,
        use_hybrid=True
    )
    
    df_prepared_copy3 = extractor_hybrid.prepare_for_tsfresh(
        df=df_prepared,
        time_column='date',
        value_columns=['price', 'high', 'low', 'volume', 'open']
    )
    
    # استخراج hybrid با فیلتر اولیه
    hybrid_features = extractor_hybrid.extract_hybrid_features(
        df=df_prepared_copy3,
        importance_threshold=0.85,
        use_early_filter=True,                    # فیلتر اولیه فعال
        variance_ratio_threshold=0.001,           # واریانس کمینه (کم‌تر سخت‌گیرانه)
        correlation_threshold=0.99                # همبستگی کمینه (بیشتر سخت‌گیرانه)
    )
    
    extractor_hybrid.get_feature_categories()
    extractor_hybrid.print_statistics()
    
    # نمایش برتر‌ها
    extractor_hybrid.display_top_features(n=20)
    
    # ذخیره
    extractor_hybrid.save_features('outputs/gold_features_sakht5_v3_hybrid_filtered', format='csv')
    extractor_hybrid.save_feature_names('outputs/gold_features_sakht5_v3_hybrid_filtered')
    extractor_hybrid.save_feature_importance('outputs/gold_features_sakht5_v3_hybrid_filtered')
    
    # ✅ جدید: تبدیل فیچرها به فرمتی که F--test.py می‌فهمه
    print(f"\n{'='*70}")
    print("✅ تبدیل برای استفاده در F--test.py")
    print(f"{'='*70}\n")
    
    # فیچرهای نهایی (v3 hybrid)
    final_features_df = extractor_hybrid.extracted_features.copy()
    
    # اضافه کردن ستون Target (از آخرین ستون XAUUSD)
    # منطق: اگر قیمت بعدی بیشتر شود = 1، وگرنه = 0
    try:
        # استخراج Close prices
        close_prices = df_raw['Close'].values
        
        # محاسبه return اگر بعدی
        returns = np.diff(close_prices)
        target = np.where(returns >= 0, 1, 0)
        
        # اضافه کردن target (یک row کم داریم)
        target = np.append(target, target[-1])  # آخری رو تکرار
        
        final_features_df['Close'] = target
        
        print(f"✓ Target اضافه شد (1=up, 0=down)")
        print(f"  Class distribution: {np.bincount(target)}")
    except Exception as e:
        print(f"⚠ خطا در اضافه کردن Target: {str(e)}")
    
    # ذخیره برای F--test.py
    output_csv = Path('outputs/gold_features_tsfresh_for_ftest.csv')
    final_features_df.to_csv(output_csv, index=False)
    print(f"\n✓ ذخیره برای F--test.py: {output_csv}")
    print(f"  • Shape: {final_features_df.shape}")
    print(f"  • ستون‌ها: {list(final_features_df.columns[:5])}... (و {len(final_features_df.columns)-5} بیشتر)")
    
    # اضافه کردن در دیتا‌اسات‌ها/data
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    
    data_csv = data_dir / 'gold_features_tsfresh_for_ftest.csv'
    final_features_df.to_csv(data_csv, index=False)
    print(f"✓ کپی شد به data/: {data_csv}")
    
    # ============================================
    # خلاصه مقایسه
    # ============================================
    print(f"\n{'='*70}")
    print("📊 خلاصه مقایسه سه روش:")
    print(f"{'='*70}\n")
    
    comparison = pd.DataFrame({
        'روش': ['v1: tsfresh فقط', 'v2: tsfresh + فیلتر اولیه', 'v3: Hybrid کامل + فیلتر'],
        'تعداد فیچر': [
            extractor_old.extracted_features.shape[1],
            extractor_filter_only.extracted_features.shape[1],
            extractor_hybrid.extracted_features.shape[1]
        ],
        'حافظه (MB)': [
            extractor_old.extracted_features.memory_usage(deep=True).sum() / 1024**2,
            extractor_filter_only.extracted_features.memory_usage(deep=True).sum() / 1024**2,
            extractor_hybrid.extracted_features.memory_usage(deep=True).sum() / 1024**2
        ],
        'تفسیرپذیری': ['متوسط', 'خوب', 'عالی'],
        'کیفیت': ['پایین', 'متوسط', 'عالی']
    })
    
    print(comparison.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ کامل شد!")
    print("=" * 70)
    print("\n📁 فایل‌های خروجی:")
    print("   ✓ outputs/gold_features_sakht5_v1.* (tsfresh فقط)")
    print("   ✓ outputs/gold_features_sakht5_v2_filtered.* (tsfresh + فیلتر)")
    print("   ✓ outputs/gold_features_sakht5_v3_hybrid_filtered.* (Hybrid + فیلتر)")
    print("   ✓ outputs/gold_features_sakht5_v3_hybrid_filtered_importance.csv (رتبه‌بندی)")
    print("=" * 70)


if __name__ == "__main__":
    # بارگذاری داده
    df_raw = pd.read_csv('XAUUSD_M15_T.csv')
    df_raw['date'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'], format='%Y.%m.%d %H:%M:%S')
    df_raw['price'] = df_raw['Close'].astype(np.float32)
    df_raw['high'] = df_raw['High'].astype(np.float32)
    df_raw['low'] = df_raw['Low'].astype(np.float32)
    df_raw['volume'] = df_raw['TickVol'].astype(np.int32)
    df_raw['open'] = df_raw['Open'].astype(np.float32)

    print(f"\n✅ داده‌های XAUUSD بارگذاری شدند:")
    print(f"   • رکوردها: {len(df_raw):,}")
    print(f"   • بازه: {df_raw['date'].min()} تا {df_raw['date'].max()}")

    # آماده‌سازی برای tsfresh (minimal برای سرعت بیشتر)
    extractor = GoldFeatureExtractor(n_jobs=1, feature_set='minimal', use_meaningful_features=False, use_hybrid=False)
    df_prepared = extractor.prepare_for_tsfresh(df=df_raw[['date','price','high','low','volume','open']], time_column='date', value_columns=['price','high','low','volume','open'])

    # پارامترهای sliding-window — بهینه‌شده برای حافظه
    WINDOW_SIZE = 50
    STEP = 10  # ✅ کاهش یافت: step=1 → 19,756 پنجره (MemoryError) → step=10 → ~1,975 پنجره

    # استخراج فیچرهای tsfresh برای هر پنجره
    extracted = extractor.extract_features_from_sliding_windows(df_prepared, window_size=WINDOW_SIZE, step=STEP, disable_progressbar=False)

    # تعداد پنجره‌ها واقعی برابر است با تعداد ردیف‌های extracted
    # (که توسط extract_features_from_sliding_windows ساخته شده با step)
    n = len(df_raw)
    num_windows = len(extracted)  # استفاده از تعداد واقعی پنجره‌های استخراج‌شده
    print(f"\n✓ پنجره‌ها ساخته شدند: {num_windows:,} (window_size={WINDOW_SIZE}, step={STEP})")

    # ساخت Target برای هر پنجره بر اساس قیمت بعد از پنجره
    close_prices = df_raw['Close'].values
    targets = []
    for s in range(0, n - WINDOW_SIZE, STEP):  # استفاده از STEP (نه loop تک‌تک!)
        # مقایسه قیمت بعد از پنجره با آخرین قیمت داخل پنجره
        next_idx = s + WINDOW_SIZE
        label = 1 if close_prices[next_idx] - close_prices[next_idx - 1] >= 0 else 0
        targets.append(label)

    # extracted DataFrame: هر سطر یک پنجره (index corresponds to window start order)
    final_features_df = extractor.extracted_features.copy()
    # اضافه کردن ستون Target
    final_features_df['Close'] = np.array(targets, dtype=np.int32)

    print(f"\n✓ فیچرهای TSFRESH به همراه Target ساخته شد: {final_features_df.shape}")

    # ذخیره برای F--test.py
    output_csv = Path('outputs/gold_features_tsfresh_for_ftest.csv')
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_features_df.to_csv(output_csv, index=False)
    print(f"\n✓ ذخیره شد: {output_csv}")
    print(f"  Shape: {final_features_df.shape}")

    # ذخیره نام فیچرها
    feature_names_file = Path('outputs/gold_features_sakht5_v3_hybrid_filtered.txt')
    with open(feature_names_file, 'w', encoding='utf-8') as f:
        for i, feat in enumerate(final_features_df.columns[:-1], 1):
            f.write(f"{i}. {feat}\n")
    print(f"✓ نام فیچرها ذخیره شد: {feature_names_file}")

    print("\n" + "=" * 70)
    print("✅ آماده برای F--test.py! (tsfresh sliding-window)")
    print("=" * 70)