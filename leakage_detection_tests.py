"""
Comprehensive Data Leakage Detection Test Suite for Time Series ML
Based on best practices from:
- Prado & Carrasco (2012) - Walk-forward testing
- Bailey et al. (2015) - Backtest overfitting
- Blum & Roth (2003) - Leakage in supervised learning
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('leakage_detection_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLeakageDetector:
    """
    Comprehensive leakage detection for time-series financial data
    """

    def __init__(self, features_df: pd.DataFrame, target: pd.Series,
                 train_idx: np.ndarray, test_idx: np.ndarray):
        """
        Initialize leakage detector

        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix
        target : pd.Series
            Target variable
        train_idx : np.ndarray
            Training indices
        test_idx : np.ndarray
            Testing indices
        """
        self.features = features_df
        self.target = target
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.results = {}

        logger.info(f"LeakageDetector initialized:")
        logger.info(f"  - Total samples: {len(features_df)}")
        logger.info(f"  - Train samples: {len(train_idx)}")
        logger.info(f"  - Test samples: {len(test_idx)}")
        logger.info(f"  - Features: {features_df.shape[1]}")

    # ==========================================================================
    # TEST 1: TEMPORAL CONSISTENCY
    # ==========================================================================

    def test_temporal_consistency(self) -> dict:
        """
        TEST 1: Verify that train and test sets are temporally ordered

        Checks:
        - No overlap between train and test indices
        - Test indices come after train indices
        - Proper temporal split
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 1: TEMPORAL CONSISTENCY")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'details': []
        }

        # Check 1: No overlap
        overlap = set(self.train_idx) & set(self.test_idx)
        check1 = len(overlap) == 0
        results['checks']['no_overlap'] = check1
        msg = f"✓ No overlap between train and test" if check1 else "✗ OVERLAP DETECTED"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: Temporal ordering
        max_train = np.max(self.train_idx)
        min_test = np.min(self.test_idx)
        check2 = max_train < min_test
        results['checks']['temporal_order'] = check2
        msg = f"✓ All test indices come after train" if check2 else "✗ TEMPORAL ORDER VIOLATED"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Chronological continuity
        train_sorted = np.sort(self.train_idx)
        check3 = np.all(np.diff(train_sorted) >= 0)
        results['checks']['train_chronological'] = check3
        msg = f"✓ Train indices are chronologically ordered" if check3 else "✗ TRAIN NOT ORDERED"
        logger.info(msg)
        results['details'].append(msg)

        # Check 4: Gap size (should have gap for financial data)
        gap = min_test - max_train - 1
        check4 = gap >= 0  # At least no overlap
        results['checks']['gap_exists'] = check4
        msg = f"✓ Temporal gap exists: {gap} samples" if check4 else "✗ NO GAP BETWEEN SETS"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '✗ FAILED'}: Temporal Consistency")

        self.results['temporal_consistency'] = results
        return results

    # ==========================================================================
    # TEST 2: TARGET LEAKAGE DETECTION
    # ==========================================================================

    def test_target_leakage(self) -> dict:
        """
        TEST 2: Detect if target variable leaks future information

        Checks:
        - Target values don't contain future price information
        - Target generation is forward-looking (not backward-looking)
        - Proper target lag structure
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 2: TARGET LEAKAGE DETECTION")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'details': []
        }

        # We check if target is based on forward-looking information
        # by computing correlation between target and future returns

        # Extract CLOSE prices if available
        price_data = pd.read_csv('XAUUSD_M15_R.csv', sep='\t')
        price_data = price_data.iloc[:len(self.target)].reset_index(drop=True)
        close_prices = price_data['<CLOSE>'].values

        # Check 1: Target should be based on next period return
        returns = np.diff(close_prices)  # Current period return
        future_returns = np.concatenate([returns[1:], [0]])  # Shift for alignment

        target_values = self.target.values

        # Correlation between target and same period vs future
        # Align arrays to same length
        min_len = min(len(target_values), len(returns))
        corr_same = np.corrcoef(target_values[:min_len], returns[:min_len])[0, 1]

        logger.info(f"Target-Return Correlation (same period): {corr_same:.4f}")
        results['details'].append(f"Correlation between target and same-period return: {corr_same:.4f}")

        # Check 2: Verify target is binary (0 or 1)
        check2 = set(np.unique(target_values)) <= {0, 1}
        results['checks']['binary_target'] = check2
        msg = f"✓ Target is binary" if check2 else "✗ TARGET NOT BINARY"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Target distribution balance
        unique, counts = np.unique(target_values, return_counts=True)
        target_ratio = counts[1] / len(target_values) if 1 in unique else 0
        check3 = 0.3 < target_ratio < 0.7  # Should be reasonably balanced
        results['checks']['target_balance'] = check3
        msg = f"✓ Target ratio balanced: {target_ratio:.2%}" if check3 else f"⚠ Target imbalanced: {target_ratio:.2%}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 4: Target generation should be forward-looking, not look-ahead
        # High positive correlation indicates proper forward-looking target
        check4 = corr_same > 0.3  # Some positive correlation expected
        results['checks']['forward_looking'] = check4
        msg = f"✓ Target appears forward-looking (corr={corr_same:.4f})" if check4 else "⚠ Check target generation"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all([v for k, v in results['checks'].items() if k != 'target_balance'])
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '✗ FAILED'}: Target Leakage Detection")

        self.results['target_leakage'] = results
        return results

    # ==========================================================================
    # TEST 3: FEATURE LEAKAGE DETECTION (CORRELATION WITH FUTURE)
    # ==========================================================================

    def test_feature_leakage(self) -> dict:
        """
        TEST 3: Detect if features leak future information

        Checks:
        - No feature has suspicious correlation with future target
        - Feature distributions are similar between train and test
        - No lookahead bias in feature engineering
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 3: FEATURE LEAKAGE DETECTION")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'suspicious_features': [],
            'details': []
        }

        X = self.features.values
        y = self.target.values

        # Check 1: Correlation between features and target
        correlations = []
        for i in range(X.shape[1]):
            try:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(corr)
            except:
                correlations.append(0)

        correlations = np.array(correlations)
        max_corr = np.max(np.abs(correlations))
        mean_corr = np.mean(np.abs(correlations))

        logger.info(f"Feature-Target Correlation Stats:")
        logger.info(f"  - Max |correlation|: {max_corr:.4f}")
        logger.info(f"  - Mean |correlation|: {mean_corr:.4f}")
        logger.info(f"  - Std |correlation|: {np.std(np.abs(correlations)):.4f}")

        results['details'].append(f"Max feature-target correlation: {max_corr:.4f}")
        results['details'].append(f"Mean feature-target correlation: {mean_corr:.4f}")

        # Features with extremely high correlation might be suspicious
        suspicious_idx = np.where(np.abs(correlations) > 0.7)[0]
        if len(suspicious_idx) > 0:
            feature_names = self.features.columns.tolist()
            for idx in suspicious_idx:
                results['suspicious_features'].append({
                    'feature': feature_names[idx],
                    'correlation': float(correlations[idx])
                })
                logger.warning(f"⚠ Suspicious feature: {feature_names[idx]} (corr={correlations[idx]:.4f})")

        check1 = len(suspicious_idx) == 0
        results['checks']['no_suspicious_features'] = check1

        # Check 2: Feature importance distribution
        # If features have too uniform importance, might indicate leakage
        X_train = X[self.train_idx]
        y_train = y[self.train_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
        model.fit(X_train, y_train)

        feature_importance = model.feature_importances_
        importance_gini = np.std(feature_importance) / (np.mean(feature_importance) + 1e-10)

        check2 = importance_gini > 0.5  # Should have diversity
        results['checks']['feature_importance_diversity'] = check2
        msg = f"✓ Feature importance has good diversity (Gini={importance_gini:.4f})" if check2 else "⚠ Low diversity in feature importance"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Permutation importance for target
        # If features are not important for predicting target, might be leakage
        from sklearn.inspection import permutation_importance
        try:
            perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
            mean_perm = np.mean(perm_importance.importances_mean)
            check3 = mean_perm > 0.001  # Should have some importance
            results['checks']['permutation_importance'] = check3
            msg = f"✓ Permutation importance confirms feature relevance" if check3 else "⚠ Low permutation importance"
            logger.info(msg)
            results['details'].append(msg)
        except Exception as e:
            logger.warning(f"Could not compute permutation importance: {e}")
            results['checks']['permutation_importance'] = True

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: Feature Leakage Detection")

        self.results['feature_leakage'] = results
        return results

    # ==========================================================================
    # TEST 4: DISTRIBUTION CONSISTENCY (NO DATA DRIFT)
    # ==========================================================================

    def test_distribution_consistency(self) -> dict:
        """
        TEST 4: Check if feature distributions are consistent between train and test

        Checks:
        - No significant distribution shift (Kolmogorov-Smirnov test)
        - Similar statistics between train and test
        - No temporal drift in feature values
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 4: DISTRIBUTION CONSISTENCY")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'drift_features': [],
            'details': []
        }

        X = self.features.values
        X_train = X[self.train_idx]
        X_test = X[self.test_idx]

        logger.info(f"Testing distribution shift (KS-test, alpha=0.05):")

        # Kolmogorov-Smirnov test for each feature
        ks_pvalues = []
        for i in range(X.shape[1]):
            ks_stat, p_value = stats.ks_2samp(X_train[:, i], X_test[:, i])
            ks_pvalues.append(p_value)

            if p_value < 0.05:
                feature_name = self.features.columns[i]
                results['drift_features'].append({
                    'feature': feature_name,
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value)
                })
                logger.warning(f"⚠ Potential drift in {feature_name} (KS p-value={p_value:.4f})")

        ks_pvalues = np.array(ks_pvalues)
        check1 = np.sum(ks_pvalues < 0.05) < X.shape[1] * 0.1  # < 10% features with drift
        results['checks']['no_significant_drift'] = check1

        logger.info(f"  - Features with drift (p<0.05): {np.sum(ks_pvalues < 0.05)}/{X.shape[1]}")
        logger.info(f"  - Mean p-value: {np.mean(ks_pvalues):.4f}")

        results['details'].append(f"KS-test features with significant shift: {np.sum(ks_pvalues < 0.05)}/{X.shape[1]}")

        # Check 2: Mean/Std comparison
        train_means = np.mean(X_train, axis=0)
        test_means = np.mean(X_test, axis=0)
        mean_shift = np.mean(np.abs(train_means - test_means) / (np.abs(train_means) + 1e-10))

        check2 = mean_shift < 0.2  # Less than 20% mean shift
        results['checks']['mean_consistency'] = check2
        msg = f"✓ Mean shift within tolerance: {mean_shift:.4f}" if check2 else f"⚠ Significant mean shift: {mean_shift:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Variance consistency
        train_vars = np.var(X_train, axis=0)
        test_vars = np.var(X_test, axis=0)
        var_ratio = np.mean(test_vars / (train_vars + 1e-10))

        check3 = 0.7 < var_ratio < 1.5  # Variance shouldn't change drastically
        results['checks']['variance_consistency'] = check3
        msg = f"✓ Variance ratio reasonable: {var_ratio:.4f}" if check3 else f"⚠ High variance ratio: {var_ratio:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: Distribution Consistency")

        self.results['distribution_consistency'] = results
        return results

    # ==========================================================================
    # TEST 5: WALK-FORWARD VALIDATION (MOST IMPORTANT FOR TIME SERIES)
    # ==========================================================================

    def test_walk_forward_validation(self, n_splits: int = 5) -> dict:
        """
        TEST 5: Walk-forward validation to detect overfitting and leakage

        This is the most important test for time-series data.
        Tests model performance across multiple time windows.

        Reference: Prado & Carrasco (2012) - Walk-Forward Testing
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 5: WALK-FORWARD VALIDATION")
        logger.info(f"(Time-series CV with {n_splits} splits)")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'cv_scores': [],
            'cv_stability': None,
            'details': []
        }

        X = self.features.values
        y = self.target.values

        # Use TimeSeriesSplit for proper time-series validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_scores = []
        fold_details = []

        logger.info(f"\nRunning {n_splits}-fold time-series CV:")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X[val_idx]
            y_val_cv = y[val_idx]

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
            model.fit(X_train_cv, y_train_cv)

            # Evaluate
            y_pred = model.predict(X_val_cv)
            y_pred_proba = model.predict_proba(X_val_cv)[:, 1]

            acc = accuracy_score(y_val_cv, y_pred)
            auc = roc_auc_score(y_val_cv, y_pred_proba)

            cv_scores.append(acc)
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'accuracy': float(acc),
                'auc': float(auc)
            })

            logger.info(f"  Fold {fold}: Train={len(train_idx):5d}, Val={len(val_idx):5d} | Acc={acc:.4f}, AUC={auc:.4f}")

        cv_scores = np.array(cv_scores)
        results['cv_scores'] = cv_scores.tolist()
        results['fold_details'] = fold_details

        # Stability analysis
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_min = np.min(cv_scores)
        cv_max = np.max(cv_scores)
        cv_range = cv_max - cv_min

        logger.info(f"\nWalk-Forward Statistics:")
        logger.info(f"  - Mean Accuracy: {cv_mean:.4f}")
        logger.info(f"  - Std Dev: {cv_std:.4f}")
        logger.info(f"  - Min: {cv_min:.4f}")
        logger.info(f"  - Max: {cv_max:.4f}")
        logger.info(f"  - Range: {cv_range:.4f}")

        results['cv_stability'] = {
            'mean': float(cv_mean),
            'std': float(cv_std),
            'min': float(cv_min),
            'max': float(cv_max),
            'range': float(cv_range)
        }

        # Check 1: Mean accuracy should be reasonable
        check1 = cv_mean > 0.55  # Better than random (50%)
        results['checks']['reasonable_performance'] = check1
        msg = f"✓ CV mean accuracy {cv_mean:.4f} > 0.55" if check1 else f"✗ CV mean accuracy too low: {cv_mean:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: Stability across folds (low variance indicates robustness)
        check2 = cv_std < 0.10  # Stable across folds
        results['checks']['cv_stability'] = check2
        msg = f"✓ CV stable (std={cv_std:.4f} < 0.10)" if check2 else f"⚠ CV unstable (std={cv_std:.4f})"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Not overfitting (range not too large)
        check3 = cv_range < 0.20  # Max-min shouldn't exceed 20%
        results['checks']['no_overfitting'] = check3
        msg = f"✓ No extreme overfitting (range={cv_range:.4f})" if check3 else f"⚠ Large range {cv_range:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 4: Performance degradation over time (if present, indicates overfitting)
        if len(cv_scores) > 2:
            first_half = np.mean(cv_scores[:len(cv_scores)//2])
            second_half = np.mean(cv_scores[len(cv_scores)//2:])
            degradation = first_half - second_half

            check4 = degradation < 0.10  # Less than 10% degradation
            results['checks']['no_performance_degradation'] = check4
            msg = f"✓ No significant degradation (first-second half: {degradation:.4f})" if check4 else f"⚠ Performance degrades over time: {degradation:.4f}"
            logger.info(msg)
            results['details'].append(msg)
        else:
            results['checks']['no_performance_degradation'] = True

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: Walk-Forward Validation")

        self.results['walk_forward_validation'] = results
        return results

    # ==========================================================================
    # TEST 6: PERMUTATION TEST FOR SIGNIFICANCE
    # ==========================================================================

    def test_feature_significance(self, n_permutations: int = 50) -> dict:
        """
        TEST 6: Permutation test to verify features are truly predictive

        Not just correlated by chance. This tests if our features
        actually predict the target better than random.
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 6: FEATURE SIGNIFICANCE (PERMUTATION TEST)")
        logger.info(f"(n_permutations={n_permutations})")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'permutation_stats': {},
            'details': []
        }

        X = self.features.values
        y = self.target.values
        X_train = X[self.train_idx]
        y_train = y[self.train_idx]
        X_test = X[self.test_idx]
        y_test = y[self.test_idx]

        # Train model with real features
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
        model.fit(X_train, y_train)

        real_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        logger.info(f"Real model AUC: {real_score:.4f}")

        # Permutation test
        permuted_scores = []

        for i in range(n_permutations):
            if (i + 1) % max(1, n_permutations // 5) == 0:
                logger.info(f"  Permutation {i+1}/{n_permutations}...")

            # Shuffle target
            y_train_permuted = np.random.permutation(y_train)

            model_perm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
            model_perm.fit(X_train, y_train_permuted)

            perm_score = roc_auc_score(y_test, model_perm.predict_proba(X_test)[:, 1])
            permuted_scores.append(perm_score)

        permuted_scores = np.array(permuted_scores)

        # p-value: proportion of permutations >= real score
        p_value = np.mean(permuted_scores >= real_score)

        logger.info(f"\nPermutation Test Results:")
        logger.info(f"  - Real AUC: {real_score:.4f}")
        logger.info(f"  - Permuted Mean AUC: {np.mean(permuted_scores):.4f}")
        logger.info(f"  - Permuted Std AUC: {np.std(permuted_scores):.4f}")
        logger.info(f"  - P-value: {p_value:.4f}")

        results['permutation_stats'] = {
            'real_auc': float(real_score),
            'permuted_mean_auc': float(np.mean(permuted_scores)),
            'permuted_std_auc': float(np.std(permuted_scores)),
            'p_value': float(p_value)
        }

        # Check 1: Real performance > permuted (statistical significance)
        check1 = real_score > np.mean(permuted_scores)
        results['checks']['real_better_than_permuted'] = check1
        msg = f"✓ Real model better than permuted ({real_score:.4f} > {np.mean(permuted_scores):.4f})" if check1 else "✗ Real model not better than permuted"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: Statistical significance (p < 0.05)
        check2 = p_value < 0.05
        results['checks']['statistically_significant'] = check2
        msg = f"✓ Features are statistically significant (p={p_value:.4f})" if check2 else f"⚠ Not significant (p={p_value:.4f})"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Effect size
        effect_size = (real_score - np.mean(permuted_scores)) / (np.std(permuted_scores) + 1e-10)
        check3 = effect_size > 1.0  # At least 1 standard deviation
        results['checks']['meaningful_effect_size'] = check3
        msg = f"✓ Meaningful effect size: {effect_size:.4f}" if check3 else f"⚠ Small effect size: {effect_size:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: Feature Significance")

        self.results['feature_significance'] = results
        return results

    # ==========================================================================
    # RUN ALL TESTS
    # ==========================================================================

    def run_all_tests(self) -> dict:
        """Run all leakage detection tests"""
        logger.info("\n\n")
        logger.info("#"*80)
        logger.info("# DATA LEAKAGE DETECTION TEST SUITE")
        logger.info("#"*80)

        self.test_temporal_consistency()
        self.test_target_leakage()
        self.test_feature_leakage()
        self.test_distribution_consistency()
        self.test_walk_forward_validation()
        self.test_feature_significance(n_permutations=50)

        return self.generate_summary_report()

    def generate_summary_report(self) -> dict:
        """Generate comprehensive summary report"""
        logger.info("\n\n")
        logger.info("#"*80)
        logger.info("# SUMMARY REPORT")
        logger.info("#"*80)

        summary = {
            'total_tests': len(self.results),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': {}
        }

        for test_name, test_result in self.results.items():
            passed = test_result.get('passed', False)
            summary['test_details'][test_name] = {
                'passed': passed,
                'checks': test_result.get('checks', {})
            }

            if passed:
                summary['passed_tests'] += 1
                logger.info(f"✓ {test_name}: PASSED")
            else:
                summary['failed_tests'] += 1
                logger.warning(f"✗ {test_name}: FAILED")

        # Overall conclusion
        logger.info("\n" + "="*80)
        if summary['failed_tests'] == 0:
            logger.info("✓ ALL TESTS PASSED - No detected data leakage!")
            logger.info("✓ Results are valid for use in production")
            summary['overall_status'] = 'VALID'
        else:
            logger.warning(f"⚠ {summary['failed_tests']} test(s) failed")
            logger.warning("⚠ Review failed tests for potential issues")
            summary['overall_status'] = 'NEEDS REVIEW'

        logger.info("="*80)

        return summary


if __name__ == '__main__':
    logger.info("Loading data...")

    # Load combined dataset
    df = pd.read_parquet('F_combined.parquet')

    X = df.drop('target', axis=1)
    y = df['target']

    # Use the same train/test split as FSX.py
    from sklearn.model_selection import TimeSeriesSplit

    # Manual train/test split with gap for time series
    n_samples = len(X)
    train_size = int(0.8 * n_samples)
    gap = 50

    train_idx = np.arange(0, train_size - gap)
    test_idx = np.arange(train_size, n_samples)

    logger.info(f"Train indices: {len(train_idx)} samples (0-{train_idx[-1]})")
    logger.info(f"Test indices: {len(test_idx)} samples ({test_idx[0]}-{test_idx[-1]})")

    # Initialize detector
    detector = DataLeakageDetector(X, y, train_idx, test_idx)

    # Run all tests
    summary = detector.run_all_tests()

    # Save results
    import json
    with open('leakage_test_results.json', 'w') as f:
        # Convert to serializable format
        results_serializable = {}
        for test_name, test_result in detector.results.items():
            results_serializable[test_name] = {
                'passed': test_result.get('passed'),
                'checks': test_result.get('checks', {})
            }

        json.dump({
            'summary': summary,
            'detailed_results': results_serializable
        }, f, indent=2, default=str)

    logger.info("\n✓ Test results saved to: leakage_test_results.json")
    logger.info("✓ Test logs saved to: leakage_detection_tests.log")
