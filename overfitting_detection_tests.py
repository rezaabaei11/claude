"""
Comprehensive Overfitting Detection Test Suite for Time Series ML
Based on best practices from:
- Hastie et al. (2009) - Elements of Statistical Learning
- Bishop (2006) - Pattern Recognition and Machine Learning
- Ng (2019) - Machine Learning best practices
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, cross_validate, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('overfitting_detection_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OverfittingDetector:
    """
    Comprehensive overfitting detection for ML models
    """

    def __init__(self, features_df: pd.DataFrame, target: pd.Series,
                 train_idx: np.ndarray, test_idx: np.ndarray):
        """Initialize overfitting detector"""
        self.features = features_df
        self.target = target
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.results = {}

        logger.info(f"OverfittingDetector initialized:")
        logger.info(f"  - Total samples: {len(features_df)}")
        logger.info(f"  - Train samples: {len(train_idx)}")
        logger.info(f"  - Test samples: {len(test_idx)}")
        logger.info(f"  - Features: {features_df.shape[1]}")

    # ==========================================================================
    # TEST 1: LEARNING CURVES
    # ==========================================================================

    def test_learning_curves(self) -> dict:
        """
        TEST 1: Generate learning curves to visualize overfitting

        Checks:
        - Training accuracy increases with more data
        - Validation accuracy converges to training accuracy
        - Gap between train and validation is not too large
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 1: LEARNING CURVES")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'details': [],
            'train_sizes': [],
            'train_scores': [],
            'val_scores': []
        }

        X = self.features.values
        y = self.target.values
        X_train = X[self.train_idx]
        y_train = y[self.train_idx]

        logger.info("Generating learning curves...")

        # Create custom train/val split from training data
        train_split = int(0.8 * len(X_train))
        X_train_lc = X_train[:train_split]
        y_train_lc = y_train[:train_split]
        X_val_lc = X_train[train_split:]
        y_val_lc = y_train[train_split:]

        # Use different training set sizes
        train_sizes = np.linspace(0.1, 1.0, 8)
        train_scores = []
        val_scores = []

        for train_size in train_sizes:
            n_samples = int(train_size * len(X_train_lc))
            X_subset = X_train_lc[:n_samples]
            y_subset = y_train_lc[:n_samples]

            model = RandomForestClassifier(n_estimators=100, random_state=42,
                                          max_depth=15, n_jobs=-1)
            model.fit(X_subset, y_subset)

            train_acc = accuracy_score(y_subset, model.predict(X_subset))
            val_acc = accuracy_score(y_val_lc, model.predict(X_val_lc))

            train_scores.append(train_acc)
            val_scores.append(val_acc)

            logger.info(f"  Train size: {train_size:.1%} ({n_samples:5d}) | "
                       f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Gap: {train_acc - val_acc:.4f}")

        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)

        results['train_sizes'] = train_sizes.tolist()
        results['train_scores'] = train_scores.tolist()
        results['val_scores'] = val_scores.tolist()

        # Check 1: Training accuracy should increase
        check1 = train_scores[-1] > train_scores[0]
        results['checks']['train_accuracy_improves'] = check1
        msg = f"✓ Training accuracy improves" if check1 else "⚠ Training accuracy doesn't improve"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: Validation accuracy should improve
        check2 = val_scores[-1] > val_scores[0]
        results['checks']['val_accuracy_improves'] = check2
        msg = f"✓ Validation accuracy improves" if check2 else "⚠ Validation accuracy doesn't improve"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Final gap should be reasonable
        final_gap = train_scores[-1] - val_scores[-1]
        check3 = final_gap < 0.15  # Gap should be < 15%
        results['checks']['reasonable_gap'] = check3
        msg = f"✓ Final train-val gap reasonable: {final_gap:.4f}" if check3 else f"⚠ Large gap: {final_gap:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 4: Convergence (gap should decrease at end)
        gap_early = train_scores[2] - val_scores[2]
        gap_late = train_scores[-1] - val_scores[-1]
        converging = gap_late < gap_early
        results['checks']['converging'] = converging
        msg = f"✓ Curves converging (gap decreases)" if converging else "⚠ Curves diverging"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: Learning Curves")

        self.results['learning_curves'] = results
        return results

    # ==========================================================================
    # TEST 2: TRAIN VS TEST PERFORMANCE GAP
    # ==========================================================================

    def test_train_test_gap(self) -> dict:
        """
        TEST 2: Measure train vs test performance gap

        Checks:
        - Test accuracy is reasonable (not much worse than train)
        - No dramatic drop in performance
        - Generalization is good
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 2: TRAIN VS TEST PERFORMANCE GAP")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'metrics': {},
            'details': []
        }

        X = self.features.values
        y = self.target.values
        X_train = X[self.train_idx]
        y_train = y[self.train_idx]
        X_test = X[self.test_idx]
        y_test = y[self.test_idx]

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42,
                                      max_depth=15, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

        test_acc = accuracy_score(y_test, model.predict(X_test))
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        logger.info(f"Accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}")
        logger.info(f"AUC      - Train: {train_auc:.4f}, Test: {test_auc:.4f}")

        acc_gap = train_acc - test_acc
        auc_gap = train_auc - test_auc

        results['metrics'] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'accuracy_gap': float(acc_gap),
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'auc_gap': float(auc_gap)
        }

        logger.info(f"Accuracy Gap: {acc_gap:.4f} ({acc_gap*100:.2f}%)")
        logger.info(f"AUC Gap: {auc_gap:.4f}")

        # Check 1: Accuracy gap should be small
        check1 = acc_gap < 0.10  # < 10%
        results['checks']['small_accuracy_gap'] = check1
        msg = f"✓ Accuracy gap reasonable: {acc_gap:.4f}" if check1 else f"✗ Large accuracy gap: {acc_gap:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: AUC gap should be small
        check2 = auc_gap < 0.10
        results['checks']['small_auc_gap'] = check2
        msg = f"✓ AUC gap reasonable: {auc_gap:.4f}" if check2 else f"✗ Large AUC gap: {auc_gap:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Test performance should be decent
        check3 = test_acc > 0.55  # Better than random
        results['checks']['decent_test_performance'] = check3
        msg = f"✓ Test accuracy decent: {test_acc:.4f}" if check3 else f"✗ Poor test accuracy: {test_acc:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 4: Test should not be much worse than train
        check4 = test_acc > train_acc * 0.90  # Test >= 90% of train
        results['checks']['test_not_worse'] = check4
        msg = f"✓ Test performance not much worse" if check4 else f"⚠ Test much worse than train"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '✗ FAILED'}: Train vs Test Gap")

        self.results['train_test_gap'] = results
        return results

    # ==========================================================================
    # TEST 3: CROSS-VALIDATION CONSISTENCY
    # ==========================================================================

    def test_cv_consistency(self, n_splits: int = 5) -> dict:
        """
        TEST 3: Check cross-validation consistency

        Checks:
        - CV scores are stable (low variance)
        - No extreme outliers in CV
        - Model generalizes consistently
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 3: CROSS-VALIDATION CONSISTENCY")
        logger.info(f"(n_splits={n_splits})")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'cv_scores': [],
            'cv_stats': {},
            'details': []
        }

        X = self.features.values
        y = self.target.values
        X_train = X[self.train_idx]
        y_train = y[self.train_idx]

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_scores = []

        logger.info(f"Running {n_splits}-fold time-series CV:")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            X_train_cv = X_train[train_idx]
            y_train_cv = y_train[train_idx]
            X_val_cv = X_train[val_idx]
            y_val_cv = y_train[val_idx]

            model = RandomForestClassifier(n_estimators=100, random_state=42,
                                          max_depth=15, n_jobs=-1)
            model.fit(X_train_cv, y_train_cv)

            score = accuracy_score(y_val_cv, model.predict(X_val_cv))
            cv_scores.append(score)

            logger.info(f"  Fold {fold}: {score:.4f}")

        cv_scores = np.array(cv_scores)
        results['cv_scores'] = cv_scores.tolist()

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_min = np.min(cv_scores)
        cv_max = np.max(cv_scores)
        cv_range = cv_max - cv_min

        results['cv_stats'] = {
            'mean': float(cv_mean),
            'std': float(cv_std),
            'min': float(cv_min),
            'max': float(cv_max),
            'range': float(cv_range)
        }

        logger.info(f"\nCV Statistics:")
        logger.info(f"  - Mean: {cv_mean:.4f}")
        logger.info(f"  - Std: {cv_std:.4f}")
        logger.info(f"  - Range: {cv_range:.4f}")

        # Check 1: Low variance
        check1 = cv_std < 0.10
        results['checks']['low_cv_variance'] = check1
        msg = f"✓ Low CV variance: {cv_std:.4f}" if check1 else f"⚠ High CV variance: {cv_std:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: No extreme outliers
        check2 = cv_range < 0.25  # Range < 25%
        results['checks']['no_extreme_outliers'] = check2
        msg = f"✓ No extreme outliers (range={cv_range:.4f})" if check2 else f"⚠ Extreme variation"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Minimum score reasonable
        check3 = cv_min > 0.50  # > random baseline
        results['checks']['min_score_reasonable'] = check3
        msg = f"✓ Minimum score reasonable: {cv_min:.4f}" if check3 else f"✗ Worst fold poor"
        logger.info(msg)
        results['details'].append(msg)

        # Check 4: Coefficient of variation
        cv_coeff = cv_std / cv_mean if cv_mean > 0 else 0
        check4 = cv_coeff < 0.10  # < 10% CV
        results['checks']['low_cv_coefficient'] = check4
        msg = f"✓ Low CV coefficient: {cv_coeff:.4f}" if check4 else f"⚠ High CV coefficient: {cv_coeff:.4f}"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: CV Consistency")

        self.results['cv_consistency'] = results
        return results

    # ==========================================================================
    # TEST 4: MODEL COMPLEXITY ANALYSIS
    # ==========================================================================

    def test_model_complexity(self) -> dict:
        """
        TEST 4: Analyze model complexity vs performance

        Checks:
        - Model complexity matches data size
        - Not too many features for data size
        - Feature ratio is reasonable
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 4: MODEL COMPLEXITY ANALYSIS")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'metrics': {},
            'details': []
        }

        X = self.features.values
        y = self.target.values
        X_train = X[self.train_idx]
        y_train = y[self.train_idx]

        n_samples = len(X_train)
        n_features = X.shape[1]
        samples_per_feature = n_samples / n_features

        logger.info(f"Data Complexity:")
        logger.info(f"  - Samples: {n_samples}")
        logger.info(f"  - Features: {n_features}")
        logger.info(f"  - Samples per Feature: {samples_per_feature:.2f}")

        results['metrics'] = {
            'n_samples': int(n_samples),
            'n_features': int(n_features),
            'samples_per_feature': float(samples_per_feature)
        }

        # Check 1: Sufficient samples for features (>10:1)
        check1 = samples_per_feature > 10  # 10 samples per feature minimum
        results['checks']['sufficient_samples'] = check1
        msg = f"✓ Sufficient samples: {samples_per_feature:.2f} per feature" if check1 else f"✗ Too few samples per feature"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: Model parameters reasonable for data
        # For RF with max_depth=15: approx 2^15 = 32K parameters
        max_depth = 15
        approx_leaf_nodes = 2 ** max_depth
        check2 = approx_leaf_nodes < n_samples  # Leaves < samples
        results['checks']['model_params_reasonable'] = check2
        msg = f"✓ Model parameters reasonable" if check2 else f"✗ Model too complex for data"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Feature dimensionality not excessive
        check3 = n_features < n_samples / 5  # Features < 20% of samples
        results['checks']['feature_dimensionality_ok'] = check3
        msg = f"✓ Feature dimensionality OK ({n_features}/{n_samples})" if check3 else f"⚠ Many features"
        logger.info(msg)
        results['details'].append(msg)

        # Check 4: Feature importance distribution
        model = RandomForestClassifier(n_estimators=100, random_state=42,
                                      max_depth=15, n_jobs=-1)
        model.fit(X_train, y_train)

        importance = model.feature_importances_
        top_5_importance = np.sum(np.sort(importance)[-5:])
        check4 = top_5_importance < 0.6  # Top 5 should not dominate
        results['checks']['feature_importance_distributed'] = check4
        msg = f"✓ Feature importance distributed" if check4 else f"⚠ Few features dominate"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: Model Complexity")

        self.results['model_complexity'] = results
        return results

    # ==========================================================================
    # TEST 5: FEATURE IMPORTANCE STABILITY
    # ==========================================================================

    def test_feature_stability(self) -> dict:
        """
        TEST 5: Check feature importance stability across subsets

        Checks:
        - Feature rankings are stable
        - Top features consistent
        - No random fluctuations
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 5: FEATURE IMPORTANCE STABILITY")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'details': [],
            'top_features': {}
        }

        X = self.features.values
        y = self.target.values
        X_train = X[self.train_idx]
        y_train = y[self.train_idx]

        # Train multiple models on subsets
        feature_rankings = []
        n_iterations = 10

        logger.info(f"Training {n_iterations} models on random subsets...")

        for i in range(n_iterations):
            # Random subset (80% of training data)
            n_subset = int(0.8 * len(X_train))
            subset_idx = np.random.choice(len(X_train), n_subset, replace=False)

            X_subset = X_train[subset_idx]
            y_subset = y_train[subset_idx]

            model = RandomForestClassifier(n_estimators=100, random_state=42,
                                          max_depth=15, n_jobs=-1)
            model.fit(X_subset, y_subset)

            # Get feature ranking
            ranking = np.argsort(model.feature_importances_)[::-1]
            feature_rankings.append(ranking)

        # Check stability of top features
        top_k = 10
        top_features_lists = [ranking[:top_k] for ranking in feature_rankings]

        # Count frequency of each feature in top 10
        feature_counts = {}
        for features in top_features_lists:
            for feat_idx in features:
                feature_counts[feat_idx] = feature_counts.get(feat_idx, 0) + 1

        # Features that appear in top 10 frequently
        stable_features = {f: c for f, c in feature_counts.items() if c >= 5}  # >= 50%

        logger.info(f"Features in top 10: {len(feature_counts)}")
        logger.info(f"Stable features (>50% of runs): {len(stable_features)}")

        results['top_features'] = {
            'total_in_top10': int(len(feature_counts)),
            'stable_features': int(len(stable_features))
        }

        # Check 1: Sufficient stable features
        check1 = len(stable_features) >= top_k * 0.7  # At least 70% stable
        results['checks']['stable_features'] = check1
        msg = f"✓ Most top features stable (70%+)" if check1 else f"⚠ Unstable feature rankings"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: Some features always in top
        always_top = {f: c for f, c in feature_counts.items() if c == n_iterations}
        check2 = len(always_top) >= 3  # At least 3 consistently top features
        results['checks']['consistent_top_features'] = check2
        msg = f"✓ Consistent top features: {len(always_top)}" if check2 else f"⚠ No consistent top features"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Features are not evenly distributed
        # Train one more model to check importance distribution
        model_final = RandomForestClassifier(n_estimators=100, random_state=42,
                                            max_depth=15, n_jobs=-1)
        model_final.fit(X_train, y_train)
        importance_final = model_final.feature_importances_
        importance_std = np.std(importance_final)
        check3 = importance_std > 0.005  # Features not all equal
        results['checks']['features_not_all_equal'] = check3
        msg = f"✓ Features vary in importance" if check3 else f"⚠ All features equally important"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: Feature Stability")

        self.results['feature_stability'] = results
        return results

    # ==========================================================================
    # TEST 6: BOOTSTRAP AGGREGATION STABILITY
    # ==========================================================================

    def test_bootstrap_stability(self, n_bootstrap: int = 30) -> dict:
        """
        TEST 6: Bootstrap aggregation to measure prediction stability

        Checks:
        - Predictions are stable across bootstrap samples
        - Low prediction variance
        - Model not overfitting to specific samples
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 6: BOOTSTRAP AGGREGATION STABILITY")
        logger.info(f"(n_bootstrap={n_bootstrap})")
        logger.info("="*80)

        results = {
            'passed': True,
            'checks': {},
            'metrics': {},
            'details': []
        }

        X = self.features.values
        y = self.target.values
        X_train = X[self.train_idx]
        y_train = y[self.train_idx]
        X_test = X[self.test_idx]
        y_test = y[self.test_idx]

        # Get predictions from multiple bootstrap samples
        predictions = []
        scores = []

        logger.info(f"Training {n_bootstrap} models on bootstrap samples...")

        for i in range(n_bootstrap):
            if (i + 1) % 10 == 0:
                logger.info(f"  Bootstrap {i+1}/{n_bootstrap}...")

            # Bootstrap sample
            boot_idx = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot = X_train[boot_idx]
            y_boot = y_train[boot_idx]

            model = RandomForestClassifier(n_estimators=100, random_state=42,
                                          max_depth=15, n_jobs=-1)
            model.fit(X_boot, y_boot)

            # Get predictions
            pred_proba = model.predict_proba(X_test)[:, 1]
            predictions.append(pred_proba)

            score = roc_auc_score(y_test, pred_proba)
            scores.append(score)

        predictions = np.array(predictions)
        scores = np.array(scores)

        # Calculate prediction variance
        pred_std = np.std(predictions, axis=0)
        pred_std_mean = np.mean(pred_std)
        pred_std_max = np.max(pred_std)

        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  - Mean prediction std: {pred_std_mean:.4f}")
        logger.info(f"  - Max prediction std: {pred_std_max:.4f}")
        logger.info(f"  - Score mean: {np.mean(scores):.4f}")
        logger.info(f"  - Score std: {np.std(scores):.4f}")

        results['metrics'] = {
            'pred_std_mean': float(pred_std_mean),
            'pred_std_max': float(pred_std_max),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores))
        }

        # Check 1: Low prediction variance
        check1 = pred_std_mean < 0.15  # Low uncertainty
        results['checks']['low_prediction_variance'] = check1
        msg = f"✓ Low prediction variance: {pred_std_mean:.4f}" if check1 else f"⚠ High variance"
        logger.info(msg)
        results['details'].append(msg)

        # Check 2: Stable scores
        check2 = np.std(scores) < 0.05  # Stable AUC
        results['checks']['stable_scores'] = check2
        msg = f"✓ Stable scores (std={np.std(scores):.4f})" if check2 else f"⚠ Unstable"
        logger.info(msg)
        results['details'].append(msg)

        # Check 3: Predictions not too uncertain
        check3 = pred_std_max < 0.35
        results['checks']['reasonable_max_variance'] = check3
        msg = f"✓ Max variance reasonable: {pred_std_max:.4f}" if check3 else f"⚠ Some predictions very uncertain"
        logger.info(msg)
        results['details'].append(msg)

        results['passed'] = all(results['checks'].values())
        logger.info(f"\n{'✓ PASSED' if results['passed'] else '⚠ WARNING'}: Bootstrap Stability")

        self.results['bootstrap_stability'] = results
        return results

    # ==========================================================================
    # RUN ALL TESTS
    # ==========================================================================

    def run_all_tests(self) -> dict:
        """Run all overfitting detection tests"""
        logger.info("\n\n")
        logger.info("#"*80)
        logger.info("# OVERFITTING DETECTION TEST SUITE")
        logger.info("#"*80)

        self.test_learning_curves()
        self.test_train_test_gap()
        self.test_cv_consistency()
        self.test_model_complexity()
        self.test_feature_stability()
        self.test_bootstrap_stability()

        return self.generate_summary_report()

    def generate_summary_report(self) -> dict:
        """Generate summary report"""
        logger.info("\n\n")
        logger.info("#"*80)
        logger.info("# OVERFITTING ANALYSIS SUMMARY")
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
            logger.info("✓ NO SIGNIFICANT OVERFITTING DETECTED!")
            logger.info("✓ Model generalizes well")
            summary['overall_status'] = 'NO OVERFITTING'
        else:
            logger.warning(f"⚠ {summary['failed_tests']} test(s) indicate potential overfitting")
            summary['overall_status'] = 'OVERFITTING DETECTED'

        logger.info("="*80)

        return summary


if __name__ == '__main__':
    logger.info("Loading data...")

    # Load combined dataset
    df = pd.read_parquet('F_combined.parquet')

    X = df.drop('target', axis=1)
    y = df['target']

    # Create train/test split
    n_samples = len(X)
    train_size = int(0.8 * n_samples)
    gap = 50

    train_idx = np.arange(0, train_size - gap)
    test_idx = np.arange(train_size, n_samples)

    logger.info(f"Train indices: {len(train_idx)} samples")
    logger.info(f"Test indices: {len(test_idx)} samples")

    # Initialize detector
    detector = OverfittingDetector(X, y, train_idx, test_idx)

    # Run all tests
    summary = detector.run_all_tests()

    # Save results
    import json
    with open('overfitting_test_results.json', 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': detector.results
        }, f, indent=2, default=str)

    logger.info("\n✓ Test results saved to: overfitting_test_results.json")
    logger.info("✓ Test logs saved to: overfitting_detection_tests.log")
