"""
Improved Model Training with Best Practices for Overfitting Reduction
Based on authoritative sources:
- GridSearchCV and hyperparameter tuning (2024)
- Early stopping with Gradient Boosting (2024)
- D.A.R.T. and XGBoost/LightGBM advances
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedModelTrainer:
    """Train models with best practices to reduce overfitting"""

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}

        logger.info(f"ImprovedModelTrainer initialized:")
        logger.info(f"  - Train: {len(X_train)} samples, {X_train.shape[1]} features")
        logger.info(f"  - Test: {len(X_test)} samples")

    # ==========================================================================
    # STRATEGY 1: OPTIMIZED RANDOM FOREST (Reduced max_depth)
    # ==========================================================================

    def train_optimized_random_forest(self) -> dict:
        """
        Random Forest with reduced complexity (max_depth=8 instead of 15)
        Based on: Hayes et al. (2024) - Regularization in Random Forests
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 1: OPTIMIZED RANDOM FOREST")
        logger.info("="*80)

        logger.info("Training optimized Random Forest with max_depth=8...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,  # Reduced from 15
            min_samples_leaf=20,  # Added
            min_samples_split=50,  # Added
            max_features='sqrt',  # Feature selection
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            warm_start=False
        )

        model.fit(self.X_train, self.y_train)

        # Evaluate
        train_acc = accuracy_score(self.y_train, model.predict(self.X_train))
        train_auc = roc_auc_score(self.y_train, model.predict_proba(self.X_train)[:, 1])
        test_acc = accuracy_score(self.y_test, model.predict(self.X_test))
        test_auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])

        results = {
            'model': model,
            'train_accuracy': train_acc,
            'train_auc': train_auc,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'accuracy_gap': train_acc - test_acc,
            'auc_gap': train_auc - test_auc
        }

        logger.info(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Gap: {results['accuracy_gap']:.4f}")
        logger.info(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Gap: {results['auc_gap']:.4f}")

        self.results['optimized_rf'] = results
        return results

    # ==========================================================================
    # STRATEGY 2: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
    # ==========================================================================

    def train_tuned_random_forest(self) -> dict:
        """
        Random Forest with GridSearchCV for optimal hyperparameters
        Based on: TraininData (2024) - GridSearchCV best practices
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 2: HYPERPARAMETER TUNING (GridSearchCV)")
        logger.info("="*80)

        logger.info("Performing GridSearchCV for Random Forest hyperparameters...")

        # Define parameter grid
        param_grid = {
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [10, 15, 20, 30],
            'min_samples_split': [30, 50, 70],
            'max_features': ['sqrt', 'log2']
        }

        # Base model
        rf = RandomForestClassifier(
            n_estimators=100,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        # GridSearchCV with time-series split
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        logger.info(f"Testing {len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * len(param_grid['min_samples_split']) * len(param_grid['max_features'])} parameter combinations...")

        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        # Evaluate
        train_acc = accuracy_score(self.y_train, best_model.predict(self.X_train))
        train_auc = roc_auc_score(self.y_train, best_model.predict_proba(self.X_train)[:, 1])
        test_acc = accuracy_score(self.y_test, best_model.predict(self.X_test))
        test_auc = roc_auc_score(self.y_test, best_model.predict_proba(self.X_test)[:, 1])

        results = {
            'model': best_model,
            'best_params': best_params,
            'best_cv_score': grid_search.best_score_,
            'train_accuracy': train_acc,
            'train_auc': train_auc,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'accuracy_gap': train_acc - test_acc,
            'auc_gap': train_auc - test_auc
        }

        logger.info(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Gap: {results['accuracy_gap']:.4f}")
        logger.info(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Gap: {results['auc_gap']:.4f}")

        self.results['tuned_rf'] = results
        return results

    # ==========================================================================
    # STRATEGY 3: GRADIENT BOOSTING WITH EARLY STOPPING
    # ==========================================================================

    def train_gradient_boosting(self) -> dict:
        """
        Gradient Boosting with early stopping to prevent overfitting
        Based on: TowardsDataScience (2024) - Early Stopping in Boosting
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 3: GRADIENT BOOSTING WITH EARLY STOPPING")
        logger.info("="*80)

        logger.info("Training Gradient Boosting with early stopping...")

        model = GradientBoostingClassifier(
            n_estimators=500,  # Large number, will be reduced by early stopping
            max_depth=5,  # Shallow trees
            learning_rate=0.05,  # Slower learning
            min_samples_leaf=20,  # Regularization
            min_samples_split=50,  # Regularization
            subsample=0.8,  # Stochastic boosting
            random_state=42,
            validation_fraction=0.2,  # Use 20% for early stopping
            n_iter_no_change=50,  # Early stopping rounds
            tol=1e-4
        )

        model.fit(self.X_train, self.y_train)

        n_estimators_used = model.n_estimators_
        logger.info(f"Early stopping at {n_estimators_used} iterations (out of {model.n_estimators})")

        # Evaluate
        train_acc = accuracy_score(self.y_train, model.predict(self.X_train))
        train_auc = roc_auc_score(self.y_train, model.predict_proba(self.X_train)[:, 1])
        test_acc = accuracy_score(self.y_test, model.predict(self.X_test))
        test_auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])

        results = {
            'model': model,
            'n_estimators_used': n_estimators_used,
            'train_accuracy': train_acc,
            'train_auc': train_auc,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'accuracy_gap': train_acc - test_acc,
            'auc_gap': train_auc - test_auc
        }

        logger.info(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Gap: {results['accuracy_gap']:.4f}")
        logger.info(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Gap: {results['auc_gap']:.4f}")

        self.results['gradient_boosting'] = results
        return results

    # ==========================================================================
    # STRATEGY 4: XGBOOST WITH EARLY STOPPING
    # ==========================================================================

    def train_xgboost(self) -> dict:
        """
        XGBoost with early stopping and regularization
        Based on: MacAluso (2024) - XGBoost hyperparameter tuning
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 4: XGBOOST WITH EARLY STOPPING")
        logger.info("="*80)

        logger.info("Training XGBoost with early stopping...")

        # Split validation set from training data
        split_idx = int(0.8 * len(self.X_train))
        X_train_split = self.X_train[:split_idx]
        y_train_split = self.y_train[:split_idx]
        X_val_split = self.X_train[split_idx:]
        y_val_split = self.y_train[split_idx:]

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,  # Shallow trees
            learning_rate=0.05,  # Slower learning
            subsample=0.8,  # Stochastic boosting
            colsample_bytree=0.8,  # Column subsampling
            min_child_weight=5,  # Regularization
            reg_lambda=1.0,  # L2 regularization
            reg_alpha=0.5,  # L1 regularization
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        eval_set = [(X_val_split, y_val_split)]

        model.fit(
            X_train_split,
            y_train_split,
            eval_set=eval_set,
            callbacks=[xgb.early_stopping(rounds=50)]
        )

        n_estimators_used = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        logger.info(f"Early stopping at {n_estimators_used} iterations")

        # Evaluate on full training set
        train_acc = accuracy_score(self.y_train, model.predict(self.X_train))
        train_auc = roc_auc_score(self.y_train, model.predict_proba(self.X_train)[:, 1])
        test_acc = accuracy_score(self.y_test, model.predict(self.X_test))
        test_auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])

        results = {
            'model': model,
            'n_estimators_used': n_estimators_used,
            'train_accuracy': train_acc,
            'train_auc': train_auc,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'accuracy_gap': train_acc - test_acc,
            'auc_gap': train_auc - test_auc
        }

        logger.info(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Gap: {results['accuracy_gap']:.4f}")
        logger.info(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Gap: {results['auc_gap']:.4f}")

        self.results['xgboost'] = results
        return results

    # ==========================================================================
    # STRATEGY 5: LIGHTGBM WITH EARLY STOPPING
    # ==========================================================================

    def train_lightgbm(self) -> dict:
        """
        LightGBM with early stopping and leaf-wise tree growth
        Based on: Neptune AI (2024) - XGBoost vs LightGBM comparison
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 5: LIGHTGBM WITH EARLY STOPPING")
        logger.info("="*80)

        logger.info("Training LightGBM with early stopping...")

        # Split validation set
        split_idx = int(0.8 * len(self.X_train))
        X_train_split = self.X_train[:split_idx]
        y_train_split = self.y_train[:split_idx]
        X_val_split = self.X_train[split_idx:]
        y_val_split = self.y_train[split_idx:]

        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_data_in_leaf=20,
            lambda_l1=1.0,
            lambda_l2=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(
            X_train_split,
            y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            eval_metric='logloss',
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(period=0)
            ]
        )

        n_estimators_used = model.best_iteration
        logger.info(f"Early stopping at {n_estimators_used} iterations")

        # Evaluate on full training set
        train_acc = accuracy_score(self.y_train, model.predict(self.X_train))
        train_auc = roc_auc_score(self.y_train, model.predict_proba(self.X_train)[:, 1])
        test_acc = accuracy_score(self.y_test, model.predict(self.X_test))
        test_auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])

        results = {
            'model': model,
            'n_estimators_used': n_estimators_used,
            'train_accuracy': train_acc,
            'train_auc': train_auc,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'accuracy_gap': train_acc - test_acc,
            'auc_gap': train_auc - test_auc
        }

        logger.info(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Gap: {results['accuracy_gap']:.4f}")
        logger.info(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Gap: {results['auc_gap']:.4f}")

        self.results['lightgbm'] = results
        return results

    # ==========================================================================
    # RUN ALL STRATEGIES AND COMPARE
    # ==========================================================================

    def train_all_strategies(self) -> dict:
        """Train all strategies and compare results"""
        logger.info("\n\n")
        logger.info("#"*80)
        logger.info("# IMPROVED MODEL TRAINING - ALL STRATEGIES")
        logger.info("#"*80)

        self.train_optimized_random_forest()
        self.train_tuned_random_forest()
        self.train_gradient_boosting()
        try:
            self.train_xgboost()
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}, skipping...")
        try:
            self.train_lightgbm()
        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}, skipping...")

        return self.generate_comparison_report()

    def generate_comparison_report(self) -> dict:
        """Generate comparison report of all strategies"""
        logger.info("\n\n")
        logger.info("#"*80)
        logger.info("# MODEL COMPARISON REPORT")
        logger.info("#"*80)

        comparison = {}

        for strategy_name, strategy_result in self.results.items():
            logger.info(f"\n{strategy_name.upper()}:")
            logger.info(f"  Train Acc: {strategy_result['train_accuracy']:.4f}")
            logger.info(f"  Test Acc:  {strategy_result['test_accuracy']:.4f}")
            logger.info(f"  Gap:       {strategy_result['accuracy_gap']:.4f}")
            logger.info(f"  Train AUC: {strategy_result['train_auc']:.4f}")
            logger.info(f"  Test AUC:  {strategy_result['test_auc']:.4f}")

            comparison[strategy_name] = {
                'train_accuracy': strategy_result['train_accuracy'],
                'test_accuracy': strategy_result['test_accuracy'],
                'accuracy_gap': strategy_result['accuracy_gap'],
                'train_auc': strategy_result['train_auc'],
                'test_auc': strategy_result['test_auc'],
                'auc_gap': strategy_result['auc_gap']
            }

        # Find best strategy
        best_strategy = min(comparison.items(), key=lambda x: x[1]['accuracy_gap'])
        logger.info(f"\n{'='*80}")
        logger.info(f"BEST STRATEGY: {best_strategy[0]}")
        logger.info(f"Accuracy Gap: {best_strategy[1]['accuracy_gap']:.4f}")
        logger.info(f"{'='*80}")

        return {
            'comparison': comparison,
            'best_strategy': best_strategy[0],
            'best_gap': best_strategy[1]['accuracy_gap']
        }


if __name__ == '__main__':
    logger.info("Loading data...")

    df = pd.read_parquet('F_combined.parquet')

    X = df.drop('target', axis=1)
    y = df['target']

    # Create train/test split
    n_samples = len(X)
    train_size = int(0.8 * n_samples)
    gap = 50

    train_idx = np.arange(0, train_size - gap)
    test_idx = np.arange(train_size, n_samples)

    X_train = X.iloc[train_idx].values
    y_train = y.iloc[train_idx].values
    X_test = X.iloc[test_idx].values
    y_test = y.iloc[test_idx].values

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train models
    trainer = ImprovedModelTrainer(X_train, y_train, X_test, y_test)
    comparison = trainer.train_all_strategies()

    # Save results
    import json
    with open('improved_model_results.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info("\n✓ Results saved to: improved_model_results.json")
