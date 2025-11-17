"""
Walk-Forward Validation Implementation for Time-Series Machine Learning
Uses expanding windows for realistic production-like validation
Compares Original Model vs Improved Model
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import pyarrow.parquet as pq
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('walk_forward_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """Implement Walk-Forward Validation for time-series models"""

    def __init__(self, X, y, initial_train_size=0.5, test_size=0.1, gap=50):
        """
        Initialize Walk-Forward Validator

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        initial_train_size : float
            Initial training window as fraction of data (default 50%)
        test_size : float
            Test window size as fraction of data (default 10%)
        gap : int
            Number of samples to skip between train and test (prevent leakage)
        """
        self.X = X
        self.y = y
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.gap = gap
        self.n_samples = len(X)

        self.results = {
            'original_model': [],
            'improved_model': [],
            'comparison': {}
        }

        logger.info(f"Walk-Forward Validator initialized:")
        logger.info(f"  - Total samples: {self.n_samples}")
        logger.info(f"  - Initial train size: {int(initial_train_size * self.n_samples)}")
        logger.info(f"  - Test size per fold: {int(test_size * self.n_samples)}")
        logger.info(f"  - Gap between train/test: {gap}")

    def run_walk_forward(self, verbose=True):
        """
        Execute Walk-Forward Validation with expanding windows
        """
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD VALIDATION - EXPANDING WINDOWS")
        logger.info("="*80)

        initial_train_idx = int(self.initial_train_size * self.n_samples)
        test_size_samples = int(self.test_size * self.n_samples)

        fold_number = 0
        current_train_end = initial_train_idx

        # Generate walk-forward folds
        folds = []
        while current_train_end + self.gap + test_size_samples < self.n_samples:
            train_start = 0
            train_end = current_train_end
            test_start = current_train_end + self.gap
            test_end = test_start + test_size_samples

            folds.append({
                'fold': fold_number,
                'train_range': (train_start, train_end),
                'test_range': (test_start, test_end),
                'train_size': train_end - train_start,
                'test_size': test_end - test_start
            })

            # Expand training window
            current_train_end += test_size_samples
            fold_number += 1

        logger.info(f"\nGenerated {len(folds)} walk-forward folds")
        logger.info("\nFold Schedule:")
        for fold in folds:
            logger.info(f"  Fold {fold['fold']}: Train [{fold['train_range'][0]}-{fold['train_range'][1]}] "
                       f"→ Test [{fold['test_range'][0]}-{fold['test_range'][1]}]")

        # Run each fold
        for fold_info in folds:
            self._run_fold(fold_info)

        # Generate summary statistics
        self._generate_summary()

        logger.info("\nWalk-Forward Validation Complete!")
        return folds

    def _run_fold(self, fold_info):
        """Run a single fold of walk-forward validation"""
        fold = fold_info['fold']
        train_start, train_end = fold_info['train_range']
        test_start, test_end = fold_info['test_range']

        # Get train/test data
        X_train = self.X[train_start:train_end]
        y_train = self.y[train_start:train_end]
        X_test = self.X[test_start:test_end]
        y_test = self.y[test_start:test_end]

        logger.info(f"\n{'─'*80}")
        logger.info(f"FOLD {fold}: Train={len(X_train)} | Test={len(X_test)} | Gap={self.gap}")
        logger.info(f"{'─'*80}")

        # =====================================================================
        # ORIGINAL MODEL (max_depth=15) - BASELINE
        # =====================================================================
        logger.info("\n▶ Original Model (max_depth=15):")

        model_orig = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,  # Original problematic depth
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )

        model_orig.fit(X_train, y_train)

        y_pred_orig = model_orig.predict(X_test)
        y_proba_orig = model_orig.predict_proba(X_test)[:, 1]

        orig_metrics = {
            'fold': fold,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy_score(y_test, y_pred_orig),
            'auc': roc_auc_score(y_test, y_proba_orig),
            'precision': precision_score(y_test, y_pred_orig),
            'recall': recall_score(y_test, y_pred_orig),
            'f1': f1_score(y_test, y_pred_orig)
        }

        logger.info(f"  Accuracy:  {orig_metrics['accuracy']:.4f}")
        logger.info(f"  AUC:       {orig_metrics['auc']:.4f}")
        logger.info(f"  Precision: {orig_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {orig_metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {orig_metrics['f1']:.4f}")

        self.results['original_model'].append(orig_metrics)

        # =====================================================================
        # IMPROVED MODEL (max_depth=8) - OPTIMIZED
        # =====================================================================
        logger.info("\n▶ Improved Model (max_depth=8):")

        model_impr = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,  # Reduced depth
            min_samples_leaf=20,  # Regularization
            min_samples_split=50,  # Regularization
            max_features='sqrt',  # Feature selection
            random_state=42,
            n_jobs=-1
        )

        model_impr.fit(X_train, y_train)

        y_pred_impr = model_impr.predict(X_test)
        y_proba_impr = model_impr.predict_proba(X_test)[:, 1]

        impr_metrics = {
            'fold': fold,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy_score(y_test, y_pred_impr),
            'auc': roc_auc_score(y_test, y_proba_impr),
            'precision': precision_score(y_test, y_pred_impr),
            'recall': recall_score(y_test, y_pred_impr),
            'f1': f1_score(y_test, y_pred_impr)
        }

        logger.info(f"  Accuracy:  {impr_metrics['accuracy']:.4f}")
        logger.info(f"  AUC:       {impr_metrics['auc']:.4f}")
        logger.info(f"  Precision: {impr_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {impr_metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {impr_metrics['f1']:.4f}")

        self.results['improved_model'].append(impr_metrics)

        # =====================================================================
        # COMPARISON
        # =====================================================================
        acc_diff = impr_metrics['accuracy'] - orig_metrics['accuracy']
        auc_diff = impr_metrics['auc'] - orig_metrics['auc']

        logger.info(f"\n📊 Fold {fold} Comparison:")
        logger.info(f"  Accuracy Difference: {acc_diff:+.4f}")
        logger.info(f"  AUC Difference:      {auc_diff:+.4f}")

        if acc_diff > 0:
            logger.info(f"  ✅ Improved model BETTER by {abs(acc_diff):.4f}")
        elif acc_diff < 0:
            logger.info(f"  ⚠️  Improved model slightly worse by {abs(acc_diff):.4f}")
        else:
            logger.info(f"  ➡️  Models equivalent")

    def _generate_summary(self):
        """Generate summary statistics for all folds"""
        logger.info("\n\n" + "="*80)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("="*80)

        orig_results = pd.DataFrame(self.results['original_model'])
        impr_results = pd.DataFrame(self.results['improved_model'])

        # Original model statistics
        logger.info("\n▶ ORIGINAL MODEL (max_depth=15):")
        logger.info(f"  Accuracy:  {orig_results['accuracy'].mean():.4f} ± {orig_results['accuracy'].std():.4f}")
        logger.info(f"  AUC:       {orig_results['auc'].mean():.4f} ± {orig_results['auc'].std():.4f}")
        logger.info(f"  Precision: {orig_results['precision'].mean():.4f} ± {orig_results['precision'].std():.4f}")
        logger.info(f"  Recall:    {orig_results['recall'].mean():.4f} ± {orig_results['recall'].std():.4f}")
        logger.info(f"  F1-Score:  {orig_results['f1'].mean():.4f} ± {orig_results['f1'].std():.4f}")

        # Improved model statistics
        logger.info("\n▶ IMPROVED MODEL (max_depth=8):")
        logger.info(f"  Accuracy:  {impr_results['accuracy'].mean():.4f} ± {impr_results['accuracy'].std():.4f}")
        logger.info(f"  AUC:       {impr_results['auc'].mean():.4f} ± {impr_results['auc'].std():.4f}")
        logger.info(f"  Precision: {impr_results['precision'].mean():.4f} ± {impr_results['precision'].std():.4f}")
        logger.info(f"  Recall:    {impr_results['recall'].mean():.4f} ± {impr_results['recall'].std():.4f}")
        logger.info(f"  F1-Score:  {impr_results['f1'].mean():.4f} ± {impr_results['f1'].std():.4f}")

        # Comparison
        logger.info("\n▶ COMPARISON (Improved - Original):")
        acc_improvement = impr_results['accuracy'].mean() - orig_results['accuracy'].mean()
        auc_improvement = impr_results['auc'].mean() - orig_results['auc'].mean()
        precision_improvement = impr_results['precision'].mean() - orig_results['precision'].mean()
        recall_improvement = impr_results['recall'].mean() - orig_results['recall'].mean()
        f1_improvement = impr_results['f1'].mean() - orig_results['f1'].mean()

        logger.info(f"  Accuracy:  {acc_improvement:+.4f}")
        logger.info(f"  AUC:       {auc_improvement:+.4f}")
        logger.info(f"  Precision: {precision_improvement:+.4f}")
        logger.info(f"  Recall:    {recall_improvement:+.4f}")
        logger.info(f"  F1-Score:  {f1_improvement:+.4f}")

        # Store summary
        self.results['comparison'] = {
            'n_folds': len(self.results['original_model']),
            'original_accuracy_mean': orig_results['accuracy'].mean(),
            'original_accuracy_std': orig_results['accuracy'].std(),
            'original_auc_mean': orig_results['auc'].mean(),
            'original_auc_std': orig_results['auc'].std(),
            'improved_accuracy_mean': impr_results['accuracy'].mean(),
            'improved_accuracy_std': impr_results['accuracy'].std(),
            'improved_auc_mean': impr_results['auc'].mean(),
            'improved_auc_std': impr_results['auc'].std(),
            'accuracy_improvement': acc_improvement,
            'auc_improvement': auc_improvement,
            'f1_improvement': f1_improvement
        }

    def plot_results(self):
        """Create comprehensive visualization of walk-forward validation results"""
        logger.info("\nGenerating Walk-Forward Validation visualizations...")

        orig_results = pd.DataFrame(self.results['original_model'])
        impr_results = pd.DataFrame(self.results['improved_model'])

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Plot 1: Accuracy over folds
        ax1 = fig.add_subplot(gs[0, 0])
        folds = orig_results['fold'].values
        ax1.plot(folds, orig_results['accuracy'], 'o-', label='Original (depth=15)', linewidth=2, markersize=8)
        ax1.plot(folds, impr_results['accuracy'], 's-', label='Improved (depth=8)', linewidth=2, markersize=8)
        ax1.axhline(orig_results['accuracy'].mean(), color='C0', linestyle='--', alpha=0.5)
        ax1.axhline(impr_results['accuracy'].mean(), color='C1', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Fold Number', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Accuracy Progression Across Folds', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.5, 0.8])

        # Plot 2: AUC over folds
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(folds, orig_results['auc'], 'o-', label='Original (depth=15)', linewidth=2, markersize=8)
        ax2.plot(folds, impr_results['auc'], 's-', label='Improved (depth=8)', linewidth=2, markersize=8)
        ax2.axhline(orig_results['auc'].mean(), color='C0', linestyle='--', alpha=0.5)
        ax2.axhline(impr_results['auc'].mean(), color='C1', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Fold Number', fontsize=11)
        ax2.set_ylabel('AUC Score', fontsize=11)
        ax2.set_title('AUC Progression Across Folds', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.65, 0.85])

        # Plot 3: Precision over folds
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(folds, orig_results['precision'], 'o-', label='Original (depth=15)', linewidth=2, markersize=8)
        ax3.plot(folds, impr_results['precision'], 's-', label='Improved (depth=8)', linewidth=2, markersize=8)
        ax3.axhline(orig_results['precision'].mean(), color='C0', linestyle='--', alpha=0.5)
        ax3.axhline(impr_results['precision'].mean(), color='C1', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Fold Number', fontsize=11)
        ax3.set_ylabel('Precision Score', fontsize=11)
        ax3.set_title('Precision Progression Across Folds', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Recall over folds
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(folds, orig_results['recall'], 'o-', label='Original (depth=15)', linewidth=2, markersize=8)
        ax4.plot(folds, impr_results['recall'], 's-', label='Improved (depth=8)', linewidth=2, markersize=8)
        ax4.axhline(orig_results['recall'].mean(), color='C0', linestyle='--', alpha=0.5)
        ax4.axhline(impr_results['recall'].mean(), color='C1', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Fold Number', fontsize=11)
        ax4.set_ylabel('Recall Score', fontsize=11)
        ax4.set_title('Recall Progression Across Folds', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # Plot 5: F1-Score over folds
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(folds, orig_results['f1'], 'o-', label='Original (depth=15)', linewidth=2, markersize=8)
        ax5.plot(folds, impr_results['f1'], 's-', label='Improved (depth=8)', linewidth=2, markersize=8)
        ax5.axhline(orig_results['f1'].mean(), color='C0', linestyle='--', alpha=0.5)
        ax5.axhline(impr_results['f1'].mean(), color='C1', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Fold Number', fontsize=11)
        ax5.set_ylabel('F1-Score', fontsize=11)
        ax5.set_title('F1-Score Progression Across Folds', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)

        # Plot 6: Summary statistics comparison
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')

        summary_text = f"""
WALK-FORWARD VALIDATION SUMMARY
(All Folds Combined)

ORIGINAL MODEL (max_depth=15):
  Accuracy:  {orig_results['accuracy'].mean():.4f} ± {orig_results['accuracy'].std():.4f}
  AUC:       {orig_results['auc'].mean():.4f} ± {orig_results['auc'].std():.4f}
  Precision: {orig_results['precision'].mean():.4f} ± {orig_results['precision'].std():.4f}
  Recall:    {orig_results['recall'].mean():.4f} ± {orig_results['recall'].std():.4f}
  F1-Score:  {orig_results['f1'].mean():.4f} ± {orig_results['f1'].std():.4f}

IMPROVED MODEL (max_depth=8):
  Accuracy:  {impr_results['accuracy'].mean():.4f} ± {impr_results['accuracy'].std():.4f}
  AUC:       {impr_results['auc'].mean():.4f} ± {impr_results['auc'].std():.4f}
  Precision: {impr_results['precision'].mean():.4f} ± {impr_results['precision'].std():.4f}
  Recall:    {impr_results['recall'].mean():.4f} ± {impr_results['recall'].std():.4f}
  F1-Score:  {impr_results['f1'].mean():.4f} ± {impr_results['f1'].std():.4f}

IMPROVEMENT:
  Accuracy:  {impr_results['accuracy'].mean() - orig_results['accuracy'].mean():+.4f}
  AUC:       {impr_results['auc'].mean() - orig_results['auc'].mean():+.4f}
  F1-Score:  {impr_results['f1'].mean() - orig_results['f1'].mean():+.4f}
        """

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Walk-Forward Validation Results: Original vs Improved Model',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.savefig('walk_forward_validation_results.png', dpi=150, bbox_inches='tight')
        logger.info("✅ Saved: walk_forward_validation_results.png")
        plt.close()

    def save_results(self):
        """Save results to JSON file"""
        with open('wfv_results.json', 'w') as f:
            # Convert DataFrames to dict for JSON serialization
            results_to_save = {
                'original_model': self.results['original_model'],
                'improved_model': self.results['improved_model'],
                'comparison': self.results['comparison'],
                'timestamp': datetime.now().isoformat()
            }
            json.dump(results_to_save, f, indent=2)

        logger.info("✅ Saved: wfv_results.json")

def main():
    logger.info("\n" + "╔" + "="*78 + "╗")
    logger.info("║" + " "*15 + "WALK-FORWARD VALIDATION IMPLEMENTATION" + " "*25 + "║")
    logger.info("╚" + "="*78 + "╝\n")

    # Load data
    logger.info("Loading data...")
    try:
        data = pq.read_table('/home/user/claude/F_combined.parquet').to_pandas()
        X = data.drop(['target', 'datetime'], axis=1, errors='ignore').values
        y = data['target'].values

        logger.info(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return

    # Initialize and run WFV
    validator = WalkForwardValidator(
        X, y,
        initial_train_size=0.5,  # Start with 50% of data for training
        test_size=0.1,           # Test on 10% of data each fold
        gap=50                   # 50-sample gap to prevent leakage
    )

    # Run walk-forward validation
    validator.run_walk_forward()

    # Generate visualizations
    validator.plot_results()

    # Save results
    validator.save_results()

    logger.info("\n" + "="*80)
    logger.info("✅ Walk-Forward Validation Complete!")
    logger.info("="*80)

if __name__ == "__main__":
    main()
