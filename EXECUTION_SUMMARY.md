# Feature Selection Execution Summary

## Overview
The FE.py script has been successfully executed on November 15, 2025 at 11:37-11:39 UTC.

## Configuration
- **Script**: FE.py (Feature Selection using LightGBM)
- **Dataset**: XAUUSD_M15_R.csv (Gold Price 15-minute time series data)
- **Number of Batches**: 5
- **Features Analyzed**: 6 main features (open, high, low, close, tickvol, spread)
- **Samples per Batch**: ~6,818 rows
- **Train/Test Split**: 80/20 with 50-sample gap

## Dependencies Installed
- pandas 2.3.3
- numpy 2.3.4
- lightgbm 4.6.0
- scikit-learn 1.7.2
- scipy 1.16.3
- psutil 7.1.3

## Execution Results

### Files Generated
30 output files created in the `feature_selection_results/` directory:

For each of the 5 batches:
- `batch_X_ranking_TIMESTAMP.csv` - Feature importance rankings
- `batch_X_strong.csv` - Strong features
- `batch_X_medium.csv` - Medium features
- `batch_X_weak.csv` - Weak features
- `batch_X_metadata.json` - Batch metadata and statistics

### Feature Rankings (Batch 1)
Based on final scores from ensemble ranking:

1. **tickvol** - 0.691 (Strongest)
2. **close** - 0.474
3. **low** - 0.465
4. **high** - 0.453
5. **open** - 0.382
6. **spread** - 0.209 (Weakest)

### Analysis Performed

For each batch, the following analyses were conducted:

1. **Multicollinearity Detection**
   - High correlation pairs identified
   - Condition index calculated
   - Adaptive penalty applied

2. **Null Importance Test**
   - 3 actual runs vs 20 null runs
   - Statistical significance testing
   - Z-score and p-value calculation

3. **Boosting Ensemble**
   - GOSS (Gradient-based One-Side Sampling)
   - DART (Dropouts meet Multiple Additive Regression Trees)
   - Extra Trees ensemble

4. **Feature Fraction Analysis**
   - By-node, by-tree, and combined strategies

5. **Adversarial Validation**
   - Time series validation with proper gap
   - Distribution shift detection

6. **RFE (Recursive Feature Elimination)**
   - Target: 20 features (limited by dataset size)

7. **Cross-Validation Multi-Metric**
   - 2 splits with time series split
   - 50-sample gap for leakage prevention

8. **Stability Bootstrap**
   - 10 bootstrap iterations
   - Feature stability scoring

### Key Findings

- **Constant Features Removed**: 1 feature per batch (likely a date/time column)
- **Significant Features (Null Test)**: 0 across all batches (indicating weak predictive power)
- **High Correlation Pairs**: 6 pairs detected in all batches
- **Stable Features**: 1-3 features per batch showed stability
- **Mean CV Score**: ~0.49 (approximately random performance)

### Performance Metrics

- **Total Execution Time**: ~90 seconds
- **Processing per Batch**: ~18 seconds
- **Memory Usage**: Optimized with dtype reduction and garbage collection

## Conclusion

The script executed successfully and performed comprehensive feature selection analysis on gold price data. The results indicate:

1. All features were analyzed across 5 different time periods
2. `tickvol` (tick volume) showed the highest importance consistently
3. Price-based features (close, low, high, open) had moderate importance
4. `spread` showed the lowest importance
5. The model achieved approximately 49% accuracy, suggesting the features alone may not be sufficient for strong prediction

## Output Files Location

All results are saved in: `/home/runner/work/claude/claude/feature_selection_results/`

## Log File

Detailed execution log available at: `feature_selection.log`

---
Generated on: 2025-11-15 11:39 UTC
