# Baseline Execution Results

## Execution Summary
- **Start Time:** 2025-11-15 13:03:50
- **End Time:** 2025-11-15 13:04:28
- **Total Duration:** ~38 seconds
- **Status:** Completed successfully

## Batch Results

### Batch 1
- Features tested: 6
- Mean CV Score: 0.4928 (49.28% accuracy)
- Std CV Score: 0.0050
- Significant features (Gain): 0
- Significant features (Split): 0
- Stable features (Gain): 1
- High correlation pairs: 6
- Condition Index: 211.51

### Batch 2
- Features tested: 6
- Mean CV Score: 0.5080 (50.80% accuracy)
- Std CV Score: 0.0041
- Significant features (Gain): 0
- Significant features (Split): 0
- Stable features (Gain): 2
- High correlation pairs: 6
- Condition Index: 187.65

### Batch 3
- Features tested: 6
- Mean CV Score: 0.5113 (51.13% accuracy)
- Std CV Score: 0.0140
- Significant features (Gain): 0
- Significant features (Split): 0
- Stable features (Gain): 3
- High correlation pairs: 6
- Condition Index: 427.81

### Batch 4
- Features tested: 6
- Mean CV Score: 0.5033 (50.33% accuracy)
- Std CV Score: 0.0072
- Significant features (Gain): 0
- Significant features (Split): 0
- Stable features (Gain): 2
- High correlation pairs: 6
- Condition Index: 112.22

### Batch 5
- Features tested: 6
- Mean CV Score: 0.4620 (46.20% accuracy)
- Std CV Score: 0.0187
- Significant features (Gain): 0
- Significant features (Split): 0
- Stable features (Gain): 3
- High correlation pairs: 6
- Condition Index: 574.51

## Overall Statistics
- **Average CV Score across all batches:** 0.4955 (49.55%)
- **Average Std CV Score:** 0.0098
- **Total features evaluated:** 6 per batch
- **Significant features found:** 0 (CRITICAL ISSUE!)
- **Average stable features:** 2.2

## Key Issues Identified
1. **NO significant features found in null importance test** - All features failed significance test
2. **Low accuracy** - ~50% is barely better than random (50% baseline for binary classification)
3. **High multicollinearity** - All batches show 6 high correlation pairs
4. **Very high condition indices** - Indicating severe multicollinearity problems
5. **Only using basic OHLC data** - No sophisticated feature engineering

## Feature Rankings (Batch 1)
1. tickvol: 0.691
2. close: 0.474
3. low: 0.465
4. high: 0.453
5. open: 0.382
6. spread: 0.209
