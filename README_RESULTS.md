# نتایج اجرای کد / Code Execution Results

[فارسی](#فارسی) | [English](#english)

---

## فارسی

### خلاصه اجرا

کد `FE.py` با موفقیت اجرا شد و نتایج به طور کامل تحلیل شدند.

### فایل‌های مهم

1. **`analysis_results.md`** - تحلیل جامع به فارسی ⭐
   - توضیحات کامل نتایج
   - تحلیل فنی و آماری
   - توصیه‌های بهبود
   - مثال‌های کد

2. **`EXECUTION_SUMMARY_EN.md`** - خلاصه کامل به انگلیسی
   - نتایج اجرا
   - جداول مقایسه
   - توصیه‌های تکنیکال

3. **`feature_selection_results/`** - پوشه نتایج
   - 25 فایل CSV و JSON
   - رتبه‌بندی فیچرها برای 5 batch
   - متادیتا و آمار

4. **`feature_selection.log`** - لاگ کامل اجرا

### نتایج کلیدی

✅ **اجرا موفق:** 5 batch پردازش شد  
⚠️ **عملکرد مدل:** 0.4955 (نزدیک به تصادفی - نیاز به بهبود)  
📊 **بهترین فیچر:** tickvol (0.691)  
🔍 **مشکل اصلی:** تعداد فیچرها خیلی کم است (فقط 6)

### پیشنهادات فوری

1. اضافه کردن اندیکاتورهای تکنیکال (RSI, MACD, Bollinger Bands)
2. ساخت فیچرهای مشتق شده (returns, ranges, lags)
3. استفاده از TSFresh برای فیچرهای پیشرفته
4. استفاده از داده با کیفیت بالاتر

### چطور از نتایج استفاده کنم؟

```bash
# مشاهده رتبه‌بندی فیچرها (Batch 1)
cat feature_selection_results/batch_1_ranking_*.csv

# مشاهده فیچرهای قوی
cat feature_selection_results/batch_1_strong.csv

# مشاهده آمار کامل
cat feature_selection_results/batch_1_metadata.json
```

### نمودار نتایج

| فیچر | امتیاز | تفسیر |
|------|--------|-------|
| tickvol | 0.691 | 🟢 بسیار قوی |
| close | 0.474 | 🟡 متوسط |
| low | 0.465 | 🟡 متوسط |
| high | 0.453 | 🟡 متوسط |
| open | 0.382 | 🟡 ضعیف |
| spread | 0.209 | 🔴 خیلی ضعیف (ثابت) |

---

## English

### Execution Summary

The code `FE.py` was successfully executed and results comprehensively analyzed.

### Important Files

1. **`analysis_results.md`** - Comprehensive analysis in Persian ⭐
   - Complete results explanation
   - Technical and statistical analysis
   - Improvement recommendations
   - Code examples

2. **`EXECUTION_SUMMARY_EN.md`** - Complete summary in English
   - Execution results
   - Comparison tables
   - Technical recommendations

3. **`feature_selection_results/`** - Results directory
   - 25 CSV and JSON files
   - Feature rankings for 5 batches
   - Metadata and statistics

4. **`feature_selection.log`** - Complete execution log

### Key Results

✅ **Execution Success:** 5 batches processed  
⚠️ **Model Performance:** 0.4955 (near-random - needs improvement)  
📊 **Best Feature:** tickvol (0.691)  
🔍 **Main Issue:** Too few features (only 6)

### Immediate Recommendations

1. Add technical indicators (RSI, MACD, Bollinger Bands)
2. Create derived features (returns, ranges, lags)
3. Use TSFresh for advanced features
4. Use higher quality data

### How to Use the Results?

```bash
# View feature rankings (Batch 1)
cat feature_selection_results/batch_1_ranking_*.csv

# View strong features
cat feature_selection_results/batch_1_strong.csv

# View complete statistics
cat feature_selection_results/batch_1_metadata.json
```

### Results Chart

| Feature | Score | Interpretation |
|---------|-------|----------------|
| tickvol | 0.691 | 🟢 Very Strong |
| close | 0.474 | 🟡 Medium |
| low | 0.465 | 🟡 Medium |
| high | 0.453 | �� Medium |
| open | 0.382 | 🟡 Weak |
| spread | 0.209 | 🔴 Very Weak (constant) |

---

## Quick Start

### View Results
```bash
# Summary statistics
python3 << 'PYEOF'
import pandas as pd
import json

# Load results
with open('feature_selection_results/batch_1_metadata.json') as f:
    meta = json.load(f)

ranking = pd.read_csv('feature_selection_results/batch_1_ranking_20251115_114317.csv')

print("Batch 1 Statistics:")
print(f"  CV Score: {meta['mean_cv_score']:.4f}")
print(f"  Total Features: {meta['n_total']}")
print("\nFeature Rankings:")
print(ranking.to_string(index=False))
PYEOF
```

### Compare All Batches
```bash
# Load and compare all batches
python3 << 'PYEOF'
import pandas as pd
import json

batches = []
for i in range(1, 6):
    with open(f'feature_selection_results/batch_{i}_metadata.json') as f:
        batches.append(json.load(f))

df = pd.DataFrame(batches)
print("All Batches Comparison:")
print(df[['batch_id', 'mean_cv_score', 'n_strong', 'n_weak']].to_string(index=False))
PYEOF
```

---

## Next Steps

### For Better Results:

1. **Feature Engineering**
   - Add technical indicators library (TA-Lib)
   - Create rolling window features
   - Add lag features (1, 5, 10, 15 periods)

2. **Data Enhancement**
   - Get data with variable spreads
   - Include more time periods
   - Add external features (market sentiment)

3. **Model Optimization**
   - Hyperparameter tuning (Optuna, GridSearch)
   - Try ensemble methods
   - Experiment with different algorithms

4. **Advanced Methods**
   - Use TSFresh for automated feature extraction
   - Apply AutoML (AutoGluon, H2O)
   - Consider deep learning (LSTM, Transformers)

---

## Support

For questions or issues:
- Check `analysis_results.md` for detailed Persian analysis
- Check `EXECUTION_SUMMARY_EN.md` for English summary
- Review the documentation in `md1.md`

**Status:** ✅ Code executed successfully  
**Date:** November 15, 2025  
**Execution Time:** ~38 seconds
