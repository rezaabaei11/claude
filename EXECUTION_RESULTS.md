# نتایج اجرای کد پایتون / Python Code Execution Results

## خلاصه / Summary

کد پایتون `FE.py` با موفقیت اجرا شد و نتایج تحلیل ویژگی‌ها (Feature Selection) تولید شد.

The Python code `FE.py` has been executed successfully and feature selection analysis results have been generated.

## جزئیات اجرا / Execution Details

- **تاریخ اجرا / Execution Date**: 2025-11-15 09:00 UTC
- **مدت زمان اجرا / Total Duration**: ~40 ثانیه / seconds
- **وضعیت / Status**: ✅ موفق / Successful
- **خروجی / Exit Code**: 0

## ورودی‌های استفاده شده / Input Files Used

1. **XAUUSD_M15_R.csv**: داده‌های قیمت طلا (XAUUSD) با فریم زمانی 15 دقیقه‌ای
   - تعداد سطرها: 34,091 ردیف
   - ستون‌ها: DATE, TIME, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD

## پردازش انجام شده / Processing Performed

اسکریپت پایتون تحلیل انتخاب ویژگی (Feature Selection) را بر روی داده‌های XAUUSD انجام داده است:

1. **بارگذاری و پیش‌پردازش داده‌ها**: خواندن CSV و ایجاد target variable
2. **تقسیم به 5 دسته (Batch)**: داده‌ها به 5 batch تقسیم شده‌اند
3. **تحلیل ویژگی‌ها برای هر batch**:
   - Null Importance Testing
   - Multicollinearity Detection
   - Boosting Ensemble Analysis
   - Feature Fraction Analysis
   - Adversarial Validation
   - RFE (Recursive Feature Elimination)
   - Cross-Validation
   - Stability Bootstrap Analysis

## نتایج تولید شده / Generated Results

### فایل‌های خروجی / Output Files

تمام نتایج در پوشه `feature_selection_results/` ذخیره شده‌اند:

#### برای هر Batch (1-5):

1. **batch_X_ranking_TIMESTAMP.csv**: رتبه‌بندی کامل ویژگی‌ها با امتیاز نهایی
2. **batch_X_strong.csv**: ویژگی‌های قوی (Strong features)
3. **batch_X_medium.csv**: ویژگی‌های متوسط (Medium features)
4. **batch_X_weak.csv**: ویژگی‌های ضعیف (Weak features)
5. **batch_X_metadata.json**: متادیتا شامل آمار و معیارهای ارزیابی

### نمونه نتایج Batch 1:

**ویژگی‌های قوی (Strong Features)**:
```
1. tickvol (امتیاز: 0.691)
2. close (امتیاز: 0.474)
3. low (امتیاز: 0.465)
4. high (امتیاز: 0.453)
5. open (امتیاز: 0.449)
6. spread (امتیاز: 0.445)
```

**معیارهای ارزیابی**:
- تعداد کل ویژگی‌ها: 6
- ویژگی‌های قوی: 6
- ویژگی‌های متوسط: 0
- ویژگی‌های ضعیف: 3
- میانگین امتیاز CV: 0.493
- انحراف معیار CV: 0.005

### لاگ اجرا / Execution Log

فایل `feature_selection.log` شامل لاگ کامل اجرا می‌باشد.

## نتیجه‌گیری / Conclusion

✅ کد پایتون با موفقیت اجرا شد و تحلیل کامل انتخاب ویژگی را بر روی داده‌های XAUUSD انجام داد.

✅ The Python code executed successfully and performed comprehensive feature selection analysis on XAUUSD data.

### یافته‌های کلیدی / Key Findings:

1. **tickvol** (حجم تیک) بهترین ویژگی با امتیاز 0.691 است
2. قیمت‌های بسته شدن (close)، پایین (low)، و بالا (high) ویژگی‌های قوی هستند
3. همه 6 ویژگی پایه در دسته strong قرار گرفته‌اند
4. همبستگی بالا (multicollinearity) بین برخی ویژگی‌ها وجود دارد (6 جفت)
5. هیچ ویژگی معنادار با آزمون Null Importance پیدا نشد (احتمالاً به دلیل تعداد کم ویژگی‌ها)

## وابستگی‌های نصب شده / Installed Dependencies

```
- pandas==2.3.3
- numpy==2.3.4
- lightgbm==4.6.0
- scikit-learn==1.7.2
- scipy==1.16.3
```

## دستورات مورد استفاده / Commands Used

```bash
# نصب وابستگی‌ها
pip3 install pandas numpy lightgbm scikit-learn scipy

# اجرای اسکریپت
python3 FE.py
```

---

**تاریخ تهیه گزارش / Report Date**: 2025-11-15
