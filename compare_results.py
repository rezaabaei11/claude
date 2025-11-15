import json
import pandas as pd
import glob

def compare_batches(baseline_dir, improved_dir):
    results = {
        'batch': [],
        'baseline_cv_score': [],
        'improved_cv_score': [],
        'baseline_cv_std': [],
        'improved_cv_std': [],
        'baseline_time': [],
        'improved_time': [],
        'baseline_memory': [],
        'improved_memory': [],
        'baseline_significant': [],
        'improved_significant': [],
        'baseline_stable': [],
        'improved_stable': []
    }
    
    for batch_id in range(1, 6):
        baseline_file = f"{baseline_dir}/batch_{batch_id}_metadata.json"
        improved_file = f"{improved_dir}/batch_{batch_id}_metadata.json"
        
        try:
            with open(baseline_file) as f:
                baseline = json.load(f)
            with open(improved_file) as f:
                improved = json.load(f)
            
            results['batch'].append(batch_id)
            results['baseline_cv_score'].append(baseline['mean_cv_score'])
            results['improved_cv_score'].append(improved['mean_cv_score'])
            results['baseline_cv_std'].append(baseline['std_cv_score'])
            results['improved_cv_std'].append(improved['std_cv_score'])
            results['baseline_time'].append(baseline['execution_time_sec'])
            results['improved_time'].append(improved['execution_time_sec'])
            results['baseline_memory'].append(baseline['memory_used_mb'])
            results['improved_memory'].append(improved['memory_used_mb'])
            
            # Extract significant and stable features from logs
            results['baseline_significant'].append(0)  # placeholder
            results['improved_significant'].append(0)  # placeholder
            results['baseline_stable'].append(0)  # placeholder
            results['improved_stable'].append(0)  # placeholder
            
        except Exception as e:
            print(f"Error processing batch {batch_id}: {e}")
            continue
    
    df = pd.DataFrame(results)
    return df

df = compare_batches('feature_selection_results_baseline', 'feature_selection_results')
print("\n" + "="*80)
print("مقایسه نتایج Baseline و Improved")
print("="*80)
print(df.to_string(index=False))

print("\n" + "="*80)
print("خلاصه بهبودها")
print("="*80)

avg_baseline_score = df['baseline_cv_score'].mean()
avg_improved_score = df['improved_cv_score'].mean()
avg_baseline_time = df['baseline_time'].mean()
avg_improved_time = df['improved_time'].mean()
avg_baseline_memory = df['baseline_memory'].mean()
avg_improved_memory = df['improved_memory'].mean()

print(f"\n1. دقت (CV Score):")
print(f"   Baseline: {avg_baseline_score:.4f}")
print(f"   Improved: {avg_improved_score:.4f}")
print(f"   تغییر: {((avg_improved_score - avg_baseline_score) / avg_baseline_score * 100):+.2f}%")

print(f"\n2. زمان اجرا (ثانیه):")
print(f"   Baseline: {avg_baseline_time:.2f}s")
print(f"   Improved: {avg_improved_time:.2f}s")
print(f"   تغییر: {((avg_improved_time - avg_baseline_time) / avg_baseline_time * 100):+.2f}%")

print(f"\n3. مصرف حافظه (MB):")
print(f"   Baseline: {avg_baseline_memory:.2f} MB")
print(f"   Improved: {avg_improved_memory:.2f} MB")
print(f"   تغییر: {((avg_improved_memory - avg_baseline_memory) / avg_baseline_memory * 100):+.2f}%")

df.to_csv('comparison_results.csv', index=False)
print("\n✓ نتایج در فایل comparison_results.csv ذخیره شد")
