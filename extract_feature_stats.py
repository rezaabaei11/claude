import re

def extract_stats(log_file):
    stats = {
        'batches': [],
        'significant_gain': [],
        'above_99_gain': [],
        'significant_split': [],
        'above_99_split': [],
        'stable_gain': [],
        'stable_split': []
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract batch-wise stats
    batch_num = 0
    for line in content.split('\n'):
        if 'Batch' in line and 'starting' in line:
            batch_num += 1
        elif 'Gain - Significant:' in line:
            match = re.search(r'Significant: (\d+), Above 99th: (\d+)', line)
            if match:
                stats['batches'].append(batch_num)
                stats['significant_gain'].append(int(match.group(1)))
                stats['above_99_gain'].append(int(match.group(2)))
        elif 'Split - Significant:' in line:
            match = re.search(r'Significant: (\d+), Above 99th: (\d+)', line)
            if match:
                stats['significant_split'].append(int(match.group(1)))
                stats['above_99_split'].append(int(match.group(2)))
        elif 'Stable features' in line:
            match = re.search(r'\(gain\): (\d+), \(split\): (\d+)', line)
            if match:
                stats['stable_gain'].append(int(match.group(1)))
                stats['stable_split'].append(int(match.group(2)))
    
    return stats

print("="*80)
print("تحلیل دقیق نتایج Feature Testing")
print("="*80)

baseline_stats = extract_stats('baseline_run.log')
improved_stats = extract_stats('improved_run.log')

print("\nBaseline Results:")
print("-" * 80)
print(f"Batch | Sig.Gain | 99th Gain | Sig.Split | 99th Split | Stable Gain | Stable Split")
print("-" * 80)
for i in range(len(baseline_stats['batches'])):
    print(f"  {baseline_stats['batches'][i]}   |    {baseline_stats['significant_gain'][i]}     |     {baseline_stats['above_99_gain'][i]}     |     {baseline_stats['significant_split'][i]}     |      {baseline_stats['above_99_split'][i]}     |      {baseline_stats['stable_gain'][i]}      |       {baseline_stats['stable_split'][i]}")

total_sig_gain_b = sum(baseline_stats['significant_gain'])
total_stable_gain_b = sum(baseline_stats['stable_gain'])

print(f"\nTotal Significant Features (Gain): {total_sig_gain_b}")
print(f"Total Stable Features (Gain): {total_stable_gain_b}")

print("\n" + "="*80)
print("Improved Results:")
print("-" * 80)
print(f"Batch | Sig.Gain | 99th Gain | Sig.Split | 99th Split | Stable Gain | Stable Split")
print("-" * 80)
for i in range(len(improved_stats['batches'])):
    print(f"  {improved_stats['batches'][i]}   |    {improved_stats['significant_gain'][i]}     |     {improved_stats['above_99_gain'][i]}     |     {improved_stats['significant_split'][i]}     |      {improved_stats['above_99_split'][i]}     |      {improved_stats['stable_gain'][i]}      |       {improved_stats['stable_split'][i]}")

total_sig_gain_i = sum(improved_stats['significant_gain'])
total_stable_gain_i = sum(improved_stats['stable_gain'])

print(f"\nTotal Significant Features (Gain): {total_sig_gain_i}")
print(f"Total Stable Features (Gain): {total_stable_gain_i}")

print("\n" + "="*80)
print("مقایسه اعتبار و پایداری")
print("="*80)
print(f"Significant Features (Gain):")
print(f"  Baseline: {total_sig_gain_b}")
print(f"  Improved: {total_sig_gain_i}")
print(f"  تغییر: {total_sig_gain_i - total_sig_gain_b:+d} ({((total_sig_gain_i - total_sig_gain_b) / max(1, total_sig_gain_b) * 100):+.1f}%)")

print(f"\nStable Features (Gain):")
print(f"  Baseline: {total_stable_gain_b}")
print(f"  Improved: {total_stable_gain_i}")
print(f"  تغییر: {total_stable_gain_i - total_stable_gain_b:+d} ({((total_stable_gain_i - total_stable_gain_b) / max(1, total_stable_gain_b) * 100):+.1f}%)")
