"""
Create visualizations for feature selection results
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Load data
ranking_df = pd.read_parquet('feature_selection_results/batch_0_ranking_20251117_144039.parquet')
strong_df = pd.read_csv('feature_selection_results/batch_0_strong.csv')
medium_df = pd.read_csv('feature_selection_results/batch_0_medium.csv')
weak_df = pd.read_csv('feature_selection_results/batch_0_weak.csv')

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Top 20 Features
ax1 = plt.subplot(2, 3, 1)
top_20 = ranking_df.head(20).copy()
top_20 = top_20.sort_values('final_score', ascending=True)
colors = ['#2ecc71' if i < 5 else '#3498db' if i < 10 else '#f39c12' for i in range(len(top_20))]
ax1.barh(range(len(top_20)), top_20['final_score'].values, color=colors)
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels([f.replace('high__', '') for f in top_20['feature'].values], fontsize=8)
ax1.set_xlabel('Final Score')
ax1.set_title('Top 20 Most Important Features', fontweight='bold', fontsize=12)
ax1.grid(axis='x', alpha=0.3)

# 2. Feature Distribution by Category
ax2 = plt.subplot(2, 3, 2)
categories = ['Strong\n(15)', 'Medium\n(45)', 'Weak\n(40)']
counts = [15, 45, 40]
colors_cat = ['#2ecc71', '#3498db', '#e74c3c']
wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%',
                                     colors=colors_cat, startangle=90, textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax2.set_title('Feature Distribution by Category', fontweight='bold', fontsize=12)

# 3. Score Distribution
ax3 = plt.subplot(2, 3, 3)
scores = ranking_df['final_score'].dropna().values
ax3.hist(scores, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
ax3.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.3f}')
ax3.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.3f}')
ax3.set_xlabel('Final Score')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Feature Importance Scores', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Strong Features Detail
ax4 = plt.subplot(2, 3, 4)
strong_features = ranking_df[ranking_df['feature'].isin(strong_df['feature'].values)].copy()
strong_features = strong_features.sort_values('final_score', ascending=True)
ax4.barh(range(len(strong_features)), strong_features['final_score'].values, color='#2ecc71')
ax4.set_yticks(range(len(strong_features)))
ax4.set_yticklabels([f.replace('high__', '') for f in strong_features['feature'].values], fontsize=8)
ax4.set_xlabel('Final Score')
ax4.set_title('All Strong Features (Top 15%)', fontweight='bold', fontsize=12)
ax4.grid(axis='x', alpha=0.3)

# 5. Cumulative Importance
ax5 = plt.subplot(2, 3, 5)
sorted_ranking = ranking_df.dropna(subset=['final_score']).sort_values('final_score', ascending=False).copy()
sorted_ranking['cumsum'] = sorted_ranking['final_score'].cumsum()
sorted_ranking['cumsum_pct'] = (sorted_ranking['cumsum'] / sorted_ranking['final_score'].sum()) * 100
ax5.plot(range(len(sorted_ranking)), sorted_ranking['cumsum_pct'].values, marker='o', linewidth=2, markersize=4, color='#2c3e50')
ax5.axhline(80, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
ax5.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90% Threshold')
ax5.set_xlabel('Number of Features')
ax5.set_ylabel('Cumulative Importance %')
ax5.set_title('Cumulative Feature Importance', fontweight='bold', fontsize=12)
ax5.grid(alpha=0.3)
ax5.legend()

# 6. Summary Statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
FEATURE SELECTION SUMMARY

Dataset Information:
  • Total Samples: 16,357
  • Initial Features: 100
  • Features After Filtering: 69
  • Features Dropped: 31

Results:
  • Strong Features: 15 (15%)
  • Medium Features: 45 (45%)
  • Weak Features: 40 (40%)

Model Performance:
  • CV Accuracy: 71.30%
  • CV Std Dev: 1.16%
  • Execution Time: 440.9 sec

Statistics:
  • Mean Score: {scores.mean():.4f}
  • Median Score: {np.median(scores):.4f}
  • Max Score: {scores.max():.4f}
  • Min Score: {scores.min():.4f}
"""
ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('feature_selection_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved: feature_selection_analysis.png")

# Create another figure for feature type analysis
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Analyze feature types
strong_features_list = strong_df['feature'].values
medium_features_list = medium_df['feature'].values
weak_features_list = weak_df['feature'].values

# Extract feature type categories
def categorize_feature(name):
    if 'autocorrelation' in name:
        return 'Autocorrelation'
    elif 'time_reversal' in name:
        return 'Time Reversal Asymmetry'
    elif 'symmetry' in name:
        return 'Symmetry'
    elif 'large_standard_deviation' in name:
        return 'Large Std Dev'
    elif 'location' in name:
        return 'Location (Peak/Valley)'
    elif 'derivative' in name:
        return 'Derivative'
    elif 'change' in name:
        return 'Change Metrics'
    elif 'correlation' in name or 'benford' in name or 'cid' in name:
        return 'Correlation Based'
    elif 'quantile' in name:
        return 'Quantiles'
    elif any(x in name for x in ['mean', 'std', 'median', 'variance', 'skewness', 'kurtosis']):
        return 'Distribution Stats'
    else:
        return 'Other'

# Count feature types in each category
strong_types = pd.Series([categorize_feature(f) for f in strong_features_list]).value_counts()
medium_types = pd.Series([categorize_feature(f) for f in medium_features_list]).value_counts()
weak_types = pd.Series([categorize_feature(f) for f in weak_features_list]).value_counts()

# Plot feature types
ax = axes[0, 0]
strong_types.plot(kind='barh', ax=ax, color='#2ecc71')
ax.set_title('Strong Features by Type', fontweight='bold')
ax.set_xlabel('Count')

ax = axes[0, 1]
medium_types.plot(kind='barh', ax=ax, color='#3498db')
ax.set_title('Medium Features by Type', fontweight='bold')
ax.set_xlabel('Count')

ax = axes[1, 0]
weak_types.plot(kind='barh', ax=ax, color='#e74c3c')
ax.set_title('Weak Features by Type', fontweight='bold')
ax.set_xlabel('Count')

# Feature type distribution across all categories
ax = axes[1, 1]
all_features = list(strong_features_list) + list(medium_features_list) + list(weak_features_list)
all_types = pd.Series([categorize_feature(f) for f in all_features])
type_colors = {'Strong': '#2ecc71', 'Medium': '#3498db', 'Weak': '#e74c3c'}

type_category_counts = {}
for ftype in all_types.unique():
    strong_count = sum(1 for f in strong_features_list if categorize_feature(f) == ftype)
    medium_count = sum(1 for f in medium_features_list if categorize_feature(f) == ftype)
    weak_count = sum(1 for f in weak_features_list if categorize_feature(f) == ftype)
    type_category_counts[ftype] = {'Strong': strong_count, 'Medium': medium_count, 'Weak': weak_count}

types_list = list(type_category_counts.keys())
x = np.arange(len(types_list))
width = 0.25

strong_counts = [type_category_counts[t]['Strong'] for t in types_list]
medium_counts = [type_category_counts[t]['Medium'] for t in types_list]
weak_counts = [type_category_counts[t]['Weak'] for t in types_list]

ax.bar(x - width, strong_counts, width, label='Strong', color='#2ecc71')
ax.bar(x, medium_counts, width, label='Medium', color='#3498db')
ax.bar(x + width, weak_counts, width, label='Weak', color='#e74c3c')

ax.set_xlabel('Feature Type')
ax.set_ylabel('Count')
ax.set_title('Feature Type Distribution by Category', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(types_list, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_type_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved: feature_type_analysis.png")

print("\nVisualization complete!")
