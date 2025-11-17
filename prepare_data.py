"""
Prepare data for feature selection by combining F_top100.csv features with XAUUSD historical data
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load features
logger.info("Loading features from F_top100.csv...")
features_df = pd.read_csv('F_top100.csv')
logger.info(f"Features loaded: {features_df.shape[0]} rows, {features_df.shape[1]} columns")

# Load historical data
logger.info("Loading historical data from XAUUSD_M15_R.csv...")
price_df = pd.read_csv('XAUUSD_M15_R.csv', sep='\t')
logger.info(f"Price data loaded: {price_df.shape[0]} rows, {price_df.shape[1]} columns")

# Align the dataframes
# We need to align features and prices
# Features has 16,359 rows, prices has 34,091 rows
# We'll use the first 16,359 rows from prices
min_rows = min(features_df.shape[0], price_df.shape[0])
logger.info(f"Aligning to {min_rows} rows (minimum of both datasets)")

features_aligned = features_df.iloc[:min_rows].reset_index(drop=True)
prices_aligned = price_df.iloc[:min_rows].reset_index(drop=True)

# Create target variable: 1 if close price goes up in next period, 0 otherwise
# We'll use the CLOSE price to create a binary classification target
logger.info("Creating target variable...")

# Shift close price to get next period's closing price
close_prices = prices_aligned['<CLOSE>'].values
target = np.zeros(len(close_prices), dtype=int)

# For each row, check if the next close price is higher than current close
for i in range(len(close_prices) - 1):
    if close_prices[i + 1] > close_prices[i]:
        target[i] = 1
    else:
        target[i] = 0

# The last row won't have a target, so we'll drop it
features_aligned = features_aligned.iloc[:-1].reset_index(drop=True)
target = target[:-1]

logger.info(f"Target created: {target.sum()} positive examples, {len(target) - target.sum()} negative examples")
logger.info(f"Target ratio: {target.mean():.2%} positive")

# Combine features with target
combined_df = features_aligned.copy()
combined_df['target'] = target

# Save combined dataset
output_file = 'F_combined.parquet'
logger.info(f"Saving combined dataset to {output_file}...")
combined_df.to_parquet(output_file, compression='snappy', index=False)
logger.info(f"Combined dataset saved: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")

# Also save as CSV for reference
csv_output = 'F_combined.csv'
logger.info(f"Saving combined dataset to {csv_output}...")
combined_df.to_csv(csv_output, index=False)
logger.info(f"CSV file saved: {csv_output}")

logger.info("Data preparation complete!")
