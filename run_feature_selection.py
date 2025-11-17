"""
Run feature selection on the combined dataset
"""
import sys
import os
import logging
import pandas as pd
from FSX import FeatureSelector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_selection_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("Starting Feature Selection Analysis on F_combined data")
    logger.info("="*80)

    # Load combined data
    logger.info("Loading F_combined.parquet...")
    df = pd.read_parquet('F_combined.parquet')
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Check target column
    if 'target' not in df.columns:
        logger.error('No target column found in F_combined.parquet')
        return

    # Print dataset info
    logger.info(f"Features: {df.shape[1]-1}, Samples: {df.shape[0]}")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    logger.info(f"Target class balance: {df['target'].mean():.2%} positive class")

    # Initialize FeatureSelector
    logger.info("Initializing FeatureSelector...")
    fs = FeatureSelector(
        target_column='target',
        classification=True,
        random_state=42,
        ensure_reproducible=True,
        enable_metadata_routing=False,
        use_shap=True,
        shap_sample_size=1000,
        enable_dataset_cache=True,
        max_cache_size=32,
        n_estimators=500,
        n_jobs=-1
    )

    logger.info("FeatureSelector initialized successfully!")
    logger.info("="*80)
    logger.info("Starting Feature Selection Process...")
    logger.info("="*80)

    # Run feature selection
    output_dir = 'feature_selection_results'
    try:
        fs.process_batch(
            features_df=df,
            batch_id=0,
            output_dir=output_dir,
            save_final_model=True
        )
        logger.info("="*80)
        logger.info("Feature Selection Process Completed Successfully!")
        logger.info("="*80)
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error during feature selection: {str(e)}", exc_info=True)
        return False

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
