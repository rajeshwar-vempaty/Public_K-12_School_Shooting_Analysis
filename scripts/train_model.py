#!/usr/bin/env python3
"""
Training Script
Complete pipeline to train the school shooting prediction model.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.features.encoding import FeatureEncoder
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.utils.logger import setup_logger
from src.utils.config import load_config

def main():
    """Execute full model training pipeline."""
    # Setup
    config = load_config()
    logger = setup_logger("train_model", console=True)

    logger.info("=" * 80)
    logger.info("SCHOOL SHOOTING INCIDENT PREDICTION MODEL TRAINING")
    logger.info("=" * 80)

    try:
        # Step 1: Load data
        logger.info("\n[1/8] Loading data...")
        loader = DataLoader()
        df_merged = loader.load_and_merge()

        # Step 2: Preprocess data
        logger.info("\n[2/8] Preprocessing data...")
        preprocessor = DataPreprocessor()
        # Load individual sheets for preprocessing
        data_sheets = loader.load_excel_file()
        shooter_clean = preprocessor.preprocess_shooter_data(data_sheets["shooter"])
        victim_clean = preprocessor.preprocess_victim_data(data_sheets["victim"])
        weapon_clean = preprocessor.preprocess_weapon_data(data_sheets["weapon"])
        df_processed = preprocessor.preprocess_merged_data(df_merged)

        # Step 3: Feature engineering
        logger.info("\n[3/8] Engineering features...")
        engineer = FeatureEngineer()
        df_features = engineer.full_feature_engineering_pipeline(df_processed)

        # Step 4: Split features and target
        logger.info("\n[4/8] Splitting features and target...")
        X, y = engineer.split_features_target(df_features)

        # Step 5: Split train/test
        logger.info("\n[5/8] Creating train/test split...")
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)

        # Step 6: Encode and scale
        logger.info("\n[6/8] Encoding and scaling features...")
        encoder = FeatureEncoder()
        X_train_encoded, X_train_scaled = encoder.encode_and_scale_train(X_train)
        X_test_encoded, X_test_scaled = encoder.encode_and_scale_test(X_test)

        # Step 7: Train model with hyperparameter tuning
        logger.info("\n[7/8] Training model with hyperparameter tuning...")
        model, best_params = trainer.hyperparameter_tuning(
            X_train_scaled, y_train, model_type="svm", method="grid"
        )

        # Step 8: Evaluate model
        logger.info("\n[8/8] Evaluating model...")
        evaluator = ModelEvaluator()
        results = evaluator.comprehensive_evaluation(
            model,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            feature_names=X_train.columns.tolist(),
            save_dir="data/evaluation_results",
        )

        # Save model
        logger.info("\nSaving model artifacts...")
        model_path = trainer.save_model(
            model=model,
            encoder_artifacts=encoder.get_encoder_artifacts(),
            feature_names=X_train.columns.tolist(),
        )

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nModel saved to: {model_path}")
        logger.info(f"\nTest Set Performance:")
        for metric, value in results["test_metrics"].items():
            logger.info(f"  {metric.upper()}: {value:.4f}")

    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
