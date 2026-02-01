"""
Main Execution Script for Insurance Enrollment Prediction Pipeline
Orchestrates data processing, model training, and evaluation
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import DataProcessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator, compare_multiple_models
import pandas as pd
import pickle


def run_full_pipeline(data_path='data/employee_data.csv', tune_hyperparameters=False, encoding='label'):
    """
    Run the complete ML pipeline
    
    Args:
        data_path (str): Path to the data file
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning
        encoding (str): Encoding method ('label' or 'ohe')
    """
    print("\n" + "="*80)
    print("INSURANCE ENROLLMENT PREDICTION PIPELINE")
    print(f"Mode: Full Pipeline | Encoding: {encoding}")
    print("="*80)
    
    # Step 1: Data Processing
    print("\n[STEP 1/4] DATA PROCESSING")
    print("-" * 80)
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.explore_data()
    processor.visualize_data()
    X_train, X_test, y_train, y_test = processor.preprocess_data(encoding_method=encoding)
    processor.save_processed_data()
    
    # Get processed data
    data_dict = processor.get_processed_data()
    
    # Step 2: Model Training
    print("\n[STEP 2/4] MODEL TRAINING")
    print("-" * 80)
    trainer = ModelTrainer(
        X_train, X_test, y_train, y_test,
        feature_names=data_dict['feature_names']
    )
    
    # Train all models
    models, results = trainer.train_all_models()
    
    # Optional: Hyperparameter tuning
    if tune_hyperparameters:
        print("\n[BONUS] HYPERPARAMETER TUNING")
        print("-" * 80)
        trainer.hyperparameter_tuning('XGBoost')
        trainer.hyperparameter_tuning('Random Forest')
    
    # Save best model
    trainer.save_model()
    
    # Step 3: Model Evaluation
    print("\n[STEP 3/4] MODEL EVALUATION")
    print("-" * 80)
    
    best_model = trainer.best_model
    best_model_name = trainer.best_model_name
    
    evaluator = ModelEvaluator(
        best_model, X_test, y_test, 
        model_name=best_model_name
    )
    evaluator.evaluate_all(feature_names=data_dict['feature_names'])
    
    # Compare all models
    print("\n[STEP 4/4] MODEL COMPARISON")
    print("-" * 80)
    compare_multiple_models(models, X_test, y_test)
    
    # Feature importance
    print("\n" + "="*80)
    print("TOP FEATURES")
    print("="*80)
    trainer.get_feature_importance(top_n=10)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ Model saved in: models/")
    print(f"✓ Visualizations saved in: notebooks/")
    print(f"✓ Processed data saved in: data/")
    
    return trainer, evaluator


def run_preprocessing_only(data_path='data/employee_data.csv', encoding='label'):
    """Run only data preprocessing"""
    print("\n" + "="*80)
    print("DATA PREPROCESSING ONLY")
    print(f"Encoding: {encoding}")
    print("="*80)
    
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.explore_data()
    processor.visualize_data()
    processor.preprocess_data(encoding_method=encoding)
    processor.save_processed_data()
    
    print("\n✓ Preprocessing complete!")


def run_training_only(tune=False):
    """Run only model training (requires preprocessed data)"""
    print("\n" + "="*80)
    print("MODEL TRAINING ONLY")
    print("="*80)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    # Train models
    trainer = ModelTrainer(
        X_train, X_test, y_train, y_test,
        feature_names=X_train.columns.tolist()
    )
    
    models, results = trainer.train_all_models()
    
    if tune:
        trainer.hyperparameter_tuning('XGBoost')
    
    trainer.save_model()
    trainer.get_feature_importance()
    
    print("\n✓ Training complete!")


def run_evaluation_only(model_path='models/xgboost.pkl'):
    """Run only model evaluation (requires trained model)"""
    print("\n" + "="*80)
    print("MODEL EVALUATION ONLY")
    print("="*80)
    
    # Load model
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    # Evaluate
    model_name = os.path.basename(model_path).replace('.pkl', '').replace('_', ' ').title()
    evaluator = ModelEvaluator(model, X_test, y_test, model_name=model_name)
    evaluator.evaluate_all(feature_names=X_test.columns.tolist())
    
    print("\n✓ Evaluation complete!")


def make_predictions(model_path='models/xgboost.pkl', input_csv=None):
    """
    Make predictions on new data
    
    Args:
        model_path (str): Path to saved model
        input_csv (str): Path to input CSV file
    """
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80)
    
    # Load model and preprocessors
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load label encoders if available
    label_encoders = {}
    if os.path.exists('data/label_encoders.pkl'):
        try:
            with open('data/label_encoders.pkl', 'rb') as f:
                label_encoders = pickle.load(f)
        except:
            print("Note: Could not load label encoders (might be OHE model)")
    
    # Load input data
    if input_csv is None:
        input_csv = 'data/X_test.csv'
    
    print(f"Loading input data from {input_csv}...")
    X = pd.read_csv(input_csv)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Prediction': predictions,
        'Probability': probabilities,
        'Enrollment_Status': ['Enrolled' if p == 1 else 'Not Enrolled' for p in predictions]
    })
    
    # Save predictions
    output_path = 'data/predictions.csv'
    results.to_csv(output_path, index=False)
    
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"\nSample predictions:")
    print(results.head(10))
    
    return results


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description='Insurance Enrollment Prediction ML Pipeline'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'preprocess', 'train', 'evaluate', 'predict'],
        default='full',
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/employee_data.csv',
        help='Path to input data CSV'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/xgboost.pkl',
        help='Path to model file (for evaluate/predict modes)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    
    parser.add_argument(
        '--encoding',
        type=str,
        choices=['label', 'ohe'],
        default='label',
        help='Encoding method for categorical variables (label or ohe)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV for predictions'
    )
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.mode == 'full':
        run_full_pipeline(args.data, args.tune, args.encoding)
    elif args.mode == 'preprocess':
        run_preprocessing_only(args.data, args.encoding)
    elif args.mode == 'train':
        run_training_only(args.tune)
    elif args.mode == 'evaluate':
        run_evaluation_only(args.model)
    elif args.mode == 'predict':
        make_predictions(args.model, args.input)


if __name__ == "__main__":
    # If no arguments provided, run full pipeline
    if len(sys.argv) == 1:
        run_full_pipeline(tune_hyperparameters=False)
    else:
        main()
