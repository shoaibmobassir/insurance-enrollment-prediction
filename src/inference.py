
import argparse
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import DataProcessor
from src.model_training import ModelTrainer

def run_inference():
    parser = argparse.ArgumentParser(description="Generalized Inference Script")
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'load'], 
                       help="Mode: 'train' a new model or 'load' an existing one")
    
    # Model configuration
    parser.add_argument('--model', type=str, default='random_forest',
                       help="Model type (random_forest, xgboost, gradient_boosting, logistic_regression, lightgbm)")
    parser.add_argument('--encoding', type=str, default='label', choices=['label', 'ohe'],
                       help="Encoding method for categorical variables")
    
    # Inference inputs
    parser.add_argument('--model_path', type=str, help="Path to saved model .pkl file (required for mode='load')")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV for prediction")
    parser.add_argument('--output', type=str, default='predictions.csv', help="Path to save prediction results")
    
    # Hyperparameters (only for train mode)
    parser.add_argument('--params', type=str, default='{}', 
                       help="JSON string of hyperparameters (e.g. '{\"n_estimators\": 100, \"max_depth\": 5}')")
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"INFERENCE PIPELINE | Mode: {args.mode.upper()}")
    print("="*80)
    
    # 1. Load Data
    print(f"Loading input data: {args.input}...")
    try:
        df_input = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df_input)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Preprocessing
    # Note: In a real production scenario, we'd load the preprocessor state. 
    # Here we re-instantiate DataProcessor for simplicity, assuming input has raw features.
    print(f"Preprocessing data (Encoding: {args.encoding})...")
    processor = DataProcessor(args.input) # Initialize with input file
    
    if args.mode == 'train':
        # For training, we need the full pipeline logic (split etc)
        # But this script implies 'inference on new data' usually...
        # If the user wants to TRAIN on this data, we assume it has targets.
        processor.load_data()
        X_train, X_test, y_train, y_test = processor.preprocess_data(encoding_method=args.encoding)
        feature_names = X_train.columns.tolist()
        
    else:
        # Load mode: We just process the input df without splitting
        # We need to handle the case where we don't fit_transform but just transform
        # For simplicity in this demo, we'll re-use the preprocess logic which fits on the data
        # WARNING: In strict prod, you load the encoder. 
        # Here we assume the input data represents a batch large enough to encode, 
        # or we just re-run the processor.
        
        # We need to handle OHE alignment if columns mismatch.
        # This is a complex part of "Generalized Inference". 
        # We will assume the user provides a dataset compatible with training rules.
        
        # Hack: Pass dummy target if column missing?
        if 'enrolled' not in df_input.columns:
            df_input['enrolled'] = 0 # Dummy
            
        # Reprocess
        # We leverage the existing class but simplified
        processor.df = df_input
        X, y = processor.preprocess_data(encoding_method=args.encoding, test_size=0.0) # 0.0 returns full X/y
        # X is the processed dataframe
        X_inference = X
        y_inference = y # Not used if pure inference
        
    # 3. Model Handling
    model = None
    
    if args.mode == 'load':
        if not args.model_path:
            print("Error: --model_path is required for load mode")
            return
            
        print(f"Loading model from {args.model_path}...")
        try:
            with open(args.model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
            
    elif args.mode == 'train':
        print(f"Initializing {args.model}...")
        
        # Parse params
        try:
            custom_params = json.loads(args.params)
            if custom_params:
                print(f"Custom Hyperparameters: {custom_params}")
        except json.JSONDecodeError:
            print("Error parsing --params JSON. Using defaults.")
            custom_params = {}
            
        # Initialize Trainer
        trainer = ModelTrainer(X_train, X_test, y_train, y_test, feature_names)
        
        # Map string to trainer method or direct model init
        # We'll use the trainer methods for convenience but override params if possible
        # Since trainer methods hardcode params, we might need to instantate directly here for full flexibility
        # Let's instantiate directly for the "Generalized" requirement
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
        
        if args.model == 'random_forest':
            model = RandomForestClassifier(random_state=42, **custom_params)
        elif args.model == 'xgboost':
            model = XGBClassifier(random_state=42, **custom_params)
        elif args.model == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42, **custom_params)
        elif args.model == 'lightgbm':
            model = LGBMClassifier(random_state=42, **custom_params)
        elif args.model == 'logistic_regression':
            model = LogisticRegression(random_state=42, **custom_params)
        else:
            print(f"Unknown model: {args.model}")
            return
            
        print("Training model...")
        model.fit(X_train, y_train)
        print("✓ Training complete")
        
        # Prepare for prediction (predict on test set for validation output?)
        # The prompt implies running the inference script chooses model ... and runs that.
        # So we likely want to predict on the INPUT file.
        # But if input file was used for training, we are predicting on training data?
        # Let's assume input file is for Training AND we save the model.
        
        # Save model
        save_path = f"models/custom_{args.model}_{args.encoding}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved to {save_path}")
        
        # For consistency, set X_inference to X_test (or full X) to generate some output?
        # User request: "run that". Could mean run training OR run inference.
        # Let's predict on the X_test partition to show "results"
        X_inference = X_test

    # 4. Prediction
    print("Generating predictions...")
    if args.mode == 'train':
        # Predict on Test set
        probs = model.predict_proba(X_inference)[:, 1]
        preds = model.predict(X_inference)
        output_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds, 'Probability': probs})
    else:
        # Predict on Full Input (Load Mode)
        # Ensure X_inference columns match model features
        # This is tricky with OHE - need alignment.
        # Simple try/except for column mismatch
        try:
            probs = model.predict_proba(X_inference)[:, 1]
            preds = model.predict(X_inference)
            
            # Combine with original input if possible, or just IDs
            output_df = df_input.copy()
            output_df['Predicted_Enrollment'] = preds
            output_df['Probability'] = probs
        except ValueError as e:
            print(f"Error during prediction (likely feature mismatch): {e}")
            return

    # 5. Save Output
    output_df.to_csv(args.output, index=False)
    print(f"✓ Predictions saved to {args.output}")

if __name__ == "__main__":
    run_inference()
