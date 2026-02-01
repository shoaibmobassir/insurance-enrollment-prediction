
import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import DataProcessor
from model_training import ModelTrainer

def run_experiment():
    print("="*80)
    print("COMPREHENSVE EXPERIMENT: ALL MODELS x ALL ENCODINGS")
    print("="*80)
    
    experiment_results = []
    models_to_tune = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']
    encodings = ['label', 'ohe']
    
    for encoding in encodings:
        print(f"\n\n>>> PROCESSING PIPELINE: ENCODING = {encoding.upper()}")
        print("-" * 60)
        
        # 1. Data Processing
        processor = DataProcessor('data/employee_data.csv')
        processor.load_data()
        X_train, X_test, y_train, y_test = processor.preprocess_data(encoding_method=encoding)
        
        # 2. Initialize Trainer
        trainer = ModelTrainer(
            X_train, X_test, y_train, y_test,
            feature_names=X_train.columns.tolist()
        )
        
        # 3. Train Base Models
        print("\nTraining Base Models...")
        trainer.train_all_models()
        
        # 4. Tune Models
        print(f"\nTuning Models: {models_to_tune}...")
        for model_name in models_to_tune:
            try:
                best_model, metrics = trainer.hyperparameter_tuning(model_name)
                
                # Save results
                result_entry = metrics.copy()
                result_entry['Model'] = model_name
                result_entry['Encoding'] = encoding
                result_entry['Tuned'] = True
                
                # Cleanup complex objects for CSV
                if 'best_params' in result_entry:
                    result_entry['best_params'] = str(result_entry['best_params'])
                if 'feature_importance' in result_entry:
                    del result_entry['feature_importance']
                
                experiment_results.append(result_entry)
                
                # Save Model
                save_name = f"{model_name.replace(' ', '_').lower()}_{encoding}_tuned"
                trainer.models[save_name] = best_model # Register parameter
                trainer.save_model(model_name=save_name, output_dir='models/')
                
            except Exception as e:
                print(f"Error tuning {model_name}: {e}")
                
        # Capture baseline results too
        for model_name, metrics in trainer.results.items():
            if 'Tuned' not in model_name: # Avoid duplicates if tuned overwrite
                result_entry = metrics.copy()
                result_entry['Model'] = model_name
                result_entry['Encoding'] = encoding
                result_entry['Tuned'] = False
                 # Cleanup
                if 'feature_importance' in result_entry:
                    del result_entry['feature_importance']
                experiment_results.append(result_entry)

    # 5. Compile Report
    print("\n" + "="*80)
    print("FINAL EXPERIMENT REPORT")
    print("="*80)
    
    df_results = pd.DataFrame(experiment_results)
    
    # Select columns
    cols = ['Model', 'Encoding', 'Tuned', 'accuracy', 'roc_auc', 'f1_score']
    df_final = df_results[cols].sort_values(['roc_auc', 'accuracy'], ascending=False)
    
    print(df_final.to_string(index=False))
    
    # Save to CSV
    df_final.to_csv('experiments/final_experiment_results.csv', index=False)
    print("\n✓ Results saved to experiments/final_experiment_results.csv")

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        import traceback
        traceback.print_exc()
