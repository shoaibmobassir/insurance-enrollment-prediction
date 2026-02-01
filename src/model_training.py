"""
Model Training Module for Insurance Enrollment Prediction
Implements multiple ML algorithms with cross-validation and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Comprehensive model training pipeline with multiple algorithms
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Initialize the model trainer
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            feature_names: List of feature names
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_baseline_model(self):
        """Train baseline Logistic Regression model"""
        print("\n" + "="*80)
        print("TRAINING BASELINE MODEL: Logistic Regression")
        print("="*80)
        
        # Train model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = lr_model.predict(self.X_test)
        y_pred_proba = lr_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(lr_model, self.X_train, self.y_train, 
                                   cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Store results
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'] = metrics
        
        self._print_metrics('Logistic Regression', metrics)
        
        return lr_model, metrics
    
    def train_random_forest(self):
        """Train Random Forest Classifier"""
        print("\n" + "="*80)
        print("TRAINING MODEL: Random Forest")
        print("="*80)
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, 
                                   cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        metrics['feature_importance'] = dict(zip(
            self.feature_names, 
            rf_model.feature_importances_
        ))
        
        # Store results
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = metrics
        
        self._print_metrics('Random Forest', metrics)
        
        # Log to MLflow
        self._log_to_mlflow(rf_model, 'Random Forest', metrics, rf_model.get_params())
        
        return rf_model, metrics
    
    def train_xgboost(self):
        """Train XGBoost Classifier"""
        print("\n" + "="*80)
        print("TRAINING MODEL: XGBoost")
        print("="*80)
        
        # Train model
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = xgb_model.predict(self.X_test)
        y_pred_proba = xgb_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, 
                                   cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        metrics['feature_importance'] = dict(zip(
            self.feature_names, 
            xgb_model.feature_importances_
        ))
        
        # Store results
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = metrics
        
        self._print_metrics('XGBoost', metrics)
        
        # Log to MLflow
        self._log_to_mlflow(xgb_model, 'XGBoost', metrics, xgb_model.get_params())
        
        return xgb_model, metrics
    
    def train_lightgbm(self):
        """Train LightGBM Classifier"""
        print("\n" + "="*80)
        print("TRAINING MODEL: LightGBM")
        print("="*80)
        
        # Train model
        lgbm_model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgbm_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = lgbm_model.predict(self.X_test)
        y_pred_proba = lgbm_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(lgbm_model, self.X_train, self.y_train, 
                                   cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        metrics['feature_importance'] = dict(zip(
            self.feature_names, 
            lgbm_model.feature_importances_
        ))
        
        # Store results
        self.models['LightGBM'] = lgbm_model
        self.results['LightGBM'] = metrics
        
        self._print_metrics('LightGBM', metrics)
        
        return lgbm_model, metrics
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting Classifier"""
        print("\n" + "="*80)
        print("TRAINING MODEL: Gradient Boosting")
        print("="*80)
        
        # Train model
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = gb_model.predict(self.X_test)
        y_pred_proba = gb_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(gb_model, self.X_train, self.y_train, 
                                   cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        metrics['feature_importance'] = dict(zip(
            self.feature_names, 
            gb_model.feature_importances_
        ))
        
        # Store results
        self.models['Gradient Boosting'] = gb_model
        self.results['Gradient Boosting'] = metrics
        
        self._print_metrics('Gradient Boosting', metrics)
        
        return gb_model, metrics
    
    def train_all_models(self):
        """Train all models and compare performance"""
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        # Train each model
        self.train_baseline_model()
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_gradient_boosting()
        
        # Compare results
        self._compare_models()
        
        return self.models, self.results
    
    def hyperparameter_tuning(self, model_name='XGBoost'):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model_name (str): Name of the model to tune
        """
        print("\n" + "="*80)
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print("="*80)
        
        if model_name == 'XGBoost':
            model = XGBClassifier(random_state=42, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        elif model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'LightGBM':
            model = LGBMClassifier(random_state=42, verbose=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [15, 31, 63]
            }
        elif model_name == 'Logistic Regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear']
            }
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        print(f"Searching through {len(param_grid)} parameters...")
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Best parameters
        print(f"\n✓ Best Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  - {param}: {value}")
        
        print(f"\n✓ Best CV Score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_mean'] = grid_search.best_score_
        metrics['cv_std'] = 0.0  # GridSearchCV doesn't provide std, set to 0
        
        # Store tuned model
        tuned_name = f"{model_name} (Tuned)"
        self.models[tuned_name] = best_model
        self.results[tuned_name] = metrics
        
        self._print_metrics(tuned_name, metrics)
        
        return best_model, metrics
    
    def _calculate_metrics(self, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
    
    def _print_metrics(self, model_name, metrics):
        """Print model metrics in a formatted way"""
        print(f"\n{model_name} Performance:")
        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall:    {metrics['recall']:.4f}")
        print(f"  - F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  - ROC AUC:   {metrics['roc_auc']:.4f}")
        if 'cv_mean' in metrics:
            print(f"  - CV ROC AUC: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    def _compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics['roc_auc'],
                'CV ROC AUC': metrics.get('cv_mean', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
        
        print("\n", comparison_df.to_string(index=False))
        
        # Identify best model
        best_idx = comparison_df['ROC AUC'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n✓ Best Model: {self.best_model_name}")
        print(f"  ROC AUC: {comparison_df.loc[best_idx, 'ROC AUC']:.4f}")
        
        return comparison_df
    
    def save_model(self, model_name=None, output_dir='models/'):
        """
        Save trained model to disk
        
        Args:
            model_name (str): Name of model to save (default: best model)
            output_dir (str): Directory to save model
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        filename = f"{output_dir}{model_name.replace(' ', '_').lower()}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✓ Model saved: {filename}")
        
        # Save metrics
        metrics_filename = f"{output_dir}{model_name.replace(' ', '_').lower()}_metrics.pkl"
        with open(metrics_filename, 'wb') as f:
            pickle.dump(self.results[model_name], f)
        
        print(f"✓ Metrics saved: {metrics_filename}")
        
        return filename
    
    def get_feature_importance(self, model_name=None, top_n=10):
        """
        Get feature importance from tree-based models
        
        Args:
            model_name (str): Name of model (default: best model)
            top_n (int): Number of top features to return
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if 'feature_importance' not in self.results[model_name]:
            print(f"Feature importance not available for {model_name}")
            return None
        
        importance_dict = self.results[model_name]['feature_importance']
        importance_df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_n} Features for {model_name}:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df

    
    def _log_to_mlflow(self, model, model_name, metrics, params=None):
        """Log model run to MLflow"""
        try:
            # Set experiment
            mlflow.set_experiment("Insurance Enrollment Prediction")
            
            with mlflow.start_run(run_name=model_name, nested=True):
                # Log params (filter long params)
                if params:
                    filtered_params = {k: v for k, v in params.items() 
                                      if isinstance(v, (int, float, str)) and len(str(v)) < 100}
                    mlflow.log_params(filtered_params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Tag
                mlflow.set_tag("model_type", model_name)
                
                # Log model artifact (optional, can be large)
                # mlflow.sklearn.log_model(model, "model")
                
            print(f"  ✓ Logged {model_name} execution to MLflow")
        except Exception as e:
            print(f"  ! Note: MLflow logging skipped (optional)")

if __name__ == "__main__":
    # Example usage
    print("Loading processed data...")
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    # Initialize trainer
    trainer = ModelTrainer(X_train, X_test, y_train, y_test, 
                          feature_names=X_train.columns.tolist())
    
    # Train all models
    models, results = trainer.train_all_models()
    
    # Hyperparameter tuning for best model
    trainer.hyperparameter_tuning('XGBoost')
    
    # Save best model
    trainer.save_model()
    
    # Feature importance
    trainer.get_feature_importance()
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE!")
    print("="*80)
