"""
Model Evaluation Module for Insurance Enrollment Prediction
Provides comprehensive evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve,
                            average_precision_score)
import pickle
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations
    """
    
    def __init__(self, model, X_test, y_test, model_name='Model'):
        """
        Initialize the evaluator
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name (str): Name of the model
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    def plot_confusion_matrix(self, save_path='notebooks/'):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Enrolled', 'Enrolled'],
                   yticklabels=['Not Enrolled', 'Enrolled'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy text
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        plt.text(1, -0.3, f'Accuracy: {accuracy:.2%}', 
                ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename = f'{save_path}confusion_matrix_{self.model_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def plot_roc_curve(self, save_path='notebooks/'):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        filename = f'{save_path}roc_curve_{self.model_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, save_path='notebooks/'):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        filename = f'{save_path}pr_curve_{self.model_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
        
        return avg_precision
    
    def plot_feature_importance(self, feature_names, save_path='notebooks/', top_n=15):
        """
        Plot feature importance for tree-based models
        
        Args:
            feature_names (list): List of feature names
            save_path (str): Directory to save plot
            top_n (int): Number of top features to display
        """
        if not hasattr(self.model, 'feature_importances_'):
            print(f"Feature importance not available for {self.model_name}")
            return None
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(top_n)
        
        sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        filename = f'{save_path}feature_importance_{self.model_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
        
        return feature_importance_df
    
    def plot_prediction_distribution(self, save_path='notebooks/'):
        """Plot distribution of prediction probabilities"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram by true label
        enrolled_probs = self.y_pred_proba[self.y_test == 1]
        not_enrolled_probs = self.y_pred_proba[self.y_test == 0]
        
        axes[0].hist(not_enrolled_probs, bins=30, alpha=0.7, label='Not Enrolled (True)', 
                    color='#FF6B6B', edgecolor='black')
        axes[0].hist(enrolled_probs, bins=30, alpha=0.7, label='Enrolled (True)', 
                    color='#4ECDC4', edgecolor='black')
        axes[0].set_xlabel('Predicted Probability', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Prediction Probability Distribution by True Label', 
                         fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        data_for_box = pd.DataFrame({
            'Probability': self.y_pred_proba,
            'True Label': ['Enrolled' if y == 1 else 'Not Enrolled' for y in self.y_test]
        })
        
        sns.boxplot(data=data_for_box, x='True Label', y='Probability', 
                   palette=['#FF6B6B', '#4ECDC4'], ax=axes[1])
        axes[1].set_title('Prediction Probability by True Label', 
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Predicted Probability', fontsize=12)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f'{save_path}prediction_distribution_{self.model_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def generate_classification_report(self):
        """Generate and print classification report"""
        print("\n" + "="*80)
        print(f"CLASSIFICATION REPORT - {self.model_name}")
        print("="*80)
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=['Not Enrolled', 'Enrolled']))
    
    def evaluate_all(self, feature_names=None, save_path='notebooks/'):
        """
        Run all evaluation methods
        
        Args:
            feature_names (list): List of feature names for feature importance
            save_path (str): Directory to save plots
        """
        print("\n" + "="*80)
        print(f"COMPREHENSIVE EVALUATION - {self.model_name}")
        print("="*80)
        
        # Classification report
        self.generate_classification_report()
        
        # Visualizations
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix(save_path)
        roc_auc = self.plot_roc_curve(save_path)
        avg_precision = self.plot_precision_recall_curve(save_path)
        self.plot_prediction_distribution(save_path)
        
        if feature_names is not None:
            self.plot_feature_importance(feature_names, save_path)
        
        print(f"\n✓ Evaluation complete!")
        print(f"  - ROC AUC: {roc_auc:.4f}")
        print(f"  - Average Precision: {avg_precision:.4f}")
        
        return {
            'roc_auc': roc_auc,
            'average_precision': avg_precision
        }


def compare_multiple_models(models_dict, X_test, y_test, save_path='notebooks/'):
    """
    Compare multiple models with ROC curves on the same plot
    
    Args:
        models_dict (dict): Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test labels
        save_path (str): Directory to save plot
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['darkorange', 'green', 'red', 'purple', 'brown', 'pink']
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = f'{save_path}roc_comparison_all_models.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Loading model and test data...")
    
    # Load model
    with open('models/xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    # Evaluate
    evaluator = ModelEvaluator(model, X_test, y_test, model_name='XGBoost')
    evaluator.evaluate_all(feature_names=X_test.columns.tolist())
    
    print("\n" + "="*80)
    print("MODEL EVALUATION COMPLETE!")
    print("="*80)
