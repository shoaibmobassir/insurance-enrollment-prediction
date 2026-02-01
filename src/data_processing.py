"""
Data Processing Module for Insurance Enrollment Prediction
Handles data loading, cleaning, feature engineering, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Comprehensive data processing pipeline for insurance enrollment prediction
    """
    
    def __init__(self, data_path):
        """
        Initialize the data processor
        
        Args:
            data_path (str): Path to the CSV file containing employee data
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self):
        """Load data from CSV file"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        Returns a dictionary with key insights
        """
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        insights = {}
        
        # Basic info
        print("\n1. Dataset Overview:")
        print(f"   - Total records: {len(self.df)}")
        print(f"   - Total features: {len(self.df.columns)}")
        print(f"   - Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Data types
        print("\n2. Data Types:")
        print(self.df.dtypes)
        
        # Missing values
        print("\n3. Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   ✓ No missing values found")
            insights['missing_values'] = 0
        else:
            print(missing[missing > 0])
            insights['missing_values'] = missing.sum()
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\n4. Duplicate Rows: {duplicates}")
        insights['duplicates'] = duplicates
        
        # Target variable distribution
        print("\n5. Target Variable Distribution (enrolled):")
        enrollment_dist = self.df['enrolled'].value_counts()
        enrollment_pct = self.df['enrolled'].value_counts(normalize=True) * 100
        print(f"   - Not Enrolled (0): {enrollment_dist[0]} ({enrollment_pct[0]:.2f}%)")
        print(f"   - Enrolled (1): {enrollment_dist[1]} ({enrollment_pct[1]:.2f}%)")
        insights['enrollment_rate'] = enrollment_pct[1]
        insights['class_balance'] = enrollment_dist.to_dict()
        
        # Numerical features statistics
        print("\n6. Numerical Features Statistics:")
        numerical_cols = ['age', 'salary', 'tenure_years']
        print(self.df[numerical_cols].describe())
        
        # Categorical features
        print("\n7. Categorical Features Distribution:")
        categorical_cols = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']
        for col in categorical_cols:
            print(f"\n   {col}:")
            print(self.df[col].value_counts())
        
        insights['numerical_stats'] = self.df[numerical_cols].describe().to_dict()
        insights['categorical_counts'] = {col: self.df[col].value_counts().to_dict() 
                                         for col in categorical_cols}
        
        return insights
    
    def visualize_data(self, save_path='notebooks/'):
        """
        Create visualizations for EDA
        
        Args:
            save_path (str): Directory to save plots
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # 1. Target distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        sns.countplot(data=self.df, x='enrolled', ax=axes[0], palette='viridis')
        axes[0].set_title('Enrollment Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Enrolled')
        axes[0].set_ylabel('Count')
        
        # Pie chart
        enrollment_counts = self.df['enrolled'].value_counts()
        axes[1].pie(enrollment_counts, labels=['Not Enrolled', 'Enrolled'], 
                   autopct='%1.1f%%', startangle=90, colors=['#FF6B6B', '#4ECDC4'])
        axes[1].set_title('Enrollment Rate', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}enrollment_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: enrollment_distribution.png")
        plt.close()
        
        # 2. Numerical features distribution
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        numerical_cols = ['age', 'salary', 'tenure_years']
        for idx, col in enumerate(numerical_cols):
            sns.histplot(data=self.df, x=col, hue='enrolled', kde=True, ax=axes[idx], palette='viridis')
            axes[idx].set_title(f'{col.replace("_", " ").title()} Distribution by Enrollment', 
                               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}numerical_distributions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: numerical_distributions.png")
        plt.close()
        
        # 3. Categorical features vs enrollment
        categorical_cols = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            pd.crosstab(self.df[col], self.df['enrolled'], normalize='index').plot(
                kind='bar', ax=axes[idx], color=['#FF6B6B', '#4ECDC4'], rot=45
            )
            axes[idx].set_title(f'Enrollment Rate by {col.replace("_", " ").title()}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col.replace("_", " ").title())
            axes[idx].set_ylabel('Proportion')
            axes[idx].legend(['Not Enrolled', 'Enrolled'])
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(f'{save_path}categorical_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: categorical_analysis.png")
        plt.close()
        
        # 4. Correlation heatmap
        # Create a copy with encoded categorical variables for correlation
        df_encoded = self.df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
        
        plt.figure(figsize=(12, 8))
        correlation_matrix = df_encoded.drop('employee_id', axis=1).corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: correlation_heatmap.png")
        plt.close()
        
        print("\n✓ All visualizations generated successfully!")
    
    def engineer_features(self):
        """
        Perform feature engineering
        Creates new features and transforms existing ones
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        # Create a copy for processing
        self.df_processed = self.df.copy()
        
        # 1. Age groups
        self.df_processed['age_group'] = pd.cut(
            self.df_processed['age'], 
            bins=[0, 30, 40, 50, 100], 
            labels=['Young', 'Middle', 'Senior', 'Veteran']
        )
        print("✓ Created age_group feature")
        
        # 2. Salary bins
        self.df_processed['salary_bin'] = pd.qcut(
            self.df_processed['salary'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        print("✓ Created salary_bin feature")
        
        # 3. Tenure categories
        self.df_processed['tenure_category'] = pd.cut(
            self.df_processed['tenure_years'],
            bins=[-1, 2, 5, 10, 100],
            labels=['New', 'Intermediate', 'Experienced', 'Veteran']
        )
        print("✓ Created tenure_category feature")
        
        # 4. Binary encoding for has_dependents
        self.df_processed['has_dependents_binary'] = (
            self.df_processed['has_dependents'] == 'Yes'
        ).astype(int)
        print("✓ Encoded has_dependents as binary")
        
        # 5. Interaction features
        self.df_processed['salary_per_tenure'] = (
            self.df_processed['salary'] / (self.df_processed['tenure_years'] + 1)
        )
        print("✓ Created salary_per_tenure interaction feature")
        
        print(f"\n✓ Feature engineering complete. New shape: {self.df_processed.shape}")
        
        return self.df_processed
    
    def preprocess_data(self, test_size=0.2, random_state=42, encoding_method='label'):
        """
        Preprocess data for model training
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            encoding_method (str): 'label' or 'ohe' for One-Hot Encoding
        """
        print("\n" + "="*80)
        print("DATA PREPROCESSING")
        print("="*80)
        
        if self.df_processed is None:
            self.engineer_features()
        
        # Drop employee_id as it's not a feature
        df_model = self.df_processed.drop('employee_id', axis=1)
        
        # Separate features and target
        X = df_model.drop('enrolled', axis=1)
        y = df_model['enrolled']
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        print(f"\nEncoding {len(categorical_cols)} categorical features using {encoding_method} encoding...")
        
        if encoding_method == 'ohe':
            # Use One-Hot Encoding
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            # Ensure boolean columns are converted to int 
            # (pd.get_dummies might return bool in newer pandas versions)
            X = X.astype(int) if X.select_dtypes(include='bool').shape[1] > 0 else X
            print(f"  ✓ One-Hot Encoding applied. Feature count: {X.shape[1]}")
            
            # Note: For OHE, we don't save LabelEncoders, but we should save feature names later
            self.label_encoders = {} 
            
        else:
            # Default: Label Encoding
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
                print(f"  ✓ Encoded: {col}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data with stratification
        print(f"\nSplitting data (test_size={test_size})...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"  ✓ Training set: {self.X_train.shape[0]} samples")
        print(f"  ✓ Test set: {self.X_test.shape[0]} samples")
        
        # Scale numerical features
        numerical_cols = ['age', 'salary', 'tenure_years', 'salary_per_tenure']
        
        print(f"\nScaling numerical features...")
        self.X_train[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        print(f"  ✓ Scaled {len(numerical_cols)} numerical features")
        
        print("\n✓ Data preprocessing complete!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_processed_data(self):
        """Return processed data for model training"""
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
    
    def save_processed_data(self, output_dir='data/'):
        """Save processed data for reproducibility"""
        import pickle
        
        print(f"\nSaving processed data to {output_dir}...")
        
        # Save train/test splits
        self.X_train.to_csv(f'{output_dir}X_train.csv', index=False)
        self.X_test.to_csv(f'{output_dir}X_test.csv', index=False)
        self.y_train.to_csv(f'{output_dir}y_train.csv', index=False)
        self.y_test.to_csv(f'{output_dir}y_test.csv', index=False)
        
        # Save scaler and encoders
        with open(f'{output_dir}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{output_dir}label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print("✓ Processed data saved successfully!")


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor('data/employee_data.csv')
    
    # Load and explore data
    processor.load_data()
    insights = processor.explore_data()
    
    # Generate visualizations
    processor.visualize_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = processor.preprocess_data()
    
    # Save processed data
    processor.save_processed_data()
    
    print("\n" + "="*80)
    print("DATA PROCESSING COMPLETE!")
    print("="*80)
