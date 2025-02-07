import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from data_preparation import load_data, preprocess_data, generate_synthetic_data

def create_model_pipeline(model, preprocessor):
    """
    Create a machine learning pipeline.
    
    Args:
        model: Machine learning classifier
        preprocessor: Data preprocessor
    
    Returns:
        Pipeline: Preprocessing and model pipeline
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate multiple classification models.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
        preprocessor: Data preprocessor
    
    Returns:
        dict: Model performance metrics
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        pipeline = create_model_pipeline(model, preprocessor)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluation metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'cross_val_mean': cv_scores.mean(),
            'cross_val_std': cv_scores.std()
        }
        
        # Save the best model
        if name == 'Gradient Boosting':
            joblib.dump(pipeline, 'best_churn_model.joblib')
    
    return results

def plot_model_comparison(results):
    """
    Create a bar plot comparing model performances.
    
    Args:
        results (dict): Model performance metrics
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, list(results.keys()), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Print results
    for model, metrics in results.items():
        print(f"\n{model} Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    
    # Plot model comparison
    plot_model_comparison(results)
    
    print("\nBest model saved as 'best_churn_model.joblib'")
    print("Model comparison plot saved as 'model_comparison.png'")

if __name__ == '__main__':
    main()
