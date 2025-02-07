import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    """
    Load the trained machine learning model.
    
    Args:
        model_path (str): Path to the saved model
    
    Returns:
        Trained model pipeline
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return None

def predict_churn(model, customer_data):
    """
    Predict customer churn probability.
    
    Args:
        model: Trained model pipeline
        customer_data (pd.DataFrame): Customer features
    
    Returns:
        np.ndarray: Churn predictions and probabilities
    """
    if model is None:
        return None
    
    # Predict probabilities
    churn_proba = model.predict_proba(customer_data)
    churn_prediction = model.predict(customer_data)
    
    return churn_prediction, churn_proba

def create_sample_customers():
    """
    Create sample customer data for prediction.
    
    Returns:
        pd.DataFrame: Sample customer features
    """
    sample_customers = pd.DataFrame({
        'tenure': [12, 24, 36, 6],
        'MonthlyCharges': [50, 80, 100, 90],
        'TotalCharges': [600, 1920, 3600, 540],
        'gender': ['Male', 'Female', 'Male', 'Female'],
        'Partner': ['Yes', 'No', 'Yes', 'No'],
        'Dependents': ['No', 'Yes', 'No', 'Yes'],
        'PhoneService': ['Yes', 'No', 'Yes', 'No'],
        'MultipleLines': ['Yes', 'No phone service', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service', 'No']
    })
    
    return sample_customers

def main():
    # Load the trained model
    model = load_model('best_churn_model.joblib')
    
    if model is not None:
        # Create sample customer data
        sample_customers = create_sample_customers()
        
        # Predict churn
        churn_prediction, churn_proba = predict_churn(model, sample_customers)
        
        # Display results
        results = pd.DataFrame({
            'Customer': range(1, len(sample_customers) + 1),
            'Churn Prediction': churn_prediction,
            'Churn Probability': churn_proba[:, 1]
        })
        
        print("Churn Predictions:")
        print(results)
        
        # Optional: Save predictions
        results.to_csv('churn_predictions.csv', index=False)
        print("\nPredictions saved to 'churn_predictions.csv'")

if __name__ == '__main__':
    main()
