import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """
    Load customer churn dataset.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def preprocess_data(df):
    """
    Preprocess the customer churn dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        tuple: Preprocessed X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Define numeric and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                             'MultipleLines', 'InternetService', 'OnlineSecurity']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic customer churn dataset.
    
    Args:
        n_samples (int): Number of samples to generate
    
    Returns:
        pd.DataFrame: Synthetic customer churn dataset
    """
    np.random.seed(42)
    
    data = {
        'tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Synthetic churn logic
    df['Churn'] = ((df['tenure'] < 12) & (df['MonthlyCharges'] > 70) | 
                   (df['InternetService'] == 'Fiber optic')).astype(int)
    
    return df

def main():
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Save dataset
    df.to_csv('customer_churn_data.csv', index=False)
    print("Synthetic dataset created successfully!")

if __name__ == '__main__':
    main()
