import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

@st.cache_data
def prepare_data(df, target_col, id_cols=None):
    """Prepare data for modeling"""
    try:
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Remove ID columns if specified
        if id_cols:
            df = df.drop(id_cols, axis=1)
        
        # Handle missing values in features first
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # For numeric columns, fill NaN with median
        for col in numeric_cols:
            if col != target_col:  # Skip target column
                if df[col].isna().any():
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
                    st.warning(f"Filled {df[col].isna().sum()} missing values in '{col}' with median value: {median_value:.2f}")
        
        # For categorical columns, fill NaN with mode
        for col in categorical_cols:
            if col != target_col:  # Skip target column
                if df[col].isna().any():
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
                    st.warning(f"Filled {df[col].isna().sum()} missing values in '{col}' with most frequent value: {mode_value}")
        
        # Handle target column separately
        if df[target_col].isna().any():
            nan_count = df[target_col].isna().sum()
            st.warning(f"Found {nan_count} missing values in target column '{target_col}'. These rows will be removed.")
            df = df.dropna(subset=[target_col])
        
        # Convert target to binary (0/1)
        target_mapping = {
            'yes': 1, 'no': 0,
            'true': 1, 'false': 0,
            '1': 1, '0': 0,
            1: 1, 0: 0,
            'y': 1, 'n': 0,
            True: 1, False: 0
        }
        
        # Convert target to string and lowercase for mapping
        df[target_col] = df[target_col].astype(str).str.lower().str.strip()
        
        # Map target values and handle any unmapped values
        y = df[target_col].map(target_mapping)
        
        # Check if any values weren't mapped properly
        unmapped = y.isna()
        if unmapped.any():
            unmapped_values = df.loc[unmapped, target_col].unique()
            error_msg = f"""
            Found invalid values in target column '{target_col}': {unmapped_values}
            
            Valid values are:
            - Yes/No, Y/N
            - True/False
            - 1/0
            
            Please clean your data and try again.
            """
            st.error(error_msg)
            raise ValueError(f"Invalid target values found: {unmapped_values}")
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        
        # Apply label encoding to remaining categorical columns
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        return X, y, label_encoders
    
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        st.info("""
        Data preparation failed. Please check:
        1. Your target column contains valid values (Yes/No, Y/N, 1/0, True/False)
        2. All feature columns have appropriate data types
        3. If there are too many missing values, consider cleaning your data first
        
        Example of valid target values:
        - Yes, No
        - Y, N
        - 1, 0
        - True, False
        """)
        raise e

def train_model(X, y, test_size=0.2, random_state=42):
    """Train the model and return results"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'report': report,
        'conf_matrix': conf_matrix,
        'feature_importance': pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    }

def plot_confusion_matrix(conf_matrix):
    """Plot confusion matrix using plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    
    return fig

def main():
    st.title("Customer Churn Prediction Dashboard")
    
    # File upload
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your customer data (CSV)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display data info
            st.subheader("Dataset Information")
            
            # Display missing values information
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            
            if not missing_values.empty:
                st.warning("Missing Values Found:")
                missing_df = pd.DataFrame({
                    'Column': missing_values.index,
                    'Missing Values': missing_values.values,
                    'Percentage': (missing_values.values / len(df) * 100).round(2)
                })
                st.write(missing_df)
            
            # Display raw data
            st.subheader("Data Preview")
            st.write(df.head())
            
            # Data preparation settings
            st.sidebar.header("Model Settings")
            
            # Select target column
            target_col = st.sidebar.selectbox(
                "Select Target Column (Churn)",
                df.columns
            )
            
            # Select ID columns to exclude
            id_cols = st.sidebar.multiselect(
                "Select ID Columns to Exclude",
                df.columns
            )
            
            # Model parameters
            test_size = st.sidebar.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.1
            )
            
            if st.sidebar.button("Train Model"):
                # Store the column information in session state
                st.session_state['target_col'] = target_col
                st.session_state['id_cols'] = id_cols
                
                # Prepare data
                X, y, label_encoders = prepare_data(df, target_col, id_cols)
                
                # Store the encoders and feature names in session state
                st.session_state['label_encoders'] = label_encoders
                st.session_state['feature_names'] = X.columns.tolist()
                
                # Train model and get results
                results = train_model(X, y, test_size=test_size)
                
                # Store the model and scaler in session state
                st.session_state['model'] = results['model']
                st.session_state['scaler'] = results['scaler']
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Model Performance")
                    
                    # Display metrics
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision (Churn)', 'Recall (Churn)', 'F1-Score (Churn)'],
                        'Value': [
                            f"{results['accuracy']:.2%}",
                            f"{results['report'].get('1', results['report'].get(1, {})).get('precision', 0):.2%}",
                            f"{results['report'].get('1', results['report'].get(1, {})).get('recall', 0):.2%}",
                            f"{results['report'].get('1', results['report'].get(1, {})).get('f1-score', 0):.2%}"
                        ]
                    })
                    st.table(metrics_df)
                    
                    # Plot confusion matrix
                    st.plotly_chart(plot_confusion_matrix(results['conf_matrix']))
                
                with col2:
                    st.subheader("Feature Importance")
                    
                    # Plot feature importance
                    fig = px.bar(
                        results['feature_importance'].head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Most Important Features'
                    )
                    st.plotly_chart(fig)
            
            # Only show prediction interface if model has been trained
            if 'model' in st.session_state:
                st.subheader("Make Predictions")
                
                # Create two columns for input fields
                col1, col2 = st.columns(2)
                input_data = {}
                
                # Get feature names and their types from the original dataframe
                feature_names = st.session_state['feature_names']
                
                # Create input fields for each feature
                for i, feature in enumerate(feature_names):
                    with col1 if i % 2 == 0 else col2:
                        if feature in st.session_state['label_encoders']:
                            # For categorical features, show original values
                            le = st.session_state['label_encoders'][feature]
                            original_values = le.classes_
                            value = st.selectbox(
                                f"Select {feature}",
                                options=original_values,
                                key=f"input_{feature}"
                            )
                            input_data[feature] = le.transform([value])[0]
                        else:
                            # For numerical features, show a number input
                            # Get the original column from the dataframe for min/max values
                            col_data = df[feature]
                            min_val = float(col_data.min())
                            max_val = float(col_data.max())
                            mean_val = float(col_data.mean())
                            
                            value = st.number_input(
                                f"Enter {feature}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                key=f"input_{feature}"
                            )
                            input_data[feature] = value
                
                if st.button("Predict", key="predict_button"):
                    try:
                        # Prepare input data
                        input_df = pd.DataFrame([input_data])
                        
                        # Ensure columns are in the same order as during training
                        input_df = input_df[st.session_state['feature_names']]
                        
                        # Scale input data
                        input_scaled = st.session_state['scaler'].transform(input_df)
                        
                        # Make prediction
                        prediction = st.session_state['model'].predict(input_scaled)[0]
                        probability = st.session_state['model'].predict_proba(input_scaled)[0][1]
                        
                        # Display prediction
                        st.write("---")
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Churn Prediction",
                                "Yes" if prediction == 1 else "No"
                            )
                        with col2:
                            st.metric(
                                "Churn Probability",
                                f"{probability:.2%}"
                            )
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.info("""
                        Prediction failed. Please check:
                        1. All input values are within expected ranges
                        2. All required fields are filled
                        3. The input data matches the format of the training data
                        """)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("""
            Please ensure your data:
            1. Is in CSV format
            2. Contains a target column for churn prediction (Yes/No, Y/N, 1/0, True/False)
            3. Has both numerical and categorical features
            
            Example of valid target values:
            - Yes, No
            - Y, N
            - 1, 0
            - True, False
            """)
    
    else:
        st.info("""
        👋 Welcome to the Customer Churn Prediction Dashboard!
        
        To get started:
        1. Upload your CSV file using the sidebar
        2. Select the target column (churn indicator)
        3. Identify any ID columns to exclude
        4. Adjust model settings
        5. Click 'Train Model'
        
        Your CSV should contain:
        - A target column indicating churn (Yes/No, Y/N, 1/0, True/False)
        - Customer features (both numerical and categorical)
        - Optional ID columns
        
        Example format:
        | CustomerID | Age | Tenure | MonthlyCharges | Churn |
        |------------|-----|--------|----------------|-------|
        | 1         | 35  | 24     | 65.5          | No    |
        | 2         | 42  | 12     | 89.9          | Yes   |
        """)

if __name__ == "__main__":
    main()
