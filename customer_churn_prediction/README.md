# Customer Churn Prediction Project

## Overview
This project demonstrates a machine learning approach to predicting customer churn using synthetic data. It includes data preparation, model training, and prediction scripts.

## Project Structure
- `data_preparation.py`: Generate and preprocess synthetic customer data
- `model_training.py`: Train multiple classification models and compare performance
- `predict.py`: Load trained model and make churn predictions

## Key Features
- Synthetic data generation
- Multiple model training (Logistic Regression, Random Forest, Gradient Boosting)
- Model performance visualization
- Churn probability prediction

## Setup and Installation
1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Generate synthetic data
```bash
python data_preparation.py
```

2. Train models
```bash
python model_training.py
```

3. Make predictions
```bash
python predict.py
```

## Output Files
- `customer_churn_data.csv`: Synthetic dataset
- `best_churn_model.joblib`: Trained machine learning model
- `model_comparison.png`: Model performance visualization
- `churn_predictions.csv`: Predictions for sample customers

## Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Cross-validation scores

## Next Steps
- Collect real-world customer data
- Fine-tune hyperparameters
- Implement more advanced feature engineering
- Explore deep learning models
