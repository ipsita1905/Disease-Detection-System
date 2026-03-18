import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import os

def load_data():
    # For demo, we'll create a sample dataset similar to Pima Indians Diabetes
    # In real scenario, load from CSV
    np.random.seed(42)
    n_samples = 1000
    
    # Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
        'BloodPressure': np.random.normal(70, 15, n_samples).clip(0, 122),
        'SkinThickness': np.random.normal(20, 15, n_samples).clip(0, 99),
        'Insulin': np.random.normal(80, 115, n_samples).clip(0, 846),
        'BMI': np.random.normal(32, 8, n_samples).clip(0, 67),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
        'Age': np.random.normal(33, 12, n_samples).clip(21, 81),
        'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    }
    
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    # Handle missing values (replace 0 with median for certain columns)
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_replace:
        df[col] = df[col].replace(0, df[col].median())
    
    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = sklearn.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Data preprocessing completed.")