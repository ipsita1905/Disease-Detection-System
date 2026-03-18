# AI-Based Early Disease Detection System
It is an AI-Based system that helps patients to identify their diseases according to their symptoms.

## 📋 Overview

This project implements an AI-powered early disease detection system that analyzes patient medical data to predict the risk of diabetes. The system uses machine learning algorithms to provide early warnings, helping healthcare professionals make timely interventions and improve patient outcomes.

## 🎯 Background

Healthcare systems often face delays in detecting diseases due to limited diagnostic resources and manual processes. AI can assist doctors by analyzing medical data to detect early signs of diseases, enabling preventive care and better treatment outcomes.

## ✨ Features

- **🔬 Machine Learning Model**: Random Forest classifier for disease prediction
- **📊 Dataset Preprocessing**: Automated data cleaning, scaling, and feature engineering
- **🎛️ Interactive Dashboard**: Streamlit-based web interface for real-time predictions
- **📈 Model Evaluation**: Comprehensive accuracy metrics (Accuracy, Precision, Recall, F1-Score)
- **💾 Data Persistence**: Save/load trained models and preprocessing scalers
- **🔧 Configurable Training**: YAML-based configuration for different environments

## 🏗️ Architecture

```
AI-Disease-Detection/
├── src/
│   ├── preprocessing.py    # Data preprocessing and feature engineering
│   ├── train.py           # Model training and evaluation
│   ├── evaluate.py        # Standalone model evaluation
│   └── app.py            # Streamlit web application
├── data/
│   ├── X_train.npy       # Preprocessed training features
│   ├── X_test.npy        # Preprocessed test features
│   ├── y_train.npy       # Training labels
│   └── y_test.npy        # Test labels
├── models/
│   ├── disease_model.pkl # Trained ML model
│   └── scaler.pkl       # Feature scaler
├── pdtrain.yaml          # ML training configuration
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🗃️ Dataset

The system uses medical predictor variables similar to the Pima Indians Diabetes Dataset:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1) - Target

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-disease-detection
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run data preprocessing**
   ```bash
   python src/preprocessing.py
   ```

5. **Train the model**
   ```bash
   python src/train.py
   ```

## 🎮 Usage

### Training Pipeline

1. **Data Preprocessing**
   ```bash
   python src/preprocessing.py
   ```
   - Generates sample medical dataset
   - Handles missing values and outliers
   - Scales features using StandardScaler
   - Saves processed data to `data/` directory

2. **Model Training**
   ```bash
   python src/train.py
   ```
   - Trains Random Forest classifier
   - Evaluates model performance
   - Saves trained model to `models/` directory

3. **Model Evaluation**
   ```bash
   python src/evaluate.py
   ```
   - Loads test data and trained model
   - Computes evaluation metrics
   - Displays classification report

### Web Application

Launch the interactive prediction dashboard:

```bash
streamlit run src/app.py
```

The dashboard will be available at `http://localhost:8501`

**Features:**
- Input patient medical parameters
- Real-time disease risk prediction
- Probability scores
- Model information display

## 📊 Model Performance

The Random Forest model achieves the following metrics on the test set:

- **Accuracy**: ~85-90%
- **Precision**: ~80-85%
- **Recall**: ~75-80%
- **F1-Score**: ~78-82%

*Note: Actual metrics may vary based on random seed and data split.*

## 🔧 Configuration

### ML Training Configuration (`pdtrain.yaml`)

```yaml
framework: sklearn
framework_version: "1.3.0"
python_version: py310
entry_point: train.py

# Hyperparameters
hyperparameters:
  n_estimators: 100
  random_state: 42

# Environment variables
env:
  MY_VAR: value
```

### Model Parameters

- **Algorithm**: Random Forest Classifier
- **n_estimators**: 100 trees
- **Random State**: 42 (for reproducibility)
- **Feature Scaling**: StandardScaler

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Pima Indians Diabetes Dataset for inspiration
- Scikit-learn for machine learning algorithms
- Streamlit for the web application framework
- Open source community for tools and libraries

## 📞 Support

For questions or support, please open an issue in the GitHub repository.

## 🔮 Future Enhancements

- [ ] Support for additional diseases
- [ ] Integration with medical imaging
- [ ] Real-time data streaming
- [ ] Multi-model ensemble predictions
- [ ] API endpoints for integration
- [ ] Advanced visualization dashboard
- [ ] Model explainability features

---

**⚠️ Disclaimer**: This is a demonstration project for educational purposes. Not intended for actual medical diagnosis. Always consult healthcare professionals for medical decisions.