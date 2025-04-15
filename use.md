# Lung Cancer Detection Project

## Overview
This project implements a machine learning model to detect potential lung cancer cases based on patient data and symptoms. The model achieves ~96% accuracy using both original and synthetic datasets.

## Setup & Installation

### Prerequisites
- Python 3.12 or higher
- PIP Package Manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/crbsdndr/cancer_detection
cd lung_cancer_detection
```

2. **Create and activate virtual environment**
```bash
# Create virtual environment
python -m venv cancer_detection

# Activate virtual environment
# On Windows:
cancer_detection\Scripts\activate
# On Linux/Mac:
source cancer_detection/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## Using the Model

### Via Python Code
```python
import joblib
import pandas as pd

# Load the model from the correct path
model = joblib.load('models/lung_cancer_detector.joblib')  # Root models directory

# Prepare patient data
patient_data = pd.DataFrame({
    'age': [65],
    'smoking': [1],
    'yellow_fingers': [1],
    'anxiety': [1],
    'peer_pressure': [0],
    'chronic_disease': [1],
    'fatigue': [1],
    'allergy': [0],
    'wheezing': [1],
    'alcohol_consuming': [0],
    'coughing': [1],
    'shortness_of_breath': [1],
    'swallowing_difficulty': [0],
    'chest_pain': [1]
})

# Normalize age (required preprocessing)
min_age = 1
max_age = 100
patient_data['age'] = (patient_data['age'] - min_age) / (max_age - min_age)

# Get prediction
prediction = model.predict(patient_data)
result = "Positive" if prediction[0] == 1 else "Negative"
print(f"Lung Cancer Prediction: {result}")
```

### Via REST API
The model is also available through a FastAPI REST API in the `api/` directory. See `api/main.py` for implementation details.

### Input Features
- `age`: Patient's age (must be between 3-100 years)
- Binary features (must be 0 or 1):
  - `smoking`: Smoking history
  - `yellow_fingers`: Presence of yellow fingers
  - `anxiety`: Presence of anxiety
  - `peer_pressure`: Experience of peer pressure
  - `chronic_disease`: Presence of chronic disease
  - `fatigue`: Presence of fatigue
  - `allergy`: Presence of allergies
  - `wheezing`: Presence of wheezing
  - `alcohol_consuming`: Alcohol consumption
  - `coughing`: Presence of coughing
  - `shortness_of_breath`: Difficulty breathing
  - `swallowing_difficulty`: Difficulty swallowing
  - `chest_pain`: Presence of chest pain

### Model Output
- `0`: Negative (No lung cancer detected)
- `1`: Positive (Lung cancer detected)

## Project Structure
```
lung_cancer_detection/
├── api/                     # FastAPI REST API
│   ├── main.py             # API implementation
│   ├── datasets/           # API-specific datasets
│   └── models/             # API model copy
├── models/                 # Trained model
├── requirements.txt        # Python dependencies
└── README.md              # Documentation
```

## Important Notes
- Model accuracy: ~96%
- Age must be between 1 and 100 years
- All features except age must be binary (0 or 1)
- The model is trained on both original and synthetic data for improved accuracy

## License
Open Source
