# Lung Cancer Detection Project

## Overview
This project implements a machine learning model to detect potential lung cancer cases based on patient data and symptoms.

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone [repository-url]
cd lung_cancer_detection
```

2. **Create and activate virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## Using the Model

### Making Predictions
Here's how to use the trained model for predictions:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('./models/lung_cancer_detector.joblib')

# Prepare sample input data
sample_data = pd.DataFrame({
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

# Scale age (required preprocessing)
min_age = 21
max_age = 80
sample_data['age'] = (sample_data['age'] - min_age) / (max_age - min_age)

# Get prediction
prediction = model.predict(sample_data)
result = "Positive" if prediction[0] == 1 else "Negative"
print(f"Lung Cancer Prediction: {result}")
```

### Input Features
- `age`: Patient's age (21-80 years)
- All other features are binary (0 or 1):
  - smoking
  - yellow_fingers
  - anxiety
  - peer_pressure
  - chronic_disease
  - fatigue
  - allergy
  - wheezing
  - alcohol_consuming
  - coughing
  - shortness_of_breath
  - swallowing_difficulty
  - chest_pain

### Model Output
- 0: Negative (No lung cancer detected)
- 1: Positive (Lung cancer detected)

## Notes
- Current model accuracy: ~96%
- Age values must be between 21 and 80 years
- All features except age should be binary (0 or 1)
- The model uses both original and synthetic data for improved accuracy

## License
Open Source
