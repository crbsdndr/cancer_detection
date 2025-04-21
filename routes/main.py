from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from io import StringIO 
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import uvicorn
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
application = FastAPI()

# Security setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY = os.getenv("API_KEY")
print(API_KEY)
# Define file paths for model and training dataset
MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()), "models/cancer_detection_model.keras")
TRAIN_DATASET_PATH = os.path.join(os.path.dirname(os.getcwd()), "datasets/lung_survey_synthetics.csv")

# Global variables to store loaded model and preprocessing objects
model = None
scaler_global = None
label_encoder_global = None

# Constant defining the expected feature order for model input
FEATURE_ORDER = [
    "age", "smoking", "yellow_fingers", "anxiety", "peer_pressure",
    "chronic_disease", "fatigue", "allergy", "wheezing",
    "alcohol_consuming", "coughing", "shortness_of_breath",
    "swallowing_difficulty", "chest_pain"
]

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate API key from request headers"""
    if not API_KEY:
        logger.error("API Key not configured in environment")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error"
        )
    if api_key != API_KEY:
        logger.warning("Invalid API Key attempt")
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API Key"
        )
    return api_key

class InputData(BaseModel):
    """Pydantic model for input data validation."""
    age: int
    smoking: bool
    yellow_fingers: bool
    anxiety: bool
    peer_pressure: bool
    chronic_disease: bool
    fatigue: bool
    allergy: bool
    wheezing: bool
    alcohol_consuming: bool
    coughing: bool
    shortness_of_breath: bool
    swallowing_difficulty: bool
    chest_pain: bool

def convert_json_to_dataframe(json_string: str) -> pd.DataFrame:
    """Convert JSON string to pandas DataFrame."""
    return pd.read_json(StringIO(json_string))

def format_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to integers and ensure age is float."""
    dataframe_copy = dataframe.copy()
    boolean_columns = dataframe_copy.select_dtypes(include='bool').columns
    dataframe_copy[boolean_columns] = dataframe_copy[boolean_columns].astype(int)
    if "age" in dataframe_copy.columns:
        dataframe_copy["age"] = dataframe_copy["age"].astype(float)
    return dataframe_copy

def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    dataframe_copy = dataframe.copy()
    dataframe_copy.columns = [column.strip().lower().replace(" ", "_") for column in dataframe_copy.columns]
    return dataframe_copy

def scale_age_column(dataframe: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """Scale the age column using the provided scaler."""
    if "age" not in dataframe.columns:
        return dataframe
    dataframe_copy = dataframe.copy()
    dataframe_copy[["age"]] = scaler.transform(dataframe_copy[["age"]])
    return dataframe_copy

def decode_prediction_label(prediction: int, label_encoder: LabelEncoder) -> str:
    """Decode integer prediction to original label."""
    try:
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as error:
        logger.error(f"Error during label decoding: {error}")
        return str(prediction)

def load_dependencies():
    """Load model, scaler, and label encoder."""
    global model, scaler_global, label_encoder_global

    # Load model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as error:
            logger.error(f"Error loading model: {error}")
            model = None
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")
        model = None

    # Load training data and fit scaler and label encoder
    if os.path.exists(TRAIN_DATASET_PATH):
        try:
            train_dataset_raw = pd.read_csv(TRAIN_DATASET_PATH)
            train_dataset = standardize_column_names(train_dataset_raw)
            scaler_global = MinMaxScaler()
            scaler_global.fit(train_dataset[["age"]])
            label_encoder_global = LabelEncoder()
            label_encoder_global.fit(train_dataset["lung_cancer"])
            logger.info("Training data loaded and preprocessors fitted.")
        except Exception as error:
            logger.error(f"Error loading or processing training data: {error}")
            scaler_global = None
            label_encoder_global = None
    else:
        logger.error(f"Training dataset file not found at {TRAIN_DATASET_PATH}")
        scaler_global = None
        label_encoder_global = None

def preprocess_input_data(input_json_string: str) -> np.ndarray:
    """Preprocess input JSON string into an array for the model."""
    input_dataframe = convert_json_to_dataframe(input_json_string)
    input_dataframe_formatted = format_data_types(input_dataframe)
    if scaler_global and ("age" in input_dataframe_formatted.columns):
        input_dataframe_scaled = scale_age_column(input_dataframe_formatted, scaler_global)
    else:
        input_dataframe_scaled = input_dataframe_formatted
        if "age" in input_dataframe_scaled.columns:
            logger.warning("Skipping age scaling as scaler was not initialized.")
    # Ensure all required features are present
    missing_features = set(FEATURE_ORDER) - set(input_dataframe_scaled.columns)
    if missing_features:
        raise KeyError(f"Missing features: {missing_features}")
    input_dataframe_final = input_dataframe_scaled[FEATURE_ORDER]
    return input_dataframe_final.to_numpy(dtype=np.float32)

@application.on_event("startup")
async def startup_event():
    """Initialize application dependencies on startup."""
    # Check if API Key is configured
    if not API_KEY:
        logger.warning("API Key not found in environment variables")
    
    load_dependencies()
    if model is None:
        logger.warning("Model failed to load. Prediction endpoint will be unavailable.")
    if scaler_global is None:
        logger.warning("Scaler failed to initialize. Age scaling will not be performed.")
    if label_encoder_global is None:
        logger.warning("LabelEncoder failed to initialize. Prediction decoding will not be performed.")

@application.post("/predict/")
async def predict(
    data: InputData,
    api_key: str = Depends(verify_api_key)  # Add API Key dependency
):
    """Endpoint for making lung cancer predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

    try:
        input_dictionary = [data.model_dump()]
        input_json_string = json.dumps(input_dictionary)
        input_array = preprocess_input_data(input_json_string)
        
        raw_prediction = model.predict(input_array)
        probability = raw_prediction[0][0]
        integer_prediction = 1 if probability >= 0.5 else 0
        
        if label_encoder_global:
            prediction_label = decode_prediction_label(integer_prediction, label_encoder_global)
        else:
            prediction_label = "Yes" if integer_prediction == 1 else "No"
            logger.warning("Using default label mapping as LabelEncoder was not initialized.")
        
        return {
            "prediction_label": prediction_label,
            "probability": float(probability)
        }
    except KeyError as key_error:
        logger.error(f"Missing expected column during preprocessing: {key_error}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: Missing data - {key_error}")
    except Exception as error:
        logger.error(f"Prediction error: {error}")
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {error}")

if __name__ == "__main__":
    uvicorn.run(application, host="127.0.0.1", port=8000)