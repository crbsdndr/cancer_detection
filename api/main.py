import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from dotenv import load_dotenv # Added import

# Load variables from the .env file into the environment
# Call this BEFORE you try to access environment variables
load_dotenv() # Added line

# Basic Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Security Configuration ---
API_KEY_NAME = "X-API-KEY"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Now os.getenv will read from .env if the variable is set there
API_KEY = os.getenv("API_KEY", "YOUR_SECRET_API_KEY_HERE")
if API_KEY == "YOUR_SECRET_API_KEY_HERE":
    logger.warning("Using default placeholder API key. Please set the API_KEY environment variable (or in .env file) for security.")

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_cancer_detector.joblib")
ORIGINAL_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "original_dataset.csv")
SYNTHETIC_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "synthetic_dataset.csv")

# --- Global Variable for Assets ---
# Stores loaded assets like the model and datasets
assets: Dict[str, Any] = {
    "model": None,
    "original_dataset": None,
    "synthetic_dataset": None,
    "min_age": 0,
    "max_age": 100
}

# --- Asset Loading Helper Functions ---

def load_model(path: str) -> Optional[Any]:
    """Loads a machine learning model from a .joblib file."""
    try:
        model = joblib.load(path)
        logger.info(f"Model successfully loaded from: {path}")
        return model
    except FileNotFoundError:
        logger.warning(f"Model file not found at: {path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes DataFrame column names (lowercase, strip whitespace, replace space with underscore)."""
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def load_dataset(path: str) -> Optional[pd.DataFrame]:
    """Loads a dataset from a CSV file and standardizes its column names."""
    try:
        df = pd.read_csv(path)
        df = standardize_columns(df)
        logger.info(f"Dataset successfully loaded from: {path}. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        logger.warning(f"Dataset file not found at: {path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load dataset from {path}: {e}")
        return None


# --- API Key Validation Dependency ---

async def get_api_key(api_key_header: str = Security(api_key_header_auth)):
    """Dependency function to validate the API key provided in the header."""
    if not api_key_header:
        logger.warning("API key header missing.")
        raise HTTPException(
            status_code=401,
            detail="Missing API Key header (X-API-KEY)",
        )
    # Use the loaded API_KEY (can be from .env or default)
    if api_key_header != API_KEY:
        logger.warning("Invalid API key received.")
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key",
        )
    return api_key_header

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Lung Cancer Prediction API",
    description="API to predict lung cancer risk based on user input data, using a pre-trained Random Forest model. Requires API Key authentication for protected endpoints.",
    version="1.3.1", # Version updated for .env config
    dependencies=[Depends(get_api_key)] # Default dependency for most endpoints
)

# --- Startup Event Handler ---

@app.on_event("startup")
def load_all_assets():
    """Loads model and datasets on application startup."""
    logger.info("Starting asset loading process on application startup...")
    assets["model"] = load_model(MODEL_PATH)
    assets["original_dataset"] = load_dataset(ORIGINAL_DATASET_PATH)
    assets["synthetic_dataset"] = load_dataset(SYNTHETIC_DATASET_PATH)

    # Calculate min/max age from combined datasets if available
    if assets["original_dataset"] is not None and assets["synthetic_dataset"] is not None:
        try:
            combined_df = pd.concat([assets["synthetic_dataset"], assets["original_dataset"]], ignore_index=True)
            if "age" in combined_df.columns:
                assets["min_age"] = int(combined_df["age"].min())
                assets["max_age"] = int(combined_df["age"].max())
                logger.info(f"Min/Max age calculated: Min={assets['min_age']}, Max={assets['max_age']}")
            else:
                 logger.warning("Column 'age' not found in combined dataset for min/max calculation.")
        except Exception as e:
            logger.error(f"Failed to calculate min/max age from combined datasets: {e}")

    # Log status of loaded assets
    if assets["model"] is None:
         logger.critical("CRITICAL: Model failed to load! /predict endpoint will not function.")
    if assets["original_dataset"] is None or assets["synthetic_dataset"] is None:
         logger.warning("WARNING: One or both datasets failed to load. Age scaling might use defaults.")
    logger.info("Asset loading on application startup finished.")


# --- Pydantic Model for Input ---

class InputData(BaseModel):
    """Input data schema for the /predict endpoint."""
    age: int
    smoking: bool
    yellow_fingers: bool
    anxiety: bool
    peer_pressure: bool
    chronic_disease: bool
    fatigue: bool  # Renamed from 'fatigue ' if there was a typo
    allergy: bool # Renamed from 'allergy ' if there was a typo
    wheezing: bool
    alcohol_consuming: bool # Renamed from 'alcohol consuming'
    coughing: bool
    shortness_of_breath: bool # Renamed from 'shortness of breath'
    swallowing_difficulty: bool # Renamed from 'swallowing difficulty'
    chest_pain: bool # Renamed from 'chest pain'

    class Config:
        schema_extra = {
            "example": {
                "age": 60,
                "smoking": True,
                "yellow_fingers": False,
                "anxiety": False,
                "peer_pressure": True,
                "chronic_disease": True,
                "fatigue": False,
                "allergy": True,
                "wheezing": True,
                "alcohol_consuming": False,
                "coughing": True,
                "shortness_of_breath": False,
                "swallowing_difficulty": False,
                "chest_pain": True
            }
        }


# --- Preprocessing Function ---

def preprocess_input(input_data: InputData) -> pd.DataFrame:
    """
    Performs preprocessing on the input data:
    1. Converts boolean features to integers (0 or 1).
    2. Scales the 'age' feature using min-max scaling based on loaded dataset ranges.
    3. Reorders columns to match the model's expected feature order.
    """
    model = assets.get("model")
    if model is None:
        # This should ideally not happen if startup checks are in place, but good practice
        raise ValueError("Model is not available for preprocessing.")

    input_dict = input_data.dict()
    processed_df = pd.DataFrame([input_dict])

    # Convert boolean columns to integers (0 or 1)
    bool_cols = processed_df.select_dtypes(include='bool').columns
    for col in bool_cols:
        processed_df[col] = processed_df[col].astype(int)

    # Min-Max scale the 'age' feature
    min_age = assets.get("min_age", 0) # Default fallback
    max_age = assets.get("max_age", 100) # Default fallback
    current_age = processed_df.loc[0, "age"]

    # Adjust effective min/max to handle input age outside training range
    # This prevents division by zero if min_age == max_age
    effective_max_age = max(max_age, current_age)
    effective_min_age = min(min_age, current_age)

    if effective_max_age == effective_min_age:
        # Handle edge case where min and max are the same (e.g., only one data point)
        # or if input age matches the single min/max value.
        # Assign a neutral value (e.g., 0.5) or 0 depending on context.
        scaled_age = 0.0 if effective_max_age == 0 else 0.5
    else:
        scaled_age = (current_age - effective_min_age) / (effective_max_age - effective_min_age)
    processed_df.loc[0, "age"] = scaled_age

    # Ensure column order matches the model's expected features
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        try:
            # Reindex ensures all expected columns are present and in the correct order
            # Missing columns will be filled with `fill_value` (0 in this case)
            # Extra columns in processed_df will be dropped
            processed_df = processed_df.reindex(columns=model_features, fill_value=0)
        except Exception as e:
             # Catch potential errors during reindexing
             raise ValueError(f"Error during feature column reordering based on model's feature_names_in_: {e}. Input columns after bool conversion: {processed_df.columns.tolist()}")
    else:
        # Fallback if the model object doesn't have feature_names_in_
        logger.warning("Model does not have 'feature_names_in_' attribute. Relying on Pydantic model's field order for features.")
        # Use the order defined in the Pydantic model as the expected order
        expected_cols_from_input = list(InputData.__fields__.keys())
        try:
            # Select columns in the order defined by Pydantic
            processed_df = processed_df[expected_cols_from_input]
        except KeyError as e:
            # This error would mean a field defined in Pydantic is somehow missing
            # from the DataFrame created from input_data.dict(), which is unlikely
            # but good to handle.
            raise ValueError(f"Expected column from Pydantic model not found in DataFrame: {e}")

    # Final checks after preprocessing
    if processed_df.isnull().values.any():
        missing_cols = processed_df.columns[processed_df.isnull().any()].tolist()
        raise ValueError(f"Processed data contains null values after preprocessing in columns: {missing_cols}")
    if processed_df.empty:
        raise ValueError("Resulting preprocessed DataFrame is empty.")

    return processed_df


# --- API Endpoints ---

# Unprotected root endpoint
@app.get("/", tags=["General"], dependencies=[]) # Explicitly empty dependencies for unprotected route
def read_root():
    """ Root endpoint of the API (unprotected). """
    return {"message": "Welcome to the Random Forest Model API for Lung Health Prediction"}

# Prediction endpoint (protected by API Key)
@app.post("/predict", tags=["Prediction"])
async def predict(
    input_data: InputData,
    # API key validation is handled by the default dependency set on the app
    # api_key: str = Security(get_api_key) # No need to repeat if set globally
):
    """
    Main endpoint for making lung cancer predictions.
    Requires API Key authentication via the 'X-API-KEY' header.

    - **Input**: JSON object matching the InputData schema.
    - **Output**: JSON object with prediction label, value, probability, status,
                 and the processed features used for the prediction.
    """
    model = assets.get("model")
    if model is None:
        logger.error("Prediction attempt failed: Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available or failed to load. Cannot make predictions.")

    try:
        logger.info(f"Received prediction request: {input_data.dict()}")
        processed_data = preprocess_input(input_data)
        logger.info("Input data preprocessing successful.")
        # logger.debug(f"Data sent to model: {processed_data.iloc[0].to_dict()}") # Optional: log preprocessed data

        # Make prediction and get probabilities
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        logger.info(f"Raw prediction: {prediction[0]}, Raw probabilities: {prediction_proba[0]}")

        # Extract the probability of the positive class (usually index 1)
        if prediction_proba.shape[1] > 1:
             # Standard case for binary classification
             probability_positive = float(prediction_proba[0][1])
        else:
             # Handle cases where predict_proba might only return one value
             # (less common for standard classifiers but possible)
             probability_positive = float(prediction_proba[0][0]) if int(prediction[0]) == 1 else 0.0
             logger.warning("Model predict_proba returned only one probability value. Inferring positive probability based on prediction.")

        prediction_value = int(prediction[0])
        prediction_label = "YES" if prediction_value == 1 else "NO" # Assuming 1 is positive class

        logger.info(f"Prediction result: Label={prediction_label}, Value={prediction_value}, Probability={probability_positive:.4f}")

        return {
            "prediction_label": prediction_label,
            "prediction_value": prediction_value,
            "probability": probability_positive,
            "status": "success",
            # Include processed features for debugging/transparency
            "processed_features_for_prediction": processed_data.iloc[0].to_dict()
        }
    except ValueError as ve:
        # Handle errors specifically from preprocessing or data validation
        logger.error(f"Value Error during prediction processing: {ve}")
        raise HTTPException(status_code=400, detail=f"Error in input data or preprocessing: {ve}")
    except Exception as e:
        # Catch any other unexpected errors during prediction
        logger.exception(f"An internal server error occurred during prediction: {e}") # Use exception to log stack trace
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {e}")

# Status endpoint (protected by API Key)
@app.get("/status", tags=["Debugging"])
async def get_status(
    # API key validation is handled by the default dependency set on the app
    # api_key: str = Security(get_api_key) # No need to repeat if set globally
):
    """
    Debugging endpoint to check the status of loaded assets (model, datasets).
    Requires API Key authentication via the 'X-API-KEY' header.
    """
    model_feature_names = "N/A"
    model_type_str = "N/A"
    if assets["model"]:
        model_type_str = str(type(assets["model"]))
        if hasattr(assets["model"], 'feature_names_in_'):
            try:
                model_feature_names = list(assets["model"].feature_names_in_)
            except Exception as e:
                logger.error(f"Failed to get feature_names_in_ from model: {e}")
                model_feature_names = f"Error retrieving feature names: {e}"
        else:
             model_feature_names = "Model does not have 'feature_names_in_' attribute"

    original_shape = None
    original_cols = None
    if assets["original_dataset"] is not None:
        original_shape = assets["original_dataset"].shape
        original_cols = list(assets["original_dataset"].columns)

    synthetic_shape = None
    synthetic_cols = None
    if assets["synthetic_dataset"] is not None:
        synthetic_shape = assets["synthetic_dataset"].shape
        synthetic_cols = list(assets["synthetic_dataset"].columns)

    return {
        "model_loaded_successfully": assets["model"] is not None,
        "original_dataset_loaded_successfully": assets["original_dataset"] is not None,
        "synthetic_dataset_loaded_successfully": assets["synthetic_dataset"] is not None,
        "model_type": model_type_str,
        "model_expected_features": model_feature_names,
        "original_dataset_shape": original_shape,
        "original_dataset_columns": original_cols,
        "synthetic_dataset_shape": synthetic_shape,
        "synthetic_dataset_columns": synthetic_cols,
        "age_scaling_min_used": assets.get("min_age"),
        "age_scaling_max_used": assets.get("max_age"),
    }

# --- Running the app with Uvicorn (if this file is executed directly) ---
if __name__ == "__main__":
    import uvicorn
    # It's generally recommended to run Uvicorn from the command line,
    # but this allows running the script directly (e.g., python main.py)
    logger.info("Running FastAPI application with Uvicorn...")
    # Ensure the app object matches the Uvicorn string 'main:app' if your file is main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
