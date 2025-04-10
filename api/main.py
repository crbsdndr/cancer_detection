import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

# Basic Logging Configuration
# Set up logging to display INFO messages and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Path Configuration ---
# Get the directory where this script is located for safe relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Full path to the model .joblib file
MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_cancer_detector.joblib")
# Full path to the original dataset file
ORIGINAL_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "original_dataset.csv")
# Full path to the synthetic dataset file
SYNTHETIC_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "synthetic_dataset.csv")

# --- Global Variable for Assets ---
# Use a dictionary to store loaded assets (model, datasets, etc.)
# This allows centralized access to loaded assets during the application lifecycle.
assets: Dict[str, Any] = {
    "model": None,
    "original_dataset": None,
    "synthetic_dataset": None,
    "min_age": 0, # Default minimum age if datasets are not loaded
    "max_age": 100 # Default maximum age if datasets are not loaded
}

# --- Asset Loading Helper Functions ---
def load_model(path: str) -> Optional[Any]:
    """
    Loads a machine learning model from a .joblib file.

    Args:
        path (str): The full path to the .joblib model file.

    Returns:
        Optional[Any]: The loaded model object if successful, None otherwise.
                       Failure can occur if the file is not found or due to other loading errors.
    """
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
    """
    Standardizes DataFrame column names.
    Converts to lowercase, removes leading/trailing spaces,
    and replaces spaces with underscores.

    Args:
        df (pd.DataFrame): The DataFrame whose column names will be standardized.

    Returns:
        pd.DataFrame: The DataFrame with standardized column names.
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def load_dataset(path: str) -> Optional[pd.DataFrame]:
    """
    Loads a dataset from a CSV file and standardizes its column names.

    Args:
        path (str): The full path to the .csv dataset file.

    Returns:
        Optional[pd.DataFrame]: The loaded Pandas DataFrame if successful,
                                  None if the file is not found or an error occurs.
    """
    try:
        df = pd.read_csv(path)
        df = standardize_columns(df) # Standardize immediately after loading
        logger.info(f"Dataset successfully loaded from: {path}. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        logger.warning(f"Dataset file not found at: {path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load dataset from {path}: {e}")
        return None

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Lung Cancer Prediction API",
    description="API to predict lung cancer risk based on user input data, using a pre-trained Random Forest model.",
    version="1.2.0", # Version updated due to translation and changes
    # Contact and license info can be added here if needed
    # contact={"name": "Your Name", "email": "your.email@example.com"},
    # license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"}
)

# --- Startup Event Handler ---
@app.on_event("startup")
def load_all_assets():
    """
    Function executed once when the FastAPI application starts.
    Responsible for loading the machine learning model and necessary datasets.
    Also calculates the age range (min/max) from the datasets if available.
    """
    logger.info("Starting asset loading process on application startup...")

    # Load the model
    assets["model"] = load_model(MODEL_PATH)

    # Load datasets
    assets["original_dataset"] = load_dataset(ORIGINAL_DATASET_PATH)
    assets["synthetic_dataset"] = load_dataset(SYNTHETIC_DATASET_PATH)

    # Calculate min/max age ONLY if both datasets were loaded successfully
    if assets["original_dataset"] is not None and assets["synthetic_dataset"] is not None:
        try:
            # Combine both datasets to get the overall age range
            combined_df = pd.concat([assets["synthetic_dataset"], assets["original_dataset"]], ignore_index=True)
            # Ensure the 'age' column exists before trying to access it
            if "age" in combined_df.columns:
                assets["min_age"] = int(combined_df["age"].min())
                assets["max_age"] = int(combined_df["age"].max())
                logger.info(f"Min/Max age successfully calculated from combined datasets: Min={assets['min_age']}, Max={assets['max_age']}")
            else:
                 logger.warning("Column 'age' not found in combined dataset. Cannot calculate min/max age.")
        except Exception as e:
            # Catch specific errors if possible (e.g., TypeError if 'age' is not numeric)
            logger.error(f"Failed to calculate min/max age from combined datasets: {e}")
            # Retain default values if calculation fails

    # Critical checks after attempting to load all assets
    if assets["model"] is None:
         logger.critical("CRITICAL: Model failed to load! /predict endpoint will not function.")
    if assets["original_dataset"] is None or assets["synthetic_dataset"] is None:
         logger.warning("WARNING: One or both datasets failed to load. Age scaling might use default values (0-100) or an inaccurate range.")

    logger.info("Asset loading on application startup finished.")

# --- Pydantic Model for Input ---
class InputData(BaseModel):
    """
    Input data schema expected by the /predict endpoint.
    Defines the data types for each feature required by the model.
    FastAPI uses this for automatic request body validation.
    """
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

    # Example data for automatic API documentation (Swagger UI / ReDoc)
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
    Performs preprocessing on the input data (from Pydantic model)
    to match the format expected by the machine learning model.

    Preprocessing Steps:
    1. Convert Pydantic data to a Pandas DataFrame.
    2. Convert boolean values (True/False) to numeric (1/0).
    3. Apply Min-Max scaling to the 'age' feature based on the age range
       calculated from datasets during startup (or default values).
    4. Adjust the DataFrame's column order to exactly match the
       feature order seen by the model during training (`feature_names_in_`).

    Args:
        input_data (InputData): The Pydantic object containing the input data from the request.

    Returns:
        pd.DataFrame: A Pandas DataFrame with a single row of processed data,
                      ready to be used for prediction by the model.

    Raises:
        ValueError: If the model is unavailable (should be handled at the endpoint),
                    if an error occurs during column reindexing, or if the
                    preprocessing result contains null values or is empty.
    """
    model = assets.get("model")
    if model is None:
        # This should not happen if startup was successful, but as a safeguard
        # This error will be caught at the endpoint and converted to an HTTPException
        raise ValueError("Model is not available for preprocessing.")

    # 1. Convert Pydantic to DataFrame
    input_dict = input_data.dict()
    processed_df = pd.DataFrame([input_dict])

    # 2. Convert Boolean to Numeric (0 or 1)
    bool_cols = processed_df.select_dtypes(include='bool').columns
    for col in bool_cols:
        processed_df[col] = processed_df[col].astype(int)
    logger.debug(f"Data after boolean conversion: {processed_df.iloc[0].to_dict()}")


    # 3. Age Scaling (Min-Max Scaling)
    # Retrieve min/max age calculated during startup
    min_age = assets.get("min_age", 0) # Use default 0 if not in assets
    max_age = assets.get("max_age", 100) # Use default 100 if not in assets
    current_age = processed_df.loc[0, "age"]

    # Adjust range for scaling:
    # If input age is outside the training data range, the effective range is expanded
    # so the scaled value remains between 0 and 1.
    effective_max_age = max(max_age, current_age)
    effective_min_age = min(min_age, current_age)

    # Avoid division by zero if min_age == max_age (constant data case or input equals bounds)
    if effective_max_age == effective_min_age:
        # If range is 0, set to a midpoint (0.5) or 0 if max was 0
        scaled_age = 0.0 if effective_max_age == 0 else 0.5
        logger.warning(f"Effective age range (min/max) is the same: {effective_min_age}. Age scaled to {scaled_age}.")
    else:
        scaled_age = (current_age - effective_min_age) / (effective_max_age - effective_min_age)

    processed_df.loc[0, "age"] = scaled_age
    logger.debug(f"Age after scaling (min={effective_min_age}, max={effective_max_age}): {scaled_age}")


    # 4. Feature Adjustment (Reindex according to model features)
    # This is a crucial step to ensure the input data has the exact same columns
    # and order as the model saw during training.
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        logger.debug(f"Features expected by model (from feature_names_in_): {model_features}")
        try:
            # Reindex DataFrame:
            # - Reorders columns according to model_features.
            # - Adds columns present in the model but not in input (filled with 0, though Pydantic should ensure all are present).
            # - Removes columns present in input but not in the model (should not happen with Pydantic).
            processed_df = processed_df.reindex(columns=model_features, fill_value=0)
        except Exception as e:
             # Catch errors if reindexing fails (e.g., column name mismatch)
             raise ValueError(f"Error during feature column reordering: {e}. Current input columns: {processed_df.columns.tolist()}")
    else:
        # Fallback if the model doesn't have feature_names_in_ (e.g., older model or different type)
        logger.warning("Model does not have 'feature_names_in_' attribute. Relying on Pydantic column order. Ensure this is consistent with model training.")
        # Assume Pydantic's column order is correct.
        expected_cols_from_input = list(InputData.__fields__.keys())
        # Ensure only columns defined in InputData are used
        try:
            processed_df = processed_df[expected_cols_from_input]
        except KeyError as e:
            raise ValueError(f"Expected column from Pydantic not found in DataFrame: {e}")


    logger.debug(f"Final data after preprocessing: {processed_df.iloc[0].to_dict()}")
    logger.debug(f"Final data columns after preprocessing: {processed_df.columns.tolist()}")

    # Final validation before returning the preprocessed data
    if processed_df.isnull().values.any():
        missing_cols = processed_df.columns[processed_df.isnull().any()].tolist()
        logger.error(f"Processed data contains null values in columns: {missing_cols}")
        raise ValueError(f"Processed data contains null values after preprocessing in columns: {missing_cols}")
    if processed_df.empty:
        logger.error("Resulting preprocessed DataFrame is empty.")
        raise ValueError("Resulting preprocessed DataFrame is empty.")

    return processed_df

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """
    Root endpoint of the API.
    Provides a simple welcome message to confirm the API is running.
    """
    return {"message": "Welcome to the Random Forest Model API for Lung Health Prediction"}

@app.post("/predict", tags=["Prediction"])
def predict(input_data: InputData):
    """
    Main endpoint for making lung cancer predictions.

    Accepts patient data in JSON format (matching the InputData model),
    performs preprocessing, runs the prediction model, and returns the result.

    Args:
        input_data (InputData): Patient input data validated by Pydantic.

    Returns:
        dict: The prediction result containing:
              - prediction_label (str): "YES" or "NO".
              - prediction_value (int): 1 (if cancer) or 0 (if not).
              - probability (float): Probability of the positive prediction (cancer).
              - status (str): "success".
              - processed_features_for_prediction (dict): Features used after
                preprocessing (for debugging purposes).

    Raises:
        HTTPException 503: If the model was not loaded successfully during startup.
        HTTPException 400: If an error occurs during input data preprocessing (e.g., ValueError).
        HTTPException 500: If any other internal server error occurs during prediction.
    """
    model = assets.get("model")
    if model is None:
        # If the model is missing, return Service Unavailable
        logger.error("/predict endpoint called but model is not available.")
        raise HTTPException(status_code=503, detail="Model is not available or failed to load. Please try again later or contact the administrator.")

    try:
        # 1. Preprocess the input data
        logger.info(f"Received prediction request: {input_data.dict()}")
        processed_data = preprocess_input(input_data)
        logger.info("Input data preprocessing successful.")

        # 2. Make prediction using the loaded model
        prediction = model.predict(processed_data)
        # Get probabilities for each class [prob_class_0, prob_class_1]
        prediction_proba = model.predict_proba(processed_data)
        logger.info(f"Raw prediction: {prediction[0]}, Raw probabilities: {prediction_proba[0]}")


        # 3. Format the prediction result
        # Get the probability of the positive class (cancer).
        # Assume class 1 is positive (cancer). Check `model.classes_` if necessary.
        # `prediction_proba[0]` is the probability array for the first (only) input.
        # `[1]` gets the probability of the second class (index 1).
        # Add a check in case predict_proba returns only one probability (rare).
        if prediction_proba.shape[1] > 1:
             probability_positive = float(prediction_proba[0][1])
        else:
             # If only 1 class prob, it's for that class.
             # If that class is 0, positive prob is 0. If it's 1, positive prob is that prob.
             probability_positive = float(prediction_proba[0][0]) if int(prediction[0]) == 1 else 0.0
             logger.warning("Model predict_proba returned only one probability value.")


        prediction_value = int(prediction[0])
        prediction_label = "YES" if prediction_value == 1 else "NO"

        logger.info(f"Prediction result: Label={prediction_label}, Value={prediction_value}, Probability={probability_positive:.4f}")

        return {
            "prediction_label": prediction_label,
            "prediction_value": prediction_value,
            "probability": probability_positive,
            "status": "success",
            # Include processed features to aid debugging if results are unexpected
            "processed_features_for_prediction": processed_data.iloc[0].to_dict()
        }

    except ValueError as ve:
        # Catch specific errors from preprocessing
        logger.error(f"Error during preprocessing or input data validation: {ve}")
        # Return Bad Request as the issue is with the input or its processing
        raise HTTPException(status_code=400, detail=f"Error in input data or preprocessing: {ve}")
    except Exception as e:
        # Catch all other unexpected errors
        logger.exception(f"An internal server error occurred during prediction: {e}") # Log the full traceback
        # Return Internal Server Error
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {e}")

# --- Debugging/Information Endpoint ---
@app.get("/status", tags=["Debugging"])
def get_status():
    """
    Debugging endpoint to check the loading status of all assets
    (model and datasets) and basic information about them.

    Useful for ensuring the application started correctly and all
    required components are available.

    Returns:
        dict: Status information including model/dataset loading status,
              model type, expected features (if available), dataset shapes,
              dataset column names, and the age range used for scaling.
    """
    model_feature_names = None
    model_type_str = None
    if assets["model"]:
        model_type_str = str(type(assets["model"]))
        if hasattr(assets["model"], 'feature_names_in_'):
            try:
                model_feature_names = list(assets["model"].feature_names_in_)
            except Exception as e:
                logger.error(f"Failed to get feature_names_in_ from model: {e}")
                model_feature_names = f"Error retrieving: {e}"


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
# This block is usually commented out if you run using `uvicorn main:app --reload`
# However, it's useful if you want to run the script directly (`python your_script_name.py`)
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" makes the API accessible from outside the local machine/container
    # port=8000 is the default, can be changed
    logger.info("Running FastAPI application with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
