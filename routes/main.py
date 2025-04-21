from fastapi import FastAPI, HTTPException, Security, Depends
from typing import Optional, List, Dict, Any
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
import joblib
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-KEY"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

API_KEY = os.getenv("API_KEY", "YOUR_SECRET_API_KEY_HERE")
if API_KEY == "YOUR_SECRET_API_KEY_HERE":
    logger.warning("Using default placeholder API key. Please set the API_KEY environment variable (or in .env file) for security.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_cancer_detector.joblib")
ORIGINAL_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "original_dataset.csv")
SYNTHETIC_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "synthetic_dataset.csv")

assets: Dict[str, Any] = {
    "model": None,
    "original_dataset": None,
    "synthetic_dataset": None,
    "min_age": 0,
    "max_age": 100
}

def load_model(path: str) -> Optional[Any]:
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
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def load_dataset(path: str) -> Optional[pd.DataFrame]:
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

async def get_api_key(api_key_header: str = Security(api_key_header_auth)):
    if not api_key_header:
        logger.warning("API key header missing.")
        raise HTTPException(
            status_code=401,
            detail="Missing API Key header (X-API-KEY)",
        )
    if api_key_header != API_KEY:
        logger.warning("Invalid API key received.")
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key",
        )
    return api_key_header

app = FastAPI(
    title="Lung Cancer Prediction API",
    description="API to predict lung cancer risk based on user input data, using a pre-trained Random Forest model. Requires API Key authentication for protected endpoints.",
    version="1.3.1",
    dependencies=[Depends(get_api_key)]
)

@app.on_event("startup")
def load_all_assets():
    logger.info("Starting asset loading process on application startup...")
    assets["model"] = load_model(MODEL_PATH)
    assets["original_dataset"] = load_dataset(ORIGINAL_DATASET_PATH)
    assets["synthetic_dataset"] = load_dataset(SYNTHETIC_DATASET_PATH)

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

    if assets["model"] is None:
        logger.critical("CRITICAL: Model failed to load! /predict endpoint will not function.")
    if assets["original_dataset"] is None or assets["synthetic_dataset"] is None:
        logger.warning("WARNING: One or both datasets failed to load. Age scaling might use defaults.")
    logger.info("Asset loading on application startup finished.")

class InputData(BaseModel):
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

def preprocess_input(input_data: InputData) -> pd.DataFrame:
    model = assets.get("model")
    if model is None:
        raise ValueError("Model is not available for preprocessing.")

    input_dict = input_data.dict()
    processed_df = pd.DataFrame([input_dict])

    bool_cols = processed_df.select_dtypes(include='bool').columns
    for col in bool_cols:
        processed_df[col] = processed_df[col].astype(int)

    min_age = assets.get("min_age", 0)
    max_age = assets.get("max_age", 100)
    current_age = processed_df.loc[0, "age"]

    effective_max_age = max(max_age, current_age)
    effective_min_age = min(min_age, current_age)

    if effective_max_age == effective_min_age:
        scaled_age = 0.0 if effective_max_age == 0 else 0.5
    else:
        scaled_age = (current_age - effective_min_age) / (effective_max_age - effective_min_age)
    processed_df.loc[0, "age"] = scaled_age

    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        try:
            processed_df = processed_df.reindex(columns=model_features, fill_value=0)
        except Exception as e:
             raise ValueError(f"Error during feature column reordering based on model's feature_names_in_: {e}. Input columns after bool conversion: {processed_df.columns.tolist()}")
    else:
        logger.warning("Model does not have 'feature_names_in_' attribute. Relying on Pydantic model's field order for features.")
        expected_cols_from_input = list(InputData.__fields__.keys())
        try:
            processed_df = processed_df[expected_cols_from_input]
        except KeyError as e:
            raise ValueError(f"Expected column from Pydantic model not found in DataFrame: {e}")

    if processed_df.isnull().values.any():
        missing_cols = processed_df.columns[processed_df.isnull().any()].tolist()
        raise ValueError(f"Processed data contains null values after preprocessing in columns: {missing_cols}")
    if processed_df.empty:
        raise ValueError("Resulting preprocessed DataFrame is empty.")

    return processed_df

@app.get("/", tags=["General"], dependencies=[])
def read_root():
    return {"message": "Welcome to the Random Forest Model API for Lung Health Prediction"}

@app.post("/predict", tags=["Prediction"])
async def predict(input_data: InputData):
    model = assets.get("model")
    if model is None:
        logger.error("Prediction attempt failed: Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available or failed to load. Cannot make predictions.")

    try:
        logger.info(f"Received prediction request: {input_data.dict()}")
        processed_data = preprocess_input(input_data)
        logger.info("Input data preprocessing successful.")
        # logger.debug(f"Data sent to model: {processed_data.iloc[0].to_dict()}")

        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        logger.info(f"Raw prediction: {prediction[0]}, Raw probabilities: {prediction_proba[0]}")

        if prediction_proba.shape[1] > 1:
            probability_positive = float(prediction_proba[0][1])
        else:
            probability_positive = float(prediction_proba[0][0]) if int(prediction[0]) == 1 else 0.0
            logger.warning("Model predict_proba returned only one probability value. Inferring positive probability based on prediction.")

        prediction_value = int(prediction[0])
        prediction_label = "YES" if prediction_value == 1 else "NO"

        logger.info(f"Prediction result: Label={prediction_label}, Value={prediction_value}, Probability={probability_positive:.4f}")

        return {
            "prediction_label": prediction_label,
            "prediction_value": prediction_value,
            "probability": probability_positive,
            "status": "success",
            "processed_features_for_prediction": processed_data.iloc[0].to_dict()
        }
    except ValueError as ve:
        logger.error(f"Value Error during prediction processing: {ve}")
        raise HTTPException(status_code=400, detail=f"Error in input data or preprocessing: {ve}")
    except Exception as e:
        logger.exception(f"An internal server error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {e}")

@app.get("/status", tags=["Debugging"])
async def get_status():
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

if __name__ == "__main__":
    import uvicorn
    logger.info("Running FastAPI application with Uvicorn...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)