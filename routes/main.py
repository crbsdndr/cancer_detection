from fastapi import FastAPI, HTTPException, Depends, Security
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Any
import tensorflow as tf
from groq import Groq
import pandas as pd
import numpy as np
import logging
import uvicorn
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_KEY_NAME = "X-API-Key"
ENV_VARS = {
    "API_KEY": os.getenv("API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY_DENDRA"),
    "GROQ_MODEL": os.getenv("GROQ_MODEL"),
    "MODEL_PATH": os.path.join(os.path.dirname(os.getcwd()), os.getenv("MODEL_PATH")),
    "TRAIN_DATASET_PATH": os.path.join(os.path.dirname(os.getcwd()), os.getenv("TRAIN_DATASET_PATH"))
}

FEATURES = [
    "age", "smoking", "yellow_fingers", "anxiety", "peer_pressure",
    "chronic_disease", "fatigue", "allergy", "wheezing",
    "alcohol_consuming", "coughing", "shortness_of_breath",
    "swallowing_difficulty", "chest_pain"
]

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

class CombinedResponse(BaseModel):
    prediction_label: str
    probability: float
    input_features: Dict[str, Any]
    explanation: str | None = None
    explanation_status: str

class CancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        
    def load(self):
        try:
            self.model = tf.keras.models.load_model(ENV_VARS["MODEL_PATH"])
            df = pd.read_csv(ENV_VARS["TRAIN_DATASET_PATH"])
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[["age"]])
            
            self.encoder = LabelEncoder()
            self.encoder.fit(df["lung_cancer"])
            
            logger.info("Model and preprocessors loaded")
        except Exception as e:
            logger.error(f"Loading failed: {e}")
            raise

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=ENV_VARS["GROQ_API_KEY"]) if ENV_VARS["GROQ_API_KEY"] else None
        
    def explain(self, data: Dict, prediction: str, prob: float) -> str:
        if not self.client:
            raise HTTPException(500, "Groq client not configured")
            
        features = [f"age {data['age']} years"] + [
            f"{k.replace('_', ' ')}: {'yes' if v else 'no'}" 
            for k, v in data.items() if k != 'age'
        ]
        
        prompt = (
            f"{', '.join(features)}. "
            f"Prediction: {prediction} ({prob:.2f} probability). "
            "Explain this prediction focusing on key risk factors."
        )
        
        response = self.client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": "Explain lung cancer predictions based on patient data."
            }, {
                "role": "user",
                "content": prompt
            }],
            model=ENV_VARS["GROQ_MODEL"]
        )
        return response.choices[0].message.content

app = FastAPI(title="Lung Cancer Prediction API")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
predictor = CancerPredictor()
groq = GroqClient()

def verify_key(api_key: str = Security(api_key_header)):
    if api_key != ENV_VARS["API_KEY"]:
        logger.warning(f"Invalid key: {api_key[:5]}...")
        raise HTTPException(403, "Invalid API Key")
    return api_key

@app.on_event("startup")
async def startup():
    try:
        predictor.load()
    except Exception:
        logger.critical("Initialization failed - endpoints disabled")

@app.post("/predict_and_explain", response_model=CombinedResponse)
async def predict(data: InputData, _=Depends(verify_key)):
    try:
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        df = df.astype({k: int for k in FEATURES if k != 'age'})
        
        df["age"] = predictor.scaler.transform(df[["age"]])
        pred = predictor.model.predict(df[FEATURES].to_numpy(np.float32))[0]
        
        prob = float(pred[0] if len(pred) == 1 else pred[1])
        label = predictor.encoder.inverse_transform([int(prob >= 0.5)])[0]
        
        explanation, status = None, "not_attempted"
        try:
            explanation = groq.explain(input_dict, label, prob)
            status = "success"
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            status = "failed"
            
        return CombinedResponse(
            prediction_label=label,
            probability=prob,
            input_features=input_dict,
            explanation=explanation,
            explanation_status=status
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, "Prediction failed")

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST"), port=int(os.getenv("PORT")))