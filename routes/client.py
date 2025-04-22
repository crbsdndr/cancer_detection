from dotenv import load_dotenv
import requests
import json
import os

class CancerAPIClient:
    def __init__(self):
        load_dotenv()
        self.url = "http://127.0.0.1:8000/predict_and_explain"
        self.api_key = os.getenv("API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        self._validate_config()

    def _validate_config(self):
        if not self.api_key:
            print("Error: API_KEY not found in environment")
            exit(1)

    def _handle_error(self, e):
        error_msg = f"Request failed: {str(e)}"
        if isinstance(e, requests.exceptions.RequestException) and e.response:
            error_msg += f"\nStatus: {e.response.status_code}"
            try:
                error_msg += f"\nResponse: {e.response.json()}"
            except ValueError:
                error_msg += f"\nResponse: {e.response.text}"
        print(error_msg)

    def predict(self, data):
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=data  # Gunakan parameter json untuk auto-serialization
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._handle_error(e)
            return None

if __name__ == "__main__":
    client = CancerAPIClient()
    
    sample_data = {
        "age": 70,
        "smoking": False,
        "yellow_fingers": True,
        "anxiety": True,
        "peer_pressure": False,
        "chronic_disease": True,
        "fatigue": True,
        "allergy": False,
        "wheezing": False,
        "alcohol_consuming": True,
        "coughing": False,
        "shortness_of_breath": False,
        "swallowing_difficulty": True,
        "chest_pain": False
    }

    if result := client.predict(sample_data):
        print("Prediction Result:")
        print(json.dumps(result, indent=2))