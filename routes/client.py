import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("API_KEY")  # Pastikan key ini sesuai dengan nama di file .env

# API endpoint for lung cancer prediction
api_url = "http://127.0.0.1:8000/predict/"

# Sample data for testing the prediction API
sample_data = {
    "age": 1,
    "smoking": True,
    "yellow_fingers": False,
    "anxiety": False,
    "peer_pressure": False,
    "chronic_disease": False,
    "fatigue": False,
    "allergy": False,
    "wheezing": False,
    "alcohol_consuming": False,
    "coughing": False,
    "shortness_of_breath": False,
    "swallowing_difficulty": False,
    "chest_pain": False
}

def test_prediction():
    """
    Test function to send a POST request to the prediction API and handle responses.
    Prints the prediction result or error details if the request fails.
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key  # Ini sesuai dengan permintaan Anda
        }

        # Send POST request with sample data and headers
        response = requests.post(api_url, json=sample_data, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Process successful response
        prediction_result = response.json()
        print("Request successful!")
        print("Prediction response:")
        print(json.dumps(prediction_result, indent=4))

    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error: Could not connect to the server at {api_url}")
        print("Please ensure the FastAPI server is running.")
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Error Detail: {response.json()}")
        except json.JSONDecodeError:
            print(f"Response Content: {response.text}")
            
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request: {req_err}")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_prediction()
