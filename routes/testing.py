import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# API endpoint URL
# Make sure this matches your running FastAPI server address
url = "http://localhost:8000/predict"

# --- API Key Configuration ---
# !!! IMPORTANT: Get the API key from the same source as your FastAPI app !!!
# Best practice: Use an environment variable.
# Defaulting to the placeholder used in the FastAPI app for demonstration.
API_KEY_NAME = "X-API-KEY"
API_KEY = os.getenv("API_KEY", "YOUR_SECRET_API_KEY_HERE")
if API_KEY == "YOUR_SECRET_API_KEY_HERE":
    print("WARNING: Using default placeholder API key for testing. Ensure this matches the server's key.")

# Input data for prediction (example)
# Adjust these values as needed for your tests
data = {
    "age": 45,
    "smoking": True,
    "yellow_fingers": False,
    "anxiety": True,
    "peer_pressure": False,
    "chronic_disease": True,
    "fatigue": True,
    "allergy": False,
    "wheezing": True,
    "alcohol_consuming": True,
    "coughing": True,
    "shortness_of_breath": True,
    "swallowing_difficulty": False,
    "chest_pain": True
}

# --- Request Headers ---
# Include both Content-Type for JSON and the API Key header
headers = {
    "Content-Type": "application/json",
    API_KEY_NAME: API_KEY  # Adding the API Key to the header
}

try:
    # Send POST request to the API endpoint
    print(f"Sending request to: {url}")
    print(f"Request data: {json.dumps(data, indent=2)}") # Print formatted data being sent
    response = requests.post(
        url,
        headers=headers,         # Pass the headers dictionary
        data=json.dumps(data)    # Convert the Python dictionary to a JSON string
    )

    # Print the response status code
    print(f"\nResponse Status Code: {response.status_code}")

    # Try to print the JSON response, handle potential errors if response is not JSON
    try:
        print("Response JSON:", response.json()) # Decode the JSON response
    except json.JSONDecodeError:
        # If the response is not valid JSON (e.g., HTML error page from the server)
        print("JSON Decode Error: Failed to decode the response from the server.")
        print("Raw Response Text:", response.text) # Print raw text if JSON decoding fails

except requests.exceptions.ConnectionError as conn_err:
    # Handle connection errors specifically (e.g., server not running)
    print(f"\nConnection Error: Could not connect to the server at {url}. Is the server running?")
    print(f"Details: {conn_err}")
except requests.exceptions.RequestException as req_err:
    # Handle other potential errors during the request (e.g., timeout, invalid URL)
    print(f"\nRequest Error: An error occurred during the request.")
    print(f"Details: {req_err}")
except Exception as e:
    # Handle any other unexpected errors during script execution
    print(f"\nAn unexpected error occurred: {str(e)}")

