import requests
import json

# API endpoint URL
url = "http://localhost:8000/predict" # Make sure this matches your API server address

# Input data for prediction
# NOTE: Removed square brackets as InputData expects a single object, not a list
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

try:
    # Send POST request to the API endpoint
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"}, # Set the content type header
        data=json.dumps(data) # Convert the Python dictionary to a JSON string
    )

    # Print the response status code and the JSON response body
    print("Status Code:", response.status_code)
    print("Response:", response.json()) # Decode the JSON response

except requests.exceptions.ConnectionError as conn_err:
    # Handle connection errors specifically
    print(f"Connection Error: Could not connect to the server at {url}. Is the server running?")
    print(f"Details: {conn_err}")
except requests.exceptions.RequestException as req_err:
    # Handle other request-related errors
    print(f"Request Error: An error occurred during the request.")
    print(f"Details: {req_err}")
except json.JSONDecodeError:
    # Handle errors if the response is not valid JSON
    print("JSON Decode Error: Failed to decode the response from the server.")
    print("Raw Response Text:", response.text)
except Exception as e:
    # Handle any other unexpected errors
    print("An unexpected error occurred:", str(e))

