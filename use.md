
# Lung Cancer Prediction API - Technical Documentation

This document provides technical instructions on how to build and run the Lung Cancer Prediction API using the provided source files. This API uses a machine learning model (Random Forest) trained on survey data to predict the likelihood of lung cancer based on input features.

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Python 3.11 or higher:** The backend of the application is written in Python.
*   **pip:** Python package installer (usually included with Python).
*   **Virtual Environment (recommended):** It is best practice to create a virtual environment to manage project dependencies. You can create one using:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

## 2. Setup

Follow these steps to set up the application:

### 2.1. Install Dependencies

Navigate to the directory containing the source files and install the required Python libraries. You can use the provided list of libraries (from the first source) as a guide. A comprehensive list of necessary libraries can be installed using pip:

```bash
pip install fastapi uvicorn joblib pandas numpy scikit-learn python-dotenv
pip install scikit-optimize  # Required for model training script
pip install kaggle          # Required for downloading the dataset (if needed)
```

**Note:** You might need to install additional libraries based on the specific requirements of the scripts if the provided list is not exhaustive.

### 2.2. Obtain and Organize Datasets

The application relies on two datasets: the original survey data and potentially synthetic data.

*   **Original Dataset:** The file `survey lung cancer.csv` (renamed to `lung_cancer.csv` by the training script) contains the original survey data. Ensure this file is available. The FastAPI application, by default, looks for `original_dataset.csv` in a `datasets` directory. You might need to rename or move the file accordingly.
*   **Synthetic Dataset:** The file `lung_synrhetics_old.csv` contains synthetic data. Similarly, the FastAPI application looks for `synthetic_dataset.csv` in the `datasets` directory. You might need to rename or move this file.

It is recommended to create a `datasets` directory in the project root and place `original_dataset.csv` and `synthetic_dataset.csv` inside it.

### 2.3. Train the Machine Learning Model (Optional but Recommended)

The provided notebook `lung_cancer_detection.ipynb` contains the code for data preprocessing and training the lung cancer detection model using a Random Forest classifier.

1.  **Run the Notebook:** Execute the cells in the `lung_cancer_detection.ipynb` notebook. This notebook performs the following key steps:
    *   Installs necessary libraries like `scikit-optimize` and `kaggle`.
    *   Downloads the lung cancer dataset from Kaggle (requires Kaggle API key setup).
    *   Loads and preprocesses both original and synthetic datasets.
    *   Performs data exploration.
    *   Splits the combined data into training and testing sets.
    *   Tunes the hyperparameters of a Random Forest classifier using Bayesian optimization.
    *   Trains the final model with the best hyperparameters.
    *   Evaluates the trained model.
    *   **Saves the trained model to `./models/lung_cancer_detector.joblib`**.

2.  **Ensure Model is Saved:** After running the notebook, verify that the trained model file `lung_cancer_detector.joblib` exists in the `models` directory created by the script.

**Note:** If you do not run the training script, the FastAPI application will attempt to load a pre-existing model at the specified path.

### 2.4. Configure API Key (Security)

The FastAPI application implements API key-based authentication for its protected endpoints (`/predict`, `/status`).

1.  **Set Environment Variable:** It is recommended to set the API key as an environment variable named `API_KEY`. You can do this directly in your terminal (for the current session) or by creating a `.env` file in the project root directory with the following content:
    ```
    API_KEY=your_secret_api_key_here
    ```
    Replace `your_secret_api_key_here` with your desired secret API key. The `main.py` script uses the `dotenv` library to load this variable.

2.  **Default Placeholder:** If the `API_KEY` environment variable is not set, the application will use a default placeholder API key `"YOUR_SECRET_API_KEY_HERE"` and log a warning. **For production environments, it is crucial to set a strong, unique API key.**

## 3. Running the Application

Once the setup is complete, you can run the FastAPI application:

1.  **Navigate to the Project Root:** Open your terminal and navigate to the root directory of your project (where the `main.py` file is located).

2.  **Run the FastAPI Server:** Execute the following command to start the Uvicorn server:
    ```bash
    uvicorn main:app --reload
    ```
    *   `main`: Refers to the `main.py` file.
    *   `app`: Refers to the FastAPI application instance created within `main.py`.
    *   `--reload`: Enables automatic reloading of the server when code changes are detected (useful for development).

3.  **Access the API:** The API will be accessible at `http://localhost:8000` by default.

## 4. Using the API

The API provides the following endpoints:

*   **`/` (GET):** A basic, unprotected root endpoint that returns a welcome message. You can access this directly in your web browser.
*   **`/predict` (POST):** A protected endpoint for making lung cancer predictions. It requires an `X-API-KEY` header with a valid API key. It accepts a JSON payload with the following input features (as defined in the `InputData` Pydantic model in `main.py`):
    *   `age` (integer)
    *   `smoking` (boolean)
    *   `yellow_fingers` (boolean)
    *   `anxiety` (boolean)
    *   `peer_pressure` (boolean)
    *   `chronic_disease` (boolean)
    *   `fatigue` (boolean)
    *   `allergy` (boolean)
    *   `wheezing` (boolean)
    *   `alcohol_consuming` (boolean)
    *   `coughing` (boolean)
    *   `shortness_of_breath` (boolean)
    *   `swallowing_difficulty` (boolean)
    *   `chest_pain` (boolean)

    The endpoint returns a JSON response containing the `prediction_label` ("YES" or "NO"), `prediction_value` (1 for "YES", 0 for "NO"), `probability` of the positive class, `status` ("success" or "error"), and the `processed_features_for_prediction`.

    **Example using `requests` in Python (as shown in `client.py`):**

    ```python
    import requests
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv()

    url = "http://localhost:8000/predict"
    api_key_name = "X-API-KEY"
    api_key = os.getenv("API_KEY", "YOUR_SECRET_API_KEY_HERE")
    headers = {
        "Content-Type": "application/json",
        api_key_name: api_key
    }
    data = {
        "age": 50,
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
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.status_code)
    print(response.json())
    ```

*   **`/status` (GET):** A protected debugging endpoint that returns the status of the loaded assets (model, datasets). It requires an `X-API-KEY` header. The response includes information about whether the model and datasets were loaded successfully, the model type, expected features, dataset shapes and columns, and the min/max age used for scaling.

## 5. Code Structure

Here is a brief overview of the main files:

*   **`lung_cancer_detection.ipynb`:** A Jupyter Notebook containing the code for data loading, preprocessing, exploration, model training, hyperparameter tuning, and saving the trained model.
*   **`main.py`:** The main FastAPI application file. It defines:
    *   API key-based security dependency.
    *   Paths and configuration for loading the trained model and datasets.
    *   Functions to load the model and datasets.
    *   Pydantic `InputData` model for request body validation.
    *   Preprocessing function for input data.
    *   API endpoints (`/`, `/predict`, `/status`).
    *   Startup event handler to load assets when the application starts.
*   **`client.py` (if provided or based on):** A Python script demonstrating how to send prediction requests to the `/predict` endpoint of the FastAPI application, including how to include the API key in the header.
*   **`datasets/`:** A directory (to be created) that should contain the `original_dataset.csv` and `synthetic_dataset.csv` files.
*   **`models/`:** A directory (created by the training script) that should contain the saved trained model file `lung_cancer_detector.joblib`.
*   **.env (optional):** A file to store environment variables, such as the `API_KEY`.

This documentation should help you build and run the Lung Cancer Prediction API. Remember to handle the API key securely, especially in production environments.
```
