# **Lung Cancer Prediction API**  
**Version:** 2.0  
**Last Updated:** 22/04/2025  

---

## **1. API Overview**  
A RESTful API for lung cancer risk assessment using a TensorFlow neural network (MLP_Enhanced_v2). Designed for integration into healthcare applications targeting underserved populations.  

### **Key Features**  
✔ Single endpoint (`/predict/`) for real-time predictions  
✔ API key authentication  
✔ Returns both binary classification and probability score  
✔ Local development-ready with deployment capabilities  

**Base URL:** `http://127.0.0.1:8000` (default local)  

---

## **2. Quick Start Guide**  

### **2.1 Prerequisites**  
- Python 3.12.9+  
- Required packages:  
  ```bash
  pip install fastapi uvicorn tensorflow pandas numpy scikit-learn pydantic python-dotenv
  ```

### **2.2 Setup**  
1. Clone repository:  
   ```bash
   git clone [your-repo-url]
   ```
2. File structure:  
   ```
   project_root/
   ├── models/
   │   └── cancer_detection_model.keras
   ├── datasets/
   │   └── lung_survey_synthetics.csv
   ├── .env
   └── api.py
   ```
3. Configure `.env`:  
   ```plaintext
   API_KEY=your_secure_api_key_here
   ```

### **2.3 Running the API**  
```bash
uvicorn main:app --reload
```
Access docs at: `http://127.0.0.1:8000/docs`  

---

## **3. API Reference**  

### **3.1 Authentication**  
**Header:**  
```http
X-API-Key: your_secure_api_key
```  
⚠ Returns `403 Forbidden` for invalid/missing keys  

### **3.2 Prediction Endpoint**  
**`POST /predict/`**  

#### **Request**  
```json
{
  "age": 45,
  "smoking": true,
  "yellow_fingers": false,
  "anxiety": false,
  "peer_pressure": false,
  "chronic_disease": false,
  "fatigue": false,
  "allergy": false,
  "wheezing": false,
  "alcohol_consuming": false,
  "coughing": false,
  "shortness_of_breath": false,
  "swallowing_difficulty": false,
  "chest_pain": false
}
```

#### **Response (Success)**  
```json
{
  "prediction_label": "No",
  "probability": 0.32
}
```

#### **Error Responses**  
| Code | Description | Sample Response |
|------|-------------|------------------|
| 400 | Invalid input | `{"detail": "Missing data - age"}` |
| 403 | Auth failure | `{"detail": "Invalid API Key"}` |
| 500 | Server error | `{"detail": "Prediction processing error"}` |
| 503 | Model error | `{"detail": "Model not loaded"}` |

---

## **4. Integration Examples**  

### **4.1 cURL**  
```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"age":45, "smoking":true, ..., "chest_pain":false}'
```

### **4.2 Python**  
```python
import requests

response = requests.post(
  "http://127.0.0.1:8000/predict/",
  headers={"X-API-Key": "your_key"},
  json={...}  # Input data
)
print(response.json())
```

### **4.3 JavaScript**  
```javascript
fetch("http://127.0.0.1:8000/predict/", {
  method: "POST",
  headers: { 
    "X-API-Key": "your_key",
    "Content-Type": "application/json"
  },
  body: JSON.stringify(inputData)
})
.then(res => res.json())
.then(console.log);
```

---

## **5. Deployment Guidelines**  

### **5.1 Production Hosting**  
Recommended platforms:  
- **Vercel** (for serverless)  
- **Heroku** (PaaS)  
- **AWS EC2** (full control)  

### **5.2 Security**  
🔐 **Must-do in production:**  
- Enable HTTPS  
- Implement rate limiting  
- Rotate API keys regularly  

### **5.3 Monitoring**  
Suggested tools:  
- FastAPI's built-in `/metrics`  
- Prometheus + Grafana  

---

## **6. Ethical & Compliance**  

### **6.1 Important Disclaimers**  
❗ **This is NOT a diagnostic tool**  
❗ Always recommend clinical verification  
❗ Probability scores <0.5 don't rule out cancer  

### **6.2 Data Privacy**  
- No persistent storage of input data  
- Compliant with GDPR for EU users  

---

---

## **7. Contact & Resources**  
- **Dataset**: [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)  
- **Repository:** [crbsdndr](https://github.com/crbsdndr/cancer_detection)  
- **Contact:** [yaveilhashou@gmail.com](mailto:yaveilhashou@gmail.com)  

**Developer**: Solo-developed by Dendra. No external collaborators.

**License:** Open Source  

---
