# **Lung Cancer Prediction Model: MLP_Enhanced_v2**  
**Documentation Version: 2.0**  
**Last Updated: 22/04/2025**  

---

## **1. Introduction**  

### **1.1 Background**  
The **MLP_Enhanced_v2** model is a neural network-based classifier designed to predict lung cancer risk based on demographic, behavioral, and symptomatic features. It targets underserved populations with limited access to advanced diagnostic tools, providing an initial risk assessment to encourage further medical evaluation.  

While the model architecture supports multi-class cancer prediction, only **lung cancer detection** has been implemented and deployed. Users input 14 clinical and lifestyle features, and the model outputs a binary prediction (**Yes/No**) with an associated probability score.  

### **1.2 Problem Statement**  
Early detection of lung cancer significantly improves survival rates, yet many communities lack access to diagnostic infrastructure. This model bridges the gap by offering a preliminary screening tool based on easily obtainable features such as:  
- **Demographics** (age)  
- **Behavioral factors** (smoking, alcohol consumption)  
- **Symptoms** (coughing, chest pain, shortness of breath)  

---

## **2. Dataset & Preprocessing**  

### **2.1 Data Sources**  
| Dataset Type | Source | Samples | Features | Notes |  
|-------------|--------|---------|----------|-------|  
| **Original** | [Kaggle: Lung Cancer Dataset](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) | ~300 | 16 | "GENDER" feature dropped |  
| **Synthetic** | Generated via **Deepseek R1** (671B params) | ~800 | 15 | Augmented to improve generalizability |  

### **2.2 Features Used**  
The model processes **14 input features**, all boolean (True/False) except `age` (float):  

1. **Lifestyle Factors**:  
   - `smoking`, `alcohol_consuming`, `peer_pressure`  
2. **Symptoms**:  
   - `yellow_fingers`, `wheezing`, `coughing`, `chest_pain`, `shortness_of_breath`, `swallowing_difficulty`  
3. **Medical History**:  
   - `chronic_disease`, `allergy`, `fatigue`, `anxiety`  

### **2.3 Preprocessing Pipeline**  
1. **Cleaning**:  
   - Dropped `GENDER` (irrelevant to prediction).  
   - Standardized column names (snake_case).  
   - Removed duplicates.  
2. **Scaling**: Applied `MinMaxScaler` to normalize numerical features.  
3. **Encoding**: Used `LabelEncoder` for binary features.  

*No major data quality issues were encountered.*  

---

## **3. Model Development**  

### **3.1 Algorithm Selection**  
**Model**: Enhanced Multilayer Perceptron (**MLP_Enhanced_v2**)  
**Rationale**:  
- Captures **non-linear relationships** better than linear models (e.g., Logistic Regression).  
- Flexible architecture (adjustable layers, regularization).  
- Supports **class imbalance mitigation** via dynamic weighting.  

### **3.2 Training Protocol**  
- **Data Splits**: 60% train, 20% validation, 20% test.  
- **Optimization**:  
  - Loss: Binary cross-entropy.  
  - Learning rate: Cosine decay with restarts.  
  - Early stopping (monitored by **AUC**).  
- **Class Handling**:  
  - Class weighting (1.25× bias for minority class).  
  - Label smoothing (α=0.2) to reduce overfitting.  

### **3.3 Hyperparameters**  
| Parameter | Value |  
|-----------|-------|  
| Hidden Layers | 3 (128, 64, 32 units) |  
| Dropout Rate | 0.3 |  
| Regularization | L1 (0.01), L2 (0.02) |  
| Batch Size | 32 |  

*(Manually tuned; no AutoML used.)*  

---

## **4. Performance Evaluation**  

### **4.1 Metrics**  
Primary: **AUC**, **Accuracy**, **Loss**.  
Secondary: **Precision at Recall=0.8** (imbalance-aware).  

### **4.2 Results**  
| Dataset | Accuracy | AUC | Loss |  
|---------|----------|-----|------|  
| Train | 91.6% | 0.955 | 0.568 |  
| Validation | 89.7% | 0.993 | 0.636 |  
| Test | 92.4% | 0.992 | 0.583 |  

**External Test (40% original Kaggle data)**: Performance consistent with validation.  

### **4.3 Limitations**  
- **High loss values** suggest suboptimal convergence.  
- **Synthetic data dependency**: May not fully reflect real-world distributions.  
- **Interpretability**: MLPs are "black-box" models.  

---

## **5. Deployment & Usage**  

### **5.1 Access**  
- **Web App**: [Cancer Analysis Project](https://projek-analisis-kanker.vercel.app/)  
- **API Docs**: [GitHub](https://github.com/crbsdndr/cancer_detection/blob/main/documentations/api.md)  
- **Client Script**: [client.py](https://github.com/crbsdndr/cancer_detection/blob/main/routes/client.py)  

### **5.2 Input/Output**  
- **Input**: 14 features (e.g., `{"age": 45, "smoking": True, ...}`).  
- **Output**: `{"prediction": "Yes", "probability": 0.87}`.  

### **5.3 Ethical Considerations**  
- **Not a diagnostic tool**: Always recommend clinical confirmation.  
- **Bias Risk**: Synthetic data may not represent all demographics.  

---

## **6. Future Work**  
1. **Expand Cancer Types**: Breast, prostate, etc.  
2. **Replace Synthetic Data**: Partner with clinics for real-world surveys.  
3. **Explainability**: Add SHAP/LIME for interpretability.  
4. **Loss Reduction**: Experiment with architectures (e.g., ResNet blocks).  

---

## **7. Contact & Resources**  
- **Dataset**: [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)  
- **Repository:** [crbsdndr](https://github.com/crbsdndr/cancer_detection)  
- **Contact:** [yaveilhashou@gmail.com](mailto:yaveilhashou@gmail.com)  

**Developer**: Solo-developed by Dendra. No external collaborators.

**License:** Open Source  

---
