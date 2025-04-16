# Lung Cancer Prediction – Machine Learning Model

This project is a part of a larger initiative: building a **web-based lung cancer prediction platform**. This repository focuses on the **Machine Learning (ML) model** used for predicting lung cancer likelihood based on survey data.

## 📊 Dataset

The model is trained using the publicly available dataset:  
**[Lung Cancer Survey Dataset](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)**

To enhance the data and improve generalization, **synthetic data** was generated using advanced Large Language Models (LLMs), including:

- DeepSeek V3
- Gemini 2.5 Pro (Experimental)
- Claude 3.7 Sonnet
- Others

All LLM-generated samples were **manually evaluated** to ensure their **logical consistency and relevance** to real-world cases.

---

## 🧪 Data Validation

To ensure quality and fairness:

- Logical evaluation was conducted on each synthetic sample.
- A performance audit showed that **incorrect predictions** were **not predominantly caused** by synthetic samples.
- Most misclassified samples were **not from the original dataset**, indicating robust generalization.

---

## 🤖 Model Details

- **Algorithm**: Random Forest Classifier
- **Reason for selection**:
  - Handles **mixed-type data** (boolean + numeric)
  - **Robust to imbalanced classes**
- **Bias adjustment**:
  - We deliberately **increased the proportion of cancer-positive samples** to **reduce false negatives** (i.e., cases where cancer goes undetected)
  - This may increase false positives (healthy detected as cancer), but it's **safer in a medical context**

> ⚠️ **Disclaimer:**  
> This model may still make incorrect predictions. It should not be solely relied upon for medical diagnosis. Always consult with a qualified healthcare professional.

---

## ✅ Model Performance

| Metric              | Value     |
|---------------------|-----------|
| Accuracy            | 96%       |
| Mean Absolute Error | 0.45%     |

---

## 📦 License & Usage

- ✅ **Free to use**
- ✅ **Modifiable**
- ✅ **Commercial & educational usage allowed**
- ❌ **Resale prohibited** without **significant modifications**

---

## 🙏 Final Notes

Please use this model responsibly and cautiously. Although it demonstrates high accuracy, it's not a substitute for professional diagnosis. We encourage further improvements and welcome contributions.

---

