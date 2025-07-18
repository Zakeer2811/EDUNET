
# 🎯 EDUNET – Employee Salary Classification  
> **Internship Project**  
> Predict whether an individual earns **>50 K USD** or **≤50 K USD** per year using demographic and job-related features.

---

## 📌 Problem Statement
HR departments need a fast, data-driven way to estimate salary brackets for workforce planning.  
This project delivers a **machine-learning classifier** + **Streamlit web app** that instantly returns “>50 K” or “≤50 K” for single or batch employee records.

---

## 🧩 Dataset
- **Source:** UCI Adult Census Income (`adult.csv`)  
- **Size:** 32 K rows, 14 attributes  
- **Target:** `income` (binary: ≤50 K, >50 K)  

---

## 🚀 Live Demo
Try the web app locally:

```bash
git clone https://github.com/Zakeer2811/EDUNET.git
cd EDUNET
pip install -r requirements.txt
streamlit run app.py
````

---

## 🏗️ System Approach

1. **Data Cleaning** – Handle “?” values, clip outliers.
2. **Feature Engineering** – Encode categoricals, scale numerics.
3. **Modeling** – Benchmark 5 algorithms; **Gradient Boosting** wins.
4. **Pipeline** – One `Pipeline` object (scaler + model) → reproducible.
5. **Deployment** – Streamlit front-end + CSV batch upload.

---

## 📁 Repository Structure

```
EDUNET/
├── app.py                 # Streamlit web interface
├── best_model.pkl         # Trained pipeline
├── requirements.txt       # Python dependencies
├── data/
│   └── adult 3.csv       # Raw dataset
├── notebooks/             # Optional EDA / training notebook
└── README.md              # This file
```

---

## 🧪 Quick Start (Developers)

```bash
# 1. Setup env
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Run app
streamlit run app.py
```

---

## 📈 Model Performance (test set)

| Model            | Accuracy   |
| ---------------- | ---------- |
| GradientBoosting | **87.3 %** |
| RandomForest     | 86.1 %     |
| LogisticReg      | 84.9 %     |

---

## 🧰 Tech Stack

* **Python 3.9+**
* **scikit-learn** – modeling & preprocessing
* **pandas / numpy** – data wrangling
* **Streamlit** – UI
* **joblib** – model persistence

---

## 📄 API (CSV Batch)

Upload a CSV with the same 13 columns → receive an extra column `PredictedClass`.

---

