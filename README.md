# 🎓 Student Performance Prediction System

A machine learning-based web application that predicts student academic performance using key parameters like attendance, marks, study hours, and assignment scores. Built as part of the **Software Engineering & Project Management (SEPM) Lab**.

---

## 🔗 Live Demo

 **[https://student-performance-prediction-sytem.onrender.com](https://student-performance-prediction-sytem.onrender.com)**

>  Hosted on Render free tier — may take 30–60 seconds to wake up on first visit.

---

## Team

| Name | Role |
|------|------|
| Vikirna Majumdar | Frontend Developer & ML Model Trainer |
| Akanksha Semwal | Backend Developer & Data Analyst |

---

## About the Project

Educational institutions often struggle to identify at-risk students before it is too late to intervene. This system uses machine learning to analyze student data and predict their performance category — **Excellent**, **Average**, or **At Risk** — enabling early intervention and data-driven academic support.

---

## Features

-  Role-based login — Student, Teacher, Admin
-  Student dashboard with performance prediction and confidence score
-  Grade trend charts generated with Matplotlib
-  Early warning alerts for At Risk students with recommended actions
-  CSV upload for batch prediction (Teacher)
-  Class-wide prediction report with bar chart
-  Three ML models trained and compared — Decision Tree, Random Forest, Logistic Regression
-  Admin panel to retrain models and manage users
-  Download prediction report as CSV

---

## Machine Learning

| Algorithm | Test Accuracy | CV Accuracy |
|-----------|--------------|-------------|
| Decision Tree | 87% | 81% |
| Random Forest | 86% | 87% |
| **Logistic Regression** | **93%** | **93%** |

**Features used:** `attendance`, `marks`, `study_hours`, `assignments`  
**Target classes:** `Excellent` / `Average` / `At Risk`  
**Library:** Scikit-learn

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, Jinja2 Templates |
| Backend | Python, Flask |
| ML | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib |
| Database | MySQL (in-memory for demo) |
| Deployment | Render |
| Version Control | Git & GitHub |

---

## Demo Login Credentials

| Role | Email | Password |
|------|-------|----------|
| Student | student@demo.com | student123 |
| Teacher | teacher@demo.com | teacher123 |
| Admin | admin@demo.com | admin123 |

---

## Project Structure

```
student_performance_system/
│
├── app.py                  ← Flask web app (main entry point)
├── ml_model.py             ← ML training, prediction, chart generation
├── generate_dataset.py     ← Generates synthetic training data
├── wsgi.py                 ← Gunicorn entry point for deployment
├── Procfile                ← Render deployment config
├── runtime.txt             ← Python version (3.11.0)
├── requirements.txt        ← Python dependencies
│
├── data/
│   └── student_data.csv    ← 500-record training dataset
│
├── models/
│   ├── best_model.pkl      ← Saved trained ML model
│   └── scaler.pkl          ← Saved MinMaxScaler
│
├── reports/
│   └── confusion_matrix.png
│
└── templates/
    ├── base.html
    ├── login.html
    ├── register.html
    ├── student_dashboard.html
    ├── teacher_dashboard.html
    ├── upload.html
    └── admin_dashboard.html
```

---

## Run Locally

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/Vikirna/Student-Performance-Prediction-Sytem.git
cd Student-Performance-Prediction-Sytem
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Generate training dataset:**
```bash
python generate_dataset.py
```

**Step 4 — Run the app:**
```bash
python app.py
```

**Step 5 — Open in browser:**
```
http://127.0.0.1:5000
```

---

## CSV Upload Format (for Teacher)

```csv
student_id,name,attendance,marks,study_hours,assignments
S001,Alice Johnson,92,85,5.0,90
S002,Bob Smith,55,40,1.0,45
S003,Carol White,78,68,3.5,72
```

---

## Test Results

| TC ID | Test Case | Status |
|-------|-----------|--------|
| TC-01 | Valid student login | ✅ PASS |
| TC-02 | Invalid password | ✅ PASS |
| TC-03 | Teacher role access | ✅ PASS |
| TC-04 | Student blocked from admin | ✅ PASS |
| TC-05 | Handle missing CSV values | ✅ PASS |
| TC-06 | Normalise features | ✅ PASS |
| TC-07 | Invalid CSV format error | ✅ PASS |
| TC-08 | High performer → Excellent | ✅ PASS |
| TC-09 | At Risk prediction + alert | ✅ PASS |
| TC-10 | Average performer | ✅ PASS |
| TC-11 | Generate class bar chart | ✅ PASS |
| TC-12 | Confusion matrix display | ✅ PASS |

**Model Accuracy: 93% (Logistic Regression)**  
**Backend Code Coverage: 81%**

---

## 📄 License

This project is built for academic purposes as part of the SEPM Lab curriculum.
