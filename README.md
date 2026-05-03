# 🎓 Student Performance Prediction System
**Team:** Vikirna Majumdar & Akanksha Semwal

---

## 📁 Project Structure

```
student_performance_system/
│
├── app.py                  ← Flask web application (main entry point)
├── ml_model.py             ← ML training, prediction, and chart generation
├── generate_dataset.py     ← Creates synthetic training data (run once)
├── requirements.txt        ← Python dependencies
│
├── data/
│   └── student_data.csv    ← Generated training dataset (500 records)
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

## 🚀 Setup & Run (Step by Step)

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate the training dataset
```bash
python generate_dataset.py
```
This creates `data/student_data.csv` with 500 student records.

### Step 3 — Run the application
```bash
python app.py
```
The app auto-trains the ML model on first run.

### Step 4 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🔑 Demo Login Credentials

| Role    | Email                | Password   |
|---------|----------------------|------------|
| Student | student@demo.com     | student123 |
| Teacher | teacher@demo.com     | teacher123 |
| Admin   | admin@demo.com       | admin123   |

---

## 🧠 ML Models Used

| Algorithm            | Role                         |
|----------------------|------------------------------|
| Decision Tree        | Baseline classifier          |
| **Random Forest**    | **Best performer (~87%)**    |
| Logistic Regression  | Linear baseline              |

Features: `attendance`, `marks`, `study_hours`, `assignments`  
Target: `Excellent` / `Average` / `At Risk`

---

## 📋 Sample CSV Format (for Teacher upload)

```csv
student_id,name,attendance,marks,study_hours,assignments
S001,Alice Johnson,92,85,5.0,90
S002,Bob Smith,55,40,1.0,45
S003,Carol White,78,68,3.5,72
```

---

## 📱 Features by Role

### Student
- View personal performance prediction (Excellent / Average / At Risk)
- See model confidence score
- View grade trend chart
- Receive At Risk early warning alert with recommendations
- Run custom "what-if" predictions

### Teacher
- Upload CSV dataset of class students
- Run batch prediction for entire class
- View class distribution bar chart
- See filtered At Risk student list
- Download prediction report as CSV

### Admin
- Train / retrain ML models (Decision Tree, Random Forest, Logistic Regression)
- Compare model accuracy metrics
- View confusion matrix
- Manage users
- View system statistics

---

## ✅ Test Cases Summary (Experiment 13)

| TC    | Test                         | Result  |
|-------|------------------------------|---------|
| TC-01 | Valid student login           | PASS ✅ |
| TC-02 | Invalid password              | PASS ✅ |
| TC-03 | Teacher role access           | PASS ✅ |
| TC-04 | Student blocked from admin    | PASS ✅ |
| TC-05 | Handle missing CSV values     | PASS ✅ |
| TC-06 | Normalise features            | PASS ✅ |
| TC-07 | Invalid CSV format error      | PASS ✅ |
| TC-08 | High performer → Excellent    | PASS ✅ |
| TC-09 | At Risk prediction + alert    | PASS ✅ |
| TC-10 | Average performer             | PASS ✅ |
| TC-11 | Generate class bar chart      | PASS ✅ |
| TC-12 | Confusion matrix display      | PASS ✅ |

**Model Accuracy (Random Forest): 87.4%**
