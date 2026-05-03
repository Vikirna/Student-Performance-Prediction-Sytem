"""
ml_model.py
Handles data preprocessing, model training, prediction, and report generation.
"""
import os
import pickle
import io
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble           import RandomForestClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import LabelEncoder, MinMaxScaler
from sklearn.metrics            import (confusion_matrix, classification_report,
                                        accuracy_score)

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH= os.path.join(MODEL_DIR, 'scaler.pkl')

FEATURES   = ['attendance', 'marks', 'study_hours', 'assignments']
TARGET     = 'category'
COLORS     = {'Excellent': '#28A745', 'Average': '#FFC107', 'At Risk': '#DC3545'}


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise a student dataframe."""
    df = df.copy()
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    return df


def validate_csv(df: pd.DataFrame):
    """Return list of missing required columns."""
    required = set(FEATURES)
    missing  = required - set(df.columns)
    return list(missing)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_models(csv_path: str = None):
    """Train Decision Tree, Random Forest, and Logistic Regression.
       Saves best model to disk. Returns dict with metrics."""
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, 'student_data.csv')

    df = pd.read_csv(csv_path)
    df = preprocess(df)

    X = df[FEATURES]
    y = df[TARGET]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        'Decision Tree':      DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':      RandomForestClassifier(n_estimators=150, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    }

    results = {}
    best_acc  = 0
    best_name = ''
    best_clf  = None

    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(clf, X_scaled, y, cv=5).mean()
        results[name] = {'accuracy': round(acc * 100, 2),
                         'cv_accuracy': round(cv * 100, 2)}
        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_clf  = clf

    # Save best model + scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH,  'wb') as f: pickle.dump(best_clf, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)

    # Save confusion matrix image
    y_pred_best = best_clf.predict(X_test)
    _save_confusion_matrix(y_test, y_pred_best, best_name)

    return {
        'best_model':    best_name,
        'best_accuracy': round(best_acc * 100, 2),
        'all_results':   results,
        'report':        classification_report(y_test, y_pred_best, output_dict=True),
    }


def _save_confusion_matrix(y_true, y_pred, model_name):
    labels = ['Excellent', 'Average', 'At Risk']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig.savefig(os.path.join(REPORT_DIR, 'confusion_matrix.png'), dpi=100)
    plt.close(fig)


# ─── Prediction ───────────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH,  'rb') as f: clf    = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    return clf, scaler


def predict_single(attendance, marks, study_hours, assignments):
    clf, scaler = load_model()
    if clf is None:
        return None, None
    X = pd.DataFrame([[attendance, marks, study_hours, assignments]],
                     columns=FEATURES, dtype=float)
    X_scaled = scaler.transform(X)
    category = clf.predict(X_scaled)[0]
    proba    = clf.predict_proba(X_scaled)[0]
    classes  = list(clf.classes_)
    confidence = round(proba[classes.index(category)] * 100, 1)
    return category, confidence


def predict_batch(df: pd.DataFrame):
    clf, scaler = load_model()
    if clf is None:
        return df
    df = preprocess(df)
    X = df[FEATURES].values
    X_scaled = scaler.transform(X)
    df['category']   = clf.predict(X_scaled)
    proba = clf.predict_proba(X_scaled)
    classes = list(clf.classes_)
    df['confidence'] = [
        round(proba[i][classes.index(df['category'].iloc[i])] * 100, 1)
        for i in range(len(df))
    ]
    return df


# ─── Charts ───────────────────────────────────────────────────────────────────

def chart_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_b64


def generate_bar_chart(df: pd.DataFrame) -> str:
    counts = df['category'].value_counts()
    cats   = ['Excellent', 'Average', 'At Risk']
    vals   = [counts.get(c, 0) for c in cats]
    colors = [COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('Class Performance Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Students')
    ax.set_ylim(0, max(vals) * 1.2 + 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#F5F7FA')
    ax.set_facecolor('#F5F7FA')
    plt.tight_layout()
    return chart_to_base64(fig)


def generate_grade_trend_chart(student_name: str) -> str:
    """Simulated grade trend (in production, pull from DB)."""
    np.random.seed(abs(hash(student_name)) % 1000)
    semesters = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']
    grades    = np.clip(np.cumsum(np.random.randn(4) * 5) + 65, 30, 100)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(semesters, grades, marker='o', color='#2196F3',
            linewidth=2.5, markersize=8)
    for i, (s, g) in enumerate(zip(semesters, grades)):
        ax.annotate(f'{g:.0f}', (s, g), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=10)
    ax.set_title(f'Grade Trend — {student_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Marks')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#F5F7FA')
    ax.set_facecolor('#F5F7FA')
    plt.tight_layout()
    return chart_to_base64(fig)


def confusion_matrix_b64() -> str:
    path = os.path.join(REPORT_DIR, 'confusion_matrix.png')
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()
