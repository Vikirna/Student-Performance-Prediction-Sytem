"""
app.py  —  Student Performance Prediction System
Flask web application entry point.

Run:
    python app.py
Then open:  http://127.0.0.1:5000
"""
import os
import io
import csv
import json
from functools import wraps
from datetime  import datetime

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, send_file)
import pandas as pd

from ml_model import (preprocess, validate_csv, train_models,
                      predict_single, predict_batch,
                      generate_bar_chart, generate_grade_trend_chart,
                      confusion_matrix_b64, FEATURES, COLORS)

# ─── App setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
app = Flask(__name__)
app.secret_key = 'spps-secret-key-2024'

# ─── In-memory "database" (replace with MySQL in production) ──────────────────
USERS = {
    'student@demo.com': {'password': 'student123', 'role': 'student',
                         'name': 'Akanksha Semwal',
                         'attendance': 78, 'marks': 72,
                         'study_hours': 3.5, 'assignments': 80},
    'teacher@demo.com': {'password': 'teacher123', 'role': 'teacher',
                         'name': 'Prof. Sharma'},
    'admin@demo.com':   {'password': 'admin123',   'role': 'admin',
                         'name': 'Admin User'},
}

BATCH_RESULTS   = []   # stores last batch prediction
TRAINING_STATS  = {}   # stores last training run stats
ALERTS          = []   # at-risk alerts


# ─── Auth helpers ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get('role') not in roles:
                flash('Access denied.', 'danger')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated
    return decorator


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        user     = USERS.get(email)
        if user and user['password'] == password:
            session['user'] = email
            session['role'] = user['role']
            session['name'] = user['name']
            flash(f"Welcome, {user['name']}!", 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        name     = request.form.get('name', '').strip()
        role     = request.form.get('role', 'student')
        if email in USERS:
            flash('Email already registered.', 'danger')
        else:
            USERS[email] = {'password': password, 'role': role,
                            'name': name,
                            'attendance': 75, 'marks': 65,
                            'study_hours': 3, 'assignments': 70}
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/dashboard')
@login_required
def dashboard():
    role = session['role']
    if role == 'student':
        return redirect(url_for('student_dashboard'))
    elif role == 'teacher':
        return redirect(url_for('teacher_dashboard'))
    else:
        return redirect(url_for('admin_dashboard'))


# ── Student ───────────────────────────────────────────────────────────────────

@app.route('/student')
@login_required
@role_required('student')
def student_dashboard():
    user  = USERS[session['user']]
    cat, conf = predict_single(user['attendance'], user['marks'],
                               user['study_hours'], user['assignments'])
    trend = generate_grade_trend_chart(user['name'])
    return render_template('student_dashboard.html',
                           user=user, category=cat,
                           confidence=conf, trend_chart=trend,
                           colors=COLORS)


# ── Teacher ───────────────────────────────────────────────────────────────────

@app.route('/teacher')
@login_required
@role_required('teacher', 'admin')
def teacher_dashboard():
    bar_chart = generate_bar_chart(pd.DataFrame(BATCH_RESULTS)) \
                if BATCH_RESULTS else None
    at_risk   = [r for r in BATCH_RESULTS if r.get('category') == 'At Risk']
    return render_template('teacher_dashboard.html',
                           results=BATCH_RESULTS,
                           bar_chart=bar_chart,
                           at_risk=at_risk,
                           alerts=ALERTS,
                           colors=COLORS)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
@role_required('teacher', 'admin')
def upload():
    global BATCH_RESULTS, ALERTS
    if request.method == 'POST':
        f = request.files.get('csv_file')
        if not f or not f.filename.endswith('.csv'):
            flash('Please upload a valid CSV file.', 'danger')
            return redirect(url_for('upload'))
        try:
            df = pd.read_csv(f)
            missing = validate_csv(df)
            if missing:
                flash(f"Required column(s) missing: {', '.join(missing)}", 'danger')
                return redirect(url_for('upload'))
            df = predict_batch(df)
            BATCH_RESULTS = df.to_dict(orient='records')
            # Generate alerts
            ALERTS = [{'name': r.get('name', r.get('student_id', '?')),
                       'student_id': r.get('student_id', '?'),
                       'confidence': r.get('confidence', 0),
                       'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')}
                      for r in BATCH_RESULTS if r.get('category') == 'At Risk']
            flash(f'Batch prediction complete for {len(BATCH_RESULTS)} students.', 'success')
            return redirect(url_for('teacher_dashboard'))
        except Exception as e:
            flash(f'Error processing file: {e}', 'danger')
    return render_template('upload.html')


@app.route('/download_report')
@login_required
def download_report():
    if not BATCH_RESULTS:
        flash('No prediction results available.', 'warning')
        return redirect(url_for('teacher_dashboard'))
    df = pd.DataFrame(BATCH_RESULTS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name='prediction_report.csv')


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.route('/admin')
@login_required
@role_required('admin')
def admin_dashboard():
    cm = confusion_matrix_b64()
    return render_template('admin_dashboard.html',
                           stats=TRAINING_STATS,
                           users=USERS,
                           cm_chart=cm)


@app.route('/train', methods=['POST'])
@login_required
@role_required('admin')
def train():
    global TRAINING_STATS
    try:
        TRAINING_STATS = train_models()
        flash(f"Model trained! Best: {TRAINING_STATS['best_model']} "
              f"({TRAINING_STATS['best_accuracy']}% accuracy)", 'success')
    except Exception as e:
        flash(f'Training error: {e}', 'danger')
    return redirect(url_for('admin_dashboard'))


# ── Predict (single, AJAX) ────────────────────────────────────────────────────

@app.route('/predict_single', methods=['POST'])
@login_required
def predict_single_route():
    data  = request.get_json()
    cat, conf = predict_single(
        float(data.get('attendance', 75)),
        float(data.get('marks', 65)),
        float(data.get('study_hours', 3)),
        float(data.get('assignments', 70))
    )
    return jsonify({'category': cat, 'confidence': conf})


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Auto-train if no model exists
    from ml_model import MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        print("No model found — training now...")
        try:
            stats = train_models()
            print(f"Model ready: {stats['best_model']} ({stats['best_accuracy']}%)")
            TRAINING_STATS = stats
        except Exception as e:
            print(f"Auto-train failed: {e} — run generate_dataset.py first")
    app.run(debug=True, port=5000)