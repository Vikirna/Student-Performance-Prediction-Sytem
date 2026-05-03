"""
generate_dataset.py
Creates a synthetic student_data.csv for training the ML model.
Run this once before starting the app.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 500

attendance   = np.random.uniform(30, 100, N)
marks        = np.random.uniform(20, 100, N)
study_hours  = np.random.uniform(0.5, 8,  N)
assignments  = np.random.uniform(30, 100, N)

def label(att, mrk, hrs, asgn):
    score = (att * 0.3) + (mrk * 0.4) + (hrs * 3) + (asgn * 0.1)
    if score >= 68:
        return 'Excellent'
    elif score >= 48:
        return 'Average'
    else:
        return 'At Risk'

categories = [label(a, m, h, s)
              for a, m, h, s in zip(attendance, marks, study_hours, assignments)]

df = pd.DataFrame({
    'student_id':   [f'S{str(i).zfill(3)}' for i in range(1, N + 1)],
    'name':         [f'Student_{i}'         for i in range(1, N + 1)],
    'attendance':   np.round(attendance,  1),
    'marks':        np.round(marks,       1),
    'study_hours':  np.round(study_hours, 1),
    'assignments':  np.round(assignments, 1),
    'category':     categories
})

out = os.path.join(os.path.dirname(__file__), 'data', 'student_data.csv')
df.to_csv(out, index=False)
print(f"Dataset saved → {out}  ({N} records)")
print(df['category'].value_counts().to_string())
