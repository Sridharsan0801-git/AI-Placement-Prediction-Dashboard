import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = ['CGPA', 'Internships', 'Projects', 'Certifications', 'Communication_Skill', 'Coding_Skill']

def generate_sample_data(n_samples=600):
    np.random.seed(42)
    data = {
        'CGPA': np.clip(np.random.normal(7.5, 1.2, n_samples), 5.0, 10.0),
        'Internships': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.2, 0.35, 0.25, 0.15, 0.05]),
        'Projects': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]),
        'Certifications': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05]),
        'Communication_Skill': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.35, 0.30, 0.15]),
        'Coding_Skill': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.30, 0.25, 0.15]),
        'Department': np.random.choice([
            'B.Tech. Computer Science and Engineering',
            'B.Tech. Artificial Intelligence and Data Science',
            'B.Tech. Electronics and Communication Engineering',
            'B.Tech. Mechanical Engineering',
            'B.Tech. Civil Engineering',
            'B.Tech. Electrical and Electronics Engineering',
        ], n_samples)
    }
    df = pd.DataFrame(data)
    score = (
        df['CGPA'] * 0.30 +
        df['Internships'] * 0.15 +
        df['Projects'] * 0.12 +
        df['Certifications'] * 0.08 +
        df['Communication_Skill'] * 0.15 +
        df['Coding_Skill'] * 0.20
    )
    score += np.random.normal(0, 0.4, n_samples)
    threshold = np.percentile(score, 30)
    df['Placed'] = (score > threshold).astype(int)
    return df

def train_model(df):
    X = df[FEATURE_COLS]
    y = df['Placed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    }
    best_model, best_acc, best_name = None, 0, ''
    for name, m in candidates.items():
        m.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, m.predict(X_test_s))
        if acc > best_acc:
            best_acc, best_model, best_name = acc, m, name

    y_pred = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return best_model, scaler, best_acc, best_name, cm, report, X_test_s, y_test

def predict(model, scaler, cgpa, internships, projects, certs, comm, coding):
    features = [[cgpa, internships, projects, certs, comm, coding]]
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]
    return prob

def prepare_uploaded_dataset(df):
    """
    Prepare uploaded dataset for ML training by mapping columns to expected format.
    Handles common column name variations and ensures proper data types.
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # Define column mapping (uploaded column -> expected column)
    column_mappings = {
        'cgpa': 'CGPA',
        'CGPA': 'CGPA',
        'gpa': 'CGPA',
        'grade': 'CGPA',
        'score': 'CGPA',

        'internships': 'Internships',
        'Internships': 'Internships',
        'internship': 'Internships',
        'internship_count': 'Internships',
        'internship_no': 'Internships',

        'projects': 'Projects',
        'Projects': 'Projects',
        'project': 'Projects',
        'project_count': 'Projects',
        'project_no': 'Projects',

        'certifications': 'Certifications',
        'Certifications': 'Certifications',
        'certification': 'Certifications',
        'cert_count': 'Certifications',
        'cert_no': 'Certifications',

        'communication_skill': 'Communication_Skill',
        'Communication_Skill': 'Communication_Skill',
        'communication': 'Communication_Skill',
        'comm_skill': 'Communication_Skill',
        'comm': 'Communication_Skill',

        'coding_skill': 'Coding_Skill',
        'Coding_Skill': 'Coding_Skill',
        'coding': 'Coding_Skill',
        'programming_skill': 'Coding_Skill',
        'tech_skill': 'Coding_Skill',

        'placed': 'Placed',
        'Placed': 'Placed',
        'placement': 'Placed',
        'placed_status': 'Placed',
        'status': 'Placed',
        'outcome': 'Placed',
    }

    # Rename columns based on mapping
    df.columns = df.columns.str.lower().str.strip()
    rename_dict = {}
    for col in df.columns:
        if col in column_mappings:
            rename_dict[col] = column_mappings[col]

    df = df.rename(columns=rename_dict)

    # Ensure required columns exist
    required_cols = FEATURE_COLS + ['Placed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert data types
    df['CGPA'] = pd.to_numeric(df['CGPA'], errors='coerce').clip(5.0, 10.0)
    df['Internships'] = pd.to_numeric(df['Internships'], errors='coerce').astype(int).clip(0, 4)
    df['Projects'] = pd.to_numeric(df['Projects'], errors='coerce').astype(int).clip(0, 5)
    df['Certifications'] = pd.to_numeric(df['Certifications'], errors='coerce').astype(int).clip(0, 5)
    df['Communication_Skill'] = pd.to_numeric(df['Communication_Skill'], errors='coerce').astype(int).clip(1, 5)
    df['Coding_Skill'] = pd.to_numeric(df['Coding_Skill'], errors='coerce').astype(int).clip(1, 5)
    df['Placed'] = pd.to_numeric(df['Placed'], errors='coerce').astype(int).clip(0, 1)

    # Drop rows with NaN values
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    return df
