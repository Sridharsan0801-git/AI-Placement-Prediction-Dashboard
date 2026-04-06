"""
student_store.py — In-memory student submission store.

Every time a student completes a placement prediction, their data is saved here.
Admin can view all submissions from the Dataset Management page.

In a production system you would swap this for a database (SQLite / PostgreSQL).
For this Streamlit deployment, Streamlit session state is used as a shared store
via a module-level dict (persists as long as the Streamlit process is running).
"""

import json
from datetime import datetime

# Module-level store — shared across all Streamlit sessions in the same process
_STORE: list[dict] = []


def save_student_submission(user_data: dict, prediction_data: dict, resume_data: dict) -> None:
    """
    Save a student's prediction submission.

    Args:
        user_data:       st.session_state.user_data
        prediction_data: dict with cgpa, internships, projects, certifications,
                         comm, coding, probability, prediction_pct
        resume_data:     parsed resume dict (skills, cgpa, etc.)
    """
    record = {
        # Identity
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name":            user_data.get("name", ""),
        "email":           user_data.get("email", ""),
        "register_number": user_data.get("register_number", ""),
        "department":      user_data.get("department", ""),
        "current_year":    user_data.get("current_year", ""),
        "target_role":     user_data.get("target_role", ""),

        # Academic inputs
        "CGPA":                 prediction_data.get("cgpa", 0),
        "Internships":          prediction_data.get("internships", 0),
        "Projects":             prediction_data.get("projects", 0),
        "Certifications":       prediction_data.get("certifications", 0),
        "Communication_Skill":  prediction_data.get("comm", 0),
        "Coding_Skill":         prediction_data.get("coding", 0),

        # ML output
        "Placement_Probability_%": round(prediction_data.get("probability", 0) * 100, 1),
        "Prediction_Label":        prediction_data.get("label", ""),

        # Resume insights
        "Resume_Skills":         ", ".join(resume_data.get("skills", [])),
        "Resume_Skill_Count":    len(resume_data.get("skills", [])),
        "Resume_Word_Count":     resume_data.get("word_count", 0),
        "Resume_CGPA_Detected":  resume_data.get("cgpa", "N/A"),
    }
    _STORE.append(record)


def get_all_submissions() -> list[dict]:
    """Return all student submissions (admin only)."""
    return list(_STORE)


def get_submission_count() -> int:
    return len(_STORE)


def clear_store() -> None:
    """Admin utility to wipe all submissions."""
    global _STORE
    _STORE = []


def submissions_to_df():
    """Convert submissions to a pandas DataFrame for display."""
    import pandas as pd
    if not _STORE:
        return pd.DataFrame()
    return pd.DataFrame(_STORE)
