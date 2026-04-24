import pandas as pd
import io
from datetime import datetime

# Supported file types for upload
SUPPORTED_UPLOAD_TYPES = ['csv', 'xlsx', 'xls']

def parse_placement_dataset(uploaded_file):
    """
    Parse uploaded CSV/Excel file into DataFrame.
    Returns (df, warnings_list) or (None, error_list)
    """
    warnings = []

    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, ["Unsupported file type. Please upload CSV or Excel files."]

        # Basic validation
        if df.empty:
            return None, ["Uploaded file is empty."]

        if len(df) < 10:
            warnings.append("Dataset has fewer than 10 rows. Model training may not be reliable.")

        # Check for required columns (flexible naming)
        required_patterns = {
            'cgpa': ['cgpa', 'gpa', 'grade', 'score'],
            'internships': ['internships', 'internship', 'internship_count', 'internship_no'],
            'projects': ['projects', 'project', 'project_count', 'project_no'],
            'certifications': ['certifications', 'certification', 'cert_count', 'cert_no'],
            'communication_skill': ['communication_skill', 'communication', 'comm_skill', 'comm'],
            'coding_skill': ['coding_skill', 'coding', 'programming_skill', 'tech_skill'],
            'placed': ['placed', 'placement', 'placed_status', 'status', 'outcome'],
        }

        found_columns = {}
        df_columns_lower = df.columns.str.lower().str.strip()

        for req, patterns in required_patterns.items():
            found = False
            for pattern in patterns:
                if pattern in df_columns_lower.values:
                    found = True
                    col_idx = df_columns_lower[df_columns_lower == pattern].index[0]
                    found_columns[req] = df.columns[col_idx]
                    break
            if not found:
                warnings.append(f"Could not find column for '{req}'. Expected one of: {patterns}")

        if len(found_columns) < 4:  # Require at least 4 key columns
            return None, ["Insufficient columns found. Please ensure your dataset has student performance data."]

        # Clean column names
        df.columns = df.columns.str.strip()

        return df, warnings

    except Exception as e:
        return None, [f"Error parsing file: {str(e)}"]

def extract_companies_from_dataset(df):
    """
    Extract company information from placement dataset.
    Returns dict of {company_name: criteria_dict}
    """
    company_db = {}

    # Look for company-related columns
    company_cols = ['company', 'company_name', 'employer', 'organization']
    company_col = None

    df_columns_lower = df.columns.str.lower().str.strip()
    for col in company_cols:
        if col in df_columns_lower.values:
            col_idx = df_columns_lower[df_columns_lower == col].index[0]
            company_col = df.columns[col_idx]
            break

    if company_col is None:
        return company_db  # Return empty dict if no company column

    # Group by company and analyze placement criteria
    placed_df = df[df.get('placed', df.get('Placed', 0)) == 1]

    for company in df[company_col].dropna().unique():
        company_data = placed_df[placed_df[company_col] == company]

        if len(company_data) > 0:
            # Calculate average criteria for placed students
            criteria = {}
            if 'CGPA' in company_data.columns:
                criteria['min_cgpa'] = round(company_data['CGPA'].mean() - 0.5, 1)
            if 'Internships' in company_data.columns:
                criteria['min_internships'] = max(0, int(company_data['Internships'].mean() - 0.5))
            if 'Projects' in company_data.columns:
                criteria['min_projects'] = max(0, int(company_data['Projects'].mean() - 0.5))
            if 'Communication_Skill' in company_data.columns:
                criteria['min_comm_skill'] = max(1, int(company_data['Communication_Skill'].mean() - 0.5))
            if 'Coding_Skill' in company_data.columns:
                criteria['min_coding_skill'] = max(1, int(company_data['Coding_Skill'].mean() - 0.5))

            company_db[company] = criteria

    return company_db

def get_eligible_from_db(company_db, student_profile):
    """
    Get eligible companies for a student from the dynamic company database.
    """
    eligible = []

    for company, criteria in company_db.items():
        meets_criteria = True

        if 'min_cgpa' in criteria and student_profile.get('cgpa', 0) < criteria['min_cgpa']:
            meets_criteria = False
        if 'min_internships' in criteria and student_profile.get('internships', 0) < criteria['min_internships']:
            meets_criteria = False
        if 'min_projects' in criteria and student_profile.get('projects', 0) < criteria['min_projects']:
            meets_criteria = False
        if 'min_comm_skill' in criteria and student_profile.get('comm_skill', 0) < criteria['min_comm_skill']:
            meets_criteria = False
        if 'min_coding_skill' in criteria and student_profile.get('coding_skill', 0) < criteria['min_coding_skill']:
            meets_criteria = False

        if meets_criteria:
            eligible.append(company)

    return eligible

def compute_yearly_stats(df):
    """
    Compute year-over-year placement statistics.
    """
    stats = {}

    # Look for year column
    year_cols = ['year', 'academic_year', 'batch', 'graduation_year']
    year_col = None

    df_columns_lower = df.columns.str.lower().str.strip()
    for col in year_cols:
        if col in df_columns_lower.values:
            col_idx = df_columns_lower[df_columns_lower == col].index[0]
            year_col = df.columns[col_idx]
            break

    if year_col is None:
        return stats

    # Group by year
    yearly_data = df.groupby(year_col).agg({
        'Placed': ['count', 'sum', lambda x: (x.sum() / x.count() * 100).round(1)]
    }).round(0)

    yearly_data.columns = ['total_students', 'placed_count', 'placement_rate']

    # Convert to dict
    for year in yearly_data.index:
        stats[str(year)] = {
            'total_students': int(yearly_data.loc[year, 'total_students']),
            'placed_count': int(yearly_data.loc[year, 'placed_count']),
            'placement_rate': float(yearly_data.loc[year, 'placement_rate'])
        }

    return stats

def dataset_summary(df):
    """
    Generate summary statistics for the dataset.
    """
    total_students = len(df)
    placed_count = df.get('Placed', df.get('placed', pd.Series())).sum()
    placement_rate = (placed_count / total_students * 100) if total_students > 0 else 0

    # Department distribution
    dept_col = None
    dept_cols = ['department', 'dept', 'branch', 'specialization']
    df_columns_lower = df.columns.str.lower().str.strip()
    for col in dept_cols:
        if col in df_columns_lower.values:
            col_idx = df_columns_lower[df_columns_lower == col].index[0]
            dept_col = df.columns[col_idx]
            break

    department_stats = {}
    if dept_col:
        dept_counts = df[dept_col].value_counts()
        department_stats = dept_counts.to_dict()

    return {
        'total_students': total_students,
        'placed_count': int(placed_count),
        'placement_rate': round(placement_rate, 1),
        'departments': department_stats
    }

def generate_sample_csv():
    """
    Generate a sample CSV template for admins to download.
    """
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'Register_Number': ['URK22CS1001', 'URK22CS1002', 'URK22CS1003', 'URK22CS1004', 'URK22CS1005'],
        'CGPA': [8.5, 7.8, 9.2, 8.1, 7.5],
        'Internships': [2, 1, 3, 1, 0],
        'Projects': [3, 2, 4, 2, 1],
        'Certifications': [2, 1, 3, 1, 0],
        'Communication_Skill': [4, 3, 5, 4, 3],
        'Coding_Skill': [4, 4, 5, 3, 3],
        'Placed': [1, 1, 1, 0, 0],
        'Company': ['Google', 'Microsoft', 'Amazon', '', ''],
        'Department': ['B.Tech. Computer Science and Engineering'] * 5,
        'Year': [2024] * 5
    }

    df = pd.DataFrame(sample_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()