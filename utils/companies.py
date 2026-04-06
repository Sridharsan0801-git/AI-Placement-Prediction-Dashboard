"""
companies.py — Company criteria with roles, CGPA cut-offs, and tier data.
"""

ALL_ROLES = [
    "Software Developer",
    "Data Analyst",
    "Data Scientist",
    "ML Engineer",
    "Cloud Engineer",
    "Backend Developer",
    "Full Stack Developer",
    "DevOps Engineer",
    "Embedded Systems Engineer",
    "Cybersecurity Analyst",
    "Business Analyst",
    "Consulting Analyst",
    "Product Manager",
    "iOS / Mobile Developer",
    "Hardware / VLSI Engineer",
]

COMPANY_CRITERIA = {
    # ── Tier 1 — Dream ───────────────────────────────────────────
    'Google': {
        'min_cgpa': 8.5, 'package': '₹25–45 LPA',
        'skills': ['DSA', 'System Design', 'Problem Solving'],
        'roles': ['Software Developer', 'Data Scientist', 'ML Engineer', 'Backend Developer', 'Full Stack Developer'],
        'type': 'Product', 'tier': 1, 'logo_color': '#4285F4',
    },
    'Microsoft': {
        'min_cgpa': 8.0, 'package': '₹20–40 LPA',
        'skills': ['DSA', 'OOP', 'System Design', 'Cloud'],
        'roles': ['Software Developer', 'Cloud Engineer', 'Backend Developer', 'Full Stack Developer', 'DevOps Engineer'],
        'type': 'Product', 'tier': 1, 'logo_color': '#00A4EF',
    },
    'Amazon': {
        'min_cgpa': 8.0, 'package': '₹20–35 LPA',
        'skills': ['DSA', 'System Design', 'Leadership Principles'],
        'roles': ['Software Developer', 'Cloud Engineer', 'Backend Developer', 'DevOps Engineer', 'Data Scientist'],
        'type': 'Product', 'tier': 1, 'logo_color': '#FF9900',
    },
    'Meta': {
        'min_cgpa': 8.5, 'package': '₹25–50 LPA',
        'skills': ['DSA', 'System Design', 'ML/AI'],
        'roles': ['Software Developer', 'ML Engineer', 'Data Scientist', 'Full Stack Developer', 'Backend Developer'],
        'type': 'Product', 'tier': 1, 'logo_color': '#0866FF',
    },
    'Apple': {
        'min_cgpa': 8.5, 'package': '₹22–40 LPA',
        'skills': ['System Design', 'iOS/Swift or Hardware', 'DSA'],
        'roles': ['Software Developer', 'iOS / Mobile Developer', 'Hardware / VLSI Engineer', 'Backend Developer'],
        'type': 'Product', 'tier': 1, 'logo_color': '#555555',
    },
    'Adobe': {
        'min_cgpa': 8.0, 'package': '₹18–30 LPA',
        'skills': ['DSA', 'Web Technologies', 'Problem Solving'],
        'roles': ['Software Developer', 'Full Stack Developer', 'Backend Developer', 'Data Analyst'],
        'type': 'Product', 'tier': 1, 'logo_color': '#FF0000',
    },
    'Atlassian': {
        'min_cgpa': 8.0, 'package': '₹18–32 LPA',
        'skills': ['DSA', 'System Design', 'Cloud'],
        'roles': ['Software Developer', 'DevOps Engineer', 'Backend Developer', 'Full Stack Developer'],
        'type': 'Product', 'tier': 1, 'logo_color': '#0052CC',
    },
    'Goldman Sachs': {
        'min_cgpa': 8.0, 'package': '₹18–32 LPA',
        'skills': ['DSA', 'Quantitative Aptitude', 'Python/Java'],
        'roles': ['Software Developer', 'Data Analyst', 'Business Analyst', 'Data Scientist'],
        'type': 'FinTech', 'tier': 1, 'logo_color': '#5F9EA0',
    },

    # ── Tier 2 — Target ──────────────────────────────────────────
    'Flipkart': {
        'min_cgpa': 7.5, 'package': '₹15–25 LPA',
        'skills': ['DSA', 'System Design', 'Backend Development'],
        'roles': ['Software Developer', 'Backend Developer', 'Full Stack Developer', 'Data Analyst'],
        'type': 'E-Commerce', 'tier': 2, 'logo_color': '#F74F00',
    },
    'Walmart Global Tech': {
        'min_cgpa': 7.5, 'package': '₹14–22 LPA',
        'skills': ['DSA', 'Java/Python', 'Cloud'],
        'roles': ['Software Developer', 'Cloud Engineer', 'Data Analyst', 'DevOps Engineer'],
        'type': 'Retail Tech', 'tier': 2, 'logo_color': '#0071CE',
    },
    'Zoho': {
        'min_cgpa': 7.5, 'package': '₹8–15 LPA',
        'skills': ['Java', 'Problem Solving', 'Aptitude'],
        'roles': ['Software Developer', 'Full Stack Developer', 'Backend Developer', 'Product Manager'],
        'type': 'Product', 'tier': 2, 'logo_color': '#E42527',
    },
    'Razorpay': {
        'min_cgpa': 7.5, 'package': '₹14–22 LPA',
        'skills': ['DSA', 'Backend Development', 'System Design'],
        'roles': ['Software Developer', 'Backend Developer', 'Data Analyst', 'DevOps Engineer'],
        'type': 'FinTech', 'tier': 2, 'logo_color': '#2D9CDB',
    },
    'Swiggy': {
        'min_cgpa': 7.5, 'package': '₹14–22 LPA',
        'skills': ['DSA', 'Backend Development', 'System Design'],
        'roles': ['Software Developer', 'Backend Developer', 'Data Analyst', 'ML Engineer'],
        'type': 'Product', 'tier': 2, 'logo_color': '#FC8019',
    },
    'Paytm': {
        'min_cgpa': 7.0, 'package': '₹10–18 LPA',
        'skills': ['DSA', 'Fintech', 'Mobile Development'],
        'roles': ['Software Developer', 'iOS / Mobile Developer', 'Backend Developer', 'Data Analyst'],
        'type': 'FinTech', 'tier': 2, 'logo_color': '#002970',
    },
    'Deloitte': {
        'min_cgpa': 7.0, 'package': '₹6–12 LPA',
        'skills': ['Analytics', 'Consulting', 'Domain Knowledge'],
        'roles': ['Business Analyst', 'Data Analyst', 'Consulting Analyst', 'Data Scientist'],
        'type': 'Consulting', 'tier': 2, 'logo_color': '#86BC25',
    },
    'KPMG': {
        'min_cgpa': 7.0, 'package': '₹6–11 LPA',
        'skills': ['Analytics', 'Excel', 'Communication'],
        'roles': ['Business Analyst', 'Data Analyst', 'Consulting Analyst'],
        'type': 'Consulting', 'tier': 2, 'logo_color': '#00338D',
    },
    'ThoughtWorks': {
        'min_cgpa': 7.5, 'package': '₹10–18 LPA',
        'skills': ['OOP', 'Agile', 'DSA'],
        'roles': ['Software Developer', 'Full Stack Developer', 'DevOps Engineer'],
        'type': 'Service', 'tier': 2, 'logo_color': '#F90084',
    },

    # ── Tier 3 — Safe ────────────────────────────────────────────
    'Infosys': {
        'min_cgpa': 7.0, 'package': '₹4–8 LPA',
        'skills': ['Java/Python', 'SQL', 'Communication'],
        'roles': ['Software Developer', 'Data Analyst', 'Business Analyst', 'Backend Developer'],
        'type': 'Service', 'tier': 3, 'logo_color': '#007CC3',
    },
    'TCS': {
        'min_cgpa': 6.5, 'package': '₹3.5–7 LPA',
        'skills': ['Programming Basics', 'Aptitude', 'Communication'],
        'roles': ['Software Developer', 'Business Analyst', 'Data Analyst', 'Consulting Analyst'],
        'type': 'Service', 'tier': 3, 'logo_color': '#1C4E9F',
    },
    'Wipro': {
        'min_cgpa': 6.5, 'package': '₹3.5–7 LPA',
        'skills': ['Programming Basics', 'SQL', 'Communication'],
        'roles': ['Software Developer', 'Data Analyst', 'Backend Developer', 'Cybersecurity Analyst'],
        'type': 'Service', 'tier': 3, 'logo_color': '#341C68',
    },
    'Cognizant': {
        'min_cgpa': 6.5, 'package': '₹4–8 LPA',
        'skills': ['Java/.NET', 'SQL', 'Soft Skills'],
        'roles': ['Software Developer', 'Data Analyst', 'Business Analyst', 'Full Stack Developer'],
        'type': 'Service', 'tier': 3, 'logo_color': '#1464A4',
    },
    'Capgemini': {
        'min_cgpa': 6.5, 'package': '₹4–8 LPA',
        'skills': ['Programming', 'Cloud Basics', 'Communication'],
        'roles': ['Software Developer', 'Cloud Engineer', 'DevOps Engineer', 'Business Analyst'],
        'type': 'Service', 'tier': 3, 'logo_color': '#003399',
    },
    'Accenture': {
        'min_cgpa': 6.5, 'package': '₹4.5–9 LPA',
        'skills': ['Programming', 'Analytics', 'Consulting Basics'],
        'roles': ['Software Developer', 'Data Analyst', 'Business Analyst', 'Consulting Analyst', 'Full Stack Developer'],
        'type': 'Consulting', 'tier': 3, 'logo_color': '#A100FF',
    },
    'HCL Technologies': {
        'min_cgpa': 6.5, 'package': '₹4–7 LPA',
        'skills': ['Programming', 'Networking', 'Hardware Basics'],
        'roles': ['Software Developer', 'Hardware / VLSI Engineer', 'Embedded Systems Engineer', 'Cybersecurity Analyst'],
        'type': 'Service', 'tier': 3, 'logo_color': '#0076C0',
    },
    'Tech Mahindra': {
        'min_cgpa': 6.5, 'package': '₹3.5–7 LPA',
        'skills': ['Programming', 'Telecom Basics', 'Communication'],
        'roles': ['Software Developer', 'Backend Developer', 'Data Analyst', 'Cybersecurity Analyst'],
        'type': 'Service', 'tier': 3, 'logo_color': '#ED2026',
    },
    'L&T Technology Services': {
        'min_cgpa': 6.5, 'package': '₹4–8 LPA',
        'skills': ['Embedded C', 'MATLAB', 'Engineering Domain'],
        'roles': ['Embedded Systems Engineer', 'Hardware / VLSI Engineer', 'Data Analyst'],
        'type': 'Engineering', 'tier': 3, 'logo_color': '#1A5276',
    },
    'Mphasis': {
        'min_cgpa': 6.5, 'package': '₹4–7 LPA',
        'skills': ['Java/Python', 'Cloud Basics', 'Communication'],
        'roles': ['Software Developer', 'Cloud Engineer', 'Backend Developer'],
        'type': 'Service', 'tier': 3, 'logo_color': '#6C3483',
    },
}


def get_eligible_companies(cgpa: float, role: str = None, types: list = None) -> list:
    """
    Return eligible companies filtered by CGPA, optional role, and optional company type.

    Args:
        cgpa:  Student's CGPA.
        role:  Target job role (string).  None = all roles.
        types: List of company types.     None = all types.
    """
    eligible = []
    role_lower = role.strip().lower() if role else None

    for name, info in COMPANY_CRITERIA.items():
        # CGPA gate
        if cgpa < info['min_cgpa']:
            continue
        # Type gate
        if types and info['type'] not in types:
            continue
        # Role gate
        if role_lower and role_lower != "all roles":
            company_roles_lower = [r.lower() for r in info.get('roles', [])]
            if role_lower not in company_roles_lower:
                continue
        eligible.append({'name': name, **info})

    return sorted(eligible, key=lambda x: x['min_cgpa'], reverse=True)
