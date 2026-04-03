COMPANY_CRITERIA = {
    'Google': {
        'min_cgpa': 8.5, 'package': '₹25–45 LPA',
        'skills': ['DSA', 'System Design', 'Problem Solving'],
        'type': 'Product', 'tier': 1, 'logo_color': '#4285F4'
    },
    'Microsoft': {
        'min_cgpa': 8.0, 'package': '₹20–40 LPA',
        'skills': ['DSA', 'OOP', 'System Design', 'Cloud'],
        'type': 'Product', 'tier': 1, 'logo_color': '#00A4EF'
    },
    'Amazon': {
        'min_cgpa': 8.0, 'package': '₹20–35 LPA',
        'skills': ['DSA', 'System Design', 'Leadership Principles'],
        'type': 'Product', 'tier': 1, 'logo_color': '#FF9900'
    },
    'Meta': {
        'min_cgpa': 8.5, 'package': '₹25–50 LPA',
        'skills': ['DSA', 'System Design', 'ML/AI'],
        'type': 'Product', 'tier': 1, 'logo_color': '#0866FF'
    },
    'Apple': {
        'min_cgpa': 8.5, 'package': '₹22–40 LPA',
        'skills': ['System Design', 'iOS/Swift or Hardware', 'DSA'],
        'type': 'Product', 'tier': 1, 'logo_color': '#555555'
    },
    'Adobe': {
        'min_cgpa': 8.0, 'package': '₹18–30 LPA',
        'skills': ['DSA', 'Web Technologies', 'Problem Solving'],
        'type': 'Product', 'tier': 1, 'logo_color': '#FF0000'
    },
    'Flipkart': {
        'min_cgpa': 7.5, 'package': '₹15–25 LPA',
        'skills': ['DSA', 'System Design', 'Backend Development'],
        'type': 'E-Commerce', 'tier': 2, 'logo_color': '#F74F00'
    },
    'Walmart Global Tech': {
        'min_cgpa': 7.5, 'package': '₹14–22 LPA',
        'skills': ['DSA', 'Java/Python', 'Cloud'],
        'type': 'Retail Tech', 'tier': 2, 'logo_color': '#0071CE'
    },
    'Zoho': {
        'min_cgpa': 7.5, 'package': '₹8–15 LPA',
        'skills': ['Java', 'Problem Solving', 'Aptitude'],
        'type': 'Product', 'tier': 2, 'logo_color': '#E42527'
    },
    'Paytm': {
        'min_cgpa': 7.0, 'package': '₹10–18 LPA',
        'skills': ['DSA', 'Fintech', 'Mobile Development'],
        'type': 'Fintech', 'tier': 2, 'logo_color': '#002970'
    },
    'Deloitte': {
        'min_cgpa': 7.0, 'package': '₹6–12 LPA',
        'skills': ['Analytics', 'Consulting', 'Domain Knowledge'],
        'type': 'Consulting', 'tier': 2, 'logo_color': '#86BC25'
    },
    'KPMG': {
        'min_cgpa': 7.0, 'package': '₹6–11 LPA',
        'skills': ['Analytics', 'Excel', 'Communication'],
        'type': 'Consulting', 'tier': 2, 'logo_color': '#00338D'
    },
    'Infosys': {
        'min_cgpa': 7.0, 'package': '₹4–8 LPA',
        'skills': ['Java/Python', 'SQL', 'Communication'],
        'type': 'Service', 'tier': 3, 'logo_color': '#007CC3'
    },
    'TCS': {
        'min_cgpa': 6.5, 'package': '₹3.5–7 LPA',
        'skills': ['Programming Basics', 'Aptitude', 'Communication'],
        'type': 'Service', 'tier': 3, 'logo_color': '#1C4E9F'
    },
    'Wipro': {
        'min_cgpa': 6.5, 'package': '₹3.5–7 LPA',
        'skills': ['Programming Basics', 'SQL', 'Communication'],
        'type': 'Service', 'tier': 3, 'logo_color': '#341C68'
    },
    'Cognizant': {
        'min_cgpa': 6.5, 'package': '₹4–8 LPA',
        'skills': ['Java/.NET', 'SQL', 'Soft Skills'],
        'type': 'Service', 'tier': 3, 'logo_color': '#1464A4'
    },
    'Capgemini': {
        'min_cgpa': 6.5, 'package': '₹4–8 LPA',
        'skills': ['Programming', 'Cloud Basics', 'Communication'],
        'type': 'Service', 'tier': 3, 'logo_color': '#003399'
    },
    'Accenture': {
        'min_cgpa': 6.5, 'package': '₹4.5–9 LPA',
        'skills': ['Programming', 'Analytics', 'Consulting Basics'],
        'type': 'Consulting', 'tier': 3, 'logo_color': '#A100FF'
    },
    'HCL Technologies': {
        'min_cgpa': 6.5, 'package': '₹4–7 LPA',
        'skills': ['Programming', 'Networking', 'Hardware Basics'],
        'type': 'Service', 'tier': 3, 'logo_color': '#0076C0'
    },
    'Tech Mahindra': {
        'min_cgpa': 6.5, 'package': '₹3.5–7 LPA',
        'skills': ['Programming', 'Telecom Basics', 'Communication'],
        'type': 'Service', 'tier': 3, 'logo_color': '#ED2026'
    },
}

def get_eligible_companies(cgpa, types=None):
    eligible = []
    for name, info in COMPANY_CRITERIA.items():
        if cgpa >= info['min_cgpa']:
            if types is None or info['type'] in types:
                eligible.append({'name': name, **info})
    return sorted(eligible, key=lambda x: x['min_cgpa'], reverse=True)
