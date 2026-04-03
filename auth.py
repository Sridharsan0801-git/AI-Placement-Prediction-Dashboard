import re

def validate_email(email: str) -> bool:
    """Only allow @karunya.edu.in emails"""
    pattern = r'^[a-zA-Z0-9._%+\-]+@karunya\.edu\.in$'
    return bool(re.match(pattern, email))

def parse_register_number(reg_no: str) -> dict:
    """Parse Karunya University register number"""
    pattern = r'^(URK|PRK|URM|PRM)(\d{2})([A-Z]{2,3})(\d{3,4})$'
    match = re.match(pattern, reg_no.strip().upper())

    if match:
        prefix = match.group(1)
        year_digits = match.group(2)
        dept_code = match.group(3)
        roll = match.group(4)

        program_map = {
            'URK': 'B.Tech / B.Sc (Undergraduate)',
            'URM': 'B.Tech / B.Sc (Undergraduate)',
            'PRK': 'M.Tech / M.Sc (Postgraduate)',
            'PRM': 'M.Tech / M.Sc (Postgraduate)',
        }
        program_type = program_map.get(prefix, 'Undergraduate')
        year_of_joining = 2000 + int(year_digits)

        return {
            'valid': True,
            'prefix': prefix,
            'year_of_joining': year_of_joining,
            'dept_code': dept_code,
            'roll_number': roll,
            'program_type': program_type,
        }
    return {'valid': False}
