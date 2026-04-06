"""
auth.py — Authentication & access control utilities
"""
import re

# ── Admin credentials ────────────────────────────────────────────────────────
# Change these before deploying!
ADMIN_EMAIL    = "admin@karunya.edu.in"
ADMIN_PASSWORD = "KUAdmin@2025"

# ── Email domain restriction ─────────────────────────────────────────────────
ALLOWED_DOMAIN = "karunya.edu.in"


def validate_email(email: str) -> bool:
    """Only allow @karunya.edu.in addresses."""
    pattern = rf'^[a-zA-Z0-9._%+\-]+@{re.escape(ALLOWED_DOMAIN)}$'
    return bool(re.match(pattern, email.strip()))


def parse_register_number(reg_no: str) -> dict:
    """Parse Karunya University register number (e.g. URK25AI1074)."""
    pattern = r'^(URK|PRK|URM|PRM)(\d{2})([A-Z]{2,3})(\d{3,4})$'
    match = re.match(pattern, reg_no.strip().upper())
    if match:
        prefix = match.group(1)
        year_digits = match.group(2)
        dept_code   = match.group(3)
        roll        = match.group(4)
        program_map = {
            'URK': 'B.Tech / B.Sc (Undergraduate)',
            'URM': 'B.Tech / B.Sc (Undergraduate)',
            'PRK': 'M.Tech / M.Sc (Postgraduate)',
            'PRM': 'M.Tech / M.Sc (Postgraduate)',
        }
        return {
            'valid':        True,
            'prefix':       prefix,
            'year_of_joining': 2000 + int(year_digits),
            'dept_code':    dept_code,
            'roll_number':  roll,
            'program_type': program_map.get(prefix, 'Undergraduate'),
        }
    return {'valid': False}


def extract_name(email: str) -> str:
    """
    Extract a display name from an email address.
    e.g.  sri.dharsan1074@karunya.edu.in  →  Sri Dharsan
    Rules: strip numbers, split on '.' or '_', capitalise.
    """
    local = email.split('@')[0]                         # sri.dharsan1074
    local = ''.join(c for c in local if not c.isdigit())  # sri.dharsan
    local = local.replace('.', ' ').replace('_', ' ')   # sri dharsan
    return local.title()                                 # Sri Dharsan


def is_admin(email: str, password: str) -> bool:
    """Check if credentials match admin account."""
    return email.strip().lower() == ADMIN_EMAIL.lower() and password == ADMIN_PASSWORD
