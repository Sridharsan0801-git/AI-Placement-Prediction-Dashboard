"""
╔══════════════════════════════════════════════════════════════════╗
║     KARUNYA UNIVERSITY — AI PLACEMENT INTELLIGENCE SYSTEM       ║
║                        Version 5.0                              ║
╚══════════════════════════════════════════════════════════════════╝

What's new in v5.0:
  ✦ Admin Dataset Upload System — CSV or Excel, auto-parsed
  ✦ Dynamic company list — extracted live from uploaded data
  ✦ Auto model retraining on dataset upload
  ✦ Year-over-year placement comparison charts
  ✦ Dataset statistics dashboard
  ✦ Template CSV download for admins
  ✦ Static fallback companies if no dataset uploaded yet
"""

import re, io, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.auth import validate_email, parse_register_number, extract_name, is_admin, ADMIN_EMAIL
from utils.ml_model import generate_sample_data, train_model, predict, prepare_uploaded_dataset
from utils.companies import COMPANY_CRITERIA, get_eligible_companies, ALL_ROLES
from utils.departments import DEPARTMENTS, DEPARTMENT_SCHOOLS, DEPARTMENT_SKILLS, get_skill_category
from utils.resume_parser import extract_text, parse_resume, get_support_status, STREAMLIT_TYPES
from utils.student_store import (save_student_submission, get_all_submissions,
                                  get_submission_count, clear_store, submissions_to_df)
from utils.company_dataset import (
    parse_placement_dataset, extract_companies_from_dataset,
    get_eligible_from_db, compute_yearly_stats, dataset_summary,
    generate_sample_csv, SUPPORTED_UPLOAD_TYPES,
)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="KU Placement Intelligence",
                   page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],.main{background:#080e1d!important;color:#dde3f0!important;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#0b1120!important;border-right:1px solid rgba(96,165,250,.1)!important}
[data-testid="stSidebar"] *{color:#c4cede!important}
h1,h2,h3,h4,h5,h6{font-family:'Syne',sans-serif!important;letter-spacing:-.02em}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;display:none}

[data-testid="stSidebar"] [role="radiogroup"] label{display:flex!important;align-items:center!important;padding:9px 14px!important;border-radius:9px!important;margin:2px 0!important;cursor:pointer!important;transition:background .18s!important;font-size:.88rem!important;font-weight:500!important}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(96,165,250,.08)!important}

[data-testid="metric-container"]{background:rgba(255,255,255,.035)!important;border:1px solid rgba(255,255,255,.07)!important;border-radius:14px!important;padding:1rem 1.3rem!important;transition:border-color .2s}
[data-testid="metric-container"]:hover{border-color:rgba(96,165,250,.3)!important}
[data-testid="metric-container"] label{color:#64748b!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:.06em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#f0f9ff!important;font-family:'Syne',sans-serif!important;font-size:1.65rem!important}
[data-testid="metric-container"] [data-testid="stMetricDelta"] svg{display:none}

[data-testid="stTextInput"] input{background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:10px!important;color:#f1f5f9!important;padding:.6rem .9rem!important;transition:border-color .2s,box-shadow .2s!important}
[data-testid="stTextInput"] input:focus{border-color:#3b82f6!important;box-shadow:0 0 0 3px rgba(59,130,246,.15)!important;outline:none!important}
[data-testid="stTextInput"] input::placeholder{color:#475569!important}
[data-testid="stSelectbox"]>div>div,[data-baseweb="select"]>div{background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:10px!important;color:#f1f5f9!important}

.stButton>button{background:linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:.92rem!important;padding:.65rem 1.6rem!important;transition:transform .2s,box-shadow .2s!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(59,130,246,.4)!important}
[data-testid="stDownloadButton"] button{background:rgba(255,255,255,.06)!important;border:1px solid rgba(255,255,255,.14)!important;color:#94a3b8!important;border-radius:10px!important}
[data-testid="stDownloadButton"] button:hover{background:rgba(255,255,255,.1)!important;color:#f1f5f9!important;transform:translateY(-1px)!important;box-shadow:none!important}

[data-baseweb="tab-list"]{background:rgba(255,255,255,.04)!important;border-radius:12px!important;padding:4px!important;gap:4px!important;border:1px solid rgba(255,255,255,.06)!important}
[data-baseweb="tab"]{background:transparent!important;border-radius:8px!important;color:#64748b!important;font-weight:500!important}
[aria-selected="true"][data-baseweb="tab"]{background:rgba(59,130,246,.22)!important;color:#93c5fd!important;font-weight:600!important}

[data-testid="stExpander"]{background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.07)!important;border-radius:12px!important;margin-bottom:6px!important;transition:border-color .2s!important}
[data-testid="stExpander"]:hover{border-color:rgba(96,165,250,.2)!important}
[data-testid="stExpander"] summary{color:#c4cede!important;font-weight:500!important}

[data-testid="stProgressBar"]>div{background:rgba(255,255,255,.08)!important;border-radius:999px!important;height:6px!important}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#3b82f6,#06b6d4)!important;border-radius:999px!important}
[data-testid="stDataFrameResizable"]{border-radius:12px!important;overflow:hidden!important;border:1px solid rgba(255,255,255,.08)!important}
[data-testid="stAlert"]{border-radius:12px!important;border-left-width:3px!important}
[data-testid="stMultiSelect"] [data-baseweb="tag"]{background:rgba(59,130,246,.22)!important;color:#93c5fd!important;border-radius:6px!important}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.03)!important;border:2px dashed rgba(99,179,237,.3)!important;border-radius:14px!important;transition:border-color .2s!important}
[data-testid="stFileUploader"]:hover{border-color:rgba(59,130,246,.6)!important}
hr{border-color:rgba(255,255,255,.07)!important}

/* Custom classes */
.ku-header{background:linear-gradient(135deg,#0c1628 0%,#13233f 60%,#0f1e38 100%);border:1px solid rgba(96,165,250,.18);border-radius:20px;padding:2.4rem 2rem;text-align:center;margin-bottom:1.8rem;position:relative;overflow:hidden}
.ku-header::before{content:'';position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse at 20% 50%,rgba(59,130,246,.13) 0%,transparent 55%),radial-gradient(ellipse at 80% 50%,rgba(6,182,212,.09) 0%,transparent 55%)}
.ku-header h1{font-size:1.9rem;color:#f0f9ff;margin:0;position:relative}
.ku-header p{color:#64748b;margin:.5rem 0 0;font-size:.9rem;position:relative}
.ku-card{background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.075);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem;transition:border-color .2s}
.ku-card:hover{border-color:rgba(96,165,250,.2)}
.ku-card-accent{background:linear-gradient(135deg,rgba(59,130,246,.08),rgba(6,182,212,.05));border:1px solid rgba(59,130,246,.22);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem}
.ku-card-success{background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.22);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem}
.ku-card-warn{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.22);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem}

.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.8rem;margin-bottom:1rem}
.stat-box{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:1rem;text-align:center}
.stat-box .sv{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#f0f9ff;line-height:1}
.stat-box .sl{font-size:.72rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-top:.3rem}

.co-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:.7rem;display:flex;gap:1rem;align-items:flex-start;transition:all .2s}
.co-card:hover{border-color:rgba(96,165,250,.25);background:rgba(255,255,255,.05);transform:translateY(-2px)}
.co-dot{width:40px;height:40px;border-radius:10px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.85rem;color:#fff;font-family:'Syne',sans-serif}

.badge{display:inline-block;background:rgba(59,130,246,.18);border:1px solid rgba(59,130,246,.32);color:#93c5fd;padding:3px 10px;border-radius:999px;font-size:.74rem;margin:2px;font-weight:500}
.badge-green{background:rgba(16,185,129,.12);border-color:rgba(16,185,129,.28);color:#6ee7b7}
.badge-yellow{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.28);color:#fcd34d}
.badge-red{background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.28);color:#fca5a5}
.badge-purple{background:rgba(139,92,246,.12);border-color:rgba(139,92,246,.28);color:#c4b5fd}
.badge-gold{background:rgba(234,179,8,.15);border-color:rgba(234,179,8,.35);color:#fde047}
.badge-admin{background:rgba(239,68,68,.18);border-color:rgba(239,68,68,.4);color:#fca5a5;font-weight:700}
.badge-orange{background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.35);color:#fdba74}
.badge-teal{background:rgba(20,184,166,.12);border-color:rgba(20,184,166,.28);color:#5eead4}

.section-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#475569;margin-bottom:.4rem}
.info-box{background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.2);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem}
.success-box{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#a7f3d0}
.warning-box{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#fde68a}
.danger-box{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#fca5a5}
.skill-chip{display:inline-flex;align-items:center;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);color:#6ee7b7;padding:4px 10px;border-radius:8px;font-size:.78rem;font-weight:500;margin:3px}
.access-denied{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-radius:16px;padding:3rem 2rem;text-align:center;margin:2rem 0}
.resume-required{background:rgba(245,158,11,.07);border:2px dashed rgba(245,158,11,.4);border-radius:14px;padding:1.8rem;text-align:center;margin-bottom:1rem}
.resume-ok{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.3);border-radius:14px;padding:1.2rem;margin-bottom:1rem}
.upload-zone{background:rgba(59,130,246,.05);border:2px dashed rgba(59,130,246,.3);border-radius:16px;padding:2rem;text-align:center}
.year-badge{display:inline-block;background:rgba(6,182,212,.12);border:1px solid rgba(6,182,212,.28);color:#5eead4;padding:4px 12px;border-radius:8px;font-size:.8rem;font-weight:600;margin:2px}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════
def _ply_base():
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="DM Sans", size=12),
                margin=dict(t=48, b=28, l=8, r=8),
                title_font=dict(color="#94a3b8", size=13, family="Syne"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                            bordercolor="rgba(255,255,255,.08)", borderwidth=1))

def _ply_axis():
    return dict(gridcolor="rgba(255,255,255,.06)",
                zerolinecolor="rgba(255,255,255,.06)",
                tickfont=dict(color="#64748b"))

def _chart(fig, h=320):
    fig.update_layout(**_ply_base(), height=h)
    fig.update_xaxes(**_ply_axis())
    fig.update_yaxes(**_ply_axis())
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "logged_in": False, "role": None,
    "user_data": {}, "placement_data": None,
    "model": None, "scaler": None,
    "model_accuracy": None, "model_name": None,
    "cm": None, "report": None,
    "resume_result": None, "resume_text": "",
    "resume_ready": False, "target_role": None,
    "last_prediction": None,
    # NEW v5
    "company_db": {},           # dynamic company dict from uploaded dataset
    "placement_dataset": None,  # the uploaded placement CSV/Excel as DataFrame
    "dataset_source": "sample", # "sample" | "uploaded"
    "dataset_meta": {},         # {filename, rows, companies, years, uploaded_at}
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _boot_model():
    if st.session_state.placement_data is None:
        df = generate_sample_data()
        model, scaler, acc, name, cm, report, _, _ = train_model(df)
        st.session_state.placement_data = df
        st.session_state.model = model; st.session_state.scaler = scaler
        st.session_state.model_accuracy = acc; st.session_state.model_name = name
        st.session_state.cm = cm; st.session_state.report = report

def _retrain_on_upload(df_placement: pd.DataFrame):
    """Retrain model using uploaded placement data."""
    ml_df = prepare_uploaded_dataset(df_placement)
    if len(ml_df) >= 10:
        model, scaler, acc, name, cm, report, _, _ = train_model(ml_df)
        st.session_state.placement_data = ml_df
        st.session_state.model = model; st.session_state.scaler = scaler
        st.session_state.model_accuracy = acc; st.session_state.model_name = name
        st.session_state.cm = cm; st.session_state.report = report
        return True, acc, name
    return False, 0, ""

def _get_company_db():
    """Return dynamic db if uploaded, else fall back to static criteria."""
    if st.session_state.company_db:
        return st.session_state.company_db
    return None   # signals: use static fallback

def _logout():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

def _is_admin():   return st.session_state.role == "admin"
def _is_student(): return st.session_state.role == "student"

def _access_denied():
    st.markdown("""<div class="access-denied">
        <div style="font-size:2.5rem;margin-bottom:.8rem;">🔒</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;color:#fca5a5;font-weight:700;">Access Restricted</div>
        <p style="color:#64748b;margin:.4rem 0 0;font-size:.88rem;">This section is only visible to administrators.</p>
    </div>""", unsafe_allow_html=True)

def _tier_badge(tier):
    return {1:'<span class="badge badge-gold">🌟 Tier 1</span>',
            2:'<span class="badge badge-yellow">⭐ Tier 2</span>',
            3:'<span class="badge">💫 Tier 3</span>'}.get(tier, "")

def _hex_initials(name):
    p = name.split(); return (p[0][0]+(p[1][0] if len(p)>1 else "")).upper()

def _support_badges():
    s = get_support_status()
    pdf  = '<span class="badge badge-red">📄 PDF</span>'   if s["PDF"]  else '<span class="badge">📄 PDF ✗</span>'
    docx = '<span class="badge">📝 DOCX</span>'            if s["DOCX"] else '<span class="badge">📝 DOCX ✗</span>'
    ocr  = '<span class="badge badge-purple">🖼️ OCR</span>' if s["OCR (images / scanned PDFs)"] else '<span class="badge">🖼️ OCR ✗</span>'
    return f"{pdf}{docx}{ocr}"

def _dataset_source_badge():
    if st.session_state.dataset_source == "uploaded":
        meta = st.session_state.dataset_meta
        return f'<span class="badge badge-teal">📂 {meta.get("filename","uploaded")} · {meta.get("rows",0)} rows</span>'
    return '<span class="badge">📊 Synthetic data</span>'

# ══════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ══════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("""<div class="ku-header" style="padding:3rem 2rem;">
        <h1 style="font-size:2.2rem;">🎓 Karunya University</h1>
        <p style="font-size:1rem;color:#475569;">AI-Powered Placement Intelligence System v5.0</p>
    </div>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1,1.5,1])
    with mid:
        mode = st.radio("", ["🎓 Student","🔑 Admin"], horizontal=True, label_visibility="collapsed")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if mode == "🔑 Admin":
            st.markdown('<div class="section-label">ADMIN LOGIN</div>', unsafe_allow_html=True)
            ae = st.text_input("Admin Email", placeholder=ADMIN_EMAIL)
            ap = st.text_input("Admin Password", type="password")
            if st.button("Login as Admin  →", use_container_width=True):
                if is_admin(ae, ap):
                    with st.spinner("Initialising…"):
                        _boot_model()
                    st.session_state.logged_in = True
                    st.session_state.role      = "admin"
                    st.session_state.user_data = {"email": ADMIN_EMAIL, "name": "Admin"}
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
        else:
            st.markdown('<div class="section-label">STUDENT LOGIN</div>', unsafe_allow_html=True)
            email  = st.text_input("Email ID", placeholder="yourname@karunya.edu.in")
            reg_no = st.text_input("Register Number", placeholder="URK25AI1074")
            dept   = st.selectbox("Department / Programme", ["— Select —"]+DEPARTMENTS)
            yr     = st.selectbox("Current Year", ["— Select —","1st Year","2nd Year","3rd Year","4th Year"])
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button("Login  →", use_container_width=True):
                errs = []
                if not email: errs.append("Email is required.")
                elif not validate_email(email): errs.append("Only **@karunya.edu.in** emails accepted.")
                parsed = None
                if not reg_no: errs.append("Register number is required.")
                else:
                    parsed = parse_register_number(reg_no)
                    if not parsed["valid"]: errs.append("Invalid register number — e.g. **URK25AI1074**")
                if dept == "— Select —": errs.append("Please select your department.")
                if yr   == "— Select —": errs.append("Please select your current year.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    with st.spinner("Initialising…"):
                        _boot_model()
                    st.session_state.logged_in = True
                    st.session_state.role      = "student"
                    st.session_state.user_data = {
                        "email": email.strip().lower(), "name": extract_name(email),
                        "register_number": reg_no.upper().strip(), "department": dept,
                        "school": DEPARTMENT_SCHOOLS.get(dept,"Karunya University"),
                        "year_of_joining": parsed["year_of_joining"],
                        "program_type": parsed["program_type"],
                        "roll_number": parsed["roll_number"], "current_year": yr,
                    }
                    st.rerun()

        st.markdown("""<div class="info-box" style="margin-top:1rem;">
            <b>📋 Register Number:</b> <code>URK25AI1074</code><br>
            <span style="color:#475569;font-size:.82rem;">URK=Undergrad · 25=Year · AI=Dept · 1074=Roll</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════
def dashboard_page():
    user = st.session_state.user_data
    df   = st.session_state.placement_data
    name = user.get("name","User")

    if _is_admin():
        sub_count = get_submission_count()
        db_badge  = _dataset_source_badge()
        st.markdown(f"""<div class="ku-header">
            <h1>Admin Dashboard 🛡️</h1>
            <p>Full analytics · {sub_count} student submission{"s" if sub_count!=1 else ""} · {db_badge}</p>
        </div>""", unsafe_allow_html=True)

        rate  = df["Placed"].mean()*100 if "Placed" in df.columns else 0
        avg_c = df["CGPA"].mean()       if "CGPA"   in df.columns else 0
        placed = int(df["Placed"].sum()) if "Placed" in df.columns else 0
        total  = len(df)
        cdb    = _get_company_db()
        company_count = len(cdb) if cdb else len(COMPANY_CRITERIA)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Placement Rate",    f"{rate:.1f}%")
        c2.metric("Average CGPA",      f"{avg_c:.2f}")
        c3.metric("Dataset Records",   f"{total:,}")
        c4.metric("Companies Tracked", f"{company_count}")
        c5.metric("Live Submissions",  f"{sub_count}")

        # Year-over-year comparison (if uploaded dataset has year column)
        pds = st.session_state.placement_dataset
        if pds is not None and "year" in pds.columns and pds["year"].nunique() > 1:
            st.markdown("### 📅 Year-over-Year Placement Comparison")
            yoy = compute_yearly_stats(pds)
            if not yoy.empty:
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig = px.bar(yoy, x="Year", y="Placement Rate %",
                        title="Placement Rate by Year",
                        color="Placement Rate %", color_continuous_scale="Blues",
                        text="Placement Rate %")
                    fig.update_traces(texttemplate="%{text}%", textposition="outside")
                    _chart(fig, 320)
                with ch2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=yoy["Year"], y=yoy["Avg Package LPA"],
                        mode="lines+markers", name="Avg Package",
                        line=dict(color="#3b82f6", width=2),
                        marker=dict(size=8)))
                    fig2.add_trace(go.Scatter(x=yoy["Year"], y=yoy["Avg CGPA"],
                        mode="lines+markers", name="Avg CGPA",
                        line=dict(color="#06b6d4", width=2, dash="dot"),
                        marker=dict(size=8), yaxis="y2"))
                    fig2.update_layout(**_ply_base(), height=320,
                        title_text="Package & CGPA Trends",
                        yaxis=dict(**_ply_axis(), title="Avg Package (LPA)"),
                        yaxis2=dict(**_ply_axis(), title="Avg CGPA", overlaying="y", side="right"))
                    st.plotly_chart(fig2, use_container_width=True)

                st.dataframe(yoy, use_container_width=True, hide_index=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            fig = go.Figure(go.Pie(values=[placed,total-placed],labels=["Placed","Not Placed"],
                hole=0.58, marker_colors=["#3b82f6","#1e293b"],
                textinfo="percent", hovertemplate="%{label}: %{value}<extra></extra>"))
            fig.update_layout(**_ply_base(), title_text="Placement Split", height=270)
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            col = "CGPA" if "CGPA" in df.columns else df.columns[0]
            col_placed = "Placed" if "Placed" in df.columns else None
            if col_placed:
                fig2 = px.histogram(df, x=col, color=col_placed,
                    color_discrete_map={0:"#ef4444",1:"#3b82f6"},
                    barmode="overlay", opacity=0.72, title="CGPA Distribution")
                _chart(fig2, 270)

        # Company distribution if uploaded
        if pds is not None:
            st.markdown("### 🏢 Company Distribution (Uploaded Dataset)")
            top_co = pds[pds["placed"]==1]["company"].value_counts().head(15).reset_index()
            top_co.columns = ["Company","Hired"]
            fig3 = px.bar(top_co, x="Hired", y="Company", orientation="h",
                title="Top 15 Hiring Companies", color="Hired",
                color_continuous_scale="Blues")
            fig3.update_layout(**_ply_base(), height=480, yaxis_categoryorder="total ascending")
            fig3.update_xaxes(**_ply_axis()); fig3.update_yaxes(**_ply_axis())
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.markdown(f"""<div class="ku-header">
            <h1>Welcome, {name}! 👋</h1>
            <p>{user['department']}&nbsp;·&nbsp;{user['school']}</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="ku-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2.5rem;margin-bottom:.8rem;">📊</div>
            <p style="color:#64748b;font-size:.9rem;">Use the sidebar to navigate to
            <b style="color:#93c5fd;">Placement Prediction</b>,
            <b style="color:#93c5fd;">Company Eligibility</b>, or
            <b style="color:#93c5fd;">My Account</b>.</p>
        </div>""", unsafe_allow_html=True)
        acc = st.session_state.model_accuracy or 0
        c1,c2,c3 = st.columns(3)
        c1.metric("ML Model",         st.session_state.model_name or "—")
        c2.metric("Model Accuracy",   f"{acc*100:.1f}%")
        c3.metric("Training Records", f"{len(df):,}")

# ══════════════════════════════════════════════════════════════════
# PAGE: ADMIN DATASET UPLOAD  (the star of v5.0)
# ══════════════════════════════════════════════════════════════════
def admin_dataset_page():
    if _is_student():
        _access_denied()
        return

    st.markdown("""<div class="ku-header">
        <h1>📂 Dataset Upload System</h1>
        <p>Upload your annual placement CSV or Excel — companies update automatically</p>
    </div>""", unsafe_allow_html=True)

    # ── Current dataset status ────────────────────────────────────
    if st.session_state.dataset_source == "uploaded":
        meta = st.session_state.dataset_meta
        pds  = st.session_state.placement_dataset
        s    = dataset_summary(pds)

        st.markdown(f"""<div class="ku-card-success">
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#6ee7b7;margin-bottom:.8rem;">
                ✅ Live Dataset Active — {meta.get('filename','')}
            </div>
            <div class="stat-grid">
                <div class="stat-box"><div class="sv">{s['total_students']:,}</div><div class="sl">Students</div></div>
                <div class="stat-box"><div class="sv">{s['placed_students']:,}</div><div class="sl">Placed</div></div>
                <div class="stat-box"><div class="sv">{s['placement_rate']}%</div><div class="sl">Rate</div></div>
                <div class="stat-box"><div class="sv">{s['unique_companies']}</div><div class="sl">Companies</div></div>
                <div class="stat-box"><div class="sv">{s['avg_cgpa']}</div><div class="sl">Avg CGPA</div></div>
                <div class="stat-box"><div class="sv">₹{s['avg_package']}</div><div class="sl">Avg Pkg LPA</div></div>
            </div>
            <div style="font-size:.8rem;color:#475569;margin-top:.4rem;">
                Uploaded: {meta.get('uploaded_at','')} &nbsp;·&nbsp;
                Years: {', '.join(str(y) for y in s['years']) or 'N/A'} &nbsp;·&nbsp;
                Top company: <b style="color:#6ee7b7;">{s['top_company']}</b>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="ku-card-warn">
            <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:#fcd34d;margin-bottom:.4rem;">
                ⚠️ Using synthetic sample data
            </div>
            <p style="color:#94a3b8;font-size:.85rem;margin:0;">
                Upload a real placement dataset below to activate dynamic companies and year-comparison analytics.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "📤  Upload New Dataset",
        "📋  Preview Data",
        "📊  Company Analytics",
        "📅  Year Comparison",
    ])

    # ── TAB 1: Upload ─────────────────────────────────────────────
    with t1:
        lc, rc = st.columns([1.4, 1])

        with lc:
            st.markdown("#### Upload Placement Dataset")
            st.markdown(f"""<div class="info-box" style="margin-bottom:1rem;">
                <b>Accepted formats:</b>
                <span class="badge badge-green">CSV</span>
                <span class="badge badge-green">Excel (.xlsx)</span>
                {"<span class='badge badge-yellow'>⚠️ openpyxl required for Excel</span>" if True else ""}<br><br>
                <b>Required columns:</b>
                <code>student_name</code> · <code>cgpa</code> · <code>company</code><br><br>
                <b>Optional columns:</b>
                <code>internships</code> · <code>projects</code> · <code>certifications</code> ·
                <code>communication_skill</code> · <code>coding_skill</code> ·
                <code>placed</code> · <code>role</code> · <code>package_lpa</code> ·
                <code>company_type</code> · <code>department</code> · <code>year</code><br><br>
                <span style="color:#475569;font-size:.8rem;">
                Column names are flexible — the system auto-detects common variations.
                </span>
            </div>""", unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Drop your placement dataset here",
                type=SUPPORTED_UPLOAD_TYPES,
                key="placement_uploader",
            )

            if uploaded:
                with st.spinner("Parsing dataset…"):
                    df_upload, warns = parse_placement_dataset(uploaded)

                if df_upload is None:
                    for w in warns:
                        st.error(w)
                else:
                    if warns:
                        for w in warns:
                            st.warning(w)

                    # Extract companies
                    with st.spinner("Extracting companies…"):
                        cdb = extract_companies_from_dataset(df_upload)

                    # Retrain model
                    with st.spinner("Retraining ML model on new data…"):
                        ok, new_acc, new_name = _retrain_on_upload(df_upload)

                    # Save to session
                    from datetime import datetime as _dt
                    st.session_state.placement_dataset = df_upload
                    st.session_state.company_db        = cdb
                    st.session_state.dataset_source    = "uploaded"
                    st.session_state.dataset_meta      = {
                        "filename":    uploaded.name,
                        "rows":        len(df_upload),
                        "companies":   len(cdb),
                        "uploaded_at": _dt.now().strftime("%Y-%m-%d %H:%M"),
                    }

                    s = dataset_summary(df_upload)
                    st.markdown(f"""<div class="ku-card-success" style="margin-top:1rem;">
                        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#6ee7b7;font-size:1rem;margin-bottom:.6rem;">
                            🎉 Dataset uploaded successfully!
                        </div>
                        <div class="stat-grid">
                            <div class="stat-box"><div class="sv">{s['total_students']:,}</div><div class="sl">Students</div></div>
                            <div class="stat-box"><div class="sv">{s['placed_students']:,}</div><div class="sl">Placed</div></div>
                            <div class="stat-box"><div class="sv">{s['placement_rate']}%</div><div class="sl">Rate</div></div>
                            <div class="stat-box"><div class="sv">{len(cdb)}</div><div class="sl">Companies</div></div>
                        </div>
                        {"<div style='font-size:.8rem;color:#6ee7b7;margin-top:.5rem;'>✅ Model retrained: <b>"+new_name+"</b> · Accuracy: <b>"+str(round(new_acc*100,1))+"%</b></div>" if ok else "<div style='color:#fcd34d;font-size:.8rem;'>⚠️ Not enough rows to retrain model — using previous.</div>"}
                    </div>""", unsafe_allow_html=True)

        with rc:
            st.markdown("#### Download Template")
            st.markdown("""<div class="ku-card" style="text-align:center;padding:1.5rem;">
                <div style="font-size:2rem;margin-bottom:.5rem;">📄</div>
                <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:#f0f9ff;margin-bottom:.4rem;">
                    Sample CSV Template
                </div>
                <p style="color:#64748b;font-size:.82rem;margin-bottom:1rem;">
                    Download, fill with real data, and upload above.
                    Contains all supported columns with example rows.
                </p>
            </div>""", unsafe_allow_html=True)
            st.download_button(
                "📥 Download Template CSV",
                generate_sample_csv(),
                "placement_template.csv", "text/csv",
                use_container_width=True,
            )

            st.markdown("#### Reset to Synthetic Data")
            if st.session_state.dataset_source == "uploaded":
                if st.button("🔄 Reset to Sample Data", use_container_width=True):
                    st.session_state.placement_dataset = None
                    st.session_state.company_db        = {}
                    st.session_state.dataset_source    = "sample"
                    st.session_state.dataset_meta      = {}
                    st.session_state.placement_data    = None
                    _boot_model()
                    st.success("Reset to synthetic data.")
                    st.rerun()

    # ── TAB 2: Preview ────────────────────────────────────────────
    with t2:
        pds = st.session_state.placement_dataset
        if pds is None:
            st.info("Upload a dataset in the Upload tab to see a preview here.")
        else:
            st.markdown(f"**{len(pds):,} rows · {len(pds.columns)} columns**")
            st.dataframe(pds.head(50), use_container_width=True)
            st.download_button("📥 Download Cleaned Dataset",
                pds.to_csv(index=False), "placement_cleaned.csv", "text/csv",
                use_container_width=True)

    # ── TAB 3: Company Analytics ──────────────────────────────────
    with t3:
        cdb = _get_company_db()
        if not cdb:
            st.info("Upload a dataset to see dynamic company analytics.")
        else:
            st.markdown(f"**{len(cdb)} companies extracted from dataset**")

            # Build display dataframe
            rows = []
            for cname, info in cdb.items():
                rows.append({
                    "Company":       cname,
                    "Min CGPA":      info["min_cgpa"],
                    "Avg CGPA":      info["avg_cgpa"],
                    "Package":       info["package"],
                    "Tier":          info["tier"],
                    "Type":          info["type"],
                    "Students":      info["student_count"],
                    "Placed":        info["placed_count"],
                    "Placement %":   info["placement_rate"],
                })
            co_df = pd.DataFrame(rows).sort_values("Placed", ascending=False)
            st.dataframe(co_df, use_container_width=True, hide_index=True)

            ca, cb = st.columns(2)
            with ca:
                fig = px.bar(co_df.head(15).sort_values("Placed"),
                    x="Placed", y="Company", orientation="h",
                    title="Top 15 Companies by Placements",
                    color="Placement %", color_continuous_scale="Blues")
                fig.update_layout(**_ply_base(), height=480,
                                  yaxis_categoryorder="total ascending")
                fig.update_xaxes(**_ply_axis()); fig.update_yaxes(**_ply_axis())
                st.plotly_chart(fig, use_container_width=True)
            with cb:
                type_counts = co_df.groupby("Type")["Placed"].sum().reset_index()
                fig2 = go.Figure(go.Pie(values=type_counts["Placed"],
                    labels=type_counts["Type"], hole=0.5,
                    hovertemplate="%{label}: %{value}<extra></extra>"))
                fig2.update_layout(**_ply_base(), title_text="Placements by Company Type", height=480)
                st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 4: Year Comparison ────────────────────────────────────
    with t4:
        pds = st.session_state.placement_dataset
        if pds is None or "year" not in pds.columns:
            st.info("Upload a dataset with a **year** column to see year-over-year comparisons.")
        else:
            yoy = compute_yearly_stats(pds)
            if yoy.empty:
                st.info("Not enough year data to compare.")
            else:
                st.dataframe(yoy, use_container_width=True, hide_index=True)

                ca, cb = st.columns(2)
                with ca:
                    fig = px.line(yoy, x="Year", y="Placement Rate %",
                        markers=True, title="Placement Rate Trend",
                        color_discrete_sequence=["#3b82f6"])
                    fig.update_traces(line_width=2, marker_size=8)
                    _chart(fig, 320)
                with cb:
                    fig2 = px.bar(yoy, x="Year", y=["Total Students","Placed"],
                        barmode="group", title="Students vs Placed by Year",
                        color_discrete_sequence=["#475569","#3b82f6"])
                    _chart(fig2, 320)

                ca2, cb2 = st.columns(2)
                with ca2:
                    fig3 = px.line(yoy, x="Year", y="Avg CGPA",
                        markers=True, title="Average CGPA Trend",
                        color_discrete_sequence=["#06b6d4"])
                    fig3.update_traces(line_width=2, marker_size=8)
                    _chart(fig3, 280)
                with cb2:
                    fig4 = px.line(yoy, x="Year", y="Avg Package LPA",
                        markers=True, title="Average Package Trend (LPA)",
                        color_discrete_sequence=["#8b5cf6"])
                    fig4.update_traces(line_width=2, marker_size=8)
                    _chart(fig4, 280)

# ══════════════════════════════════════════════════════════════════
# PAGE: COMPANY ELIGIBILITY  (now dynamic)
# ══════════════════════════════════════════════════════════════════
def company_page():
    st.markdown("""<div class="ku-header">
        <h1>🏢 Company Eligibility</h1>
        <p>Discover companies that match your CGPA and target role</p>
    </div>""", unsafe_allow_html=True)

    cdb = _get_company_db()
    using_dynamic = cdb is not None

    if using_dynamic:
        st.markdown(f"""<div class="success-box" style="margin-bottom:1rem;">
            ✅ Showing <b>{len(cdb)} companies</b> extracted from the uploaded placement dataset.
            {_dataset_source_badge()}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="warning-box" style="margin-bottom:1rem;">
            ⚠️ Showing <b>static company list</b>. Admin can upload a placement dataset to make this dynamic.
        </div>""", unsafe_allow_html=True)

    lc, rc = st.columns([1,3])
    with lc:
        cgpa  = st.slider("Your CGPA", 5.0, 10.0, 7.5, 0.1)
        role_opts = ["All Roles"] + ALL_ROLES
        cur   = st.session_state.target_role
        idx   = role_opts.index(cur) if cur and cur in role_opts else 0
        sel   = st.selectbox("🎯 Target Job Role", role_opts, index=idx)
        st.session_state.target_role = None if sel == "All Roles" else sel

        all_types = sorted(set(info["type"] for info in (cdb or COMPANY_CRITERIA).values()))
        types = st.multiselect("Company Type", all_types, default=all_types[:min(4,len(all_types))])

    with rc:
        role_arg = st.session_state.target_role

        if using_dynamic:
            eligible = get_eligible_from_db(cdb, cgpa, role=role_arg, types=types if types else None)
        else:
            eligible = get_eligible_companies(cgpa, role=role_arg, types=types if types else None)

        t1l = [c for c in eligible if c["tier"]==1]
        t2l = [c for c in eligible if c["tier"]==2]
        t3l = [c for c in eligible if c["tier"]==3]

        role_pill = f'<span class="badge badge-purple">🎯 {role_arg}</span>' if role_arg else ""
        st.markdown(f"""<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem;">
            <span class="badge badge-green">✓ Eligible: {len(eligible)}</span>
            <span class="badge badge-gold">🌟 Tier 1: {len(t1l)}</span>
            <span class="badge badge-yellow">⭐ Tier 2: {len(t2l)}</span>
            <span class="badge">💫 Tier 3: {len(t3l)}</span>
            {role_pill}
        </div>""", unsafe_allow_html=True)

        tab1,tab2,tab3 = st.tabs([f"🌟 Dream ({len(t1l)})",f"⭐ Target ({len(t2l)})",f"💫 Safe ({len(t3l)})"])

        def show_co(lst):
            if not lst:
                st.info("Adjust CGPA, role, or type filters.")
                return
            for c in lst:
                init    = _hex_initials(c["name"])
                roles_h = "".join(f'<span class="badge badge-purple" style="font-size:.7rem;">{r}</span>' for r in c.get("roles",[]))
                skills_t= " · ".join(c.get("skills",[]))
                # Extra stats from dynamic db
                extra = ""
                if using_dynamic:
                    extra = f'<span class="badge badge-teal" style="font-size:.7rem;">🎓 {c.get("student_count",0)} students</span> <span class="badge badge-green" style="font-size:.7rem;">✓ {c.get("placement_rate",0)}% placed</span>'
                st.markdown(f"""<div class="co-card">
                    <div class="co-dot" style="background:{c['logo_color']};">{init}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;">
                            <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f9ff;">{c['name']}</span>
                            {_tier_badge(c['tier'])}
                            <span class="badge badge-green">{c['package']}</span>
                            {extra}
                        </div>
                        <div style="font-size:.78rem;color:#475569;margin:.3rem 0;">
                            <b style="color:#64748b;">Min CGPA:</b> {c['min_cgpa']} &nbsp;·&nbsp;
                            <b style="color:#64748b;">Type:</b> {c['type']}
                        </div>
                        <div style="font-size:.76rem;color:#64748b;margin-bottom:.4rem;"><b>Skills:</b> {skills_t}</div>
                        <div>{roles_h}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with tab1: show_co(t1l)
        with tab2: show_co(t2l)
        with tab3: show_co(t3l)

    # CGPA requirements chart
    st.markdown("### 📊 CGPA Requirements")
    src = cdb if using_dynamic else COMPANY_CRITERIA
    df_c = pd.DataFrame([{"Company":n,"Min CGPA":i["min_cgpa"],"Type":i["type"]} for n,i in src.items()])
    fig = px.bar(df_c.sort_values("Min CGPA"), x="Min CGPA", y="Company",
        color="Type", orientation="h", title="CGPA Cut-off by Company",
        color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(**_ply_base(), height=max(500, len(df_c)*28), yaxis_categoryorder="total ascending")
    fig.update_xaxes(**_ply_axis()); fig.update_yaxes(**_ply_axis())
    fig.add_vline(x=cgpa, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"  Your CGPA: {cgpa}", annotation_font_color="#f59e0b")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: PLACEMENT PREDICTION
# ══════════════════════════════════════════════════════════════════
def prediction_page():
    st.markdown("""<div class="ku-header">
        <h1>🎯 Placement Prediction</h1>
        <p>Upload your résumé first — the AI analyses it before predicting your placement chances</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("## Step 1 — Upload Your Résumé")
    st.markdown(f"""<div class="info-box" style="margin-bottom:1rem;">
        <b>📎 Accepted formats:</b>&nbsp;&nbsp;{_support_badges()}
    </div>""", unsafe_allow_html=True)

    resume_file = st.file_uploader(
        "📄 Drop your résumé (PDF, DOCX, PNG, JPG, TIFF)",
        type=STREAMLIT_TYPES, key="resume_uploader")

    if resume_file:
        if True:
            with st.spinner(f"🔍 Analysing {resume_file.name}…"):
                raw_text, method = extract_text(resume_file)
                resume_data      = parse_resume(raw_text)
                st.session_state.resume_result = resume_data
                st.session_state.resume_text   = raw_text
                st.session_state.resume_ready  = len(raw_text.strip()) > 30

            if st.session_state.resume_ready and resume_data["skills"]:
                chips = "".join(f'<span class="skill-chip">✓ {s}</span>' for s in resume_data["skills"])
                auto_cgpa = resume_data.get("cgpa")
                cgpa_line = f'<span class="badge badge-green">📊 CGPA detected: {auto_cgpa}</span>' if auto_cgpa else ""
                st.markdown(f"""<div class="resume-ok">
                    <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-bottom:.5rem;">
                        <span style="font-family:'Syne',sans-serif;font-weight:700;color:#6ee7b7;">✅ Résumé Analysed</span>
                        <span class="badge badge-green">{resume_file.name}</span>
                        <span class="badge">{method}</span>
                        <span class="badge">{resume_data.get('word_count',0)} words</span>
                        {cgpa_line}
                    </div>
                    <div>{chips}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="warning-box">
                    ⚠️ Very little text extracted. Try a text-based PDF or DOCX.
                    You can still fill fields manually and predict.
                </div>""", unsafe_allow_html=True)
                st.session_state.resume_ready = True
    else:
        st.markdown("""<div class="resume-required">
            <div style="font-size:2.5rem;margin-bottom:.6rem;">📄</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#fde68a;">Résumé Required</div>
            <p style="color:#64748b;font-size:.85rem;margin:.4rem 0 0;">
                Upload your résumé above to unlock the prediction.<br>
                Supported: <b>PDF · DOCX · PNG · JPG · BMP · TIFF</b>
            </p>
        </div>""", unsafe_allow_html=True)

    if not st.session_state.resume_ready:
        return

    st.markdown("---")
    st.markdown("## Step 2 — Review & Predict")
    rd = st.session_state.resume_result or {}

    lc, rc = st.columns([1,1.15])
    with lc:
        st.markdown("#### 📋 Academic Profile")
        cgpa_def = float(rd.get("cgpa") or 7.5)
        cgpa_def = max(5.0, min(10.0, cgpa_def))
        cgpa = st.slider("CGPA (out of 10)", 5.0, 10.0, cgpa_def, 0.1)
        # ── Internship Details ────────────────────────────────────
        st.markdown("""
        <div style="margin:1rem 0 .4rem;">
            <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f9ff;">
                💼 Internship Details
            </span>
            <span style="font-size:.75rem;color:#ef4444;margin-left:.3rem;">* All fields required</span>
        </div>""", unsafe_allow_html=True)

        # Allow up to 3 internships
        num_internships = st.selectbox(
            "How many internships have you done?",
            [0, 1, 2, 3],
            index=min(rd.get("internships", 0), 3),
            help="Select 0 if you have no internships yet.",
        )

        internship_entries = []
        for i in range(num_internships):
            st.markdown(f"""
            <div style="background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.15);
                        border-radius:12px;padding:.8rem 1rem .2rem;margin:.6rem 0 .4rem;">
                <div style="font-size:.78rem;font-weight:700;color:#93c5fd;margin-bottom:.5rem;
                            text-transform:uppercase;letter-spacing:.08em;">
                    Internship #{i+1}
                </div>
            """, unsafe_allow_html=True)

            ic1, ic2 = st.columns(2)
            with ic1:
                co_name = st.text_input(
                    f"Company Name *",
                    key=f"intern_company_{i}",
                    placeholder="e.g. Zoho, TCS, Google",
                )
            with ic2:
                job_title = st.text_input(
                    f"Job Title / Role *",
                    key=f"intern_title_{i}",
                    placeholder="e.g. ML Intern, Data Science Intern",
                )

            duration = st.selectbox(
                f"Duration *",
                ["— Select —",
                 "Less than 1 month",
                 "1 month", "2 months", "3 months",
                 "4 months", "5 months", "6 months",
                 "More than 6 months"],
                key=f"intern_duration_{i}",
            )

            work_desc = st.text_area(
                f"Work Description * (what did you build / do?)",
                key=f"intern_desc_{i}",
                placeholder="e.g. Built a customer churn prediction model using Python and scikit-learn. Deployed via Flask API on AWS EC2.",
                height=90,
            )

            tech_used = st.text_input(
                f"Technologies / Tools Used *",
                key=f"intern_tech_{i}",
                placeholder="e.g. Python, Pandas, TensorFlow, MySQL, Git",
            )

            st.markdown("</div>", unsafe_allow_html=True)

            internship_entries.append({
                "company":     co_name,
                "title":       job_title,
                "duration":    duration,
                "description": work_desc,
                "tech":        tech_used,
            })

        if num_internships == 0:
            st.markdown("""
            <div style="background:rgba(100,116,139,.07);border:1px solid rgba(100,116,139,.2);
                        border-radius:10px;padding:.7rem 1rem;font-size:.82rem;color:#64748b;margin-bottom:.5rem;">
                No internships — that's okay. Focus on projects and certifications.
            </div>""", unsafe_allow_html=True)

        # ── Certificate Details ───────────────────────────────────
        st.markdown("""
        <div style="margin:1.2rem 0 .4rem;">
            <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f9ff;">
                📜 Certificate Details
            </span>
            <span style="font-size:.75rem;color:#ef4444;margin-left:.3rem;">* All fields required</span>
        </div>""", unsafe_allow_html=True)

        num_certs = st.selectbox(
            "How many certifications have you earned?",
            [0, 1, 2, 3, 4, 5],
            index=min(rd.get("certifications", 0), 5),
        )

        cert_entries = []
        for i in range(num_certs):
            st.markdown(f"""
            <div style="background:rgba(139,92,246,.06);border:1px solid rgba(139,92,246,.15);
                        border-radius:12px;padding:.8rem 1rem .2rem;margin:.6rem 0 .4rem;">
                <div style="font-size:.78rem;font-weight:700;color:#c4b5fd;margin-bottom:.5rem;
                            text-transform:uppercase;letter-spacing:.08em;">
                    Certificate #{i+1}
                </div>
            """, unsafe_allow_html=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                cert_name = st.text_input(
                    f"Certificate Name *",
                    key=f"cert_name_{i}",
                    placeholder="e.g. AWS Cloud Practitioner, Google Data Analytics",
                )
            with cc2:
                cert_issuer = st.text_input(
                    f"Issuing Authority *",
                    key=f"cert_issuer_{i}",
                    placeholder="e.g. Amazon, Google, Coursera, NPTEL, Udemy",
                )

            cert_skills = st.text_area(
                f"Key Skills Covered *",
                key=f"cert_skills_{i}",
                placeholder="e.g. Cloud computing, EC2, S3, IAM, Lambda, VPC — describe what you learned",
                height=75,
            )

            st.markdown("</div>", unsafe_allow_html=True)

            cert_entries.append({
                "name":    cert_name,
                "issuer":  cert_issuer,
                "skills":  cert_skills,
            })

        if num_certs == 0:
            st.markdown("""
            <div style="background:rgba(100,116,139,.07);border:1px solid rgba(100,116,139,.2);
                        border-radius:10px;padding:.7rem 1rem;font-size:.82rem;color:#64748b;margin-bottom:.5rem;">
                No certifications yet — consider NPTEL, Coursera, or Google free courses.
            </div>""", unsafe_allow_html=True)

        # ── Project count (still numeric — quick to fill) ─────────
        st.markdown("""
        <div style="margin:1.2rem 0 .4rem;">
            <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f9ff;">
                🔧 Projects
            </span>
        </div>""", unsafe_allow_html=True)
        projects = st.selectbox(
            "Number of personal / academic projects completed",
            list(range(11)),
            index=min(rd.get("projects", 2), 10),
            help="Include GitHub projects, college mini-projects, hackathon builds, etc.",
        )

        # ── Self-assessment ───────────────────────────────────────
        st.markdown("""
        <div style="margin:1.2rem 0 .4rem;">
            <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f9ff;">
                🧠 Skill Self-Assessment
            </span>
        </div>""", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            comm = st.select_slider(
                "Communication Skills",
                options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: ["Poor","Fair","Good","Very Good","Excellent"][x-1],
            )
        with c4:
            coding = st.select_slider(
                "Coding / DSA Level",
                options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: ["Beginner","Elementary","Intermediate","Advanced","Expert"][x-1],
            )

        # ── Validation before allowing predict ────────────────────
        def _validate_form():
            """Returns (valid: bool, error_message: str)"""
            for idx, e in enumerate(internship_entries):
                n = idx + 1
                if not e["company"].strip():
                    return False, f"Internship #{n}: Company Name is required."
                if not e["title"].strip():
                    return False, f"Internship #{n}: Job Title is required."
                if e["duration"] == "— Select —":
                    return False, f"Internship #{n}: Please select a duration."
                if not e["description"].strip():
                    return False, f"Internship #{n}: Work Description is required."
                if not e["tech"].strip():
                    return False, f"Internship #{n}: Technologies Used is required."
            for idx, c in enumerate(cert_entries):
                n = idx + 1
                if not c["name"].strip():
                    return False, f"Certificate #{n}: Certificate Name is required."
                if not c["issuer"].strip():
                    return False, f"Certificate #{n}: Issuing Authority is required."
                if not c["skills"].strip():
                    return False, f"Certificate #{n}: Key Skills Covered is required."
            return True, ""

        # ── Derive numeric counts for ML model ───────────────────
        internships    = num_internships
        certifications = num_certs

        # Extract extra skills from context text for skill page cross-fill
        def _extract_context_skills(entries_list, field_keys):
            import re
            combined = " ".join(
                e.get(k, "") for e in entries_list for k in field_keys
            )
            from utils.resume_parser import SKILL_MAP
            return [label for pat, label in SKILL_MAP.items()
                    if re.search(pat, combined, re.IGNORECASE)]

        context_skills = _extract_context_skills(
            internship_entries, ["description", "tech"]
        ) + _extract_context_skills(
            cert_entries, ["skills"]
        )
        # Merge with resume skills — deduplicated
        existing_skills = (st.session_state.resume_result or {}).get("skills", [])
        all_detected_skills = list(dict.fromkeys(existing_skills + context_skills))
        if context_skills:
            st.session_state.resume_result = {
                **(st.session_state.resume_result or {}),
                "skills": all_detected_skills,
            }

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        clicked = st.button("🔮  Predict My Chances", use_container_width=True)

        # Show validation errors immediately after click
        if clicked:
            valid, err_msg = _validate_form()
            if not valid:
                st.error(f"⚠️ {err_msg}")
                clicked = False  # block prediction

    with rc:
        if clicked:
            prob  = predict(st.session_state.model, st.session_state.scaler,
                            cgpa, internships, projects, certifications, comm, coding)
            pct   = prob * 100
            clr   = "#3b82f6" if pct>=65 else ("#f59e0b" if pct>=40 else "#ef4444")
            label = "High Chances 🎉" if pct>=65 else ("Moderate Chances ⚡" if pct>=40 else "Needs Improvement ⚠️")

            pred_payload = {
                "cgpa": cgpa, "internships": internships, "projects": projects,
                "certifications": certifications, "comm": comm, "coding": coding,
                "probability": prob, "label": label,
                # Rich context for admin records
                "internship_details": [
                    f"{e['title']} @ {e['company']} ({e['duration']}) — {e['tech']}"
                    for e in internship_entries
                ],
                "certificate_details": [
                    f"{c['name']} by {c['issuer']}"
                    for c in cert_entries
                ],
                "context_skills_extracted": context_skills,
            }
            save_student_submission({**st.session_state.user_data,
                                     "target_role":st.session_state.target_role or ""}, pred_payload, rd)
            st.session_state.last_prediction = pred_payload

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix":"%","font":{"size":38,"color":"#f0f9ff","family":"Syne"},"valueformat":".1f"},
                title={"text":"Placement Probability","font":{"color":"#64748b","size":13}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#334155","tickfont":{"color":"#475569"}},
                       "bar":{"color":clr},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                       "steps":[{"range":[0,40],"color":"rgba(239,68,68,.10)"},
                                 {"range":[40,65],"color":"rgba(245,158,11,.10)"},
                                 {"range":[65,100],"color":"rgba(59,130,246,.10)"}],
                       "threshold":{"line":{"color":"#f59e0b","width":2},"thickness":.8,"value":65}},
            ))
            fig.update_layout(**_ply_base(), height=290)
            st.plotly_chart(fig, use_container_width=True)

            if pct>=65:   st.success(f"🎉 **{label} ({pct:.1f}%)**")
            elif pct>=40: st.warning(f"⚡ **{label} ({pct:.1f}%)**")
            else:         st.error(  f"⚠️ **{label} ({pct:.1f}%)**")

            factors = {"CGPA":min(cgpa/10,1),"Internships":min(internships/4,1),
                       "Projects":min(projects/6,1),"Certifications":min(certifications/6,1),
                       "Communication":comm/5,"Coding":coding/5}
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=list(factors.values()),theta=list(factors.keys()),
                fill="toself",fillcolor="rgba(59,130,246,.18)",
                line=dict(color="#3b82f6",width=2),name="Your Profile"))
            fig2.add_trace(go.Scatterpolar(r=[.9,.75,.75,.75,.8,.9],theta=list(factors.keys()),
                fill="toself",fillcolor="rgba(6,182,212,.07)",
                line=dict(color="#06b6d4",width=1.5,dash="dot"),name="Benchmark"))
            fig2.update_layout(**_ply_base(), showlegend=True, height=310,
                title_text="Profile vs Benchmark",
                polar=dict(radialaxis=dict(visible=True,range=[0,1],
                           tickfont=dict(color="#475569",size=9),gridcolor="rgba(255,255,255,.07)"),
                           angularaxis=dict(tickfont=dict(color="#94a3b8",size=11)),bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### 💡 Recommendations")
            recs=[]
            if cgpa<7.5:         recs.append(("📚","Aim for CGPA ≥ 7.5 — unlocks more companies."))
            if internships==0:   recs.append(("💼","No internships detected — apply to internship programmes on Internshala or LinkedIn."))
            elif internships<2:  recs.append(("💼","Try to complete at least one more internship before final year."))
            if projects<3:       recs.append(("🔧","Build 3+ end-to-end projects and push them to GitHub."))
            if certifications==0:recs.append(("📜","Earn at least 1 certification — start with NPTEL (free) or Google Certificates."))
            elif certifications<2:recs.append(("📜","Aim for 2+ certifications to strengthen your profile."))
            if comm<4:           recs.append(("🗣️","Practise communication — join GDs and mock interviews."))
            if coding<4:         recs.append(("💻","Solve DSA daily on LeetCode / HackerRank."))
            if context_skills and len(context_skills) >= 3:
                recs.insert(0, ("✅", f"Good — we detected {len(context_skills)} skills from your internship & certificate details: {', '.join(context_skills[:5])}{' …' if len(context_skills)>5 else ''}."))
            elif not context_skills and not existing_skills:
                recs.append(("📄","Add more detail to your internship descriptions so we can extract your skills automatically."))
            if recs:
                for icon,text in recs:
                    st.markdown(f"""<div style="display:flex;gap:.7rem;align-items:flex-start;
                                padding:.6rem 0;border-bottom:1px solid rgba(255,255,255,.05);">
                        <span style="font-size:1.1rem;">{icon}</span>
                        <span style="color:#94a3b8;font-size:.87rem;">{text}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("🌟 Excellent profile!")

            st.markdown("""<div style="background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.2);
                        border-radius:10px;padding:.7rem 1rem;margin-top:.8rem;font-size:.78rem;color:#6ee7b7;">
                ✅ Saved to admin panel.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="ku-card" style="text-align:center;padding:3.5rem 2rem;">
                <div style="font-size:3.5rem;margin-bottom:1rem;">🤖</div>
                <div style="color:#475569;font-size:.92rem;line-height:1.7;">
                    Résumé uploaded ✅<br>Fill the fields and click<br>
                    <b style="color:#93c5fd;">Predict My Chances</b>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: SKILL RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
def skills_page():
    user = st.session_state.user_data
    st.markdown("""<div class="ku-header">
        <h1>🛠️ Skill Recommendations</h1>
        <p>Personalised learning roadmap for your programme</p>
    </div>""", unsafe_allow_html=True)

    cat = get_skill_category(user["department"])
    ds  = DEPARTMENT_SKILLS.get(cat, DEPARTMENT_SKILLS["cse"])
    all_skills = sorted(set(ds["core"]+ds["recommended"]+ds["additional"]+[
        "Python","Java","C++","JavaScript","SQL","Git","Linux","Docker",
        "Machine Learning","Deep Learning","Cloud Computing","React","Node.js"]))

    rr      = st.session_state.resume_result or {}
    prefill = [s for s in all_skills if s in rr.get("skills",[])]

    lc, rc = st.columns([1,2.1])
    with lc:
        st.markdown("#### Your Current Skills")
        if prefill:
            st.markdown(f'<div class="success-box" style="margin-bottom:.8rem;">✅ Auto-filled {len(prefill)} skills from résumé.</div>',unsafe_allow_html=True)
        current = st.multiselect("Select all skills you have:", all_skills, default=prefill or all_skills[:2])
        badges  = "".join(f'<span class="badge">{s}</span>' for s in current)
        st.markdown(f"<div style='margin-top:.6rem;'>{badges}</div>", unsafe_allow_html=True)

    with rc:
        cd = sum(1 for s in ds["core"] if s in current)
        rd = sum(1 for s in ds["recommended"] if s in current)
        t1,t2,t3,t4 = st.tabs(["🔴 Core","🟡 Recommended","🟢 Additional","⚠️ Gaps"])
        with t1:
            for s in ds["core"]: st.markdown(f"{'✅' if s in current else '❌'} &nbsp; {s}")
            st.progress(cd/max(len(ds["core"]),1)); st.caption(f"Core: {cd}/{len(ds['core'])}")
        with t2:
            for s in ds["recommended"]: st.markdown(f"{'✅' if s in current else '⬜'} &nbsp; {s}")
            st.progress(rd/max(len(ds["recommended"]),1)); st.caption(f"Recommended: {rd}/{len(ds['recommended'])}")
        with t3:
            for s in ds["additional"]: st.markdown(f"{'✅' if s in current else '⬜'} &nbsp; {s}")
        with t4:
            pri  = ds["core"]+ds["recommended"]+ds["additional"]
            gaps = [s for s in pri if s not in current][:10]
            if gaps:
                for i,s in enumerate(gaps,1):
                    tier="🔴" if s in ds["core"] else("🟡" if s in ds["recommended"] else "🟢")
                    st.markdown(f"{tier} **{i}.** {s}")
                st.markdown("---")
                st.markdown("**📚 Platforms:** Coursera · Udemy · NPTEL · LeetCode · HackerRank · GeeksforGeeks")
            else:
                st.success("🎉 You've covered almost all recommended skills!")

    st.markdown("### 🗺️ 6-Month Roadmap")
    rm = pd.DataFrame({
        "Task":  ["Core Skills","DSA & Problem Solving","Project Building","Internship","Advanced Concepts","Interview Prep"],
        "Start": pd.to_datetime(["2025-01-01","2025-01-20","2025-03-01","2025-03-15","2025-05-01","2025-05-20"]),
        "End":   pd.to_datetime(["2025-03-01","2025-03-01","2025-05-01","2025-05-01","2025-07-01","2025-07-01"]),
        "Phase": ["Phase 1","Phase 1","Phase 2","Phase 2","Phase 3","Phase 3"],
    })
    fig = px.timeline(rm,x_start="Start",x_end="End",y="Task",color="Phase",
        title="Skill Development Timeline",color_discrete_sequence=["#3b82f6","#06b6d4","#8b5cf6"])
    fig.update_yaxes(autorange="reversed")
    _chart(fig, 360)

# ══════════════════════════════════════════════════════════════════
# PAGE: STUDENT SUBMISSIONS  (admin only)
# ══════════════════════════════════════════════════════════════════
def submissions_page():
    if _is_student():
        _access_denied()
        return

    st.markdown("""<div class="ku-header">
        <h1>👥 Student Submissions</h1>
        <p>Live prediction results from all students</p>
    </div>""", unsafe_allow_html=True)

    sub_df = submissions_to_df()
    if sub_df.empty:
        st.info("No student submissions yet.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Submissions", len(sub_df))
    c2.metric("Unique Students",   sub_df["email"].nunique())
    c3.metric("Avg Probability",   f"{sub_df['Placement_Probability_%'].mean():.1f}%")
    c4.metric("Avg CGPA",          f"{sub_df['CGPA'].mean():.2f}")

    st.dataframe(sub_df, use_container_width=True, height=380)
    st.download_button("📥 Download Submissions", sub_df.to_csv(index=False),
        "student_submissions.csv","text/csv", use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        fig=px.histogram(sub_df, x="Placement_Probability_%", nbins=20,
            title="Prediction Distribution", color_discrete_sequence=["#3b82f6"])
        _chart(fig, 300)
    with cb:
        if len(sub_df) > 2:
            fig2=px.scatter(sub_df, x="CGPA", y="Placement_Probability_%",
                hover_data=["name","register_number"],
                title="CGPA vs Probability", opacity=0.8,
                color_discrete_sequence=["#06b6d4"])
            _chart(fig2, 300)

    st.markdown("---")
    if st.button("🗑️ Clear All Submissions", use_container_width=True):
        clear_store(); st.success("Cleared."); st.rerun()

# ══════════════════════════════════════════════════════════════════
# PAGE: MY ACCOUNT
# ══════════════════════════════════════════════════════════════════
def account_page():
    user = st.session_state.user_data
    name = user.get("name","User")
    st.markdown("""<div class="ku-header">
        <h1>👤 My Account</h1>
        <p>Your personal profile and placement summary</p>
    </div>""", unsafe_allow_html=True)

    if _is_admin():
        st.markdown("""<div class="ku-card-accent">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#fca5a5;margin-bottom:.5rem;">🛡️ Administrator Account</div>
            <p style="color:#94a3b8;font-size:.85rem;margin:0;">Full access to all data and analytics.</p>
        </div>""", unsafe_allow_html=True)
        return

    lc, rc = st.columns([1,2])
    with lc:
        initials = "".join(w[0].upper() for w in name.split()[:2])
        st.markdown(f"""<div class="ku-card" style="text-align:center;padding:2rem 1rem;">
            <div style="width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                        display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;
                        font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#fff;">{initials}</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#f0f9ff;">{name}</div>
            <div style="font-size:.8rem;color:#64748b;margin-top:.3rem;">{user['email']}</div>
            <div style="margin-top:.8rem;"><span class="badge badge-green">🎓 Student</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="ku-card">
            <div class="section-label">Account Details</div>
            <div style="display:grid;gap:.7rem;font-size:.83rem;margin-top:.5rem;">
                <div><span style="color:#475569;">Register No.</span><br><b style="color:#cbd5e1;">{user['register_number']}</b></div>
                <div><span style="color:#475569;">Year of Joining</span><br><b style="color:#cbd5e1;">{user['year_of_joining']}</b></div>
                <div><span style="color:#475569;">Current Year</span><br><b style="color:#cbd5e1;">{user['current_year']}</b></div>
                <div><span style="color:#475569;">Programme</span><br><b style="color:#cbd5e1;">{user['program_type']}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with rc:
        st.markdown(f"""<div class="ku-card-accent">
            <div class="section-label">Department</div>
            <div style="font-size:.9rem;color:#e2e8f0;font-weight:500;margin-top:.3rem;">{user['department']}</div>
            <div style="font-size:.78rem;color:#475569;">{user['school']}</div>
        </div>""", unsafe_allow_html=True)

        tr = st.session_state.target_role or "Not set"
        st.markdown(f"""<div class="ku-card">
            <div class="section-label">Target Job Role</div>
            <span class="badge badge-purple">🎯 {tr}</span>
            <p style="color:#475569;font-size:.78rem;margin-top:.5rem;">Set in Company Eligibility.</p>
        </div>""", unsafe_allow_html=True)

        rr     = st.session_state.resume_result or {}
        skills = rr.get("skills",[])
        if skills:
            chips = "".join(f'<span class="skill-chip">✓ {s}</span>' for s in skills)
            auto_cgpa = rr.get("cgpa")
            cgpa_line = f'<div style="font-size:.78rem;color:#6ee7b7;margin-top:.3rem;">📊 CGPA from résumé: <b>{auto_cgpa}</b></div>' if auto_cgpa else ""
            st.markdown(f"""<div class="ku-card">
                <div class="section-label">Skills from Résumé ({len(skills)})</div>
                {cgpa_line}<div style="margin-top:.5rem;">{chips}</div>
            </div>""", unsafe_allow_html=True)

        lp = st.session_state.last_prediction
        if lp:
            pct = lp["probability"]*100
            clr = "#6ee7b7" if pct>=65 else ("#fcd34d" if pct>=40 else "#fca5a5")
            st.markdown(f"""<div class="ku-card">
                <div class="section-label">Last Prediction</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:{clr};">{pct:.1f}%</div>
                <div style="font-size:.8rem;color:#64748b;margin-top:.2rem;">{lp['label']}</div>
                <div style="font-size:.76rem;color:#334155;margin-top:.3rem;">CGPA:{lp['cgpa']} · Intern:{lp['internships']} · Projects:{lp['projects']}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    if _is_admin():
        PAGES = {
            "🏠  Dashboard":            "dashboard",
            "📂  Dataset Upload":       "dataset_upload",
            "👥  Student Submissions":  "submissions",
            "🎯  Placement Prediction": "prediction",
            "🛠️  Skill Recommendations":"skills",
            "🏢  Company Eligibility":  "companies",
            "👤  My Account":           "account",
        }
    else:
        PAGES = {
            "🏠  Dashboard":            "dashboard",
            "🎯  Placement Prediction": "prediction",
            "🛠️  Skill Recommendations":"skills",
            "🏢  Company Eligibility":  "companies",
            "👤  My Account":           "account",
        }

    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:1.5rem 0 .6rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;color:#f0f9ff;">🎓 KU Placement</div>
            <div style="font-size:.72rem;color:#334155;text-transform:uppercase;letter-spacing:.1em;">Intelligence v5.0</div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        user = st.session_state.user_data
        name = user.get("name","User")
        dept = user.get("department","")
        dept_s = dept[:34]+"…" if len(dept)>36 else dept
        role_lbl = '<span class="badge badge-admin">🛡️ Admin</span>' if _is_admin() else '<span class="badge badge-green">🎓 Student</span>'
        r_status = ""
        if _is_student():
            r_status = '<span class="badge badge-green" style="font-size:.68rem;">📄 ✓</span>' \
                       if st.session_state.resume_ready else \
                       '<span class="badge badge-orange" style="font-size:.68rem;">📄 needed</span>'

        st.markdown(f"""<div style="padding:.7rem .9rem;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:10px;margin-bottom:1rem;">
            <div style="font-size:.9rem;font-weight:600;color:#e2e8f0;">{name}</div>
            <div style="font-size:.75rem;color:#64748b;">{user.get('email','')}</div>
            {"" if _is_admin() else f'<div style="font-size:.72rem;color:#475569;">{dept_s}</div>'}
            {"" if _is_admin() else f'<div style="font-size:.72rem;color:#334155;">{user.get("current_year","")}</div>'}
            <div style="margin-top:.5rem;display:flex;gap:4px;flex-wrap:wrap;">{role_lbl}{r_status}</div>
        </div>""", unsafe_allow_html=True)

        # Admin: dataset status + submission count
        if _is_admin():
            sc = get_submission_count()
            ds_badge = _dataset_source_badge()
            st.markdown(f"""<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                            border-radius:8px;padding:.6rem .8rem;font-size:.78rem;margin-bottom:.8rem;">
                {ds_badge}<br>
                <span style="color:#6ee7b7;margin-top:.3rem;display:block;">📥 <b>{sc}</b> submission{"s" if sc!=1 else ""}</span>
            </div>""", unsafe_allow_html=True)

        selected = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        if st.button("🚪  Logout", use_container_width=True):
            _logout()
        st.markdown("""<div style="text-align:center;font-size:.68rem;color:#1e293b;margin-top:.8rem;">
            © 2025 Karunya University<br>AI Placement Intelligence v5.0
        </div>""", unsafe_allow_html=True)

    page = PAGES[selected]
    if   page=="dashboard":      dashboard_page()
    elif page=="dataset_upload":  admin_dataset_page()
    elif page=="submissions":     submissions_page()
    elif page=="prediction":      prediction_page()
    elif page=="skills":          skills_page()
    elif page=="companies":       company_page()
    elif page=="account":         account_page()


if __name__ == "__main__":
    main()
