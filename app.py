"""
╔══════════════════════════════════════════════════════════════════╗
║     KARUNYA UNIVERSITY — AI PLACEMENT INTELLIGENCE SYSTEM       ║
║                        Version 4.0                              ║
╚══════════════════════════════════════════════════════════════════╝

New in v4.0:
  ✦ Resume REQUIRED before prediction runs
  ✦ Multi-format resume support: PDF · DOCX · PNG · JPG · BMP · TIFF
  ✦ OCR for scanned PDFs and image résumés (via pytesseract)
  ✦ Every student prediction is saved → admin can view all submissions
  ✦ Admin Dataset page shows live student submissions table
"""

import re
import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.auth import (
    validate_email, parse_register_number,
    extract_name, is_admin, ADMIN_EMAIL,
)
from utils.ml_model import generate_sample_data, train_model, predict
from utils.companies import COMPANY_CRITERIA, get_eligible_companies, ALL_ROLES
from utils.departments import DEPARTMENTS, DEPARTMENT_SCHOOLS, DEPARTMENT_SKILLS, get_skill_category
from utils.resume_parser import (
    extract_text, parse_resume, get_support_status,
    STREAMLIT_TYPES,
)
from utils.student_store import (
    save_student_submission, get_all_submissions,
    get_submission_count, clear_store, submissions_to_df,
)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="KU Placement Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],.main{
    background:#080e1d!important;color:#dde3f0!important;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#0b1120!important;border-right:1px solid rgba(96,165,250,.1)!important}
[data-testid="stSidebar"] *{color:#c4cede!important}
h1,h2,h3,h4,h5,h6{font-family:'Syne',sans-serif!important;letter-spacing:-.02em}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;display:none}

[data-testid="stSidebar"] [role="radiogroup"] label{
    display:flex!important;align-items:center!important;padding:9px 14px!important;
    border-radius:9px!important;margin:2px 0!important;cursor:pointer!important;
    transition:background .18s,color .18s!important;font-size:.88rem!important;font-weight:500!important}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(96,165,250,.08)!important}

[data-testid="metric-container"]{
    background:rgba(255,255,255,.035)!important;border:1px solid rgba(255,255,255,.07)!important;
    border-radius:14px!important;padding:1rem 1.3rem!important;transition:border-color .2s}
[data-testid="metric-container"]:hover{border-color:rgba(96,165,250,.3)!important}
[data-testid="metric-container"] label{color:#64748b!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:.06em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#f0f9ff!important;font-family:'Syne',sans-serif!important;font-size:1.65rem!important}
[data-testid="metric-container"] [data-testid="stMetricDelta"] svg{display:none}

[data-testid="stTextInput"] input{
    background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;
    border-radius:10px!important;color:#f1f5f9!important;padding:.6rem .9rem!important;
    transition:border-color .2s,box-shadow .2s!important}
[data-testid="stTextInput"] input:focus{border-color:#3b82f6!important;box-shadow:0 0 0 3px rgba(59,130,246,.15)!important;outline:none!important}
[data-testid="stTextInput"] input::placeholder{color:#475569!important}
[data-testid="stSelectbox"]>div>div,[data-baseweb="select"]>div{
    background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;
    border-radius:10px!important;color:#f1f5f9!important}

.stButton>button{
    background:linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%)!important;color:#fff!important;
    border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;
    font-weight:700!important;font-size:.92rem!important;padding:.65rem 1.6rem!important;
    transition:transform .2s,box-shadow .2s!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(59,130,246,.4)!important}
[data-testid="stDownloadButton"] button{
    background:rgba(255,255,255,.06)!important;border:1px solid rgba(255,255,255,.14)!important;
    color:#94a3b8!important;border-radius:10px!important}

[data-baseweb="tab-list"]{background:rgba(255,255,255,.04)!important;border-radius:12px!important;
    padding:4px!important;gap:4px!important;border:1px solid rgba(255,255,255,.06)!important}
[data-baseweb="tab"]{background:transparent!important;border-radius:8px!important;
    color:#64748b!important;font-weight:500!important}
[aria-selected="true"][data-baseweb="tab"]{background:rgba(59,130,246,.22)!important;color:#93c5fd!important;font-weight:600!important}

[data-testid="stExpander"]{background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.07)!important;
    border-radius:12px!important;margin-bottom:6px!important;transition:border-color .2s!important}
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

/* ── Custom classes ──────────────────────────────────────────── */
.ku-header{background:linear-gradient(135deg,#0c1628 0%,#13233f 60%,#0f1e38 100%);border:1px solid rgba(96,165,250,.18);border-radius:20px;padding:2.4rem 2rem;text-align:center;margin-bottom:1.8rem;position:relative;overflow:hidden}
.ku-header::before{content:'';position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse at 20% 50%,rgba(59,130,246,.13) 0%,transparent 55%),radial-gradient(ellipse at 80% 50%,rgba(6,182,212,.09) 0%,transparent 55%)}
.ku-header h1{font-size:1.9rem;color:#f0f9ff;margin:0;position:relative}
.ku-header p{color:#64748b;margin:.5rem 0 0;font-size:.9rem;position:relative}

.ku-card{background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.075);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem;transition:border-color .2s}
.ku-card:hover{border-color:rgba(96,165,250,.2)}
.ku-card-accent{background:linear-gradient(135deg,rgba(59,130,246,.08),rgba(6,182,212,.05));border:1px solid rgba(59,130,246,.22);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem}

/* Resume upload zone */
.resume-required{background:rgba(245,158,11,.07);border:2px dashed rgba(245,158,11,.4);border-radius:14px;padding:1.8rem;text-align:center;margin-bottom:1rem}
.resume-ok{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.3);border-radius:14px;padding:1.2rem;margin-bottom:1rem}

/* Company card */
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

.section-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#475569;margin-bottom:.4rem}
.info-box{background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.2);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem}
.success-box{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#a7f3d0}
.warning-box{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#fde68a}
.danger-box{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#fca5a5}

.skill-chip{display:inline-flex;align-items:center;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);color:#6ee7b7;padding:4px 10px;border-radius:8px;font-size:.78rem;font-weight:500;margin:3px}
.access-denied{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-radius:16px;padding:3rem 2rem;text-align:center;margin:2rem 0}

/* Format badges for resume */
.fmt-badge{display:inline-flex;align-items:center;gap:4px;padding:5px 12px;border-radius:8px;font-size:.76rem;font-weight:600;margin:3px}
.fmt-pdf{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.3);color:#fca5a5}
.fmt-docx{background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);color:#93c5fd}
.fmt-img{background:rgba(139,92,246,.15);border:1px solid rgba(139,92,246,.3);color:#c4b5fd}
.fmt-ok{background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);color:#6ee7b7}
.fmt-na{background:rgba(100,116,139,.15);border:1px solid rgba(100,116,139,.3);color:#94a3b8}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PLOTLY HELPERS  (no yaxis conflict)
# ══════════════════════════════════════════════════════════════════
def _ply_base():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="DM Sans", size=12),
        margin=dict(t=48, b=28, l=8, r=8),
        title_font=dict(color="#94a3b8", size=13, family="Syne"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                    bordercolor="rgba(255,255,255,.08)", borderwidth=1),
    )

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
    "resume_result": None,    # parsed resume dict
    "resume_text": "",        # raw resume text
    "resume_ready": False,    # True once a resume is successfully parsed
    "target_role": None,
    "last_prediction": None,  # last prediction dict for admin store
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

def _logout():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

def _is_admin():   return st.session_state.role == "admin"
def _is_student(): return st.session_state.role == "student"

def _access_denied():
    st.markdown("""
    <div class="access-denied">
        <div style="font-size:2.5rem;margin-bottom:.8rem;">🔒</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;color:#fca5a5;font-weight:700;">Access Restricted</div>
        <p style="color:#64748b;margin:.4rem 0 0;font-size:.88rem;">This section is only visible to administrators.</p>
    </div>""", unsafe_allow_html=True)

def _tier_badge(tier):
    m = {1:'<span class="badge badge-gold">🌟 Tier 1</span>',
         2:'<span class="badge badge-yellow">⭐ Tier 2</span>',
         3:'<span class="badge">💫 Tier 3</span>'}
    return m.get(tier,"")

def _hex_initials(name):
    p = name.split()
    return (p[0][0]+(p[1][0] if len(p)>1 else "")).upper()

def _support_badges():
    """Render format support chips."""
    s = get_support_status()
    pdf  = '<span class="fmt-badge fmt-pdf">📄 PDF</span>' if s["PDF"] else '<span class="fmt-badge fmt-na">📄 PDF ✗</span>'
    docx = '<span class="fmt-badge fmt-docx">📝 DOCX</span>' if s["DOCX"] else '<span class="fmt-badge fmt-na">📝 DOCX ✗</span>'
    ocr  = '<span class="fmt-badge fmt-img">🖼️ Images / OCR</span>' if s["OCR (images / scanned PDFs)"] else '<span class="fmt-badge fmt-na">🖼️ Images ✗</span>'
    return f"{pdf}{docx}{ocr}"

# ══════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ══════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("""
    <div class="ku-header" style="padding:3rem 2rem;">
        <h1 style="font-size:2.2rem;">🎓 Karunya University</h1>
        <p style="font-size:1rem;color:#475569;">AI-Powered Placement Intelligence System</p>
    </div>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1,1.5,1])
    with mid:
        mode = st.radio("", ["🎓 Student", "🔑 Admin"], horizontal=True, label_visibility="collapsed")
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
                    with st.spinner("Initialising analytics engine…"):
                        _boot_model()
                    st.session_state.logged_in = True
                    st.session_state.role      = "student"
                    st.session_state.user_data = {
                        "email":           email.strip().lower(),
                        "name":            extract_name(email),
                        "register_number": reg_no.upper().strip(),
                        "department":      dept,
                        "school":          DEPARTMENT_SCHOOLS.get(dept,"Karunya University"),
                        "year_of_joining": parsed["year_of_joining"],
                        "program_type":    parsed["program_type"],
                        "roll_number":     parsed["roll_number"],
                        "current_year":    yr,
                    }
                    st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <b>📋 Register Number:</b> <code>URK25AI1074</code><br>
            <span style="color:#475569;font-size:.82rem;">URK = Undergrad · 25 = Year · AI = Dept · 1074 = Roll</span>
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
        st.markdown(f"""
        <div class="ku-header">
            <h1>Admin Dashboard 🛡️</h1>
            <p>Full dataset analytics · {sub_count} student submission{"s" if sub_count!=1 else ""} received</p>
        </div>""", unsafe_allow_html=True)

        rate   = df["Placed"].mean()*100
        avg_c  = df["CGPA"].mean()
        placed = int(df["Placed"].sum())
        total  = len(df)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Placement Rate",     f"{rate:.1f}%")
        c2.metric("Average CGPA",       f"{avg_c:.2f}")
        c3.metric("Total (Synthetic)",  f"{total:,}")
        c4.metric("Placed",             f"{placed:,}")
        c5.metric("Live Submissions",   f"{sub_count}")

        ch1, ch2 = st.columns(2)
        with ch1:
            fig = go.Figure(go.Pie(values=[placed,total-placed],labels=["Placed","Not Placed"],
                hole=0.58,marker_colors=["#3b82f6","#1e293b"],textinfo="percent",
                hovertemplate="%{label}: %{value}<extra></extra>"))
            fig.update_layout(**_ply_base(), title_text="Placement Split", height=270)
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            fig2=px.histogram(df,x="CGPA",color="Placed",
                color_discrete_map={0:"#ef4444",1:"#3b82f6"},
                barmode="overlay",opacity=0.72,title="CGPA Distribution")
            _chart(fig2,270)

        st.markdown("### 📊 Full Analytics")
        t1,t2,t3=st.tabs(["📈 CGPA","💼 Internships","🧠 Skills"])
        with t1:
            ca,cb=st.columns(2)
            with ca:
                _chart(px.box(df,x="Placed",y="CGPA",color="Placed",
                    color_discrete_map={0:"#ef4444",1:"#3b82f6"},
                    title="CGPA: Placed vs Not Placed"),340)
            with cb:
                bins=pd.cut(df["CGPA"],bins=[5,6,7,8,9,10],labels=["5–6","6–7","7–8","8–9","9–10"])
                r2=df.groupby(bins,observed=False)["Placed"].mean()*100
                _chart(px.bar(x=r2.index.astype(str),y=r2.values,
                    labels={"x":"CGPA Range","y":"Placement Rate (%)"},
                    title="Placement Rate by CGPA",
                    color=r2.values,color_continuous_scale="Blues"),340)
        with t2:
            ca,cb=st.columns(2)
            with ca:
                r=df.groupby("Internships")["Placed"].mean()*100
                _chart(px.bar(x=r.index,y=r.values,
                    labels={"x":"Internships","y":"Placement Rate (%)"},
                    title="Placement Rate by Internships",
                    color=r.values,color_continuous_scale="Blues"),340)
            with cb:
                _chart(px.scatter(df,x="CGPA",y="Internships",color="Placed",
                    color_discrete_map={0:"#ef4444",1:"#3b82f6"},
                    size="Projects",title="CGPA vs Internships",opacity=0.65),340)
        with t3:
            LBL=["Beginner","Elementary","Intermediate","Advanced","Expert"]
            ca,cb=st.columns(2)
            with ca:
                r=df.groupby("Coding_Skill")["Placed"].mean()*100
                _chart(px.bar(x=LBL[:len(r)],y=r.values,
                    labels={"x":"Coding Skill","y":"Placement Rate (%)"},
                    title="Coding Skill Impact",
                    color=r.values,color_continuous_scale="Viridis"),340)
            with cb:
                r2=df.groupby("Communication_Skill")["Placed"].mean()*100
                LB2=["Poor","Fair","Good","Very Good","Excellent"]
                _chart(px.bar(x=LB2[:len(r2)],y=r2.values,
                    labels={"x":"Communication","y":"Placement Rate (%)"},
                    title="Communication Skill Impact",
                    color=r2.values,color_continuous_scale="Oranges"),340)
    else:
        st.markdown(f"""
        <div class="ku-header">
            <h1>Welcome, {name}! 👋</h1>
            <p>{user['department']}&nbsp;·&nbsp;{user['school']}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="ku-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2.5rem;margin-bottom:.8rem;">📊</div>
            <p style="color:#64748b;font-size:.9rem;">
                Use the sidebar to navigate to
                <b style="color:#93c5fd;">Placement Prediction</b>,
                <b style="color:#93c5fd;">Company Eligibility</b>, or
                <b style="color:#93c5fd;">My Account</b>.
            </p>
        </div>""", unsafe_allow_html=True)

        acc = st.session_state.model_accuracy or 0
        c1,c2,c3 = st.columns(3)
        c1.metric("ML Model",        st.session_state.model_name or "—")
        c2.metric("Model Accuracy",  f"{acc*100:.1f}%")
        c3.metric("Training Records",f"{len(df):,}")

# ══════════════════════════════════════════════════════════════════
# PAGE: MY ACCOUNT
# ══════════════════════════════════════════════════════════════════
def account_page():
    user = st.session_state.user_data
    name = user.get("name","User")

    st.markdown("""
    <div class="ku-header">
        <h1>👤 My Account</h1>
        <p>Your personal profile and placement summary</p>
    </div>""", unsafe_allow_html=True)

    if _is_admin():
        st.markdown("""
        <div class="ku-card-accent">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#fca5a5;margin-bottom:.5rem;">🛡️ Administrator Account</div>
            <p style="color:#94a3b8;font-size:.85rem;margin:0;">Full access to all student data and analytics.</p>
        </div>""", unsafe_allow_html=True)
        return

    lc, rc = st.columns([1,2])

    with lc:
        initials = "".join(w[0].upper() for w in name.split()[:2])
        st.markdown(f"""
        <div class="ku-card" style="text-align:center;padding:2rem 1rem;">
            <div style="width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                        display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;
                        font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#fff;">{initials}</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#f0f9ff;">{name}</div>
            <div style="font-size:.8rem;color:#64748b;margin-top:.3rem;">{user['email']}</div>
            <div style="margin-top:.8rem;"><span class="badge badge-green">🎓 Student</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="ku-card">
            <div class="section-label">Account Details</div>
            <div style="display:grid;gap:.7rem;font-size:.83rem;margin-top:.5rem;">
                <div><span style="color:#475569;">Register No.</span><br><b style="color:#cbd5e1;">{user['register_number']}</b></div>
                <div><span style="color:#475569;">Year of Joining</span><br><b style="color:#cbd5e1;">{user['year_of_joining']}</b></div>
                <div><span style="color:#475569;">Current Year</span><br><b style="color:#cbd5e1;">{user['current_year']}</b></div>
                <div><span style="color:#475569;">Programme</span><br><b style="color:#cbd5e1;">{user['program_type']}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with rc:
        st.markdown(f"""
        <div class="ku-card-accent">
            <div class="section-label">Department</div>
            <div style="font-size:.9rem;color:#e2e8f0;font-weight:500;margin-top:.3rem;">{user['department']}</div>
            <div style="font-size:.78rem;color:#475569;margin-top:.2rem;">{user['school']}</div>
        </div>""", unsafe_allow_html=True)

        tr = st.session_state.target_role or "Not set yet"
        st.markdown(f"""
        <div class="ku-card">
            <div class="section-label">Target Job Role</div>
            <div style="margin-top:.4rem;"><span class="badge badge-purple">🎯 {tr}</span></div>
            <p style="color:#475569;font-size:.78rem;margin-top:.5rem;">
                Set your target role in <b>Company Eligibility</b>.
            </p>
        </div>""", unsafe_allow_html=True)

        # Resume skills
        rr     = st.session_state.resume_result or {}
        skills = rr.get("skills",[])
        if skills:
            chips = "".join(f'<span class="skill-chip">✓ {s}</span>' for s in skills)
            auto_cgpa = rr.get("cgpa")
            cgpa_line = f'<div style="font-size:.78rem;color:#64748b;margin-top:.4rem;">📊 CGPA detected from résumé: <b style="color:#6ee7b7;">{auto_cgpa}</b></div>' if auto_cgpa else ""
            st.markdown(f"""
            <div class="ku-card">
                <div class="section-label">Skills from Résumé ({len(skills)})</div>
                {cgpa_line}
                <div style="margin-top:.5rem;">{chips}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ku-card">
                <div class="section-label">Skills</div>
                <p style="color:#475569;font-size:.82rem;margin-top:.4rem;">
                    Upload a résumé in <b>Placement Prediction</b> to auto-detect skills.
                </p>
            </div>""", unsafe_allow_html=True)

        # Last prediction summary
        lp = st.session_state.last_prediction
        if lp:
            pct   = lp["probability"]*100
            clr   = "#6ee7b7" if pct>=65 else ("#fcd34d" if pct>=40 else "#fca5a5")
            st.markdown(f"""
            <div class="ku-card">
                <div class="section-label">Last Prediction Result</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:{clr};">{pct:.1f}%</div>
                <div style="font-size:.8rem;color:#64748b;margin-top:.2rem;">Placement Probability · {lp['label']}</div>
                <div style="font-size:.76rem;color:#334155;margin-top:.3rem;">CGPA: {lp['cgpa']} · Internships: {lp['internships']} · Projects: {lp['projects']}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: PLACEMENT PREDICTION  (resume REQUIRED)
# ══════════════════════════════════════════════════════════════════
def prediction_page():
    st.markdown("""
    <div class="ku-header">
        <h1>🎯 Placement Prediction</h1>
        <p>Upload your résumé first — the AI analyses it before predicting your placement chances</p>
    </div>""", unsafe_allow_html=True)

    # ── STEP 1: Resume upload ─────────────────────────────────────
    st.markdown("## Step 1 — Upload Your Résumé")
    support = get_support_status()

    st.markdown(f"""
    <div class="info-box" style="margin-bottom:1rem;">
        <b>📎 Accepted formats:</b>&nbsp;&nbsp;{_support_badges()}<br>
        <span style="color:#475569;font-size:.8rem;">
        ⚠️ For image OCR and scanned PDFs, install: <code>pip install pytesseract Pillow</code> + Tesseract engine
        </span>
    </div>""", unsafe_allow_html=True)

    resume_file = st.file_uploader(
        "📄 Drop your résumé here (PDF, DOCX, PNG, JPG, BMP, TIFF)",
        type=STREAMLIT_TYPES,
        help="Required — the AI reads your résumé to auto-fill your profile.",
        key="resume_uploader",
    )

    if resume_file:
        with st.spinner(f"🔍 Extracting text from **{resume_file.name}**…"):
            raw_text, method = extract_text(resume_file)
            resume_data      = parse_resume(raw_text)

        st.session_state.resume_result = resume_data
        st.session_state.resume_text   = raw_text
        st.session_state.resume_ready  = len(raw_text.strip()) > 30

        if st.session_state.resume_ready:
            skills_html = "".join(f'<span class="skill-chip">✓ {s}</span>' for s in resume_data["skills"])
            auto_cgpa   = resume_data.get("cgpa")
            cgpa_line   = f'<span class="badge badge-green">📊 CGPA detected: {auto_cgpa}</span>' if auto_cgpa else ""
            wc          = resume_data.get("word_count",0)

            st.markdown(f"""
            <div class="resume-ok">
                <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-bottom:.5rem;">
                    <span style="font-family:'Syne',sans-serif;font-weight:700;color:#6ee7b7;">✅ Résumé Analysed</span>
                    <span class="badge badge-green">{resume_file.name}</span>
                    <span class="badge">{method}</span>
                    <span class="badge">{wc} words</span>
                    {cgpa_line}
                </div>
                <div style="margin-top:.4rem;">
                    <span style="font-size:.8rem;color:#475569;">Skills detected ({len(resume_data['skills'])}):</span><br>
                    {skills_html if skills_html else '<span style="color:#475569;font-size:.8rem;">No skills auto-detected — fill in manually below.</span>'}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <b>Very little text could be extracted</b> from this file.<br>
                <span style="font-size:.83rem;">
                Try: a text-based PDF, a DOCX, or ensure Tesseract is installed for image/scanned files.
                You can still fill in fields manually and predict below.
                </span>
            </div>""", unsafe_allow_html=True)
            st.session_state.resume_ready = True   # allow predict anyway after trying
    else:
        # No file uploaded yet — show requirement notice
        st.markdown("""
        <div class="resume-required">
            <div style="font-size:2.5rem;margin-bottom:.6rem;">📄</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#fde68a;">
                Résumé Required
            </div>
            <p style="color:#64748b;font-size:.85rem;margin:.4rem 0 0;">
                Please upload your résumé above to unlock the prediction.<br>
                Supported: <b>PDF · DOCX · PNG · JPG · BMP · TIFF</b>
            </p>
        </div>""", unsafe_allow_html=True)

    # ── Only show prediction form once resume is ready ────────────
    if not st.session_state.resume_ready:
        return

    st.markdown("---")
    st.markdown("## Step 2 — Review & Predict")

    rd = st.session_state.resume_result or {}

    lc, rc = st.columns([1, 1.15])

    with lc:
        st.markdown("#### 📋 Academic Profile")
        auto_cgpa_val = rd.get("cgpa")
        cgpa_default  = float(auto_cgpa_val) if auto_cgpa_val else 7.5
        cgpa_default  = max(5.0, min(10.0, cgpa_default))
        cgpa = st.slider("CGPA (out of 10)", 5.0, 10.0, cgpa_default, 0.1,
                         help="Auto-detected from résumé if possible.")

        st.markdown("#### 🔢 Experience")
        c1,c2 = st.columns(2)
        with c1: internships   = st.selectbox("Internships",    list(range(6)),  index=min(rd.get("internships",0),5))
        with c2: projects      = st.selectbox("Projects",       list(range(11)), index=min(rd.get("projects",2),10))
        certifications = st.selectbox("Certifications", list(range(11)), index=min(rd.get("certifications",2),10))

        st.markdown("#### 🧠 Skill Self-Assessment")
        c3,c4 = st.columns(2)
        with c3:
            comm = st.select_slider("Communication", options=[1,2,3,4,5], value=3,
                format_func=lambda x:["Poor","Fair","Good","Very Good","Excellent"][x-1])
        with c4:
            coding = st.select_slider("Coding / DSA", options=[1,2,3,4,5], value=3,
                format_func=lambda x:["Beginner","Elementary","Intermediate","Advanced","Expert"][x-1])

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        clicked = st.button("🔮  Predict My Chances", use_container_width=True)

    with rc:
        if clicked:
            prob  = predict(st.session_state.model, st.session_state.scaler,
                            cgpa, internships, projects, certifications, comm, coding)
            pct   = prob * 100
            clr   = "#3b82f6" if pct>=65 else ("#f59e0b" if pct>=40 else "#ef4444")
            label = "High Chances 🎉" if pct>=65 else ("Moderate Chances ⚡" if pct>=40 else "Needs Improvement ⚠️")

            # ── Save to admin store ───────────────────────────────
            pred_payload = {
                "cgpa": cgpa, "internships": internships, "projects": projects,
                "certifications": certifications, "comm": comm, "coding": coding,
                "probability": prob, "label": label,
            }
            user_data_with_role = {**st.session_state.user_data,
                                   "target_role": st.session_state.target_role or ""}
            save_student_submission(user_data_with_role, pred_payload, rd)
            st.session_state.last_prediction = pred_payload

            # ── Gauge ─────────────────────────────────────────────
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

            if pct>=65:   st.success(f"🎉 **{label} ({pct:.1f}%)** — Great profile, keep it up!")
            elif pct>=40: st.warning(f"⚡ **{label} ({pct:.1f}%)** — A few improvements will help.")
            else:         st.error(  f"⚠️ **{label} ({pct:.1f}%)** — Focus on internships, projects & skills.")

            # ── Radar ─────────────────────────────────────────────
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

            # ── Personalised recommendations ──────────────────────
            st.markdown("#### 💡 Recommendations")
            recs=[]
            if cgpa<7.5:         recs.append(("📚","Aim for CGPA ≥ 7.5 — unlocks significantly more companies."))
            if internships<2:    recs.append(("💼","Complete at least 2 internships before final year."))
            if projects<3:       recs.append(("🔧","Build 3+ end-to-end projects to demonstrate skills."))
            if certifications<2: recs.append(("📜","Earn 2+ certifications (AWS, Google, Microsoft, NPTEL)."))
            if comm<4:           recs.append(("🗣️","Practise communication — GDs and mock interviews help."))
            if coding<4:         recs.append(("💻","Solve 2–3 DSA problems daily on LeetCode / HackerRank."))
            if not rd.get("skills"):
                recs.append(("📄","Add more quantifiable skills & technologies to your résumé."))
            if recs:
                for icon,text in recs:
                    st.markdown(f"""
                    <div style="display:flex;gap:.7rem;align-items:flex-start;
                                padding:.6rem 0;border-bottom:1px solid rgba(255,255,255,.05);">
                        <span style="font-size:1.1rem;">{icon}</span>
                        <span style="color:#94a3b8;font-size:.87rem;">{text}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("🌟 Excellent profile! Aim for top-tier companies.")

            st.markdown("""
            <div style="background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.2);
                        border-radius:10px;padding:.7rem 1rem;margin-top:.8rem;font-size:.78rem;color:#6ee7b7;">
                ✅ Your prediction result has been saved and will appear in the Admin panel.
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="ku-card" style="text-align:center;padding:3.5rem 2rem;">
                <div style="font-size:3.5rem;margin-bottom:1rem;">🤖</div>
                <div style="color:#475569;font-size:.92rem;line-height:1.7;">
                    Résumé uploaded ✅<br>Fill in the fields and click<br>
                    <b style="color:#93c5fd;">Predict My Chances</b>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: SKILL RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
def skills_page():
    user = st.session_state.user_data
    st.markdown("""
    <div class="ku-header">
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
        "Task":  ["Core Skill Mastery","DSA & Problem Solving","Project Building","Open Source / Internship","Advanced Concepts","Interview Prep"],
        "Start": pd.to_datetime(["2025-01-01","2025-01-20","2025-03-01","2025-03-15","2025-05-01","2025-05-20"]),
        "End":   pd.to_datetime(["2025-03-01","2025-03-01","2025-05-01","2025-05-01","2025-07-01","2025-07-01"]),
        "Phase": ["Phase 1","Phase 1","Phase 2","Phase 2","Phase 3","Phase 3"],
    })
    fig = px.timeline(rm,x_start="Start",x_end="End",y="Task",color="Phase",
        title="Skill Development Timeline",color_discrete_sequence=["#3b82f6","#06b6d4","#8b5cf6"])
    fig.update_yaxes(autorange="reversed")
    _chart(fig, 380)

# ══════════════════════════════════════════════════════════════════
# PAGE: COMPANY ELIGIBILITY
# ══════════════════════════════════════════════════════════════════
def company_page():
    st.markdown("""
    <div class="ku-header">
        <h1>🏢 Company Eligibility</h1>
        <p>Discover companies that match your CGPA and target role</p>
    </div>""", unsafe_allow_html=True)

    lc, rc = st.columns([1,3])
    with lc:
        cgpa = st.slider("Your CGPA", 5.0, 10.0, 7.5, 0.1)
        role_opts = ["All Roles"]+ALL_ROLES
        cur_role  = st.session_state.target_role
        sel_idx   = role_opts.index(cur_role) if cur_role and cur_role in role_opts else 0
        sel_role  = st.selectbox("🎯 Target Job Role", role_opts, index=sel_idx)
        st.session_state.target_role = None if sel_role=="All Roles" else sel_role

        types = st.multiselect("Company Type",
            ["Product","Service","Consulting","E-Commerce","FinTech","Retail Tech","Engineering"],
            default=["Product","Service","Consulting","FinTech"])

    with rc:
        role_arg = st.session_state.target_role
        eligible = get_eligible_companies(cgpa, role=role_arg, types=types if types else None)
        t1_list  = [c for c in eligible if c["tier"]==1]
        t2_list  = [c for c in eligible if c["tier"]==2]
        t3_list  = [c for c in eligible if c["tier"]==3]

        role_pill = f'<span class="badge badge-purple">🎯 {role_arg}</span>' if role_arg else ""
        st.markdown(f"""
        <div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem;">
            <span class="badge badge-green">✓ Eligible: {len(eligible)}</span>
            <span class="badge badge-gold">🌟 Tier 1: {len(t1_list)}</span>
            <span class="badge badge-yellow">⭐ Tier 2: {len(t2_list)}</span>
            <span class="badge">💫 Tier 3: {len(t3_list)}</span>
            {role_pill}
        </div>""", unsafe_allow_html=True)

        tab1,tab2,tab3 = st.tabs([f"🌟 Dream ({len(t1_list)})",f"⭐ Target ({len(t2_list)})",f"💫 Safe ({len(t3_list)})"])

        def show_co(lst):
            if not lst:
                st.info("Adjust CGPA, role, or company type filters.")
                return
            for c in lst:
                init    = _hex_initials(c["name"])
                roles_h = "".join(f'<span class="badge badge-purple" style="font-size:.7rem;">{r}</span>' for r in c.get("roles",[]))
                skills_t= " · ".join(c.get("skills",[]))
                st.markdown(f"""
                <div class="co-card">
                    <div class="co-dot" style="background:{c['logo_color']};">{init}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;">
                            <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f9ff;">{c['name']}</span>
                            {_tier_badge(c['tier'])}
                            <span class="badge badge-green">{c['package']}</span>
                        </div>
                        <div style="font-size:.78rem;color:#475569;margin:.3rem 0;">
                            <b style="color:#64748b;">Min CGPA:</b> {c['min_cgpa']} &nbsp;·&nbsp;
                            <b style="color:#64748b;">Type:</b> {c['type']}
                        </div>
                        <div style="font-size:.76rem;color:#64748b;margin-bottom:.4rem;"><b>Skills:</b> {skills_t}</div>
                        <div>{roles_h}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with tab1: show_co(t1_list)
        with tab2: show_co(t2_list)
        with tab3: show_co(t3_list)

    # CGPA chart — Plotly yaxis fix applied
    st.markdown("### 📊 CGPA Requirements — All Companies")
    df_c = pd.DataFrame([{"Company":n,"Min CGPA":i["min_cgpa"],"Type":i["type"]} for n,i in COMPANY_CRITERIA.items()])
    fig = px.bar(df_c.sort_values("Min CGPA"),x="Min CGPA",y="Company",color="Type",
        orientation="h",title="CGPA Cut-off by Company",
        color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(**_ply_base(), height=720, yaxis_categoryorder="total ascending")
    fig.update_xaxes(**_ply_axis())
    fig.update_yaxes(**_ply_axis())
    fig.add_vline(x=cgpa,line_dash="dash",line_color="#f59e0b",
        annotation_text=f"  Your CGPA: {cgpa}",annotation_font_color="#f59e0b")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: DATASET & STUDENT SUBMISSIONS  (admin only)
# ══════════════════════════════════════════════════════════════════
def dataset_page():
    st.markdown("""
    <div class="ku-header">
        <h1>📊 Dataset & Student Submissions</h1>
        <p>Admin view — training data, live student submissions, model management</p>
    </div>""", unsafe_allow_html=True)

    if _is_student():
        _access_denied()
        return

    t1,t2,t3,t4 = st.tabs([
        "👥 Student Submissions",
        "📋 Training Dataset",
        "📤 Upload Dataset",
        "🔧 Retrain Model",
    ])

    # ── Tab 1: Live student submissions ──────────────────────────
    with t1:
        st.markdown("#### All Student Prediction Submissions")
        sub_df = submissions_to_df()
        if sub_df.empty:
            st.info("No student submissions yet. Students will appear here after they run a prediction.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Submissions",  len(sub_df))
            c2.metric("Unique Students",    sub_df["email"].nunique())
            avg_prob = sub_df["Placement_Probability_%"].mean()
            c3.metric("Avg. Probability",   f"{avg_prob:.1f}%")
            c4.metric("Avg. CGPA",          f"{sub_df['CGPA'].mean():.2f}")

            # Submissions table
            st.dataframe(sub_df, use_container_width=True, height=380)

            # Download
            st.download_button("📥 Download All Submissions",
                sub_df.to_csv(index=False), "student_submissions.csv","text/csv",
                use_container_width=True)

            # Probability distribution chart
            st.markdown("#### Placement Probability Distribution")
            fig = px.histogram(sub_df, x="Placement_Probability_%",
                nbins=20, title="Student Prediction Probability Distribution",
                color_discrete_sequence=["#3b82f6"])
            _chart(fig, 320)

            # CGPA vs Probability scatter
            if len(sub_df) > 2:
                st.markdown("#### CGPA vs Placement Probability")
                fig2 = px.scatter(sub_df, x="CGPA", y="Placement_Probability_%",
                    color="department" if "department" in sub_df.columns else None,
                    hover_data=["name","register_number","target_role"],
                    title="CGPA vs Predicted Placement Probability",
                    size_max=14, opacity=0.8)
                _chart(fig2, 360)

            st.markdown("---")
            if st.button("🗑️ Clear All Submissions", use_container_width=True):
                clear_store()
                st.success("All submissions cleared.")
                st.rerun()

    # ── Tab 2: Synthetic training dataset ────────────────────────
    with t2:
        df = st.session_state.placement_data
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Records",        f"{len(df):,}")
        c2.metric("Placed",         f"{int(df['Placed'].sum()):,}")
        c3.metric("Placement Rate", f"{df['Placed'].mean()*100:.1f}%")
        c4.metric("Model Accuracy", f"{st.session_state.model_accuracy*100:.1f}%")
        st.dataframe(df.head(50), use_container_width=True)
        st.download_button("📥 Download Training Dataset",
            df.to_csv(index=False),"karunya_placement_data.csv","text/csv",
            use_container_width=True)

    # ── Tab 3: Upload new training data ──────────────────────────
    with t3:
        st.markdown("""
        <div class="info-box" style="margin-bottom:1rem;">
            <b>Required columns:</b>
            <code>CGPA, Internships, Projects, Certifications, Communication_Skill, Coding_Skill, Placed</code>
        </div>""", unsafe_allow_html=True)
        up = st.file_uploader("Choose CSV", type="csv")
        if up:
            try:
                new_df = pd.read_csv(up)
                req = ["CGPA","Internships","Projects","Certifications","Communication_Skill","Coding_Skill","Placed"]
                miss = [c for c in req if c not in new_df.columns]
                if miss: st.error(f"Missing: {', '.join(miss)}")
                else:
                    st.success(f"✅ {len(new_df):,} records ready.")
                    st.dataframe(new_df.head(5), use_container_width=True)
                    if st.button("Import Dataset"):
                        st.session_state.placement_data = new_df
                        st.success("Imported. Go to Retrain Model.")
            except Exception as e: st.error(f"Error: {e}")

    # ── Tab 4: Retrain model ──────────────────────────────────────
    with t4:
        df = st.session_state.placement_data
        st.markdown(f"""
        <div class="ku-card" style="margin-bottom:1rem;">
            <div style="display:flex;gap:2rem;flex-wrap:wrap;font-size:.85rem;">
                <div><div class="section-label">Dataset</div><b style="color:#f0f9ff;">{len(df):,} records</b></div>
                <div><div class="section-label">Model</div><b style="color:#f0f9ff;">{st.session_state.model_name}</b></div>
                <div><div class="section-label">Accuracy</div><b style="color:#6ee7b7;">{st.session_state.model_accuracy*100:.1f}%</b></div>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("🚀 Retrain Model Now", use_container_width=True):
            with st.spinner("Training models…"):
                model,scaler,acc,name,cm,report,_,_ = train_model(df)
                st.session_state.model=model; st.session_state.scaler=scaler
                st.session_state.model_accuracy=acc; st.session_state.model_name=name
                st.session_state.cm=cm; st.session_state.report=report
            st.success(f"✅ **{name}** · Accuracy: **{acc*100:.1f}%**")
            fig=px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual"),
                x=["Not Placed","Placed"],y=["Not Placed","Placed"],
                color_continuous_scale="Blues",title="Confusion Matrix")
            _chart(fig, 400)

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
            "🎯  Placement Prediction": "prediction",
            "🛠️  Skill Recommendations":"skills",
            "🏢  Company Eligibility":  "companies",
            "📊  Dataset & Submissions":"dataset",
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
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 .6rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;color:#f0f9ff;">🎓 KU Placement</div>
            <div style="font-size:.72rem;color:#334155;text-transform:uppercase;letter-spacing:.1em;">Intelligence v4.0</div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        user = st.session_state.user_data
        name = user.get("name","User")
        dept = user.get("department","")
        dept_s = dept[:34]+"…" if len(dept)>36 else dept
        role_lbl = '<span class="badge badge-admin">🛡️ Admin</span>' if _is_admin() else '<span class="badge badge-green">🎓 Student</span>'

        # Resume status indicator in sidebar
        if _is_student():
            r_status = '<span class="badge badge-green" style="font-size:.68rem;">📄 Résumé ✓</span>' \
                       if st.session_state.resume_ready else \
                       '<span class="badge badge-orange" style="font-size:.68rem;">📄 No Résumé</span>'
        else:
            r_status = ""

        st.markdown(f"""
        <div style="padding:.7rem .9rem;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:10px;margin-bottom:1rem;">
            <div style="font-size:.9rem;font-weight:600;color:#e2e8f0;">{name}</div>
            <div style="font-size:.75rem;color:#64748b;margin-top:1px;">{user.get('email','')}</div>
            {"" if _is_admin() else f'<div style="font-size:.72rem;color:#475569;margin-top:1px;">{dept_s}</div>'}
            {"" if _is_admin() else f'<div style="font-size:.72rem;color:#334155;margin-top:2px;">{user.get("current_year","")}</div>'}
            <div style="margin-top:.5rem;display:flex;gap:4px;flex-wrap:wrap;">{role_lbl}{r_status}</div>
        </div>""", unsafe_allow_html=True)

        # Admin: show live submission count
        if _is_admin():
            sc = get_submission_count()
            st.markdown(f"""
            <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);border-radius:8px;
                        padding:.5rem .8rem;font-size:.78rem;color:#6ee7b7;margin-bottom:.8rem;">
                📥 <b>{sc}</b> student submission{"s" if sc!=1 else ""} waiting
            </div>""", unsafe_allow_html=True)

        selected = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()

        if st.button("🚪  Logout", use_container_width=True):
            _logout()

        st.markdown("""
        <div style="text-align:center;font-size:.68rem;color:#1e293b;margin-top:.8rem;">
            © 2025 Karunya University<br>AI Placement Intelligence v4.0
        </div>""", unsafe_allow_html=True)

    page = PAGES[selected]
    if   page=="dashboard":  dashboard_page()
    elif page=="prediction": prediction_page()
    elif page=="skills":     skills_page()
    elif page=="companies":  company_page()
    elif page=="dataset":    dataset_page()
    elif page=="account":    account_page()


if __name__ == "__main__":
    main()
