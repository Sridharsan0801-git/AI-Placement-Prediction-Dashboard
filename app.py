"""
╔══════════════════════════════════════════════════════════════════╗
║     KARUNYA UNIVERSITY — AI PLACEMENT INTELLIGENCE SYSTEM       ║
║                        Version 2.1                              ║
╚══════════════════════════════════════════════════════════════════╝
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

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from utils.auth import validate_email, parse_register_number
from utils.ml_model import generate_sample_data, train_model, predict, FEATURE_COLS
from utils.companies import COMPANY_CRITERIA, get_eligible_companies
from utils.departments import DEPARTMENTS, DEPARTMENT_SCHOOLS, DEPARTMENT_SKILLS, get_skill_category

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KU Placement Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

*,*::before,*::after{box-sizing:border-box}

html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],.main{
    background:#080e1d!important;color:#dde3f0!important;
    font-family:'DM Sans',sans-serif;
}
[data-testid="stSidebar"]{background:#0b1120!important;border-right:1px solid rgba(96,165,250,.1)!important}
[data-testid="stSidebar"] *{color:#c4cede!important}
h1,h2,h3,h4,h5,h6{font-family:'Syne',sans-serif!important;letter-spacing:-.02em}

#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;display:none}

[data-testid="stSidebar"] [role="radiogroup"] label{
    display:flex!important;align-items:center!important;
    padding:9px 14px!important;border-radius:9px!important;margin:2px 0!important;
    cursor:pointer!important;transition:background .18s,color .18s!important;
    font-size:.88rem!important;font-weight:500!important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(96,165,250,.08)!important}

[data-testid="metric-container"]{
    background:rgba(255,255,255,.035)!important;border:1px solid rgba(255,255,255,.07)!important;
    border-radius:14px!important;padding:1rem 1.3rem!important;transition:border-color .2s;
}
[data-testid="metric-container"]:hover{border-color:rgba(96,165,250,.3)!important}
[data-testid="metric-container"] label{color:#64748b!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:.06em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#f0f9ff!important;font-family:'Syne',sans-serif!important;font-size:1.65rem!important}
[data-testid="metric-container"] [data-testid="stMetricDelta"] svg{display:none}

[data-testid="stTextInput"] input{
    background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;
    border-radius:10px!important;color:#f1f5f9!important;padding:.6rem .9rem!important;
    font-family:'DM Sans',sans-serif!important;transition:border-color .2s,box-shadow .2s!important;
}
[data-testid="stTextInput"] input:focus{border-color:#3b82f6!important;box-shadow:0 0 0 3px rgba(59,130,246,.15)!important;outline:none!important}
[data-testid="stTextInput"] input::placeholder{color:#475569!important}

[data-testid="stSelectbox"]>div>div,[data-baseweb="select"]>div{
    background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;
    border-radius:10px!important;color:#f1f5f9!important;
}

.stButton>button{
    background:linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%)!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:.92rem!important;
    padding:.65rem 1.6rem!important;transition:transform .2s,box-shadow .2s!important;letter-spacing:.01em;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(59,130,246,.4)!important}
.stButton>button:active{transform:translateY(0)!important}

[data-testid="stDownloadButton"] button{
    background:rgba(255,255,255,.06)!important;border:1px solid rgba(255,255,255,.14)!important;
    color:#94a3b8!important;border-radius:10px!important;
}
[data-testid="stDownloadButton"] button:hover{background:rgba(255,255,255,.1)!important;color:#f1f5f9!important;transform:translateY(-1px)!important;box-shadow:none!important}

[data-baseweb="tab-list"]{
    background:rgba(255,255,255,.04)!important;border-radius:12px!important;
    padding:4px!important;gap:4px!important;border:1px solid rgba(255,255,255,.06)!important;
}
[data-baseweb="tab"]{background:transparent!important;border-radius:8px!important;color:#64748b!important;font-family:'DM Sans',sans-serif!important;font-weight:500!important;transition:background .18s,color .18s!important}
[data-baseweb="tab"]:hover{color:#94a3b8!important}
[aria-selected="true"][data-baseweb="tab"]{background:rgba(59,130,246,.22)!important;color:#93c5fd!important;font-weight:600!important}

[data-testid="stExpander"]{
    background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.07)!important;
    border-radius:12px!important;margin-bottom:6px!important;transition:border-color .2s!important;
}
[data-testid="stExpander"]:hover{border-color:rgba(96,165,250,.2)!important}
[data-testid="stExpander"] summary{color:#c4cede!important;font-weight:500!important}

[data-testid="stProgressBar"]>div{background:rgba(255,255,255,.08)!important;border-radius:999px!important;height:6px!important}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#3b82f6,#06b6d4)!important;border-radius:999px!important}

[data-testid="stDataFrameResizable"]{border-radius:12px!important;overflow:hidden!important;border:1px solid rgba(255,255,255,.08)!important}
[data-testid="stAlert"]{border-radius:12px!important;border-left-width:3px!important;font-family:'DM Sans',sans-serif!important}

[data-testid="stMultiSelect"] [data-baseweb="tag"]{background:rgba(59,130,246,.22)!important;color:#93c5fd!important;border-radius:6px!important;font-size:.8rem!important}

[data-testid="stFileUploader"]{background:rgba(255,255,255,.03)!important;border:1.5px dashed rgba(255,255,255,.14)!important;border-radius:12px!important;transition:border-color .2s!important}
[data-testid="stFileUploader"]:hover{border-color:rgba(59,130,246,.4)!important}

hr{border-color:rgba(255,255,255,.07)!important}

/* ── Custom Classes ──────────────────────────────────────────── */
.ku-header{
    background:linear-gradient(135deg,#0c1628 0%,#13233f 60%,#0f1e38 100%);
    border:1px solid rgba(96,165,250,.18);border-radius:20px;padding:2.4rem 2rem;
    text-align:center;margin-bottom:1.8rem;position:relative;overflow:hidden;
}
.ku-header::before{
    content:'';position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(ellipse at 20% 50%,rgba(59,130,246,.13) 0%,transparent 55%),
               radial-gradient(ellipse at 80% 50%,rgba(6,182,212,.09) 0%,transparent 55%),
               radial-gradient(ellipse at 50% 0%,rgba(139,92,246,.07) 0%,transparent 50%);
}
.ku-header h1{font-size:1.9rem;color:#f0f9ff;margin:0;position:relative;line-height:1.15}
.ku-header p{color:#64748b;margin:.5rem 0 0;font-size:.9rem;position:relative}

.ku-card{
    background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.075);
    border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem;transition:border-color .2s;
}
.ku-card:hover{border-color:rgba(96,165,250,.2)}
.ku-card-accent{
    background:linear-gradient(135deg,rgba(59,130,246,.08),rgba(6,182,212,.05));
    border:1px solid rgba(59,130,246,.22);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem;
}

.badge{display:inline-block;background:rgba(59,130,246,.18);border:1px solid rgba(59,130,246,.32);color:#93c5fd;padding:3px 10px;border-radius:999px;font-size:.74rem;margin:2px;font-weight:500}
.badge-green{background:rgba(16,185,129,.12);border-color:rgba(16,185,129,.28);color:#6ee7b7}
.badge-yellow{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.28);color:#fcd34d}
.badge-red{background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.28);color:#fca5a5}
.badge-purple{background:rgba(139,92,246,.12);border-color:rgba(139,92,246,.28);color:#c4b5fd}

.section-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#475569;margin-bottom:.4rem}
.info-box{background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.2);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem}
.success-box{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#a7f3d0}
.warning-box{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#fde68a}

.skill-chip{display:inline-flex;align-items:center;gap:4px;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);color:#6ee7b7;padding:4px 10px;border-radius:8px;font-size:.78rem;font-weight:500;margin:3px}
.skill-chip-miss{background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.2);color:#fca5a5}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────────────────────────
PLY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="DM Sans", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.06)", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.06)", tickfont=dict(color="#64748b")),
    margin=dict(t=48, b=28, l=8, r=8),
    title_font=dict(color="#94a3b8", size=13, family="Syne"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), bordercolor="rgba(255,255,255,.08)", borderwidth=1),
)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "logged_in": False, "user_data": {},
    "placement_data": None, "model": None, "scaler": None,
    "model_accuracy": None, "model_name": None,
    "cm": None, "report": None, "resume_result": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# RESUME PARSING
# ─────────────────────────────────────────────────────────────────
RESUME_SKILL_MAP = {
    r"\bpython\b": "Python", r"\bjava\b": "Java", r"c\+\+": "C++",
    r"\bc#\b": "C#", r"\bjavascript\b": "JavaScript", r"\btypescript\b": "TypeScript",
    r"\bkotlin\b": "Kotlin", r"\bswift\b": "Swift", r"\brust\b": "Rust",
    r"\breact\b": "React", r"\bnode\.?js\b": "Node.js", r"\bdjango\b": "Django",
    r"\bflask\b": "Flask", r"\bfastapi\b": "FastAPI", r"\bhtml\b": "HTML",
    r"\bcss\b": "CSS", r"\bvue\.?js\b": "Vue.js", r"\bangular\b": "Angular",
    r"machine learning": "Machine Learning", r"deep learning": "Deep Learning",
    r"data science": "Data Science", r"\btensorflow\b": "TensorFlow",
    r"\bpytorch\b": "PyTorch", r"\bkeras\b": "Keras",
    r"\bnlp\b|natural language processing": "NLP",
    r"computer vision": "Computer Vision", r"\bpandas\b": "Pandas",
    r"\bnumpy\b": "NumPy", r"\bscikit.?learn\b": "Scikit-Learn",
    r"\bopencv\b": "OpenCV", r"\bsql\b": "SQL", r"\bmysql\b": "MySQL",
    r"\bpostgresql\b": "PostgreSQL", r"\bmongodb\b": "MongoDB",
    r"\bhadoop\b": "Hadoop", r"\bspark\b": "Apache Spark",
    r"\baws\b": "AWS", r"\bgcp\b|google cloud": "Google Cloud",
    r"\bazure\b": "Azure", r"\bdocker\b": "Docker",
    r"\bkubernetes\b": "Kubernetes", r"\bterraform\b": "Terraform",
    r"\bjenkins\b": "Jenkins", r"\bgit\b": "Git", r"\blinux\b": "Linux",
    r"\bembedded\b": "Embedded Systems", r"\bvlsi\b": "VLSI",
    r"\bmatlab\b": "MATLAB", r"\bansys\b": "ANSYS",
    r"\bautocad\b": "AutoCAD", r"\bsolidworks\b": "SolidWorks",
    r"\bplc\b": "PLC/SCADA", r"\biot\b": "IoT",
    r"\bblockchain\b": "Blockchain", r"\bcybersecurity\b|ethical hack": "Cybersecurity",
}

_INTERNSHIP_KWS = [r"\binternship\b", r"\bintern\b", r"\btrainee\b", r"\bsummer project\b", r"\bindustry training\b"]
_PROJECT_KWS    = [r"\bproject\b", r"\bbuilt\b", r"\bdeveloped\b", r"\bimplemented\b", r"\bdesigned\b", r"\bcreated\b"]
_CERT_KWS       = [r"\bcertif", r"\bcoursera\b", r"\budemy\b", r"\bnptel\b", r"\bedx\b", r"\blinkedin learning\b", r"\bmooc\b"]

def _extract_pdf_text(file) -> str:
    if not PDF_SUPPORT:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        return " ".join(page.extract_text() or "" for page in reader.pages).lower()
    except Exception:
        return ""

def _parse_resume(text: str) -> dict:
    if not text:
        return {"skills": [], "internships": 0, "projects": 0, "certifications": 0}
    skills = [label for pat, label in RESUME_SKILL_MAP.items() if re.search(pat, text, re.IGNORECASE)]
    def cnt(kws): return min(sum(1 for kw in kws if re.search(kw, text, re.IGNORECASE)), 5)
    return {"skills": skills, "internships": cnt(_INTERNSHIP_KWS),
            "projects": min(cnt(_PROJECT_KWS), 8), "certifications": cnt(_CERT_KWS)}

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def _boot_model():
    if st.session_state.placement_data is None:
        df = generate_sample_data()
        model, scaler, acc, name, cm, report, _, _ = train_model(df)
        st.session_state.placement_data = df
        st.session_state.model = model; st.session_state.scaler = scaler
        st.session_state.model_accuracy = acc; st.session_state.model_name = name
        st.session_state.cm = cm; st.session_state.report = report

def _chart(fig, h=320):
    fig.update_layout(**PLY, height=h)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: LOGIN
# ─────────────────────────────────────────────────────────────────
def login_page():
    st.markdown("""
    <div class="ku-header" style="padding:3rem 2rem;">
        <h1 style="font-size:2.2rem;">🎓 Karunya University</h1>
        <p style="font-size:1rem;color:#475569;margin-top:.4rem;">AI-Powered Placement Intelligence System</p>
    </div>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown('<div class="section-label" style="margin-bottom:.8rem;">STUDENT PORTAL LOGIN</div>', unsafe_allow_html=True)
        email  = st.text_input("Email ID", placeholder="yourname@karunya.edu.in")
        reg_no = st.text_input("Register Number", placeholder="e.g. URK25AI1074")
        dept   = st.selectbox("Department / Programme", ["— Select your programme —"] + DEPARTMENTS)
        yr     = st.selectbox("Current Academic Year", ["— Select —","1st Year","2nd Year","3rd Year","4th Year"])
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if st.button("Login  →", use_container_width=True):
            errors = []
            if not email: errors.append("Email is required.")
            elif not validate_email(email): errors.append("Only **@karunya.edu.in** emails are accepted.")
            parsed = None
            if not reg_no: errors.append("Register number is required.")
            else:
                parsed = parse_register_number(reg_no)
                if not parsed["valid"]: errors.append("Invalid register number. Example: **URK25AI1074**")
            if dept == "— Select your programme —": errors.append("Please select your department.")
            if yr == "— Select —": errors.append("Please select your current academic year.")

            if errors:
                for e in errors: st.error(e)
            else:
                with st.spinner("Initialising analytics engine…"):
                    _boot_model()
                st.session_state.logged_in = True
                st.session_state.user_data = {
                    "email": email.strip().lower(),
                    "register_number": reg_no.upper().strip(),
                    "department": dept,
                    "school": DEPARTMENT_SCHOOLS.get(dept, "Karunya University"),
                    "year_of_joining": parsed["year_of_joining"],
                    "program_type": parsed["program_type"],
                    "roll_number": parsed["roll_number"],
                    "current_year": yr,
                }
                st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <b>📋 Register Number Format:</b> <code>URK25AI1074</code><br>
            <span style="color:#475569;font-size:.82rem;">
            <b>URK</b> = Undergrad Karunya &nbsp;·&nbsp; <b>25</b> = Year of joining
            &nbsp;·&nbsp; <b>AI</b> = Dept code &nbsp;·&nbsp; <b>1074</b> = Roll no
            </span>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────
def dashboard_page():
    user = st.session_state.user_data
    df   = st.session_state.placement_data
    name = user["email"].split("@")[0].replace(".", " ").title()

    st.markdown(f"""
    <div class="ku-header">
        <h1>Welcome back, {name}! 👋</h1>
        <p>{user['department']}&nbsp;·&nbsp;{user['school']}</p>
    </div>""", unsafe_allow_html=True)

    pc, mc = st.columns([1.15, 2.85])
    with pc:
        st.markdown(f"""
        <div class="ku-card">
            <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:#f0f9ff;margin-bottom:1rem;">👤 Student Profile</div>
            <div style="display:grid;gap:.7rem;font-size:.83rem;">
                <div><div class="section-label">Email</div><b style="color:#cbd5e1;">{user['email']}</b></div>
                <div><div class="section-label">Register No.</div><b style="color:#cbd5e1;">{user['register_number']}</b></div>
                <div><div class="section-label">Year of Joining</div><b style="color:#cbd5e1;">{user['year_of_joining']}</b></div>
                <div><div class="section-label">Current Year</div><b style="color:#cbd5e1;">{user['current_year']}</b></div>
                <div><div class="section-label">Programme</div><b style="color:#cbd5e1;">{user['program_type']}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

        acc = st.session_state.model_accuracy or 0
        st.markdown(f"""
        <div class="ku-card-accent">
            <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:#93c5fd;margin-bottom:.8rem;">🤖 ML Engine Status</div>
            <div style="font-size:.81rem;display:grid;gap:.5rem;">
                <div><span style="color:#475569;">Model</span><br><b style="color:#e2e8f0;">{st.session_state.model_name}</b></div>
                <div><span style="color:#475569;">Accuracy</span><br><b style="color:#6ee7b7;font-size:1.1rem;">{acc*100:.1f}%</b></div>
                <div><span style="color:#475569;">Training Records</span><br><b style="color:#e2e8f0;">{len(df):,}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with mc:
        rate  = df["Placed"].mean() * 100
        avg_c = df["CGPA"].mean()
        placed = int(df["Placed"].sum())
        total  = len(df)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Placement Rate", f"{rate:.1f}%",  f"+{rate-65:.1f}%")
        c2.metric("Average CGPA",   f"{avg_c:.2f}",  f"+{avg_c-7:.2f}")
        c3.metric("Total Students", f"{total:,}")
        c4.metric("Placed Students",f"{placed:,}")

        ch1, ch2 = st.columns(2)
        with ch1:
            fig = go.Figure(go.Pie(
                values=[placed, total-placed], labels=["Placed","Not Placed"],
                hole=0.58, marker_colors=["#3b82f6","#1e293b"],
                textinfo="percent", textfont_size=12,
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            fig.update_layout(**PLY, title_text="Placement Split", height=270)
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            fig2 = px.histogram(df, x="CGPA", color="Placed",
                color_discrete_map={0:"#ef4444",1:"#3b82f6"},
                barmode="overlay", opacity=0.72, title="CGPA Distribution")
            fig2.update_layout(**PLY, height=270)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📊 Detailed Analytics")
    t1, t2, t3 = st.tabs(["📈  CGPA Analysis","💼  Internship Impact","🧠  Skills Correlation"])

    with t1:
        ca, cb = st.columns(2)
        with ca:
            _chart(px.box(df, x="Placed", y="CGPA", color="Placed",
                color_discrete_map={0:"#ef4444",1:"#3b82f6"},
                title="CGPA: Placed vs Not Placed"), 340)
        with cb:
            bins = pd.cut(df["CGPA"], bins=[5,6,7,8,9,10], labels=["5–6","6–7","7–8","8–9","9–10"])
            rate2 = df.groupby(bins, observed=False)["Placed"].mean() * 100
            _chart(px.bar(x=rate2.index.astype(str), y=rate2.values,
                labels={"x":"CGPA Range","y":"Placement Rate (%)"},
                title="Placement Rate by CGPA Range",
                color=rate2.values, color_continuous_scale="Blues"), 340)

    with t2:
        ca, cb = st.columns(2)
        with ca:
            r = df.groupby("Internships")["Placed"].mean() * 100
            _chart(px.bar(x=r.index, y=r.values,
                labels={"x":"No. of Internships","y":"Placement Rate (%)"},
                title="Placement Rate by Internships",
                color=r.values, color_continuous_scale="Blues"), 340)
        with cb:
            _chart(px.scatter(df, x="CGPA", y="Internships", color="Placed",
                color_discrete_map={0:"#ef4444",1:"#3b82f6"},
                size="Projects", title="CGPA vs Internships", opacity=0.65), 340)

    with t3:
        LBL = ["Beginner","Elementary","Intermediate","Advanced","Expert"]
        ca, cb = st.columns(2)
        with ca:
            r = df.groupby("Coding_Skill")["Placed"].mean() * 100
            _chart(px.bar(x=LBL[:len(r)], y=r.values,
                labels={"x":"Coding Skill","y":"Placement Rate (%)"},
                title="Placement Rate by Coding Skill",
                color=r.values, color_continuous_scale="Viridis"), 340)
        with cb:
            r2 = df.groupby("Communication_Skill")["Placed"].mean() * 100
            LB2 = ["Poor","Fair","Good","Very Good","Excellent"]
            _chart(px.bar(x=LB2[:len(r2)], y=r2.values,
                labels={"x":"Communication Skill","y":"Placement Rate (%)"},
                title="Placement Rate by Communication",
                color=r2.values, color_continuous_scale="Oranges"), 340)

# ─────────────────────────────────────────────────────────────────
# PAGE: PLACEMENT PREDICTION  (with PDF resume upload)
# ─────────────────────────────────────────────────────────────────
def prediction_page():
    st.markdown("""
    <div class="ku-header">
        <h1>🎯 Placement Prediction</h1>
        <p>AI-powered probability analysis — upload your résumé or fill in details manually</p>
    </div>""", unsafe_allow_html=True)

    lc, rc = st.columns([1, 1.15])

    with lc:
        st.markdown("#### 📋 Academic Profile")
        cgpa = st.slider("CGPA (out of 10)", 5.0, 10.0, 7.5, 0.1)

        st.markdown("#### 📄 Résumé Upload *(auto-fills fields below)*")
        resume_file = st.file_uploader("Upload your résumé (PDF)", type=["pdf"],
            help="We scan for skills, projects, internships, and certifications.")

        resume_data = None
        if resume_file is not None:
            if not PDF_SUPPORT:
                st.error("PyPDF2 not installed. Run: `pip install PyPDF2`")
            else:
                with st.spinner("Analysing résumé…"):
                    text = _extract_pdf_text(resume_file)
                    resume_data = _parse_resume(text)
                    st.session_state.resume_result = resume_data

                if resume_data["skills"]:
                    chips = "".join(f'<span class="skill-chip">✓ {s}</span>' for s in resume_data["skills"])
                    st.markdown(f"""
                    <div class="ku-card" style="padding:1rem;">
                        <div class="section-label">Detected Skills ({len(resume_data['skills'])})</div>
                        <div style="margin-top:.5rem;">{chips}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.warning("No recognisable skills found. Try a text-based PDF.")

        st.markdown("#### 🔢 Experience & Certifications")
        ai  = (resume_data or {}).get("internships", 0)
        ap  = (resume_data or {}).get("projects", 2)
        ac  = (resume_data or {}).get("certifications", 2)

        c1, c2 = st.columns(2)
        with c1: internships = st.selectbox("Internships", list(range(6)), index=min(ai,5), help="Auto-detected from résumé.")
        with c2: projects    = st.selectbox("Projects",    list(range(11)),index=min(ap,10),help="Auto-detected from résumé.")
        certifications = st.selectbox("Certifications", list(range(11)), index=min(ac,10), help="Auto-detected from résumé.")

        st.markdown("#### 🧠 Skill Self-Assessment")
        c3, c4 = st.columns(2)
        with c3:
            comm = st.select_slider("Communication", options=[1,2,3,4,5], value=3,
                format_func=lambda x: ["Poor","Fair","Good","Very Good","Excellent"][x-1])
        with c4:
            coding = st.select_slider("Coding / DSA", options=[1,2,3,4,5], value=3,
                format_func=lambda x: ["Beginner","Elementary","Intermediate","Advanced","Expert"][x-1])

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮  Predict My Chances", use_container_width=True)

    with rc:
        if predict_clicked:
            prob = predict(st.session_state.model, st.session_state.scaler,
                           cgpa, internships, projects, certifications, comm, coding)
            pct  = prob * 100
            clr  = "#3b82f6" if pct >= 65 else ("#f59e0b" if pct >= 40 else "#ef4444")

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix":"%","font":{"size":38,"color":"#f0f9ff","family":"Syne"},"valueformat":".1f"},
                title={"text":"Placement Probability","font":{"color":"#64748b","size":13}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#334155","tickfont":{"color":"#475569"}},
                    "bar":{"color":clr}, "bgcolor":"rgba(0,0,0,0)", "borderwidth":0,
                    "steps":[
                        {"range":[0,40],  "color":"rgba(239,68,68,.10)"},
                        {"range":[40,65], "color":"rgba(245,158,11,.10)"},
                        {"range":[65,100],"color":"rgba(59,130,246,.10)"},
                    ],
                    "threshold":{"line":{"color":"#f59e0b","width":2},"thickness":.8,"value":65},
                },
            ))
            fig.update_layout(**{k:v for k,v in PLY.items() if k not in("xaxis","yaxis")}, height=290)
            st.plotly_chart(fig, use_container_width=True)

            # Verdict
            if pct >= 65:   st.success(f"🎉 **High Placement Chances ({pct:.1f}%)** — Your profile is strong. Keep it up!")
            elif pct >= 40: st.warning(f"⚡ **Moderate Chances ({pct:.1f}%)** — A few improvements will make a big difference.")
            else:           st.error(  f"⚠️ **Needs Improvement ({pct:.1f}%)** — Focus on internships, projects, and skills.")

            # Radar (profile vs benchmark)
            factors = {
                "CGPA":           min(cgpa/10, 1),
                "Internships":    min(internships/4, 1),
                "Projects":       min(projects/6, 1),
                "Certifications": min(certifications/6, 1),
                "Communication":  comm/5,
                "Coding":         coding/5,
            }
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=list(factors.values()), theta=list(factors.keys()),
                fill="toself", fillcolor="rgba(59,130,246,.18)",
                line=dict(color="#3b82f6",width=2), name="Your Profile"))
            fig2.add_trace(go.Scatterpolar(r=[.9,.75,.75,.75,.8,.9], theta=list(factors.keys()),
                fill="toself", fillcolor="rgba(6,182,212,.07)",
                line=dict(color="#06b6d4",width=1.5,dash="dot"), name="Benchmark"))
            fig2.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True,range=[0,1],tickfont=dict(color="#475569",size=9),gridcolor="rgba(255,255,255,.07)"),
                    angularaxis=dict(tickfont=dict(color="#94a3b8",size=11)),
                    bgcolor="rgba(0,0,0,0)",
                ),
                **{k:v for k,v in PLY.items() if k not in("xaxis","yaxis")},
                showlegend=True, height=320, title_text="Profile vs Benchmark",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Recommendations
            st.markdown("#### 💡 Personalised Recommendations")
            recs = []
            if cgpa < 7.5:          recs.append(("📚","Aim for CGPA ≥ 7.5 — unlocks significantly more companies."))
            if internships < 2:     recs.append(("💼","Complete at least 2 internships before final year."))
            if projects < 3:        recs.append(("🔧","Build 3+ end-to-end projects to showcase practical skills."))
            if certifications < 2:  recs.append(("📜","Earn 2+ certifications (AWS, Google, Microsoft, NPTEL)."))
            if comm < 4:            recs.append(("🗣️","Practise communication — join GDs, mock interviews, Toastmasters."))
            if coding < 4:          recs.append(("💻","Solve 2–3 DSA problems daily on LeetCode / HackerRank."))

            if recs:
                for icon, text in recs:
                    st.markdown(f"""
                    <div style="display:flex;gap:.7rem;align-items:flex-start;
                                padding:.6rem 0;border-bottom:1px solid rgba(255,255,255,.05);">
                        <span style="font-size:1.1rem;">{icon}</span>
                        <span style="color:#94a3b8;font-size:.87rem;">{text}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("🌟 Excellent profile! Maintain your standards and aim for top-tier companies.")
        else:
            st.markdown("""
            <div class="ku-card" style="text-align:center;padding:3.5rem 2rem;">
                <div style="font-size:3.5rem;margin-bottom:1rem;">🤖</div>
                <div style="color:#475569;font-size:.92rem;line-height:1.7;">
                    Upload your résumé for an instant profile scan,<br>
                    or fill in the fields manually and click<br>
                    <b style="color:#93c5fd;">Predict My Chances</b> to see your results.
                </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: SKILL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────
def skills_page():
    user = st.session_state.user_data
    st.markdown("""
    <div class="ku-header">
        <h1>🛠️ Skill Recommendations</h1>
        <p>Personalised learning roadmap for your programme</p>
    </div>""", unsafe_allow_html=True)

    cat         = get_skill_category(user["department"])
    dept_skills = DEPARTMENT_SKILLS.get(cat, DEPARTMENT_SKILLS["cse"])

    all_skills = sorted(set(
        dept_skills["core"] + dept_skills["recommended"] + dept_skills["additional"] + [
            "Python","Java","C++","JavaScript","SQL","Git","Linux","Docker",
            "Machine Learning","Deep Learning","Cloud Computing","React","Node.js",
        ]
    ))

    resume_result  = st.session_state.get("resume_result") or {}
    resume_skills  = resume_result.get("skills", [])
    prefill        = [s for s in all_skills if s in resume_skills]

    lc, rc = st.columns([1, 2.1])
    with lc:
        st.markdown("#### Your Current Skills")
        if resume_skills:
            st.markdown(f'<div class="success-box" style="margin-bottom:.8rem;">✅ Auto-filled {len(prefill)} skills from résumé upload.</div>', unsafe_allow_html=True)
        current_skills = st.multiselect("Select all skills you have:", all_skills, default=prefill if prefill else all_skills[:2])
        st.markdown(f'<div style="margin-top:.6rem;"><span class="badge">{len(current_skills)} skills selected</span></div>', unsafe_allow_html=True)
        if current_skills:
            badges = "".join(f'<span class="badge">{s}</span>' for s in current_skills)
            st.markdown(f"<div style='margin-top:.6rem;'>{badges}</div>", unsafe_allow_html=True)

    with rc:
        core_done = sum(1 for s in dept_skills["core"] if s in current_skills)
        rec_done  = sum(1 for s in dept_skills["recommended"] if s in current_skills)

        t1, t2, t3, t4 = st.tabs(["🔴  Core Skills","🟡  Recommended","🟢  Additional","⚠️  Skill Gaps"])
        with t1:
            st.markdown("**Must-have for your programme:**")
            for s in dept_skills["core"]:
                st.markdown(f"{'✅' if s in current_skills else '❌'} &nbsp; {s}")
            st.progress(core_done / max(len(dept_skills["core"]),1))
            st.caption(f"Core completion: {core_done} / {len(dept_skills['core'])}")

        with t2:
            st.markdown("**Highly recommended:**")
            for s in dept_skills["recommended"]:
                st.markdown(f"{'✅' if s in current_skills else '⬜'} &nbsp; {s}")
            st.progress(rec_done / max(len(dept_skills["recommended"]),1))
            st.caption(f"Recommended completion: {rec_done} / {len(dept_skills['recommended'])}")

        with t3:
            st.markdown("**Nice-to-have for a competitive edge:**")
            for s in dept_skills["additional"]:
                st.markdown(f"{'✅' if s in current_skills else '⬜'} &nbsp; {s}")

        with t4:
            priority = dept_skills["core"] + dept_skills["recommended"] + dept_skills["additional"]
            gaps = [s for s in priority if s not in current_skills][:10]
            if gaps:
                st.markdown("**Top skills to learn next:**")
                for i, s in enumerate(gaps, 1):
                    tier = "🔴" if s in dept_skills["core"] else ("🟡" if s in dept_skills["recommended"] else "🟢")
                    st.markdown(f"{tier} **{i}.** {s}")
                st.markdown("---")
                st.markdown("**📚 Platforms:**  Coursera · Udemy · NPTEL · LeetCode · HackerRank · GeeksforGeeks · YouTube")
            else:
                st.success("🎉 Outstanding — you've covered almost all recommended skills!")

    st.markdown("### 🗺️ 6-Month Development Roadmap")
    roadmap = pd.DataFrame({
        "Task":  ["Core Skill Mastery","DSA & Problem Solving","Project Building","Open Source / Internship","Advanced Concepts","Interview Preparation"],
        "Start": pd.to_datetime(["2025-01-01","2025-01-20","2025-03-01","2025-03-15","2025-05-01","2025-05-20"]),
        "End":   pd.to_datetime(["2025-03-01","2025-03-01","2025-05-01","2025-05-01","2025-07-01","2025-07-01"]),
        "Phase": ["Phase 1","Phase 1","Phase 2","Phase 2","Phase 3","Phase 3"],
    })
    fig = px.timeline(roadmap, x_start="Start", x_end="End", y="Task", color="Phase",
        title="Skill Development Timeline", color_discrete_sequence=["#3b82f6","#06b6d4","#8b5cf6"])
    fig.update_yaxes(autorange="reversed")
    _chart(fig, 380)

# ─────────────────────────────────────────────────────────────────
# PAGE: COMPANY ELIGIBILITY
# ─────────────────────────────────────────────────────────────────
def company_page():
    st.markdown("""
    <div class="ku-header">
        <h1>🏢 Company Eligibility</h1>
        <p>Discover which companies match your academic profile</p>
    </div>""", unsafe_allow_html=True)

    lc, rc = st.columns([1, 3])
    with lc:
        cgpa  = st.slider("Your CGPA", 5.0, 10.0, 7.5, 0.1)
        types = st.multiselect("Company Type",
            ["Product","Service","Consulting","E-Commerce","Fintech","Retail Tech"],
            default=["Product","Service","Consulting"])

    with rc:
        eligible = get_eligible_companies(cgpa, types if types else None)
        tier1 = [c for c in eligible if c["tier"] == 1]
        tier2 = [c for c in eligible if c["tier"] == 2]
        tier3 = [c for c in eligible if c["tier"] == 3]

        st.markdown(f"""
        <div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem;">
            <span class="badge badge-green">✓ Eligible: {len(eligible)}</span>
            <span class="badge badge-purple">🌟 Tier 1: {len(tier1)}</span>
            <span class="badge badge-yellow">⭐ Tier 2: {len(tier2)}</span>
            <span class="badge">💫 Tier 3: {len(tier3)}</span>
        </div>""", unsafe_allow_html=True)

        t1, t2, t3 = st.tabs([f"🌟  Dream  ({len(tier1)})", f"⭐  Target  ({len(tier2)})", f"💫  Safe  ({len(tier3)})"])

        def show_companies(lst):
            if not lst:
                st.info("Adjust your CGPA or filters to see companies in this tier.")
                return
            for c in lst:
                with st.expander(f"**{c['name']}**  ·  {c['package']}  ·  {c['type']}"):
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown(f"**Minimum CGPA:** `{c['min_cgpa']}`")
                        st.markdown(f"**Package:** {c['package']}")
                        st.markdown(f"**Company Type:** {c['type']}")
                    with cb:
                        st.markdown("**Required Skills:**")
                        for sk in c["skills"]: st.markdown(f"• {sk}")

        with t1: show_companies(tier1)
        with t2: show_companies(tier2)
        with t3: show_companies(tier3)

    st.markdown("### 📊 CGPA Requirements — All Companies")
    df_c = pd.DataFrame([{"Company":n,"Min CGPA":i["min_cgpa"],"Type":i["type"]} for n,i in COMPANY_CRITERIA.items()])
    fig = px.bar(df_c.sort_values("Min CGPA"), x="Min CGPA", y="Company", color="Type",
        orientation="h", title="CGPA Cut-off by Company",
        color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.add_vline(x=cgpa, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"  Your CGPA: {cgpa}", annotation_font_color="#f59e0b")
    fig.update_layout(**PLY, height=620, yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: DATASET MANAGEMENT
# ─────────────────────────────────────────────────────────────────
def dataset_page():
    st.markdown("""
    <div class="ku-header">
        <h1>📊 Dataset & Model Management</h1>
        <p>Upload placement data and retrain the prediction model</p>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📋  Current Dataset","📤  Upload New Dataset","🔧  Retrain Model"])

    with t1:
        df = st.session_state.placement_data
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Records",        f"{len(df):,}")
        c2.metric("Placed",         f"{int(df['Placed'].sum()):,}")
        c3.metric("Placement Rate", f"{df['Placed'].mean()*100:.1f}%")
        c4.metric("Model Accuracy", f"{st.session_state.model_accuracy*100:.1f}%")
        st.dataframe(df.head(30), use_container_width=True)
        st.download_button("📥 Download Dataset as CSV", df.to_csv(index=False),
            "karunya_placement_data.csv", "text/csv", use_container_width=True)

    with t2:
        st.markdown("""
        <div class="info-box" style="margin-bottom:1rem;">
            <b>Required CSV columns:</b><br>
            <code style="font-size:.82rem;">CGPA, Internships, Projects, Certifications, Communication_Skill, Coding_Skill, Placed</code><br>
            <span style="color:#475569;font-size:.8rem;">Placed must be 0 (not placed) or 1 (placed).</span>
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded:
            try:
                new_df = pd.read_csv(uploaded)
                req = ["CGPA","Internships","Projects","Certifications","Communication_Skill","Coding_Skill","Placed"]
                missing = [c for c in req if c not in new_df.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.success(f"✅ Valid dataset — {len(new_df):,} records loaded.")
                    st.dataframe(new_df.head(5), use_container_width=True)
                    if st.button("✅ Import Dataset", use_container_width=True):
                        st.session_state.placement_data = new_df
                        st.success("Dataset imported. Head to 'Retrain Model' to update the ML model.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with t3:
        df = st.session_state.placement_data
        st.markdown(f"""
        <div class="ku-card" style="margin-bottom:1.2rem;">
            <div style="display:flex;gap:2rem;flex-wrap:wrap;font-size:.85rem;">
                <div><div class="section-label">Dataset Size</div><b style="color:#f0f9ff;font-size:1.1rem;">{len(df):,} records</b></div>
                <div><div class="section-label">Current Model</div><b style="color:#f0f9ff;">{st.session_state.model_name}</b></div>
                <div><div class="section-label">Accuracy</div><b style="color:#6ee7b7;font-size:1.1rem;">{st.session_state.model_accuracy*100:.1f}%</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button("🚀 Retrain Model Now", use_container_width=True):
            with st.spinner("Training models… selecting best…"):
                model, scaler, acc, name, cm, report, _, _ = train_model(df)
                st.session_state.model = model; st.session_state.scaler = scaler
                st.session_state.model_accuracy = acc; st.session_state.model_name = name
                st.session_state.cm = cm; st.session_state.report = report
            st.success(f"✅ Retrained! **{name}** · Accuracy: **{acc*100:.1f}%**")
            fig = px.imshow(cm, text_auto=True,
                labels=dict(x="Predicted",y="Actual"),
                x=["Not Placed","Placed"], y=["Not Placed","Placed"],
                color_continuous_scale="Blues", title="Confusion Matrix")
            _chart(fig, 400)

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 .6rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;color:#f0f9ff;letter-spacing:-.02em;">🎓 KU Placement</div>
            <div style="font-size:.72rem;color:#334155;margin-top:3px;text-transform:uppercase;letter-spacing:.1em;">Intelligence System</div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        user = st.session_state.user_data
        name = user["email"].split("@")[0].replace(".", " ").title()
        dept_short = user["department"][:34] + "…" if len(user["department"]) > 36 else user["department"]
        st.markdown(f"""
        <div style="padding:.7rem .9rem;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:10px;margin-bottom:1rem;">
            <div style="font-size:.9rem;font-weight:600;color:#e2e8f0;">{name}</div>
            <div style="font-size:.75rem;color:#64748b;margin-top:1px;">{user['register_number']}</div>
            <div style="font-size:.72rem;color:#475569;margin-top:1px;">{dept_short}</div>
            <div style="font-size:.72rem;color:#334155;margin-top:2px;">{user['current_year']}</div>
        </div>""", unsafe_allow_html=True)

        PAGES = {
            "🏠  Dashboard":             "dashboard",
            "🎯  Placement Prediction":  "prediction",
            "🛠️  Skill Recommendations": "skills",
            "🏢  Company Eligibility":   "companies",
            "📊  Dataset Management":    "dataset",
        }
        selected = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()

        if st.button("🚪  Logout", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

        st.markdown("""
        <div style="text-align:center;font-size:.68rem;color:#1e293b;margin-top:.8rem;">
            © 2025 Karunya University<br>AI Placement Intelligence v2.1
        </div>""", unsafe_allow_html=True)

    page = PAGES[selected]
    if   page == "dashboard":  dashboard_page()
    elif page == "prediction": prediction_page()
    elif page == "skills":     skills_page()
    elif page == "companies":  company_page()
    elif page == "dataset":    dataset_page()


if __name__ == "__main__":
    main()
