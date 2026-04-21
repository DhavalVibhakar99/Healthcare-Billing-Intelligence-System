import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go
import os
import re
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
_deploy = ROOT_DIR / 'data' / 'cms_medicare_deploy.db'
_full   = ROOT_DIR / 'data' / 'cms_medicare.db'
DB_PATH = str(_deploy if _deploy.exists() else _full)
load_dotenv(ROOT_DIR / '.env')

st.set_page_config(
    page_title="Healthcare Billing Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── DARK MODE CUSTOM CSS ──────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #ffffff; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1c1e26;
        border: 1px solid #2d2f3e;
        border-radius: 12px;
        padding: 16px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 600;
        color: #ffffff;
        padding: 8px 0;
        border-bottom: 2px solid #2d5be3;
        margin-bottom: 20px;
    }
    
    /* Fraud finding cards */
    .finding-critical {
        background-color: #2d1515;
        border-left: 4px solid #e05252;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .finding-high {
        background-color: #2d2015;
        border-left: 4px solid #e09052;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .finding-medium {
        background-color: #2d2d15;
        border-left: 4px solid #e0d052;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    
    /* Chat container */
    .chat-container {
        background-color: #1c1e26;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d2f3e;
    }
    
    /* Chat messages */
    .chat-user {
        background-color: #2d5be3;
        border-radius: 12px 12px 4px 12px;
        padding: 10px 14px;
        margin: 8px 0;
        margin-left: 20%;
        color: white;
    }
    .chat-ai {
        background-color: #1c1e26;
        border: 1px solid #2d2f3e;
        border-radius: 12px 12px 12px 4px;
        padding: 10px 14px;
        margin: 8px 0;
        margin-right: 20%;
        color: white;
    }

    /* Divider */
    .custom-divider {
        border: none;
        border-top: 1px solid #2d2f3e;
        margin: 30px 0;
    }

    /* Dataframe */
    .dataframe { background-color: #1c1e26 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #2d2f3e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── DATABASE + CLIENT ─────────────────────────────────────
@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_resource
def get_client():
    api_key = st.secrets.get("OPENROUTER_API_KEY",
                              os.getenv("OPENROUTER_API_KEY"))
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


# ── SESSION STATE ─────────────────────────────────────────
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'dialog_q' not in st.session_state:
    st.session_state.dialog_q = ""

# ── SCHEMA ────────────────────────────────────────────────
SCHEMA = """
You are a SQL expert analyzing Medicare billing data.
DATABASE: SQLite
TABLE: medicare_billing
COLUMNS:
- provider_id: unique NPI identifier
- provider_name: LAST name or organization name. Use LIKE for partial matches.
- first_name: FIRST name only
- specialty: medical specialty
- state: two letter state abbreviation
- procedure_desc: plain English procedure description
- total_patients: number of unique patients
- avg_submitted_charge: amount provider submitted to Medicare
- avg_medicare_payment: amount Medicare actually paid
- place_of_service: F = facility, O = office
RULES:
- Always LIMIT to 15 unless asked otherwise
- Use ROUND() for decimals
- Use LIKE '%name%' for name searches
- Never use SELECT *
"""

# ── HELPERS ───────────────────────────────────────────────
def download_button(df, filename, label="Download CSV"):
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"⬇️ {label}",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

def run_query(sql):
    try:
        return pd.read_sql_query(sql, get_connection())
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()

def is_conversational(question):
    conversational_keywords = [
        'what is this', 'what does this', 'explain', 
        'tell me about', 'describe', 'how does', 
        'what are you', 'who built', 'about this'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower 
               for keyword in conversational_keywords)

def get_conversational_answer(question):
    prompt = f"""
You are an AI assistant for a Healthcare Billing Intelligence dashboard.
The dashboard analyzes 500,000 real CMS Medicare billing records from 2023.

Answer this question conversationally in 2-3 sentences:
"{question}"

Context about the data:
- Source: CMS Medicare Physician and Other Practitioners dataset 2023
- 500,000 rows sampled from 13 million total records
- Contains provider billing information, procedure codes, Medicare payments
- Used for fraud detection and billing anomaly analysis
- Key findings include Ellenberger (NP billing for surgery), 
  Phoenix Eye (wrong procedures), Johnson (upcoding)
"""
    message = get_client().chat.completions.create(
        model="google/gemma-3-4b-it:free",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.choices[0].message.content.strip()

# ── AI CHAT DIALOG ────────────────────────────────────────
@st.dialog("🤖 AI Query Assistant", width="large")
def ai_chat_dialog():
    # Deferred clear: applied before text_input renders so widget key isn't modified after instantiation
    if st.session_state.get("_clear_dialog_q"):
        st.session_state.dialog_q = ""
        del st.session_state["_clear_dialog_q"]

    st.progress(
        min(st.session_state.query_count / 20, 1.0),
        text=f"Queries used: {st.session_state.query_count} / 20",
    )

    st.markdown("**Try an example:**")
    examples = [
        "Which specialties have the highest average Medicare payment?",
        "Show me providers in New Jersey with average charge above 10000",
        "What procedures does provider with last name Ellenberger bill for?",
        "Which states have the most Medicare providers?",
        "Show me nurse practitioners billing more than 5000 on average",
    ]
    ex_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with ex_cols[i % 2]:
            if st.button(ex, key=f"dialog_ex_{i}", use_container_width=True):
                st.session_state.dialog_q = ex

    question = st.text_input(
        "Your question:",
        key="dialog_q",
        placeholder="e.g. Which specialties bill the most?",
    )

    ask_col, clear_col = st.columns([3, 1])
    with ask_col:
        ask_clicked = st.button(
            "🔍 Ask", type="primary", use_container_width=True, key="dialog_ask"
        )
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True, key="dialog_clear"):
            st.session_state.chat_history = []
            st.session_state.query_count = 0

    if ask_clicked and question:
        if st.session_state.query_count >= 20:
            st.error("Query limit reached (20/20). Click 'Clear' to reset.")
        else:
            with st.spinner("Thinking..."):
                try:
                    if is_conversational(question):
                        answer = get_conversational_answer(question)
                        st.session_state.chat_history.append({
                            "question": question,
                            "sql": None,
                            "results": None,
                            "conversational_answer": answer,
                        })
                    else:
                        filter_context = ""
                        if selected_states:
                            filter_context += f"\nOnly show data for states: {', '.join(selected_states)}"
                        if selected_specialties:
                            filter_context += f"\nOnly show data for specialties: {', '.join(selected_specialties)}"
                        prompt = (
                            f"{SCHEMA}\n{filter_context}\n"
                            f'Convert this question to SQL: "{question}"\n'
                            "Return ONLY the SQL query, no explanations, no backticks."
                        )
                        message = get_client().chat.completions.create(
                            model="google/gemma-3-4b-it:free",
                            max_tokens=500,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        sql = message.choices[0].message.content.strip()
                        sql = re.sub(r'```.*?\n', '', sql)
                        sql = re.sub(r'```', '', sql)
                        sql = sql.strip().rstrip(";")
                        results = pd.read_sql_query(sql, get_connection())
                        st.session_state.chat_history.append(
                            {"question": question, "sql": sql, "results": results}
                        )
                    st.session_state.query_count += 1
                    st.session_state["_clear_dialog_q"] = True
                except Exception as e:
                    if "rate" in str(e).lower():
                        st.error("Rate limit hit. Please wait 30 seconds.")
                    else:
                        st.error(f"Error: {e}")

    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        st.markdown("---")
        st.markdown(f"**Result:** _{latest['question']}_")
        if latest.get("conversational_answer"):
            st.markdown(latest["conversational_answer"])
        else:
            with st.expander("Generated SQL", expanded=False):
                st.code(latest["sql"], language="sql")
            if latest["results"] is not None and len(latest["results"]) > 0:
                st.dataframe(latest["results"], width='stretch')
                download_button(latest["results"], "ai_query_results.csv")
                num_cols = latest["results"].select_dtypes(include="number").columns.tolist()
                txt_cols = latest["results"].select_dtypes(include="object").columns.tolist()
                if num_cols and txt_cols:
                    fig = px.bar(
                        latest["results"],
                        x=txt_cols[0],
                        y=num_cols[0],
                        title=latest["question"],
                        color=num_cols[0],
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(
                        xaxis_tickangle=45,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                st.warning("No results found.")

# ── SIDEBAR FILTERS ───────────────────────────────────────
with st.sidebar:
    st.title("🔍 Filters")
    
    all_states = run_query(
        "SELECT DISTINCT state FROM medicare_billing ORDER BY state"
    )
    all_specialties = run_query(
        "SELECT DISTINCT specialty FROM medicare_billing ORDER BY specialty"
    )
    
    selected_states = st.multiselect(
        "States", options=all_states['state'].tolist(), default=[]
    )
    selected_specialties = st.multiselect(
        "Specialties", options=all_specialties['specialty'].tolist(), default=[]
    )
    
    def build_filter(prefix="WHERE"):
        filters = []
        if selected_states:
            states_str = ','.join([f"'{s}'" for s in selected_states])
            filters.append(f"state IN ({states_str})")
        if selected_specialties:
            specs_str = ','.join([f"'{s}'" for s in selected_specialties])
            filters.append(f"specialty IN ({specs_str})")
        if filters:
            return f"{prefix} " + " AND ".join(filters)
        return ""
    
# Filter clause will be built dynamically before each query
    
    st.markdown("---")
    if selected_states or selected_specialties:
        st.success(f"✅ Filters active")
    else:
        st.info("Showing all data")
    
    st.markdown("---")
    st.markdown("""
**About This Project**
Built to demonstrate end-to-end data science — ETL, SQL fraud detection,
statistical anomaly detection, and AI querying on real government data.

**Data Source**
CMS Medicare Part B
2023 · 500,000 records sampled from 13M rows

**Tech Stack**
Python · Pandas · SQLite
Streamlit · Plotly · Groq Llama AI
""")
    st.markdown("---")
    st.markdown("**📊 Quick Stats**")
    total = run_query("SELECT COUNT(*) as n FROM medicare_billing")
    st.metric("Records", f"{total['n'][0]:,}")

# ══════════════════════════════════════════════════════════
# SECTION 1 — HEADER
# ══════════════════════════════════════════════════════════
col_title, col_links = st.columns([3, 1])

with col_title:
    st.markdown("""
    <div style='padding: 30px 0 10px 0;'>
        <h1 style='font-size:38px; font-weight:700; color:#ffffff; margin:0;'>
            🏥 Medicare Fraud Detection Dashboard
        </h1>
        <p style='font-size:16px; color:#8b8fa8; margin:8px 0 0 0;'>
            End-to-end fraud detection on 500,000 real CMS Medicare billing
            records — anomaly detection, SQL analytics, and AI-powered querying
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_links:
    st.markdown("""
    <div style='padding:30px 0 10px 0; text-align:right;'>
        <a href='https://github.com/DhavalVibhakar99/Healthcare-Billing-Intelligence-System'
           target='_blank'
           style='color:#2d5be3; text-decoration:none; font-size:14px;'>
            📂 View Source Code
        </a><br>
        <a href='https://www.linkedin.com/in/dhavalvibhakar99'
           target='_blank'
           style='color:#2d5be3; text-decoration:none; font-size:14px;'>
            👤 Dhaval Vibhakar
        </a><br>
        <span style='color:#8b8fa8; font-size:12px;'>
            Python · Streamlit · Plotly · SQLite · OpenRouter Gemma AI
        </span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🤖 Ask AI", type="primary", use_container_width=True, key="open_ai_dialog"):
        ai_chat_dialog()

# ── KPI ROW ───────────────────────────────────────────────
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.metric("Total Records", "500,000", delta="2023 Dataset")
with col2:
    specs = run_query("SELECT COUNT(DISTINCT specialty) as n FROM medicare_billing")
    st.metric("Specialties", f"{specs['n'][0]}")
with col3:
    states = run_query("SELECT COUNT(DISTINCT state) as n FROM medicare_billing")
    st.metric("States", f"{states['n'][0]}")
with col4:
    provs = run_query("SELECT COUNT(DISTINCT provider_id) as n FROM medicare_billing")
    st.metric("Providers", f"{provs['n'][0]:,}")
with col5:
    avg_c = run_query("SELECT ROUND(AVG(avg_submitted_charge),2) as n FROM medicare_billing")
    st.metric("Avg Submitted", f"${avg_c['n'][0]:,}")
with col6:
    avg_p = run_query("SELECT ROUND(AVG(avg_medicare_payment),2) as n FROM medicare_billing")
    st.metric("Avg Paid", f"${avg_p['n'][0]:,}")
with col7:
    payment_rate = round((avg_p['n'][0] / avg_c['n'][0]) * 100, 1)
    st.metric("Payment Rate", f"{payment_rate}%",
              delta="of submitted charge",
              delta_color="inverse")

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 2 — FRAUD FINDINGS + ANOMALY STATS
# ══════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🚨 Key Fraud Findings</div>",
            unsafe_allow_html=True)

left_col, right_col = st.columns([1.3, 1])

with left_col:
    findings = [
        {
            "risk": "CRITICAL",
            "css": "finding-critical",
            "icon": "🔴",
            "provider": "Ellenberger",
            "detail": "Nurse Practitioner billing for spinal fusion surgery",
            "z_score": "71.9",
            "impact": "USD 28,056 avg charge",
            "status": "Confirmed Suspicious"
        },
        {
            "risk": "CRITICAL",
            "css": "finding-critical",
            "icon": "🔴",
            "provider": "Phoenix Eye Surgical Center",
            "detail": "Eye facility — zero eye procedures, 100% spinal billing",
            "z_score": "N/A",
            "impact": "USD 18,669 avg Medicare payment",
            "status": "Confirmed Suspicious"
        },
        {
            "risk": "HIGH",
            "css": "finding-high",
            "icon": "🟠",
            "provider": "Johnson",
            "detail": "Routine ED visit upcoded — billed USD 23,866",
            "z_score": "69.2",
            "impact": "USD 23,866 avg charge",
            "status": "Under Investigation"
        },
        {
            "risk": "HIGH",
            "css": "finding-high",
            "icon": "🟠",
            "provider": "Fleming",
            "detail": "Single provider billing across 8 different specialties",
            "z_score": "45.1",
            "impact": "Multiple specialty billing",
            "status": "Under Investigation"
        },
        {
            "risk": "MEDIUM",
            "css": "finding-medium",
            "icon": "🟡",
            "provider": "Buonocore — NJ",
            "detail": "Highest charge in New Jersey — USD 79,714 average",
            "z_score": "N/A",
            "impact": "USD 79,714 avg charge",
            "status": "Flagged for Review"
        },
    ]

    st.caption(
        "Z-Score measures how far a provider's billing deviates from their "
        "specialty average. Z > 2 = top 2.5% of billers — statistically abnormal."
    )

    for f in findings:
        st.markdown(f"""
        <div class='{f["css"]}'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-weight:600; font-size:15px;'>
                    {f["icon"]} {f["provider"]}
                    <span style='font-size:11px; background:#ffffff22; 
                    border-radius:4px; padding:2px 8px; margin-left:8px;'>
                        {f["risk"]}
                    </span>
                </span>
                <span style='font-size:12px; color:#aaaaaa;'>{f["status"]}</span>
            </div>
            <div style='margin-top:8px; font-size:13px; color:#dddddd;'>
                {f["detail"]}
            </div>
            <div style='margin-top:6px; font-size:12px; color:#aaaaaa;'>
                Z-Score: {f["z_score"]} &nbsp;|&nbsp; Financial Impact: {f["impact"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.markdown("**Anomaly Detection Summary**")

    filter_clause = build_filter()
    and_clause = filter_clause.replace("WHERE", "AND") if filter_clause else ""

    a1, a2 = st.columns(2)
    with a1:
        flagged = run_query(f"""
            WITH stats AS (
                SELECT specialty,
                       AVG(avg_submitted_charge) as mean,
                       AVG(avg_submitted_charge*avg_submitted_charge) -
                       AVG(avg_submitted_charge)*AVG(avg_submitted_charge) as var
                FROM medicare_billing GROUP BY specialty
            )
            SELECT COUNT(DISTINCT m.provider_id) as n
            FROM medicare_billing m
            JOIN stats s ON m.specialty = s.specialty
            {filter_clause.replace('state', 'm.state').replace('specialty', 'm.specialty') if filter_clause else ''}
            AND (m.avg_submitted_charge - s.mean) /
                NULLIF(SQRT(s.var),0) > 2
        """)
        st.metric("Flagged Providers", f"{flagged['n'][0]:,}",
                  delta="z-score > 2")

        scope = run_query(f"""
            SELECT COUNT(DISTINCT provider_id) as n
            FROM medicare_billing
            {filter_clause if filter_clause else "WHERE 1=1"}
            AND specialty IN (
                'Nurse Practitioner','Physician Assistant',
                'Family Practice','Internal Medicine'
            )
            AND (procedure_desc LIKE '%surgery%'
                 OR procedure_desc LIKE '%fusion%'
                 OR procedure_desc LIKE '%replacement%')
            AND avg_submitted_charge > 1000
        """)
        st.metric("Scope Violations", f"{scope['n'][0]:,}",
                  delta="non-surgical billing surgical")

    with a2:
        high_gap = run_query(f"""
            SELECT COUNT(DISTINCT provider_id) as n
            FROM medicare_billing
            {filter_clause if filter_clause else "WHERE 1=1"}
            AND (avg_submitted_charge - avg_medicare_payment) /
                NULLIF(avg_submitted_charge, 0) > 0.95
        """)
        st.metric("Extreme Gap", f"{high_gap['n'][0]:,}",
                  delta=">95% gap rate")

        multi = run_query(f"""
            SELECT COUNT(*) as n FROM (
                SELECT provider_id FROM medicare_billing
                {filter_clause}
                GROUP BY provider_id
                HAVING COUNT(DISTINCT specialty) > 3
            )
        """)
        st.metric("Multi-Specialty", f"{multi['n'][0]:,}",
                  delta=">3 specialties")

    st.markdown("---")

    # Mini anomaly chart
    top_anomalies = run_query("""
        WITH stats AS (
            SELECT specialty,
                   AVG(avg_submitted_charge) as mean,
                   AVG(avg_submitted_charge*avg_submitted_charge) -
                   AVG(avg_submitted_charge)*AVG(avg_submitted_charge) as var
            FROM medicare_billing GROUP BY specialty
        )
        SELECT m.provider_name,
               ROUND((AVG(m.avg_submitted_charge) - s.mean) /
                     NULLIF(SQRT(s.var),0), 1) as z_score
        FROM medicare_billing m
        JOIN stats s ON m.specialty = s.specialty
        GROUP BY m.provider_name, m.specialty
        HAVING z_score > 2
        ORDER BY z_score DESC
        LIMIT 5
    """)

    fig = px.bar(
        top_anomalies,
        x='z_score',
        y='provider_name',
        orientation='h',
        color='z_score',
        color_continuous_scale='Reds',
        title='Top 5 Anomalies by Z-Score',
        labels={'z_score': 'Z-Score', 'provider_name': ''}
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("Source: CMS Medicare Physician & Other Practitioners 2023")

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 3 — SPECIALTY ANALYSIS
# ══════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📊 Specialty Analysis</div>",
            unsafe_allow_html=True)

filter_clause = build_filter()  # ← rebuild here

spec_df = run_query(f"""
    SELECT specialty,
           ROUND(SUM(avg_medicare_payment), 2) as total_payment,
           ROUND(AVG(avg_submitted_charge), 2) as avg_charge,
           ROUND(AVG(avg_medicare_payment), 2) as avg_payment,
           COUNT(DISTINCT provider_id) as providers
    FROM medicare_billing
    {filter_clause}
    GROUP BY specialty
    ORDER BY total_payment DESC
    LIMIT 15
""")

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        spec_df, x='total_payment', y='specialty',
        orientation='h',
        color='total_payment',
        color_continuous_scale='Blues',
        title='Top 15 Specialties by Total Medicare Payment',
        labels={'total_payment': 'Total Payment (USD)', 'specialty': ''}
    )
    fig.update_layout(
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, width='stretch')

with col2:
    fig2 = px.bar(
        spec_df.head(10), x='specialty',
        y=['avg_charge', 'avg_payment'],
        barmode='group',
        title='Submitted Charge vs Medicare Payment',
        color_discrete_map={
            'avg_charge': '#e05252',
            'avg_payment': '#2d5be3'
        },
        labels={'value': 'Amount (USD)', 'specialty': ''}
    )
    fig2.update_layout(
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=45
    )
    st.plotly_chart(fig2, width='stretch')

st.caption("Source: CMS Medicare Physician & Other Practitioners 2023")
download_button(spec_df, "specialty_analysis.csv")

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 4 — GEOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🗺️ Geographic Analysis</div>",
            unsafe_allow_html=True)
filter_clause = build_filter()  # ← rebuild here

state_df = run_query(f"""
    SELECT state,
           ROUND(AVG(avg_medicare_payment), 2) as avg_payment,
           ROUND(AVG(avg_submitted_charge), 2) as avg_charge,
           COUNT(DISTINCT provider_id) as providers
    FROM medicare_billing
    {filter_clause}
    GROUP BY state
    ORDER BY avg_payment DESC
    LIMIT 20
""")

col1, col2 = st.columns(2)

with col1:
    fig = px.choropleth(
        state_df,
        locations='state',
        locationmode='USA-states',
        color='avg_payment',
        scope='usa',
        title='Average Medicare Payment by State',
        color_continuous_scale='Blues',
        labels={'avg_payment': 'Avg Payment (USD)'}
    )
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, width='stretch')

with col2:
    fig2 = px.bar(
        state_df.head(15),
        x='avg_charge', y='state',
        orientation='h',
        color='avg_charge',
        color_continuous_scale='Oranges',
        title='Top 15 States by Avg Submitted Charge',
        labels={'avg_charge': 'Avg Charge (USD)', 'state': ''}
    )
    fig2.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig2, width='stretch')

st.caption("Source: CMS Medicare Physician & Other Practitioners 2023")
download_button(state_df, "geographic_analysis.csv")

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 5 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🔬 Anomaly Detection</div>",
            unsafe_allow_html=True)

filter_clause = build_filter()
where_clause = filter_clause if filter_clause else ""
and_clause = filter_clause.replace("WHERE", "AND") if filter_clause else ""

anomaly_df = run_query(f"""
    WITH stats AS (
        SELECT specialty,
               AVG(avg_submitted_charge) as mean,
               AVG(avg_submitted_charge*avg_submitted_charge) -
               AVG(avg_submitted_charge)*AVG(avg_submitted_charge) as var
        FROM medicare_billing
        GROUP BY specialty
    )
    SELECT m.provider_name,
           m.specialty,
           m.state,
           ROUND(AVG(m.avg_submitted_charge), 2) as avg_charge,
           ROUND(s.mean, 2) as specialty_mean,
           ROUND((AVG(m.avg_submitted_charge) - s.mean) /
                 NULLIF(SQRT(s.var), 0), 2) as z_score
    FROM medicare_billing m
    JOIN stats s ON m.specialty = s.specialty
    {where_clause}
    GROUP BY m.provider_name, m.specialty, m.state
    HAVING z_score > 2
    ORDER BY z_score DESC
    LIMIT 15
""")

col1, col2 = st.columns([1.2, 1])

with col1:
    fig = px.bar(
        anomaly_df,
        x='z_score',
        y=anomaly_df['provider_name'] + ' (' + anomaly_df['specialty'] + ')',
        orientation='h',
        color='z_score',
        color_continuous_scale='Reds',
        title='Top 15 Statistical Outliers',
        labels={'x': 'Z-Score', 'y': ''}
    )
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown("**Outlier Details**")
    st.dataframe(
        anomaly_df[['provider_name', 'specialty',
                    'state', 'avg_charge', 'z_score']],
        width='stretch',
        height=460
    )

st.caption("Source: CMS Medicare Physician & Other Practitioners 2023")
download_button(anomaly_df, "anomaly_detection.csv")

# ── FOOTER ────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px; color:#8b8fa8; font-size:13px;'>
    Healthcare Billing Intelligence System &nbsp;|&nbsp;
    Built by Dhaval Vibhakar &nbsp;|&nbsp;
    Data: CMS Medicare 2023 &nbsp;|&nbsp;
    500,000 records analyzed
</div>
""", unsafe_allow_html=True)

# ── PREVIOUS QUERIES HISTORY ──────────────────────────────
if st.session_state.chat_history:
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>💬 Previous Queries</div>",
        unsafe_allow_html=True,
    )
    for idx, item in enumerate(reversed(st.session_state.chat_history)):
        query_num = len(st.session_state.chat_history) - idx
        row_count = len(item["results"])
        st.markdown(
            f"<div class='chat-user'>{item['question']}</div>",
            unsafe_allow_html=True,
        )
        result_text = (
            f"Found {row_count} result{'s' if row_count != 1 else ''}"
            if row_count > 0 else "No results found"
        )
        st.markdown(
            f"<div class='chat-ai'>{result_text} &nbsp;"
            f"<small style='color:#8b8fa8;'>Query #{query_num}</small></div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Full results & SQL — Query #{query_num}"):
            st.code(item["sql"], language="sql")
            if row_count > 0:
                st.dataframe(item["results"], width='stretch')
                download_button(
                    item["results"],
                    f"query_{query_num}_results.csv",
                    label=f"Download Query #{query_num} CSV",
                )