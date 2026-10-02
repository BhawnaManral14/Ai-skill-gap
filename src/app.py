"""
Streamlit Dashboard for AI Skill-Gap & Job Readiness Predictor.
Provides an interactive, visually stunning UI for uploading resumes, selecting target roles,
viewing competency breakdowns, analyzing weak skill gaps, and exploring a custom 4-week roadmap.
"""

import io
import json
import os
import requests
import streamlit as st

# Direct module imports for fallback or local execution
try:
    from src.parser import extract_text_from_file, ResumeParsingError
    from src.analyzer import analyze_resume, load_job_profiles
except ImportError:
    try:
        from parser import extract_text_from_file, ResumeParsingError
        from analyzer import analyze_resume, load_job_profiles
    except ImportError:
        # Will rely purely on API if local files aren't in pythonpath
        pass


# ---------------------------------------------------------
# Page Configuration & Modern Custom Styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Skill-Gap & Job Readiness Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle-text {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }

    /* Metric Card Styling */
    .metric-container-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.95));
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .metric-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .badge-green {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }

    .badge-amber {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
    }

    .badge-red {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }

    /* Progress bar item container */
    .skill-row {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .skill-row:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .skill-header-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }

    .skill-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #1E293B;
    }

    .skill-level-tag {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
    }

    .skill-score-num {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.95rem;
        color: #0F172A;
    }

    /* Cards */
    .card-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        height: 100%;
    }

    .roadmap-card {
        background: #FFFFFF;
        border-radius: 14px;
        border-top: 4px solid #3B82F6;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-left: 1px solid #E2E8F0;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px -2px rgba(0,0,0,0.04);
    }

    .roadmap-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 3px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    .missing-skill-pill {
        display: inline-flex;
        align-items: center;
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }

    .resource-tag {
        display: inline-block;
        background: #F1F5F9;
        color: #475569;
        font-size: 0.75rem;
        border-radius: 4px;
        padding: 2px 6px;
        margin: 2px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Sample Resumes for Instant Testing
# ---------------------------------------------------------

SAMPLE_RESUMES = {
    "Select a pre-loaded sample...": None,
    "Sample: Mid-Level Data Scientist": (
        "Sarah Jenkins\n"
        "Data Scientist with 3 years of experience specializing in statistical modeling and machine learning.\n"
        "Core Skills: Python, Scikit-Learn, Pandas, NumPy, SQL (PostgreSQL), Regression, Random Forest, XGBoost.\n"
        "Experience:\n"
        "• Developed churn prediction models using LightGBM and Scikit-learn, improving customer retention by 14%.\n"
        "• Authored complex SQL queries with CTEs and window functions to aggregate weekly user metrics.\n"
        "• Executed A/B tests and hypothesis testing (t-tests, ANOVA) to evaluate UI variants.\n"
        "• Visualized findings using Matplotlib, Seaborn, and built interactive dashboards in Plotly.\n"
        "Education: B.S. in Applied Mathematics and Statistics."
    ),
    "Sample: Junior Data Analyst": (
        "Alex Rivera\n"
        "Junior Data Analyst passionate about business intelligence and exploratory data analysis.\n"
        "Core Skills: Microsoft Excel (VLOOKUP, Pivot Tables, Power Query), SQL (MySQL), Tableau, PowerBI, EDA.\n"
        "Experience:\n"
        "• Cleaned transactional databases removing duplicates and imputing missing records for 100K+ rows.\n"
        "• Built interactive Tableau and Power BI KPI dashboards for executive sales tracking.\n"
        "• Wrote relational SQL queries with joins and group by aggregations to answer operational questions.\n"
        "• Automated daily reporting spreadsheets using Excel formulas and macros.\n"
        "Education: B.A. in Business Administration."
    ),
    "Sample: Senior ML / MLOps Engineer": (
        "Marcus Zhao\n"
        "Machine Learning Engineer specializing in distributed training, model serving, and automated MLOps pipelines.\n"
        "Core Skills: Python, PyTorch, Docker, Kubernetes, MLOps, MLflow, FastAPI, CI/CD, GitHub Actions, Feature Engineering.\n"
        "Experience:\n"
        "• Deployed high-throughput real-time prediction microservices in Docker containers orchestrated with Kubernetes.\n"
        "• Architected automated continuous retraining and data drift monitoring using MLflow and Evidently AI.\n"
        "• Built asynchronous REST APIs with FastAPI and Pydantic handling 2,500 requests per second with sub-50ms latency.\n"
        "• Authored GitHub Actions CI/CD workflows for automated unit testing, linting, and container registry publishing.\n"
        "• Engineered temporal feature pipelines in PyTorch and implemented mixed-precision training.\n"
        "Education: M.S. in Computer Science."
    ),
    "Sample: Senior Software Engineer": (
        "David Chen\n"
        "Senior Software Engineer with 5+ years of experience designing and scaling distributed backend systems.\n"
        "Core Skills: Python, Java, RESTful APIs, Microservices, PostgreSQL, System Design, Docker, Git, Unit Testing.\n"
        "Experience:\n"
        "• Designed and built high-performance microservices in Python and Java processing 15M daily requests.\n"
        "• Optimized PostgreSQL database indexing, partitioning, and connection pools, cutting p99 latency by 35%.\n"
        "• Containerized 12 core services using Docker and orchestrated deployments via GitHub Actions CI/CD.\n"
        "• Implemented Redis caching and RabbitMQ asynchronous queues for resilient decoupled architectures.\n"
        "Education: B.S. in Computer Science."
    ),
    "Sample: AI / GenAI Engineer": (
        "Elena Rostova\n"
        "AI Engineer specializing in Large Language Models, agentic architectures, and production RAG pipelines.\n"
        "Core Skills: LLMs, Prompt Engineering, RAG, Vector Databases (Pinecone/Chroma), LangChain, LangGraph, Python, FastAPI.\n"
        "Experience:\n"
        "• Built enterprise RAG assistant with hybrid search, reciprocal rank fusion, and context-aware chunking.\n"
        "• Created autonomous multi-agent research workflows using LangGraph and tool calling for document synthesis.\n"
        "• Fine-tuned open-source Llama-3 models with QLoRA on domain datasets, improving format compliance by 28%.\n"
        "• Benchmarked generation faithfulness and relevancy using Ragas and deployed inference endpoints on FastAPI.\n"
        "Education: M.S. in Artificial Intelligence."
    ),
    "Sample: MERN Stack Developer": (
        "Rahul Sharma\n"
        "Full-Stack MERN Developer with 3+ years of experience building scalable web applications and REST APIs.\n"
        "Core Skills: React.js, Node.js, Express.js, MongoDB, Mongoose, Redux Toolkit, JWT, RESTful APIs, Git.\n"
        "Experience:\n"
        "• Developed and deployed responsive e-commerce web platform using React.js and Redux Toolkit.\n"
        "• Built RESTful backend microservices using Node.js and Express.js with JWT authentication and bcrypt.\n"
        "• Designed MongoDB schemas, aggregations, and indexing strategies with Mongoose ODM.\n"
        "• Implemented secure cookie-based session management, CORS configuration, and input validation.\n"
        "Education: B.Tech in Computer Science and Engineering."
    )
}


# ---------------------------------------------------------
# Helper Functions: Backend Communication & Direct Fallback
# ---------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"


def check_backend_health(api_url: str) -> bool:
    """Checks if the FastAPI backend is running and healthy."""
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False


def get_available_roles(api_url: str) -> list:
    """Fetches list of available benchmark job roles."""
    # Attempt via API first
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/roles", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("roles", [])
    except Exception:
        pass

    # Fallback to local profile loader
    try:
        profiles = load_job_profiles()
        return list(profiles.keys())
    except Exception:
        return [
            "Data Scientist",
            "Data Analyst",
            "Machine Learning Engineer",
            "Data Engineer",
            "AI / Generative AI Engineer",
            "Software Engineer",
            "Frontend Developer",
            "Full Stack Developer",
            "DevOps & Cloud Engineer",
            "Cybersecurity Analyst",
            "Product Manager (Technical)",
            "MERN Stack Developer",
            "Java Full Stack Developer",
            "Python / Django Developer",
            ".NET / C# Developer",
            "Mobile App Developer",
            "Android Developer",
            "iOS Developer",
            "QA Automation Engineer / SDET",
            "Manual QA / Software Tester",
            "UI/UX Designer",
            "Cloud Solutions Architect",
            "Site Reliability Engineer (SRE)",
            "Database Administrator (DBA)",
            "Business Analyst (IT / Tech)",
            "Scrum Master / Agile Delivery Manager",
        ]


def execute_analysis(file_bytes: bytes, filename: str, target_role: str, api_url: str, execution_mode: str) -> dict:
    """
    Submits resume for analysis either through the FastAPI HTTP endpoint or directly in-process.
    """
    if execution_mode == "FastAPI Backend (HTTP)":
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = {"target_role": target_role}
        resp = requests.post(f"{api_url.rstrip('/')}/analyze", files=files, data=data, timeout=60.0)
        
        if resp.status_code != 200:
            err_detail = "Failed to analyze resume"
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            raise Exception(f"API Error ({resp.status_code}): {err_detail}")
            
        return resp.json()

    else:
        # Direct In-Process Mode
        resume_text = extract_text_from_file(file_bytes, filename)
        result = analyze_resume(resume_text, target_role)
        return {
            "readiness_score": result["readiness_score"],
            "skill_breakdown": result["skill_breakdown"],
            "missing_skills": result["missing_skills"],
            "readiness_tier": result["readiness_tier"],
            "study_plan": result["study_plan"],
        }


# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------

st.markdown('<div class="main-title">🎯 AI Skill-Gap & Job Readiness Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Upload your resume to evaluate semantic competency match against target industry benchmarks, '
    'uncover critical skill gaps, and generate an actionable 4-week roadmap.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Evaluation Settings")

    api_url = st.text_input("FastAPI Service URL", value=DEFAULT_API_URL, help="Backend endpoint for resume processing.")
    backend_online = check_backend_health(api_url)

    if backend_online:
        st.success("🟢 FastAPI Backend: Connected")
        mode_options = ["FastAPI Backend (HTTP)", "Direct In-Process (Standalone)"]
    else:
        st.warning("🟡 FastAPI Backend: Offline (Using Direct Mode)")
        mode_options = ["Direct In-Process (Standalone)", "FastAPI Backend (HTTP)"]

    execution_mode = st.radio(
        "Execution Engine",
        mode_options,
        index=0,
        help="Choose whether to route requests through the FastAPI REST API or run the NLP model directly within Streamlit."
    )

    st.markdown("---")
    st.markdown("### 💼 Target Role Benchmark")
    available_roles = get_available_roles(api_url)
    selected_role = st.selectbox(
        "Select Target Career Track",
        available_roles,
        index=0,
        help="The benchmark profile and skill weights used to evaluate your resume."
    )

    st.markdown("---")
    st.markdown("### 📄 Resume Input")
    
    uploaded_file = st.file_uploader(
        "Upload Resume Document",
        type=["pdf", "txt"],
        help="Upload PDF or TXT formatted resume. Text layer required for PDF documents."
    )

    sample_selection = st.selectbox(
        "Or test with a sample resume:",
        list(SAMPLE_RESUMES.keys()),
        index=0
    )

    analyze_clicked = st.button("🚀 Analyze Job Readiness", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Powered by **Sentence-Transformers** (`all-MiniLM-L6-v2`), **FastAPI**, and **Streamlit**.")


# ---------------------------------------------------------
# Main Analysis Trigger & State Handling
# ---------------------------------------------------------

if analyze_clicked:
    file_bytes = None
    filename = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
    elif sample_selection != "Select a pre-loaded sample...":
        sample_text = SAMPLE_RESUMES[sample_selection]
        file_bytes = sample_text.encode("utf-8")
        filename = f"{sample_selection.replace(' ', '_').lower()}.txt"
    else:
        st.error("⚠️ Please either upload a resume (.pdf or .txt) or select one of the pre-loaded sample resumes from the sidebar.")
        st.stop()

    with st.spinner("Analyzing semantic skill competencies and calculating readiness..."):
        try:
            analysis_data = execute_analysis(
                file_bytes=file_bytes,
                filename=filename,
                target_role=selected_role,
                api_url=api_url,
                execution_mode=execution_mode,
            )
            st.session_state["analysis_result"] = analysis_data
            st.session_state["active_role"] = selected_role
            st.session_state["active_file"] = filename
            st.success("✅ Analysis completed successfully!")
        except Exception as e:
            st.error(f"❌ Analysis Failed: {str(e)}")
            st.stop()


# ---------------------------------------------------------
# Display Analysis Results View
# ---------------------------------------------------------

if "analysis_result" in st.session_state:
    res = st.session_state["analysis_result"]
    active_role = st.session_state.get("active_role", "Target Role")
    active_file = st.session_state.get("active_file", "Resume")

    readiness_score = float(res.get("readiness_score", 0.0))
    skill_breakdown = res.get("skill_breakdown", {})
    missing_skills = res.get("missing_skills", [])
    readiness_tier = res.get("readiness_tier", "")
    study_plan = res.get("study_plan", [])

    # Dynamic badge color assignment
    if readiness_score >= 80.0:
        badge_class = "badge-green"
        tier_icon = "🟢"
        meter_color = "#10B981"
    elif readiness_score >= 60.0:
        badge_class = "badge-amber"
        tier_icon = "🟡"
        meter_color = "#F59E0B"
    else:
        badge_class = "badge-red"
        tier_icon = "🔴"
        meter_color = "#EF4444"

    st.markdown("---")

    # 1. Top Metric Display
    col_metric1, col_metric2, col_metric3 = st.columns([1.2, 2.0, 1.0])
    
    with col_metric1:
        st.metric(
            label="Overall Job Readiness",
            value=f"{readiness_score}%",
            delta=f"{readiness_score - 70.0:.1f}% vs. Benchmark" if readiness_score != 70.0 else "Benchmark Baseline",
            delta_color="normal" if readiness_score >= 60.0 else "inverse",
        )

    with col_metric2:
        st.markdown(f"**Target Role:** `{active_role}` &nbsp;|&nbsp; **Source:** `{active_file}`")
        st.markdown(
            f'<div class="metric-badge {badge_class}">{tier_icon} {readiness_tier}</div>',
            unsafe_allow_html=True,
        )

    with col_metric3:
        st.metric(
            label="Core Skills Evaluated",
            value=len(skill_breakdown),
            delta=f"{len(missing_skills)} Gaps Identified",
            delta_color="inverse" if len(missing_skills) > 0 else "normal",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Progress Bars: Core Skill Competency Breakdown
    st.markdown("### 📊 Core Skill Competency Breakdown")
    st.caption("Semantic match & keyword occurrence evaluated against weighted role expectations.")

    # Render skills in two clean responsive columns
    skill_cols = st.columns(2)
    sorted_skills = sorted(skill_breakdown.items(), key=lambda x: -x[1]["score"])

    for idx, (skill_name, data) in enumerate(sorted_skills):
        col_target = skill_cols[idx % 2]
        score = float(data["score"])
        weight_pct = int(data["weight"] * 100)
        level = data.get("level", "Competent")
        matched_kws = data.get("matched_keywords", [])

        # Color coding progress bars
        if score >= 80.0:
            level_bg = "#DCFCE7"
            level_color = "#15803D"
        elif score >= 65.0:
            level_bg = "#E0F2FE"
            level_color = "#0369A1"
        elif score >= 50.0:
            level_bg = "#FEF3C7"
            level_color = "#B45309"
        else:
            level_bg = "#FEE2E2"
            level_color = "#B91C1C"

        with col_target:
            st.markdown(
                f"""
                <div class="skill-row">
                    <div class="skill-header-flex">
                        <div>
                            <span class="skill-title">{skill_name}</span>
                            <span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">(Weight: {weight_pct}%)</span>
                        </div>
                        <div>
                            <span class="skill-level-tag" style="background:{level_bg}; color:{level_color};">{level}</span>
                            <span class="skill-score-num" style="margin-left:8px;">{score}%</span>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            # Native Streamlit progress bar
            st.progress(score / 100.0)

            if matched_kws:
                kw_badges = " ".join([f"<span class='resource-tag'>✓ {k}</span>" for k in matched_kws[:4]])
                st.markdown(f"<div style='margin-top:4px;'>{kw_badges}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='margin-top:4px;'><span class='resource-tag' style='color:#EF4444;'>No direct keywords detected</span></div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Two-Column Layout: Weak Skills vs Interview Readiness Assessment
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### ⚠️ Missing / Weak Skills (< 50%)")
        if missing_skills:
            st.markdown(
                f"<div class='card-box' style='border-left: 4px solid #EF4444;'>"
                f"<p style='color:#64748B; font-size:0.9rem;'>The following competencies fell below the benchmark threshold for <strong>{active_role}</strong>. Prioritize these areas to elevate your profile.</p>",
                unsafe_allow_html=True,
            )
            for item in missing_skills:
                skill_name = item["skill"]
                score = item["score"]
                gap = item["gap"]
                weight_pct = int(item["weight"] * 100)
                missing_kws = item.get("missing_keywords", [])

                kw_str = ", ".join(missing_kws[:3]) if missing_kws else "Key frameworks & concepts"
                st.markdown(
                    f"""
                    <div style="margin-bottom: 0.85rem; padding: 0.6rem; background:#FFF5F5; border-radius:8px; border:1px solid #FED7D7;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; color:#9B2C2C;">{skill_name}</span>
                            <span style="font-size:0.8rem; font-weight:600; color:#C53030;">Current: {score}% &nbsp;|&nbsp; Gap: -{gap}%</span>
                        </div>
                        <div style="font-size:0.8rem; color:#742A2A; margin-top:3px;">
                            Role Importance: <strong>{weight_pct}%</strong>. Recommended review: <em>{kw_str}</em>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="card-box" style="border-left: 4px solid #10B981; background:#F0FDF4;">
                    <h4 style="color:#15803D; margin:0 0 0.5rem 0;">🎉 Outstanding Competency Coverage!</h4>
                    <p style="color:#166534; font-size:0.9rem; margin:0;">
                        None of your target skills scored below 50%. Your profile demonstrates solid foundational coverage across all required pillars for this position.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown("### 🎯 Interview Readiness Assessment")
        st.markdown(
            f"""
            <div class="card-box" style="border-left: 4px solid #3B82F6;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem;">
                    Status: {readiness_tier}
                </div>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.5;">
                    Your weighted profile readiness score is <strong>{readiness_score}%</strong>. 
                    Based on our AI evaluation of candidate benchmarks:
                </p>
                <ul style="font-size: 0.88rem; color: #475569; padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong>Technical Screening Probability:</strong> {"High (Estimated 85%+ pass rate)" if readiness_score >= 80 else ("Moderate (Pass rate sensitive to specific topic depth)" if readiness_score >= 60 else "Low without project portfolio reinforcement")}.</li>
                    <li><strong>Core Advantage:</strong> Strongest performance in <em>{sorted_skills[0][0]}</em> ({sorted_skills[0][1]['score']}%).</li>
                    <li><strong>Application Strategy:</strong> {"Proceed to apply immediately while polishing domain system design questions." if readiness_score >= 80 else ("Target the top 2 identified gaps for 2-3 weeks before submitting applications." if readiness_score >= 60 else "Commit to completing the 4-week project roadmap before formal technical screens.")}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Bottom Section: Actionable 4-Week Up-skilling Roadmap
    st.markdown("### 🗓️ Custom 4-Week Up-skilling Roadmap")
    st.caption("Actionable curriculum synthesized specifically to bridge your identified skill gaps.")

    roadmap_cols = st.columns(4)
    for idx, week_data in enumerate(study_plan):
        target_col = roadmap_cols[idx]
        week_num = week_data.get("week", idx + 1)
        title = week_data.get("title", f"Week {week_num}")
        theme = week_data.get("theme", "")
        focus_skills = week_data.get("focus_skills", [])
        objectives = week_data.get("weekly_objectives", [])
        project = week_data.get("hands_on_project", "")
        resources = week_data.get("key_resources", [])

        with target_col:
            st.markdown(
                f"""
                <div class="roadmap-card">
                    <div class="roadmap-badge">WEEK {week_num}</div>
                    <div style="font-weight:700; font-size:0.95rem; color:#0F172A; min-height:48px; margin-bottom:6px;">
                        {title}
                    </div>
                    <div style="font-size:0.78rem; font-style:italic; color:#64748B; margin-bottom:10px;">
                        {theme}
                    </div>
                    <hr style="margin: 6px 0; border:0; border-top:1px solid #F1F5F9;">
                    <div style="font-size:0.8rem; font-weight:700; color:#334155; margin-bottom:4px;">Core Objectives:</div>
                    <ul style="font-size:0.77rem; color:#475569; padding-left:1rem; margin-bottom:8px; line-height:1.4;">
                """,
                unsafe_allow_html=True,
            )
            for obj in objectives[:3]:
                st.markdown(f"<li style='font-size:0.77rem;'>{obj}</li>", unsafe_allow_html=True)

            st.markdown(
                f"""
                    </ul>
                    <div style="font-size:0.8rem; font-weight:700; color:#1E40AF; margin-top:8px;">🛠️ Hands-on Deliverable:</div>
                    <div style="font-size:0.76rem; color:#1E293B; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:6px; margin-top:4px;">
                        {project}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:
    # Initial empty state illustration
    st.info("👈 Upload your resume (.pdf or .txt) and choose a target role in the sidebar to begin your evaluation.")
