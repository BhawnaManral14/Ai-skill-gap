# 🎯 AI Skill-Gap & Job Readiness Predictor

A production-ready, modular web application that analyzes candidate resumes against industry benchmark role requirements using NLP semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and keyword matching.

The application calculates a weighted **Job Readiness Score (0-100%)**, provides a **Core Skill Competency Breakdown**, identifies **Missing/Weak Skills (<50%)**, evaluates **Interview Screening Probability**, and generates a customized **Actionable 4-Week Up-skilling Roadmap**.

---

## 🏗️ Architecture & Directory Structure

```
skill_gap_predictor/
├── data/
│   ├── job_profiles.json                    # Benchmark skill requirements, weights & keywords
│   ├── sample_resume_data_scientist.pdf     # Test PDF resume
│   └── sample_resume_ml_engineer.txt        # Test TXT resume
├── src/
│   ├── parser.py                            # PDF (pypdf) & TXT text extraction and cleaning
│   ├── analyzer.py                          # Embedding similarity, skill matching, readiness scoring & roadmap
│   ├── main.py                              # FastAPI backend with structured Pydantic response models
│   └── app.py                               # Streamlit dashboard interface
├── requirements.txt                         # Exact production dependencies
└── README.md                                # Execution & deployment documentation
```

---

## ⚡ Key Features

1. **Multi-Format Resume Ingestion**:
   - Parses multi-page PDF documents cleanly via `pypdf.PdfReader`.
   - Handles TXT files with resilient UTF-8 decoding and encoding fallbacks.
   - Robust error handling for corrupt files, empty inputs, or password-protected PDFs.

2. **Hybrid Semantic & Keyword Match Scoring**:
   - Uses `all-MiniLM-L6-v2` embeddings to capture localized contextual experience across resume chunks.
   - Combines semantic similarity (55%) with direct sub-skill keyword verification (45%).
   - Weighted overall job readiness score (0-100%) tailored to role priorities.

3. **Interview Readiness Classification**:
   - **80% - 100%**: *"Interview Ready - High probability of passing technical screen"*
   - **60% - 79%**: *"Nearly Ready - Target core gaps before applying"*
   - **< 60%**: *"Foundational Stage - Comprehensive project work required"*

4. **Actionable 4-Week Learning Roadmap**:
   - Rule-based curriculum generator prioritizing identified weak competencies.
   - Details weekly focus themes, learning objectives, hands-on portfolio deliverables, and practice resources.

5. **Dual Architecture (Decoupled & Standalone)**:
   - **FastAPI Backend**: Pre-warms embedding models during application startup to eliminate inference latency.
   - **Streamlit Frontend**: Interactive dashboard with real-time progress bars, color-coded badges, and a direct in-process fallback mode.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ is installed:

```bash
# Clone or navigate to the repository
cd skill_gap_predictor

# (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 2. Start the FastAPI Backend Service

Run the FastAPI server on port 8000:

```bash
# From within the skill_gap_predictor directory:
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

- **API Base URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

### 3. Launch the Streamlit Frontend

In a separate terminal window:

```bash
# From within the skill_gap_predictor directory:
streamlit run src/app.py
```

Streamlit will launch automatically at: `http://localhost:8501`.

---

## 📡 API Endpoints

### `GET /roles`
Returns all available benchmark career tracks and their skill weight profiles.

**Response Example:**
```json
{
  "roles": [
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
    "Scrum Master / Agile Delivery Manager"
  ],
  "profiles": { ... }
}
```

### `POST /analyze`
Accepts a multipart file upload (`.pdf` or `.txt`) and a `target_role` parameter.

**Request:**
- `file`: Multipart file binary (`.pdf` or `.txt`)
- `target_role`: String (e.g., `Data Scientist`, `Software Engineer`, `AI / Generative AI Engineer`, etc.)

**Response Schema (`AnalysisResponse`):**
```json
{
  "readiness_score": 83.0,
  "skill_breakdown": {
    "Machine Learning": {
      "score": 92.5,
      "weight": 0.20,
      "level": "Expert / Strong",
      "matched_keywords": ["machine learning", "scikit-learn", "xgboost", "random forest"],
      "missing_keywords": ["lightgbm", "cross-validation"]
    },
    ...
  },
  "missing_skills": [
    {
      "skill": "Deep Learning",
      "score": 42.0,
      "weight": 0.10,
      "gap": 58.0,
      "level": "Needs Improvement",
      "matched_keywords": [],
      "missing_keywords": ["pytorch", "tensorflow", "cnn", "transformers"]
    }
  ],
  "readiness_tier": "Interview Ready - High probability of passing technical screen",
  "study_plan": [
    {
      "week": 1,
      "title": "Week 1: Core Fundamentals & Theory — Focus on Deep Learning",
      "focus_skills": ["Deep Learning"],
      "theme": "Solidifying Core Foundations & Addressing High-Priority Blindspots",
      "weekly_objectives": [...],
      "hands_on_project": "...",
      "key_resources": [...]
    },
    ...
  ]
}
```

---

## 🧪 Testing with Sample Resumes

Pre-built sample resumes are included in the `data/` directory for immediate testing:

1. **Data Scientist (PDF)**: `data/sample_resume_data_scientist.pdf`
2. **Machine Learning Engineer (TXT)**: `data/sample_resume_ml_engineer.txt`

You can also test directly in the Streamlit UI using the pre-loaded sample selector in the sidebar!

### Automated Test Verification:
```bash
# Verify parsing and analysis via script
python -c "
from src.parser import extract_text_from_file
from src.analyzer import analyze_resume

with open('data/sample_resume_data_scientist.pdf', 'rb') as f:
    text = extract_text_from_file(f.read(), 'sample_resume_data_scientist.pdf')

res = analyze_resume(text, 'Data Scientist')
print('Readiness Score:', res['readiness_score'])
print('Tier:', res['readiness_tier'])
"
```
