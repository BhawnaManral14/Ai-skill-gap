"""
FastAPI Backend Application for AI Skill-Gap & Job Readiness Predictor.
Provides endpoints for retrieving benchmark job profiles and analyzing uploaded resumes (PDF/TXT).
Pre-warms the SentenceTransformer model on startup to ensure near-instantaneous inference.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Any
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local imports
try:
    from src.parser import extract_text_from_file, ResumeParsingError
    from src.analyzer import (
        get_embedding_model,
        load_job_profiles,
        analyze_resume,
    )
except ImportError:
    from parser import extract_text_from_file, ResumeParsingError
    from analyzer import (
        get_embedding_model,
        load_job_profiles,
        analyze_resume,
    )


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class RoleProfileResponse(BaseModel):
    title: str
    description: str
    skills: Dict[str, Any]


class RolesListResponse(BaseModel):
    roles: List[str]
    profiles: Dict[str, Any]


class AnalysisResponse(BaseModel):
    readiness_score: float = Field(..., description="Overall job readiness score (0-100%)")
    skill_breakdown: Dict[str, Any] = Field(..., description="Breakdown of individual skill scores and levels")
    missing_skills: List[Any] = Field(..., description="List of identified weak or missing skills (< 50%)")
    readiness_tier: str = Field(..., description="Interview readiness evaluation tier")
    study_plan: List[Dict[str, Any]] = Field(..., description="Actionable 4-week learning roadmap")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ---------------------------------------------------------
# Lifespan Event: Pre-warm Embedding Model on Startup
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the SentenceTransformer embedding model and caches job profiles
    during application startup to eliminate initial request latency.
    """
    print("[Startup] Initializing SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = get_embedding_model()
    _ = model.encode(["Warming up sentence transformer embedding engine."])
    profiles = load_job_profiles()
    print(f"[Startup] Ready! Loaded {len(profiles)} benchmark roles: {list(profiles.keys())}")
    yield
    print("[Shutdown] Cleaning up application resources.")


# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------

app = FastAPI(
    title="AI Skill-Gap & Job Readiness Predictor API",
    description="Production-ready FastAPI service for resume skill-gap extraction, semantic competency scoring, and roadmap generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Root welcoming endpoint with API metadata."""
    return {
        "service": "AI Skill-Gap & Job Readiness Predictor API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "GET /roles": "List all benchmark job roles",
            "POST /analyze": "Upload resume (PDF/TXT) and get comprehensive readiness assessment",
            "GET /health": "Service health check",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint confirming model status."""
    model = get_embedding_model()
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None),
        version="1.0.0",
    )


@app.get("/roles", response_model=RolesListResponse, tags=["Roles"])
async def get_roles():
    """
    Returns available target job profiles and their benchmark skill requirements.
    """
    try:
        profiles = load_job_profiles()
        return RolesListResponse(
            roles=list(profiles.keys()),
            profiles=profiles,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load job profiles: {str(e)}",
        )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_resume_endpoint(
    file: UploadFile = File(..., description="Resume file in .pdf or .txt format"),
    target_role: str = Form(..., description="Target job role (e.g. Data Scientist, Software Engineer, Full Stack Developer, DevOps & Cloud Engineer, etc.)"),
):
    """
    Uploads a resume file (PDF or TXT) and evaluates it against the benchmark profile of the specified target role.
    
    Returns:
    - readiness_score: Overall weighted job readiness (0-100%).
    - skill_breakdown: Individual skill competency scores, matched keywords, and levels.
    - missing_skills: Prioritized list of skills scoring below 50%.
    - readiness_tier: Interview screening readiness classification.
    - study_plan: Structured 4-week up-skilling roadmap customized to candidate gaps.
    """
    # 1. Validate file presence and format
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please upload a valid .pdf or .txt resume.",
        )

    filename = file.filename
    lower_name = filename.lower()
    if not (lower_name.endswith(".pdf") or lower_name.endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{filename}'. Allowed formats: .pdf, .txt",
        )

    # 2. Read file content
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if not file_bytes or len(file_bytes.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes).",
        )

    # 3. Extract text from file
    try:
        resume_text = extract_text_from_file(file_bytes, filename)
    except ResumeParsingError as rpe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Resume Extraction Error: {str(rpe)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while extracting resume text: {str(e)}",
        )

    # 4. Run semantic analyzer
    try:
        analysis_result = analyze_resume(resume_text, target_role.strip())
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )

    # 5. Format and return response matching exact schema
    return AnalysisResponse(
        readiness_score=analysis_result["readiness_score"],
        skill_breakdown=analysis_result["skill_breakdown"],
        missing_skills=analysis_result["missing_skills"],
        readiness_tier=analysis_result["readiness_tier"],
        study_plan=analysis_result["study_plan"],
    )


# ---------------------------------------------------------
# CLI Runner Entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
