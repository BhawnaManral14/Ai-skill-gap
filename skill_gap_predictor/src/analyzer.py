"""
Analyzer Module for AI Skill-Gap & Job Readiness Predictor.
Computes semantic similarity embeddings using sentence-transformers (all-MiniLM-L6-v2),
evaluates keyword occurrence, calculates weighted readiness scores, and generates
a targeted 4-week learning roadmap based on identified skill gaps.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Global singleton instance for sentence transformer model to prevent reload latency
_MODEL_INSTANCE: Optional[SentenceTransformer] = None
MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-loads and returns the singleton SentenceTransformer model.
    Ensures model weights are kept in memory across requests.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME)
    return _MODEL_INSTANCE


def load_job_profiles() -> Dict[str, Any]:
    """
    Loads benchmark job profiles from data/job_profiles.json.
    Searches across common relative paths to ensure reliability regardless of working directory.
    """
    possible_paths = [
        Path(__file__).resolve().parent.parent / "data" / "job_profiles.json",
        Path(__file__).resolve().parent / "data" / "job_profiles.json",
        Path.cwd() / "data" / "job_profiles.json",
        Path.cwd() / "skill_gap_predictor" / "data" / "job_profiles.json",
    ]

    for p in possible_paths:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"job_profiles.json not found in expected locations. Checked: {[str(p) for p in possible_paths]}"
    )


def chunk_resume_text(text: str, chunk_size: int = 3) -> List[str]:
    """
    Splits resume text into coherent contextual chunks (bullet points, sentences, or line segments).
    This ensures embeddings capture localized skill experience rather than averaging over the entire document.
    """
    # Split by double newlines or single newlines with bullet points
    raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # Also break down very long lines by sentence delimiters
    sentences: List[str] = []
    for line in raw_lines:
        # Split on sentence terminals followed by space
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p_clean = p.strip(" •-*–\t")
            if len(p_clean) > 10:
                sentences.append(p_clean)

    if not sentences:
        return [text[:500]] if text else ["General resume content"]

    # Group into overlapping chunks of `chunk_size` sentences
    chunks: List[str] = []
    for i in range(0, len(sentences), max(1, chunk_size - 1)):
        window = sentences[i : i + chunk_size]
        chunk_str = " ".join(window).strip()
        if len(chunk_str) > 15:
            chunks.append(chunk_str)

    return chunks if chunks else sentences


def check_keyword_occurrences(resume_text: str, keywords: List[str]) -> Tuple[List[str], List[str], float]:
    """
    Checks direct occurrences of sub-skill keywords in the resume text using word-boundary regex.
    
    Returns:
        matched_keywords: List of found keywords
        missing_keywords: List of unmentioned keywords
        keyword_score: Normalized score (0.0 to 1.0)
    """
    lower_text = resume_text.lower()
    matched = []
    missing = []

    for kw in keywords:
        # Escape special regex characters in keywords (e.g. C++, CI/CD)
        escaped_kw = re.escape(kw.lower())
        pattern = rf"(?:\b|_){escaped_kw}(?:\b|_)"
        if re.search(pattern, lower_text):
            matched.append(kw)
        else:
            missing.append(kw)

    total_kws = len(keywords)
    if total_kws == 0:
        ratio = 0.5
    else:
        # Full keyword credit reached if candidate mentions at least 4 key tools/concepts or 60% of keywords
        effective_target = min(total_kws, 4)
        ratio = min(1.0, len(matched) / max(1, effective_target))

    return matched, missing, ratio


def compute_skill_scores(
    resume_text: str,
    job_profile: Dict[str, Any],
    model: Optional[SentenceTransformer] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes individual skill match scores (0-100%) by combining semantic embedding similarity
    with exact keyword occurrence checks.
    """
    if model is None:
        model = get_embedding_model()

    chunks = chunk_resume_text(resume_text)
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    skills_data = job_profile.get("skills", {})
    skill_results: Dict[str, Dict[str, Any]] = {}

    for skill_name, skill_info in skills_data.items():
        weight = float(skill_info.get("weight", 0.1))
        description = skill_info.get("description", "")
        keywords = skill_info.get("keywords", [])

        # 1. Keyword analysis
        matched_kws, missing_kws, kw_ratio = check_keyword_occurrences(resume_text, keywords)

        # 2. Semantic query formulation
        query_text = f"{skill_name}: {description}. Keywords: {', '.join(keywords[:6])}."
        query_embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)

        # 3. Compute cosine similarities with all resume chunks
        sim_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Take the average of the top 2 most relevant chunks
        top_k = min(2, len(sim_scores))
        top_indices = np.argsort(sim_scores)[-top_k:]
        semantic_sim = float(np.mean(sim_scores[top_indices]))

        # Calibrate semantic similarity:
        # MiniLM cosine similarity typically spans 0.20 (unrelated) to 0.75 (highly matching)
        calibrated_semantic = np.clip((semantic_sim - 0.20) / (0.65 - 0.20), 0.0, 1.0)

        # 4. Blend semantic score (55%) and keyword verification (45%)
        # If at least one keyword matches, grant a confidence boost
        kw_boost = 0.05 if len(matched_kws) > 0 else 0.0
        blended_ratio = (0.55 * calibrated_semantic) + (0.45 * kw_ratio) + kw_boost
        raw_score = float(np.clip(blended_ratio * 100.0, 5.0, 98.0))
        final_score = round(raw_score, 1)

        # Determine competency level label
        if final_score >= 80.0:
            level = "Expert / Strong"
        elif final_score >= 65.0:
            level = "Proficient"
        elif final_score >= 50.0:
            level = "Moderate Competency"
        elif final_score >= 35.0:
            level = "Needs Improvement"
        else:
            level = "Critical Gap"

        skill_results[skill_name] = {
            "score": final_score,
            "weight": weight,
            "level": level,
            "matched_keywords": matched_kws,
            "missing_keywords": missing_kws,
            "semantic_similarity": round(semantic_sim, 3),
        }

    return skill_results


def calculate_overall_readiness(skill_breakdown: Dict[str, Dict[str, Any]]) -> float:
    """
    Computes the weighted overall job readiness score (0-100%).
    """
    total_weight = sum(item["weight"] for item in skill_breakdown.values())
    if total_weight <= 0:
        total_weight = 1.0

    weighted_sum = sum(item["score"] * item["weight"] for item in skill_breakdown.values())
    overall_score = weighted_sum / total_weight
    return round(float(np.clip(overall_score, 0.0, 100.0)), 1)


def determine_readiness_tier(readiness_score: float) -> str:
    """
    Assigns candidate to an interview readiness tier based on overall score.
    """
    if readiness_score >= 80.0:
        return "Interview Ready - High probability of passing technical screen"
    elif readiness_score >= 60.0:
        return "Nearly Ready - Target core gaps before applying"
    else:
        return "Foundational Stage - Comprehensive project work required"


def identify_weak_skills(skill_breakdown: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts skills scoring below 50%, prioritized by role weight (descending) and score (ascending).
    """
    weak_skills = []
    for skill_name, data in skill_breakdown.items():
        if data["score"] < 50.0:
            gap = round(100.0 - data["score"], 1)
            weak_skills.append({
                "skill": skill_name,
                "score": data["score"],
                "weight": data["weight"],
                "gap": gap,
                "level": data["level"],
                "matched_keywords": data["matched_keywords"],
                "missing_keywords": data["missing_keywords"][:5],
            })

    # Sort primarily by weight (importance) descending, then score ascending
    weak_skills.sort(key=lambda x: (-x["weight"], x["score"]))
    return weak_skills


def generate_study_plan(
    weak_skills: List[Dict[str, Any]],
    skill_breakdown: Dict[str, Dict[str, Any]],
    target_role: str,
) -> List[Dict[str, Any]]:
    """
    Generates an actionable, targeted 4-week study plan based directly on the identified weak skills.
    If few or no skills are weak, generates an advanced interview-readiness & architecture mastery plan.
    """
    # Collect names of weak skills or target focus skills
    weak_skill_names = [w["skill"] for w in weak_skills]

    # Pre-defined curriculum blueprints per skill
    curriculum_catalog = {
        "Python": {
            "fundamentals": "Data structures (dicts, sets, generators), OOP design patterns, and clean code (PEP8)",
            "applied": "Vectorized computations with NumPy & Pandas, profiling, and unit testing with pytest",
            "project": "Build an asynchronous data pipeline with type hints and automated pytest coverage",
            "topics": ["Memory management & generators", "Pandas performance optimization", "Pytest unit test suite"]
        },
        "Machine Learning": {
            "fundamentals": "Supervised learning theory, bias-variance tradeoff, cross-validation strategies, and metric selection",
            "applied": "Gradient boosting tuning (XGBoost/LightGBM), hyperparameter optimization (Optuna), and error analysis",
            "project": "End-to-end predictive modeling pipeline on Kaggle dataset with feature importance interpretation (SHAP)",
            "topics": ["Ensemble methods (XGBoost/LightGBM)", "Hyperparameter tuning with Optuna", "Model interpretability with SHAP"]
        },
        "SQL": {
            "fundamentals": "Relational algebra, subqueries, grouping sets, and multi-table joins",
            "applied": "Window functions (ROW_NUMBER, DENSE_RANK, LAG/LEAD), recursive CTEs, and query indexing strategies",
            "project": "Design a relational schema and author complex analytical cohorts & retention queries",
            "topics": ["Window functions & CTEs", "Query plan analysis & indexing", "Customer churn & cohort SQL analysis"]
        },
        "Statistics": {
            "fundamentals": "Probability distributions, central limit theorem, and hypothesis formulation",
            "applied": "Parametric/non-parametric tests (t-tests, Mann-Whitney), A/B testing methodology, and sample size calculations",
            "project": "A/B test analysis notebook with statistical significance, p-value calculation, and confidence intervals",
            "topics": ["A/B testing statistical rigor", "P-values & False Discovery Rate", "Bayesian vs. Frequentist inference"]
        },
        "Deep Learning": {
            "fundamentals": "Neural network architectures, activation functions, loss landscapes, and backpropagation",
            "applied": "PyTorch tensor operations, DataLoader creation, learning rate schedulers, and regularization",
            "project": "Implement and train a custom vision or text classifier in PyTorch with TensorBoard logging",
            "topics": ["PyTorch nn.Module & training loop", "Transfer learning techniques", "Gradient clipping & regularization"]
        },
        "NLP": {
            "fundamentals": "Tokenization, word embeddings (Word2Vec, FastText), and transformer self-attention mechanisms",
            "applied": "Hugging Face Transformers library, fine-tuning pre-trained models (BERT/RoBERTa), and text preprocessing",
            "project": "Fine-tune a Hugging Face transformer model for multi-class document classification",
            "topics": ["Transformer attention mechanics", "Hugging Face pipelines & tokenizers", "Text classification & evaluation"]
        },
        "Model Deployment": {
            "fundamentals": "REST API design principles, OpenAPI specifications, and inference serialization (ONNX/Pickle)",
            "applied": "FastAPI asynchronous endpoints, Docker containerization, and request payload validation with Pydantic",
            "project": "Dockerized FastAPI microservice serving real-time model predictions with health checks",
            "topics": ["FastAPI endpoint authoring", "Docker multi-stage builds", "Model serialization & latency benchmarking"]
        },
        "Data Visualization": {
            "fundamentals": "Visual hierarchy, grammar of graphics, and cognitive load minimization in charting",
            "applied": "Interactive charts using Plotly/Seaborn, multi-panel layouts, and custom color palettes",
            "project": "Interactive analytical dashboard with responsive filters and publication-grade charts",
            "topics": ["Plotly interactive dashboards", "Seaborn distribution plots", "Executive data storytelling"]
        },
        "Excel": {
            "fundamentals": "Logical functions (IFS, SWITCH), XLOOKUP, INDEX-MATCH, and data validation rules",
            "applied": "Power Query for ETL, advanced pivot tables, dynamic arrays, and conditional KPI alerts",
            "project": "Automated financial or operations dashboard powered by dynamic Power Query refreshes",
            "topics": ["Power Query automation", "Dynamic array formulas (FILTER, UNIQUE)", "Executive KPI dashboarding"]
        },
        "Tableau": {
            "fundamentals": "Data connections, discrete vs. continuous fields, and standard chart configurations",
            "applied": "Level of Detail (LOD) calculations (FIXED, INCLUDE, EXCLUDE), parameters, and interactive actions",
            "project": "Multi-sheet interactive Tableau workbook with drill-downs and customized user filters",
            "topics": ["LOD expressions mastery", "Dashboard UI/UX and actions", "Data blending vs. relationships"]
        },
        "PowerBI": {
            "fundamentals": "Star schema modeling, relationships (1:*, *:*), and bidirectional filtering awareness",
            "applied": "DAX measures (CALCULATE, FILTER, ALL, time-intelligence functions), and row-level security",
            "project": "Enterprise Power BI dashboard with dynamic date tables and complex time-intelligence DAX KPIs",
            "topics": ["DAX CALCULATE & context transitions", "Time intelligence formulas", "Data modeling best practices"]
        },
        "Data Cleaning": {
            "fundamentals": "Data profiling, identifying null mechanisms (MCAR, MAR, MNAR), and duplicate resolution",
            "applied": "Robust imputation strategies, regex string cleaning, datetime parsing, and outlier capping (IQR/Z-score)",
            "project": "Automated data wrangling script processing messy real-world survey/telemetry data",
            "topics": ["Handling missing data mechanisms", "Regex pattern parsing", "Outlier detection & validation rules"]
        },
        "Business Intelligence": {
            "fundamentals": "North Star metrics, unit economics, conversion funnels, and customer lifetime value (LTV)",
            "applied": "Cohort retention tables, stakeholder requirement framing, and automated alerting logic",
            "project": "Comprehensive business performance deck translating transactional logs into executive strategy",
            "topics": ["Cohort retention modeling", "Stakeholder translation frameworks", "Executive metric definition"]
        },
        "Exploratory Data Analysis": {
            "fundamentals": "Summary statistics, skewness, kurtosis, correlation heatmaps, and distribution checks",
            "applied": "Multivariate analysis, interaction effects, feature clustering, and anomaly discovery",
            "project": "Rigorous exploratory data analysis notebook uncovering 5 non-obvious business anomalies",
            "topics": ["Multivariate hypothesis exploration", "Feature correlation & collinearity", "Data profiling automation"]
        },
        "Docker": {
            "fundamentals": "Container architecture vs. VMs, layers, images, base OS selection, and daemon basics",
            "applied": "Multi-stage builds, non-root user permissions, caching layers, and docker-compose configurations",
            "project": "Production-hardened, lightweight Docker image for a machine learning API service",
            "topics": ["Multi-stage Dockerfile optimization", "Docker Compose multi-service networks", "Container security & secrets"]
        },
        "MLOps": {
            "fundamentals": "Experiment tracking concepts, artifact versioning, and reproducibility principles",
            "applied": "MLflow tracking server setup, model registry state management, DVC for dataset versioning",
            "project": "Reproducible MLflow experiment tracking pipeline with registered model stages",
            "topics": ["MLflow experiment tracking", "Data drift monitoring (Evidently AI)", "Model registry promotions"]
        },
        "PyTorch": {
            "fundamentals": "Tensors, autograd engine, computational graphs, and loss calculation",
            "applied": "Custom torch.utils.data.Dataset & DataLoader, GPU acceleration (CUDA), and optimizer tuning",
            "project": "Custom deep learning model in PyTorch trained with mixed-precision and validation checkpoints",
            "topics": ["PyTorch custom datasets & loaders", "CUDA memory management", "Learning rate scheduling & checkpoints"]
        },
        "Kubernetes": {
            "fundamentals": "Cluster topology, Pods, Deployments, ReplicaSets, and Services (ClusterIP vs. NodePort)",
            "applied": "Resource requests/limits, Horizontal Pod Autoscaling (HPA), ConfigMaps, and Helm templating",
            "project": "Deploy a containerized model on local Kubernetes (Minikube/Kind) with HPA and readiness probes",
            "topics": ["Pod manifests & resource limits", "Horizontal Pod Autoscaler (HPA)", "Kubernetes Service routing"]
        },
        "API Development": {
            "fundamentals": "HTTP protocols, status codes, RESTful resource naming, and JSON schemas",
            "applied": "FastAPI dependency injection, middleware, asynchronous route handlers, and exception handling",
            "project": "Robust REST API with input validation, rate limiting, and comprehensive OpenAPI docs",
            "topics": ["FastAPI dependency injection", "Async route handling & concurrency", "Error handling & validation"]
        },
        "CI/CD": {
            "fundamentals": "Git workflows, branching strategies, semantic versioning, and automated trigger hooks",
            "applied": "GitHub Actions workflow syntax, matrix builds, automated test execution, and container publishing",
            "project": "Complete GitHub Actions CI pipeline running linters, tests, and building Docker image on PR",
            "topics": ["GitHub Actions workflow authoring", "Automated linting & pytest in CI", "Artifact caching & registry push"]
        },
        "Feature Engineering": {
            "fundamentals": "Feature types (categorical, continuous, ordinal), target encoding, and leakage prevention",
            "applied": "Scikit-Learn ColumnTransformer & custom transformers, cyclical time features, and interaction terms",
            "project": "Reusable feature engineering pipeline preventing data leakage between train and test splits",
            "topics": ["Preventing data leakage", "Scikit-Learn Pipeline & ColumnTransformer", "Categorical & temporal feature encoding"]
        },
        "Data Pipelines & ETL": {
            "fundamentals": "Batch vs streaming ETL/ELT paradigms, data extraction patterns, and idempotency",
            "applied": "dbt model transformations, schema migrations, data validation tests, and pipeline backfills",
            "project": "Production-grade ELT pipeline with automated data quality assertions and schema testing using dbt",
            "topics": ["Idempotent pipeline design", "dbt transformation models", "Data quality testing & assertions"]
        },
        "Apache Spark": {
            "fundamentals": "Distributed computing concepts, RDD lineage, Catalyst optimizer, and partition strategies",
            "applied": "PySpark DataFrames, broadcast joins, window functions, and distributed shuffle minimization",
            "project": "Scalable PySpark processing pipeline analyzing multi-gigabyte dataset with custom partitioning",
            "topics": ["PySpark optimization & shuffles", "Broadcast joins & partitioning", "Spark SQL performance tuning"]
        },
        "Workflow Orchestration": {
            "fundamentals": "DAG construction, scheduling intervals, upstream/downstream task dependencies, and SLAs",
            "applied": "Apache Airflow custom operators, task sensors, XComs, dynamic task mapping, and retries",
            "project": "End-to-end Airflow DAG managing automated ingestion, transformation, and email alerting",
            "topics": ["Airflow DAG authoring", "Task decorators & XComs", "Dynamic task mapping & backfills"]
        },
        "Data Warehousing": {
            "fundamentals": "Dimensional modeling, Kimball methodology, star and snowflake schemas, and slowly changing dimensions (SCD)",
            "applied": "Snowflake/BigQuery clustering keys, partitioning, materialized views, and query cost optimization",
            "project": "Dimensional data warehouse model with automated SCD Type 2 handling on BigQuery or Snowflake",
            "topics": ["Star schema & dimensional modeling", "SCD Type 1 & 2 implementation", "Clustering & warehouse cost optimization"]
        },
        "Kafka & Streaming": {
            "fundamentals": "Event-driven architecture, topic partitions, producer/consumer offset semantics, and consumer groups",
            "applied": "Kafka Python/confluent-kafka producers/consumers, serialization with Avro/JSON, and stream processing",
            "project": "Real-time streaming pipeline ingesting and processing high-throughput event streams",
            "topics": ["Kafka partition & consumer group mechanics", "Exactly-once processing semantics", "Streaming event schemas"]
        },
        "Cloud Data Platforms": {
            "fundamentals": "Cloud storage architectures (S3/GCS), IAM roles/policies, and serverless compute primitives",
            "applied": "Automated data ingestion to cloud lakes, lifecycle policies, and serverless compute triggers",
            "project": "Serverless event-driven ingestion pipeline storing raw telemetry in cloud lake storage",
            "topics": ["Cloud lake architecture", "IAM least-privilege security", "Serverless pipeline triggers"]
        },
        "LLMs & Prompt Engineering": {
            "fundamentals": "Transformer architecture intuition, tokenization, temperature, top-p, and context windows",
            "applied": "Chain-of-thought, Few-Shot prompting, ReAct frameworks, system instructions, and structured JSON outputs",
            "project": "Complex multi-step reasoning agent with structured Pydantic output validation and prompt regression tests",
            "topics": ["Few-shot & Chain-of-thought prompts", "Structured schema extraction", "Context window & token optimization"]
        },
        "RAG & Vector Databases": {
            "fundamentals": "Dense vs sparse embeddings, distance metrics (Cosine, Dot, Euclidean), and vector indexing (HNSW)",
            "applied": "Chunking strategies, hybrid search (BM25 + Dense), re-ranking (Cross-encoders), and vector store retrieval",
            "project": "Production RAG knowledge assistant over complex enterprise PDFs with hybrid search and reciprocal rank fusion",
            "topics": ["Context-aware document chunking", "Hybrid search (BM25 + Vector)", "Cross-encoder re-ranking"]
        },
        "AI Frameworks & Agents": {
            "fundamentals": "Agent state graphs, memory persistence, planning loops, and tool execution protocols",
            "applied": "LangGraph/LlamaIndex multi-agent coordination, human-in-the-loop approvals, and external API tool integration",
            "project": "Multi-agent autonomous research system coordinating specialized web search, summarization, and reporting agents",
            "topics": ["LangGraph agent state management", "Tool calling & schema validation", "Human-in-the-loop workflows"]
        },
        "Model APIs & Embeddings": {
            "fundamentals": "Embedding spaces, cosine similarity, token economics, and rate-limit handling",
            "applied": "Sentence-Transformers fine-tuning, OpenAI/Gemini/Anthropic API integration, and batch embedding caching",
            "project": "Semantic search engine with local sentence-transformers embeddings and fast in-memory indexing",
            "topics": ["Sentence embeddings fine-tuning", "Semantic similarity search", "API rate-limiting & backoff"]
        },
        "Fine-Tuning & Quantization": {
            "fundamentals": "Parameter-Efficient Fine-Tuning (PEFT), LoRA adapter matrices, and 4-bit/8-bit quantization mechanics",
            "applied": "Hugging Face SFTTrainer, bitsandbytes, QLoRA instruction tuning on domain-specific datasets",
            "project": "Fine-tune an open-source LLM (Llama/Mistral) using QLoRA for custom domain instruction adherence",
            "topics": ["LoRA & QLoRA mechanics", "Instruction tuning dataset prep", "GGUF/AWQ model quantization"]
        },
        "Evaluation & Guardrails": {
            "fundamentals": "Faithfulness, answer relevancy, context precision/recall, and prompt injection vulnerabilities",
            "applied": "Ragas evaluation framework, NeMo Guardrails, input sanitization, and automated eval benchmarks",
            "project": "Automated CI/CD eval suite benchmarking RAG response quality and enforcing safety guardrails",
            "topics": ["Ragas RAG metrics", "NeMo guardrail integration", "Prompt injection & jailbreak defense"]
        },
        "Core Programming": {
            "fundamentals": "Object-oriented design, SOLID principles, memory management, and idiomatic language constructs",
            "applied": "Design patterns (Factory, Strategy, Observer), concurrency/multithreading, and asynchronous event loops",
            "project": "High-performance modular backend service built with clean architecture and SOLID design patterns",
            "topics": ["Design patterns in practice", "Concurrency & thread synchronization", "SOLID principles refactoring"]
        },
        "Data Structures & Algorithms": {
            "fundamentals": "Asymptotic analysis (Big-O), recursion, arrays, linked lists, trees, graphs, and hash maps",
            "applied": "Graph traversal (BFS/DFS), dynamic programming, divide-and-conquer, and heap-based priority queues",
            "project": "Comprehensive algorithmic library solving complex optimization and traversal challenges with unit tests",
            "topics": ["Graph algorithms (Dijkstra, BFS/DFS)", "Dynamic programming patterns", "Algorithmic complexity optimization"]
        },
        "RESTful APIs & Microservices": {
            "fundamentals": "REST constraints, HTTP verbs/status codes, idempotency, microservice communication patterns",
            "applied": "FastAPI/Spring Boot route controllers, middleware, gRPC protocol buffers, and error envelopes",
            "project": "Microservice ecosystem with REST and gRPC endpoints, API gateway routing, and contract validation",
            "topics": ["gRPC & Protocol Buffers", "RESTful API standards & versioning", "Microservice fault tolerance"]
        },
        "Databases & SQL": {
            "fundamentals": "Relational algebra, ACID transaction isolation levels, indexing (B-Tree, Hash), and normalization",
            "applied": "PostgreSQL query tuning, EXPLAIN ANALYZE, connection pooling, and ORM mapping (SQLAlchemy)",
            "project": "Highly performant database schema with connection pooling, migrations, and index-optimized queries",
            "topics": ["Transaction isolation & locks", "Index optimization (B-Tree vs GIN)", "Query execution plans (EXPLAIN)"]
        },
        "System Design & Architecture": {
            "fundamentals": "CAP theorem, horizontal vs vertical scaling, caching strategies (Write-Through, Cache-Aside), and CDN",
            "applied": "Distributed rate limiting, message queue decoupling (RabbitMQ/Kafka), and Redis caching layers",
            "project": "High-availability distributed system architecture RFC addressing 100k requests/sec concurrency",
            "topics": ["Distributed caching with Redis", "CAP theorem & consistency models", "Message queue architectural patterns"]
        },
        "Testing & Quality": {
            "fundamentals": "Testing pyramid, unit vs integration vs regression testing, and mock vs stub vs spy",
            "applied": "Automated test suites with pytest/JUnit, fixtures, parameterization, and mutation testing",
            "project": "Full test suite with 90%+ code coverage, mocking external services, and integration test suites",
            "topics": ["Test-Driven Development (TDD)", "Mocking third-party dependencies", "Integration test design"]
        },
        "JavaScript & TypeScript": {
            "fundamentals": "Event loop, closures, prototypes, ES6+ syntax, TypeScript strict mode, interfaces, and generics",
            "applied": "Type-safe async operations, custom utility types, DOM manipulation, and modular architecture",
            "project": "Type-safe utility library published with comprehensive unit tests and TypeScript declaration files",
            "topics": ["TypeScript advanced generics", "JavaScript event loop & microtasks", "Type guards & utility types"]
        },
        "React & Modern Frameworks": {
            "fundamentals": "Component lifecycle, virtual DOM, reconciliation, JSX, and immutable state updates",
            "applied": "Custom React hooks (useMemo, useCallback), Next.js App Router, and Server Components (RSC)",
            "project": "Full Next.js web application utilizing Server Components, streaming SSR, and custom interactive hooks",
            "topics": ["React Server Components & Next.js App Router", "Custom hooks & performance tuning", "Component design patterns"]
        },
        "HTML5 & Modern CSS": {
            "fundamentals": "Semantic HTML, box model, Flexbox, CSS Grid, specificity, and cascading rules",
            "applied": "Tailwind CSS utility styling, responsive breakpoints, CSS transitions/animations, and modern variables",
            "project": "Pixel-perfect, fully responsive landing page with dark mode and smooth animations using Tailwind CSS",
            "topics": ["Flexbox & CSS Grid mastery", "Tailwind CSS design systems", "Modern CSS variables & animations"]
        },
        "State Management": {
            "fundamentals": "Unidirectional data flow, normalized state, action dispatching, and selector memoization",
            "applied": "Zustand stores, Redux Toolkit slices/thunks, and TanStack Query server-state caching",
            "project": "Complex client-side dashboard with synchronized server state, optimistic updates, and offline caching",
            "topics": ["TanStack Query caching & mutations", "Zustand lightweight state stores", "Optimistic UI updates"]
        },
        "Web Performance & SEO": {
            "fundamentals": "Critical rendering path, Core Web Vitals (LCP, INP, CLS), and browser parsing pipeline",
            "applied": "Bundle analysis, dynamic imports, image optimization, font preloading, and meta tags",
            "project": "Web performance audit and optimization achieving 95+ Google Lighthouse scores across all metrics",
            "topics": ["Core Web Vitals optimization", "Code splitting & lazy loading", "Lighthouse audit remediation"]
        },
        "REST & GraphQL Integration": {
            "fundamentals": "GraphQL schema definition (SDL), queries vs mutations, REST HTTP methods, and status codes",
            "applied": "Apollo Client integration, Axios interceptors, token refresh flows, and WebSocket subscriptions",
            "project": "Data-rich client dashboard consuming GraphQL queries and real-time WebSocket event feeds",
            "topics": ["Axios interceptors & auth refresh", "GraphQL Apollo Client queries", "Real-time WebSockets"]
        },
        "Responsive Design & Accessibility": {
            "fundamentals": "WCAG 2.1 AA criteria, semantic landmark tags, screen reader navigation, and keyboard focus traps",
            "applied": "ARIA roles/states, color contrast compliance, mobile-first media queries, and touch targets",
            "project": "Fully accessible UI component library certified with zero axe-core accessibility violations",
            "topics": ["WCAG AA compliance guidelines", "ARIA attributes & keyboard navigation", "Mobile-first responsive layouts"]
        },
        "Testing & Build Tools": {
            "fundamentals": "Bundling fundamentals (Rollup, Vite, Webpack), source maps, and tree shaking",
            "applied": "Jest and React Testing Library component tests, user-event simulations, and Cypress E2E tests",
            "project": "Automated testing pipeline with Vite, Vitest, React Testing Library, and Cypress end-to-end specs",
            "topics": ["React Testing Library user-event testing", "Vite build configuration", "Cypress E2E test suites"]
        },
        "Frontend Frameworks": {
            "fundamentals": "Client vs server rendering, component composition, and responsive layout foundations",
            "applied": "Next.js/React full-stack application development, form validation, and reactive UI states",
            "project": "Interactive dashboard with responsive navigation, form validations, and server-side rendering",
            "topics": ["React component composition", "Client & Server component boundaries", "Form validation patterns"]
        },
        "Backend Development": {
            "fundamentals": "Server architecture, request/response lifecycle, middleware chains, and asynchronous I/O",
            "applied": "Express/FastAPI backend API routing, request validation, authentication middleware, and database ORM",
            "project": "Scalable backend REST API with JWT authentication, logging middleware, and database persistence",
            "topics": ["Express/FastAPI middleware architecture", "Asynchronous I/O & database pooling", "API error handling envelopes"]
        },
        "Database Management": {
            "fundamentals": "Schema design, relational foreign keys vs NoSQL document collections, and index strategies",
            "applied": "Prisma/Mongoose ORM modeling, database migrations, connection pooling, and aggregation pipelines",
            "project": "Multi-model database architecture with relational user entities and NoSQL document activity logs",
            "topics": ["Prisma/SQLAlchemy ORM modeling", "Database migration strategies", "Aggregation pipeline optimization"]
        },
        "API Design & Integration": {
            "fundamentals": "REST resource modeling, status code semantics, API pagination, and rate limiting",
            "applied": "Swagger/OpenAPI documentation, webhook dispatching, third-party payment/auth API integration",
            "project": "Full API integration gateway connecting Stripe webhooks, third-party auth, and internal services",
            "topics": ["OpenAPI & Swagger documentation", "Webhook verification & processing", "API pagination & rate-limiting"]
        },
        "Cloud & Docker": {
            "fundamentals": "Containerization benefits, image layers, cloud deployment targets, and environment variables",
            "applied": "Multi-container Docker Compose, deploying to Vercel/AWS ECS, and cloud object storage",
            "project": "Containerized full stack application deployed to cloud infrastructure with automated DNS and SSL",
            "topics": ["Docker Compose multi-service apps", "Cloud deployment pipelines", "Environment variable security"]
        },
        "Authentication & Web Security": {
            "fundamentals": "OWASP Top 10 vulnerabilities (XSS, CSRF, SQLi), encryption, hashing (bcrypt), and sessions",
            "applied": "JWT access & refresh token rotation, secure HTTP-only cookies, CORS configuration, and RBAC",
            "project": "Zero-trust authentication microservice with refresh token rotation, MFA, and role-based permissions",
            "topics": ["JWT & refresh token rotation", "OWASP Top 10 mitigation", "Role-Based Access Control (RBAC)"]
        },
        "Version Control & CI/CD": {
            "fundamentals": "Git commit conventions, branching models (GitFlow, trunk-based), and merge conflict resolution",
            "applied": "GitHub Actions multi-step CI pipelines, automated testing, lint checks, and staging deployments",
            "project": "Production CI/CD pipeline running linters, automated tests, and deploying on release tag creation",
            "topics": ["Trunk-based development with Git", "GitHub Actions workflow design", "Automated deployment triggers"]
        },
        "Testing & Debugging": {
            "fundamentals": "Unit, integration, and E2E testing scopes, test fixtures, and assertion libraries",
            "applied": "Jest/Pytest test automation, Supertest API testing, centralized logging, and error tracking (Sentry)",
            "project": "Integrated testing and observability setup covering unit, API, and error monitoring with Sentry",
            "topics": ["API testing with Supertest/Pytest", "Centralized logging with correlation IDs", "Sentry error monitoring"]
        },
        "Cloud Infrastructure": {
            "fundamentals": "Cloud networking, VPCs, subnets, route tables, security groups, and object storage",
            "applied": "AWS/GCP infrastructure architecture, IAM policy authoring, auto-scaling groups, and load balancers",
            "project": "Multi-tier highly available cloud infrastructure with private subnets, NAT gateway, and ALB",
            "topics": ["VPC & Subnet networking design", "IAM policy least privilege", "Application Load Balancer & Auto-scaling"]
        },
        "Infrastructure as Code": {
            "fundamentals": "Declarative vs imperative IaC, state files, resource graph dependencies, and drift detection",
            "applied": "Terraform modules, remote state backends (S3/DynamoDB lock), variables, and outputs",
            "project": "Reusable Terraform module library provisioning complete cloud VPC, database, and compute clusters",
            "topics": ["Terraform module design", "Remote state management & locking", "Infrastructure drift remediation"]
        },
        "Kubernetes & Containers": {
            "fundamentals": "Kubernetes control plane architecture, pods, services, deployments, and container runtimes",
            "applied": "Writing Kubernetes manifests, Helm chart packaging, ingress controllers, and horizontal pod autoscaling",
            "project": "Custom Helm chart deploying a high-availability microservice with ingress, TLS, and autoscaling",
            "topics": ["Kubernetes Deployment & Service manifests", "Helm chart authoring & templating", "Horizontal Pod Autoscaling (HPA)"]
        },
        "CI/CD Automation": {
            "fundamentals": "Pipeline stages, artifact management, cache strategies, and continuous delivery vs deployment",
            "applied": "Advanced GitHub Actions/GitLab CI, matrix testing, Docker image build & scan, and GitOps with ArgoCD",
            "project": "End-to-end GitOps deployment pipeline using GitHub Actions, container vulnerability scans, and ArgoCD",
            "topics": ["GitOps with ArgoCD", "Container vulnerability scanning (Trivy)", "Matrix build pipelines"]
        },
        "Linux & Shell Scripting": {
            "fundamentals": "Linux filesystem hierarchy, file permissions, process management, and systemd units",
            "applied": "Writing modular Bash automation scripts, sed/awk text stream processing, cron jobs, and SSH hardening",
            "project": "Automated server provisioning and maintenance script with systemd service configuration and cron monitoring",
            "topics": ["Bash scripting best practices", "Systemd service management", "Text manipulation with awk & sed"]
        },
        "Monitoring & Observability": {
            "fundamentals": "Three pillars of observability (Metrics, Logs, Traces), Prometheus pull model, and alert fatigue",
            "applied": "Prometheus metric instrumentation, Grafana dashboard authoring, and distributed tracing with OpenTelemetry",
            "project": "Comprehensive observability stack running Prometheus, Grafana, and Loki with automated alert rules",
            "topics": ["Prometheus metric types & PromQL", "Grafana executive dashboard design", "Distributed tracing with OpenTelemetry"]
        },
        "Cloud Security & Networking": {
            "fundamentals": "Zero-trust architecture, network segmentation, TLS handshakes, and secret sprawl risks",
            "applied": "HashiCorp Vault secret storage, cert-manager automated TLS renewals, and security group audits",
            "project": "Automated secret rotation system using HashiCorp Vault integrated with cloud applications",
            "topics": ["HashiCorp Vault secret management", "Automated TLS certificate management", "Network perimeter security"]
        },
        "Site Reliability Engineering": {
            "fundamentals": "Service Level Indicators (SLIs), Service Level Objectives (SLOs), and Error Budget policies",
            "applied": "Designing SLO alerting rules, incident runbooks, post-mortem templates, and chaos experiments",
            "project": "SRE operational framework with SLO dashboards, automated burn-rate alerts, and blameless post-mortem",
            "topics": ["SLI/SLO definition & calculation", "Multi-window burn-rate alerts", "Blameless post-mortem culture"]
        },
        "Network & System Security": {
            "fundamentals": "OSI 7-layer model, TCP/IP handshake, DNS, firewall rules, and intrusion detection mechanisms",
            "applied": "Packet analysis with Wireshark, configuring iptables/UFW, and host-based intrusion detection",
            "project": "Network traffic anomaly analysis report identifying malicious beaconing and port scans via Wireshark",
            "topics": ["Wireshark packet inspection", "Firewall & iptables configuration", "Intrusion Detection Systems (Snort/Suricata)"]
        },
        "SIEM & Threat Detection": {
            "fundamentals": "Security operations center (SOC) triage, log normalization, and MITRE ATT&CK framework mapping",
            "applied": "Writing Splunk SPL queries, Microsoft Sentinel KQL detection rules, and threat hunting workflows",
            "project": "Suite of custom SIEM detection rules mapped to MITRE ATT&CK techniques with alert triage playbooks",
            "topics": ["Splunk SPL & Sentinel KQL queries", "MITRE ATT&CK threat mapping", "SOC alert triage & escalation"]
        },
        "Vulnerability Assessment": {
            "fundamentals": "Common Vulnerabilities and Exposures (CVE), CVSS scoring metrics, and OWASP Top 10 vulnerabilities",
            "applied": "Running Nessus/Qualys vulnerability scans, verifying findings with Burp Suite, and prioritizing patches",
            "project": "Comprehensive enterprise vulnerability assessment report with CVSS risk scores and remediation plans",
            "topics": ["Vulnerability scanning with Nessus", "OWASP Top 10 web vulnerabilities", "CVSS scoring & patch prioritization"]
        },
        "Incident Response & Forensics": {
            "fundamentals": "NIST/SANS incident response lifecycle: Preparation, Detection, Containment, Eradication, Recovery",
            "applied": "Live system memory capture (Volatility), disk imaging, timeline analysis, and indicator of compromise (IoC) extraction",
            "project": "End-to-end simulated breach investigation and forensic report documenting threat actor lateral movement",
            "topics": ["Memory & disk forensics analysis", "Incident containment strategies", "IoC extraction & YARA rule authoring"]
        },
        "Security Compliance & Governance": {
            "fundamentals": "Compliance frameworks: NIST Cybersecurity Framework, ISO 27001, SOC 2 Type II, and GDPR",
            "applied": "Gap analysis audits, authoring security control policies, vendor risk assessments, and audit evidence gathering",
            "project": "Enterprise SOC 2 readiness audit and security policy documentation covering Access Control and Change Management",
            "topics": ["ISO 27001 / SOC 2 control mappings", "Risk assessment methodologies", "Security policy documentation"]
        },
        "Identity & Access Management": {
            "fundamentals": "Principle of least privilege, multi-factor authentication (MFA), SSO (SAML, OIDC), and directory services",
            "applied": "Active Directory / Entra ID administration, Okta identity federation, and role-based access policy authoring",
            "project": "Enterprise IAM architecture implementing SSO, conditional access policies, and automated user lifecycle provisioning",
            "topics": ["SAML 2.0 & OIDC protocols", "Conditional access policies", "Least-privilege role assignment"]
        },
        "Cloud Security": {
            "fundamentals": "Cloud shared responsibility model, Cloud Security Posture Management (CSPM), and identity boundaries",
            "applied": "Auditing AWS/Azure IAM permissions, enforcing S3 encryption/public block, and automated CIS benchmark audits",
            "project": "Automated cloud security audit script validating cloud environment against CIS Foundation Benchmarks",
            "topics": ["AWS/Azure security best practices", "CSPM posture compliance", "Cloud IAM permission boundaries"]
        },
        "Cryptography & Secure Protocols": {
            "fundamentals": "Symmetric vs asymmetric encryption, cryptographic hashing (SHA-256), digital signatures, and PKI",
            "applied": "OpenSSL certificate generation, TLS cipher suite hardening, and secure key storage implementations",
            "project": "Public Key Infrastructure (PKI) demonstration with root CA generation, certificate signing, and TLS verification",
            "topics": ["Public Key Infrastructure (PKI)", "TLS 1.3 protocol handshake", "Cryptographic key lifecycle & storage"]
        },
        "Product Strategy & Roadmapping": {
            "fundamentals": "Product vision, OKRs, North Star metrics, market sizing (TAM/SAM/SOM), and value propositions",
            "applied": "Quarterly roadmap planning, stakeholder alignment decks, competitive matrix analysis, and GTM strategy",
            "project": "Comprehensive 12-month product vision and strategic roadmap deck with measurable quarterly OKRs",
            "topics": ["OKR goal setting & alignment", "Strategic roadmapping frameworks", "Competitive market analysis"]
        },
        "Feature Prioritization & PRDs": {
            "fundamentals": "Prioritization models (RICE, Kano, MoSCoW, Value vs Effort), and user story mapping",
            "applied": "Authoring comprehensive PRDs, defining edge cases, non-functional requirements, and acceptance criteria",
            "project": "Complete, production-ready Product Requirements Document (PRD) for a complex multi-platform feature",
            "topics": ["RICE prioritization scoring", "PRD authoring & edge case analysis", "User story & acceptance criteria drafting"]
        },
        "Data Analytics & Metrics": {
            "fundamentals": "AARRR Pirate Metrics, conversion funnels, cohort retention curves, churn rate, and LTV/CAC ratios",
            "applied": "Writing SQL for cohort analysis, building retention charts in Mixpanel/Amplitude, and executive KPI reporting",
            "project": "Deep-dive product analytics dashboard evaluating user acquisition, onboarding funnel drop-off, and 30-day retention",
            "topics": ["Cohort retention & churn analysis", "Funnel conversion optimization", "Product analytics instrumentation"]
        },
        "Agile & Scrum Delivery": {
            "fundamentals": "Agile Manifesto, Scrum ceremonies (Sprint Planning, Standup, Review, Retrospective), and Kanban principles",
            "applied": "Jira board administration, writing technical sprint tickets, estimating story points, and tracking velocity/burndown",
            "project": "Complete Jira sprint backlog simulation with epic decomposition, refined user stories, and acceptance tests",
            "topics": ["Sprint planning & velocity tracking", "Jira backlog grooming & estimation", "Scrum master & product owner synergy"]
        },
        "User Experience & Customer Research": {
            "fundamentals": "Qualitative vs quantitative research, user interview methodologies, usability heuristics, and mental models",
            "applied": "Conducting customer discovery interviews, synthesizing affinity maps, and wireframing user flows in Figma",
            "project": "Customer discovery research report with interview transcripts, empathy maps, and low-fidelity Figma prototypes",
            "topics": ["Customer discovery interviewing", "Heuristic UX evaluation", "Figma wireframing & prototyping"]
        },
        "A/B Testing & Experimentation": {
            "fundamentals": "Hypothesis formulation, null hypothesis, Type I/II errors, sample size calculation, and statistical power",
            "applied": "Designing multivariate/AB experiment plans, tracking primary/guardrail metrics, and statistical significance analysis",
            "project": "A/B test design document with sample size calculations, guardrail metrics, and post-experiment decision framework",
            "topics": ["Hypothesis formulation & test design", "Statistical power & sample sizing", "Guardrail metrics & rollout strategies"]
        },
        "Stakeholder Management": {
            "fundamentals": "Stakeholder mapping (Influence vs Interest), managing expectations, and influence without authority",
            "applied": "Facilitating cross-functional trade-off discussions, executive status updates, and conflict resolution",
            "project": "Executive briefing document negotiating technical debt refactoring vs new feature roadmap delivery",
            "topics": ["Influence without authority", "Executive communication & briefings", "Conflict resolution & trade-off framing"]
        },
        "Technical Acumen": {
            "fundamentals": "Client-server architecture, database tradeoffs, API contracts, latency vs throughput, and technical debt",
            "applied": "Evaluating engineering feasibility, reviewing technical architecture proposals (RFCs), and API documentation",
            "project": "Technical feasibility analysis assessing architectural options, cost implications, and delivery timelines",
            "topics": ["System architecture evaluation", "API design review for product managers", "Technical debt vs velocity trade-offs"]
        },
        "React.js & Frontend": {
            "fundamentals": "Component lifecycle, JSX syntax, Virtual DOM, and state vs props architecture",
            "applied": "React hooks (useState, useEffect, useMemo, custom hooks), React Router v6, and context providers",
            "project": "Interactive responsive single-page application with client-side routing, protected routes, and custom hooks",
            "topics": ["React custom hooks & lifecycle", "Virtual DOM & reconciliation", "React Router dynamic routing"]
        },
        "MongoDB & NoSQL": {
            "fundamentals": "BSON document structure, collections, MongoDB Compass, and embedded documents vs references",
            "applied": "Mongoose schema design, validators, aggregation pipelines ($lookup, $group, $match), and indexing",
            "project": "Complex e-commerce data model in MongoDB with Mongoose ODM, aggregation reports, and compound indexes",
            "topics": ["Mongoose schema modeling & validation", "Aggregation pipeline optimization", "Compound indexing & Atlas search"]
        },
        "Node.js Runtime": {
            "fundamentals": "V8 engine, Node.js event loop phases, process object, Buffer, and non-blocking I/O",
            "applied": "EventEmitters, Streams (Readable/Writable), file system operations, and npm package publishing",
            "project": "High-throughput streaming file processing service in Node.js with custom EventEmitters and backpressure",
            "topics": ["Event loop & libuv threadpool", "Node.js Streams & backpressure", "Asynchronous error propagation"]
        },
        "Express.js & REST APIs": {
            "fundamentals": "Express application structure, HTTP methods, router modularity, and middleware chaining",
            "applied": "Global error handling middleware, express-validator schemas, CORS policies, and rate-limiting",
            "project": "Production REST API server with modular routing, input validation, JWT authentication, and Swagger docs",
            "topics": ["Express middleware execution pipeline", "Route grouping & controllers", "Global error handling envelopes"]
        },
        "State Management (Redux/Zustand)": {
            "fundamentals": "Single source of truth, actions, pure reducers, and immutable state updates",
            "applied": "Redux Toolkit createSlice, createAsyncThunk, Zustand lightweight stores, and selector memoization",
            "project": "Global shopping cart and user session state architecture using Redux Toolkit with persistent local storage",
            "topics": ["Redux Toolkit slices & extraReducers", "Zustand minimal stores", "Async thunk lifecycle handling"]
        },
        "Java & Core Programming": {
            "fundamentals": "JVM architecture, memory management (Heap/Stack/GC), OOP, and Collections Framework",
            "applied": "Java Streams, Lambdas, Optional, Generics, and modern multithreading with ExecutorService",
            "project": "Concurrent transaction processing engine using Java ExecutorService, custom exceptions, and Streams API",
            "topics": ["Java Streams & functional idioms", "JVM garbage collection tuning", "Multithreading & thread-safety"]
        },
        "Spring Boot & Microservices": {
            "fundamentals": "Inversion of Control (IoC), Dependency Injection, Spring Application Context, and Auto-configuration",
            "applied": "Spring Boot REST controllers, Spring Security with JWT, Eureka service discovery, and Spring Cloud Config",
            "project": "Distributed microservices system with API Gateway, Eureka discovery, and Spring Security authentication",
            "topics": ["Spring Boot dependency injection", "Spring Security filter chains", "Microservices service discovery (Eureka)"]
        },
        "Hibernate & JPA": {
            "fundamentals": "JPA entity lifecycle, annotations (@Entity, @Table, @Id), and relational mapping",
            "applied": "Entity relationships (@OneToMany, @ManyToMany), fetching strategies (Lazy/Eager), and JPQL queries",
            "project": "Enterprise persistence layer using Spring Data JPA, custom JPQL queries, and pagination/sorting",
            "topics": ["N+1 query problem & entity graphs", "JPA entity relationships", "Spring Data JPA repositories"]
        },
        "Django & Django REST Framework": {
            "fundamentals": "Django MVT architecture, ORM models, migrations, and URL dispatcher routing",
            "applied": "DRF ModelSerializers, generic APIViewsets, custom permissions, and authentication classes",
            "project": "Full-featured REST API with Django REST Framework, token authentication, and filtered pagination",
            "topics": ["DRF Viewsets & ModelSerializers", "Custom permissions & throttling", "Django ORM query optimization"]
        },
        "C# & .NET Core": {
            "fundamentals": "C# type system, Common Language Runtime (CLR), memory safety, and LINQ expressions",
            "applied": "Async/await asynchronous methods, generics, extension methods, and dependency injection in .NET Core",
            "project": "Clean Architecture backend API in .NET 8 using CQRS pattern, MediatR, and automated unit testing",
            "topics": ["C# LINQ query syntax & delegates", "Asynchronous programming in .NET", "Dependency injection in ASP.NET"]
        },
        "Flutter & Dart": {
            "fundamentals": "Dart language syntax, Widget tree hierarchy, StatelessWidget vs StatefulWidget, and build contexts",
            "applied": "BLoC / Riverpod state management, custom animations, navigation routes, and HTTP API integration",
            "project": "Cross-platform mobile app with BLoC state management, offline Hive storage, and responsive UI",
            "topics": ["BLoC pattern & event dispatching", "Custom animations & gestures", "Dart async Streams & Futures"]
        },
        "Kotlin & Modern Android": {
            "fundamentals": "Kotlin null-safety, data classes, extension functions, and Android Activity/Fragment lifecycles",
            "applied": "Jetpack Compose UI, ViewModel, StateFlow, Coroutines IO dispatchers, and Room database persistence",
            "project": "Modern Android app built with Jetpack Compose, MVVM architecture, Retrofit networking, and Room SQLite",
            "topics": ["Jetpack Compose state hoisting", "Kotlin Coroutines & StateFlow", "MVVM clean repository architecture"]
        },
        "Swift Programming": {
            "fundamentals": "Swift optionals, protocols, structs vs classes, Automatic Reference Counting (ARC), and closures",
            "applied": "SwiftUI declarative layouts, @State, @Binding, ObservableObject, URLSession async networking, and SwiftData",
            "project": "Native iOS application with SwiftUI, MVVM architecture, offline SwiftData persistence, and dark mode",
            "topics": ["SwiftUI state management", "Automatic Reference Counting (ARC)", "Async/await networking with URLSession"]
        },
        "Test Automation Frameworks (Selenium/Cypress/Playwright)": {
            "fundamentals": "Test automation pyramid, DOM locators (CSS/XPath), explicit vs implicit waits, and Page Object Model",
            "applied": "Playwright/Cypress end-to-end test suites, parallel test execution, mocking network requests, and CI reporting",
            "project": "Automated regression testing framework in Playwright with Page Object Model and Allure visual reports",
            "topics": ["Page Object Model (POM) architecture", "Robust DOM locator strategies", "Headless CI/CD test execution"]
        },
        "Manual Testing & Test Execution": {
            "fundamentals": "Software testing lifecycle (STLC), test levels, black-box techniques (Equivalence/Boundary), and test plans",
            "applied": "Writing detailed test cases in JIRA/TestRail, executing exploratory test charters, and bug lifecycle triage",
            "project": "Comprehensive QA test documentation suite including Test Strategy, Test Cases, and Bug Severity Matrix",
            "topics": ["Black-box test design techniques", "JIRA defect reporting & severity triage", "Exploratory testing charters"]
        },
        "Figma & Prototyping": {
            "fundamentals": "Visual hierarchy, typography scales, color theory, layout grids, and Figma vector editing tools",
            "applied": "Auto-layout with responsive constraints, interactive component variants, design tokens, and prototypes",
            "project": "End-to-end mobile and web design system in Figma with reusable components, tokens, and click-through prototype",
            "topics": ["Auto-layout & responsive constraints", "Interactive component variants", "Design tokens & developer handoff"]
        },
        "Cloud Architecture Design (AWS/Azure/GCP)": {
            "fundamentals": "Well-Architected Framework pillars, shared responsibility model, regions, and availability zones",
            "applied": "Multi-tier cloud topology design, VPC subnetting, Application Load Balancers, and auto-scaling groups",
            "project": "Production-ready enterprise cloud architecture blueprint with multi-region failover and Terraform IaC",
            "topics": ["Well-Architected Framework design", "Multi-AZ high availability & failover", "Cloud cost optimization strategies"]
        },
        "Database Administration (PostgreSQL/MySQL/Oracle)": {
            "fundamentals": "RDBMS internal engine mechanics, WAL logs, buffer caches, and ACID transaction concurrency",
            "applied": "EXPLAIN ANALYZE query profiling, vacuum/analyze maintenance, point-in-time recovery (PITR), and read replicas",
            "project": "Database disaster recovery drill and automated vacuuming/indexing maintenance script with monitoring",
            "topics": ["Query plan optimization (EXPLAIN)", "Point-in-Time Recovery (PITR)", "Master-replica streaming replication"]
        },
        "Requirements Elicitation & BRDs": {
            "fundamentals": "Business analysis core concept model (BACCM), stakeholder mapping, and requirement classification",
            "applied": "Conducting requirements elicitation workshops, authoring BRDs, functional specs, and process modeling (BPMN)",
            "project": "Complete Business Requirements Document (BRD) with As-Is/To-Be BPMN process flows and user stories",
            "topics": ["Requirements elicitation techniques", "BPMN process flow modeling", "Business Requirements Document (BRD) authoring"]
        },
        "Scrum Framework & Agile Ceremonies": {
            "fundamentals": "Agile Manifesto values, 12 principles, Scrum roles, events, artifacts, and definition of done (DoD)",
            "applied": "Facilitating Sprint Planning, Daily Standups, Sprint Reviews, Retrospectives, and resolving team blockers",
            "project": "Scrum Team Playbook with Definition of Ready, Definition of Done, retrospective templates, and velocity metrics",
            "topics": ["Effective retrospective facilitation", "Sprint capacity & velocity planning", "Servant leadership & impediment removal"]
        }
    }

    # Case 1: Targeted roadmap for identified weak skills
    if weak_skill_names:
        # Determine primary and secondary weak skills
        primary_skill = weak_skill_names[0]
        secondary_skill = weak_skill_names[1] if len(weak_skill_names) > 1 else primary_skill
        tertiary_skill = weak_skill_names[2] if len(weak_skill_names) > 2 else secondary_skill

        p_info = curriculum_catalog.get(primary_skill, {
            "fundamentals": f"Core concepts and principles of {primary_skill}",
            "applied": f"Hands-on exercises and standard tooling for {primary_skill}",
            "project": f"Build a focused working demonstration utilizing {primary_skill}",
            "topics": [f"{primary_skill} essentials", "Best practices", "Hands-on implementation"]
        })
        s_info = curriculum_catalog.get(secondary_skill, {
            "fundamentals": f"Core theory and syntax for {secondary_skill}",
            "applied": f"Practical workflows and problem-solving with {secondary_skill}",
            "project": f"Implement a standalone mini-project in {secondary_skill}",
            "topics": [f"{secondary_skill} fundamentals", "Applied drills", "Integration"]
        })
        t_info = curriculum_catalog.get(tertiary_skill, {
            "fundamentals": f"Core knowledge and workflows for {tertiary_skill}",
            "applied": f"Intermediate techniques and design for {tertiary_skill}",
            "project": f"Real-world application of {tertiary_skill}",
            "topics": [f"{tertiary_skill} theory", "Optimization", "Testing"]
        })

        plan = [
            {
                "week": 1,
                "title": f"Week 1: Core Fundamentals & Theory — Focus on {primary_skill}",
                "focus_skills": [primary_skill],
                "theme": "Solidifying Core Foundations & Addressing High-Priority Blindspots",
                "weekly_objectives": [
                    f"Master theoretical concepts: {p_info['fundamentals']}.",
                    f"Review official documentation and core language/framework idioms for {primary_skill}.",
                    "Complete 5 foundational exercises to build mechanical fluency."
                ],
                "hands_on_project": f"Setup dev environment and complete a guided mini-lab covering: {p_info['fundamentals']}.",
                "key_resources": p_info["topics"] + ["Official Documentation", "Interactive Coding Drills"]
            },
            {
                "week": 2,
                "title": f"Week 2: Applied Tooling & Framework Proficiency — {primary_skill} & {secondary_skill}",
                "focus_skills": [primary_skill, secondary_skill] if primary_skill != secondary_skill else [primary_skill],
                "theme": "Practical Execution & Tooling Deep Dive",
                "weekly_objectives": [
                    f"Bridge theory to practice: {p_info['applied']}.",
                    f"Deep dive into secondary target skill: {s_info['fundamentals']}.",
                    "Implement error handling, logging, and unit verification for all workflows."
                ],
                "hands_on_project": f"Develop an applied module combining {primary_skill} with realistic datasets/scenarios: {p_info['project']}.",
                "key_resources": s_info["topics"] + ["Industry Reference Architectures", "Real-world Dataset Exploration"]
            },
            {
                "week": 3,
                "title": f"Week 3: End-to-End Hands-on Project Implementation — {secondary_skill} & {tertiary_skill}",
                "focus_skills": list(set([primary_skill, secondary_skill, tertiary_skill]))[:2],
                "theme": "Multi-Skill Synthesis & Portfolio Project",
                "weekly_objectives": [
                    f"Synthesize skills into an integrated solution: {s_info['project']}.",
                    f"Address edge cases and performance bottlenecks in {tertiary_skill}: {t_info['applied']}.",
                    "Document code cleanly with README, architecture overview, and setup instructions."
                ],
                "hands_on_project": f"Build and push an end-to-end portfolio-grade project resolving real-world requirements for {target_role}.",
                "key_resources": t_info["topics"] + ["GitHub Open Source Templates", "System Design Cheat Sheets"]
            },
            {
                "week": 4,
                "title": f"Week 4: Productionization, Mock Interviews & Role Readiness — {target_role}",
                "focus_skills": weak_skill_names[:3] if weak_skill_names else ["Technical Screening"],
                "theme": "Interview Simulation, Polish & Screening Preparedness",
                "weekly_objectives": [
                    f"Conduct 3 timed mock interview sessions covering {', '.join(weak_skill_names[:3])}.",
                    "Practice articulating architectural tradeoffs, complexity analysis, and debugging scenarios.",
                    f"Perform final resume audit to clearly highlight projects built during weeks 1-3 for {target_role} applications."
                ],
                "hands_on_project": "Complete a full 60-minute take-home technical assessment simulation and publish code with Dockerized reproducibility.",
                "key_resources": [
                    f"{target_role} Technical Interview Guides",
                    "Behavioral STAR Method Articulation",
                    "Live Coding & System Design Whiteboarding"
                ]
            }
        ]
        return plan

    # Case 2: Candidate is already strong (no weak skills < 50%) -> Advanced Mastery Sprint
    top_weighted_skills = sorted(skill_breakdown.items(), key=lambda x: -x[1]["weight"])[:3]
    top_skill_names = [s[0] for s in top_weighted_skills]

    return [
        {
            "week": 1,
            "title": f"Week 1: Advanced Edge Cases & Performance Tuning — {top_skill_names[0]}",
            "focus_skills": [top_skill_names[0]],
            "theme": "From Proficient to Subject Matter Expert",
            "weekly_objectives": [
                f"Analyze advanced internals, concurrency, and memory optimization for {top_skill_names[0]}.",
                "Conduct code reviews and benchmark execution runtimes against large-scale inputs.",
                "Identify non-obvious failure modes in high-concurrency environments."
            ],
            "hands_on_project": f"Benchmark and refactor an existing {top_skill_names[0]} codebase for 3x throughput improvement.",
            "key_resources": ["Advanced Internals Whitepapers", "Profiling & Flamegraph Tools", "Production Incident Post-Mortems"]
        },
        {
            "week": 2,
            "title": f"Week 2: Scalable Architecture & System Design for {target_role}",
            "focus_skills": top_skill_names[:2],
            "theme": "High-Availability System Design & Tradeoff Analysis",
            "weekly_objectives": [
                f"Design resilient end-to-end architectures utilizing {', '.join(top_skill_names[:2])}.",
                "Evaluate tradeoffs: latency vs. cost, horizontal vs. vertical scaling, consistency vs. availability.",
                "Draft comprehensive System Architecture Documents (RFC format)."
            ],
            "hands_on_project": f"Author a production RFC and deploy an autoscaling reference pipeline on cloud infrastructure.",
            "key_resources": ["Designing Data-Intensive Applications", "Enterprise Cloud Reference Architectures", "RFC Templates"]
        },
        {
            "week": 3,
            "title": f"Week 3: Open-Source Contributions & Capstone Demonstration",
            "focus_skills": top_skill_names,
            "theme": "Industry Impact & High-Visibility Portfolio Artifacts",
            "weekly_objectives": [
                "Submit a meaningful PR or bugfix to a prominent open-source library in the ecosystem.",
                "Publish an in-depth technical case study detailing technical challenges and performance metrics.",
                "Refactor all portfolio repositories to production standards (CI/CD, 90%+ test coverage, automated docs)."
            ],
            "hands_on_project": "Publish a capstone demonstration with live interactive demo and comprehensive documentation.",
            "key_resources": ["GitHub Trending Repositories", "Technical Blog Publishing", "CI/CD Pipeline Automation"]
        },
        {
            "week": 4,
            "title": f"Week 4: Executive Technical Screening & Leadership Interview Sprint",
            "focus_skills": [f"{target_role} Leadership"],
            "theme": "Interview Mastery & Offer Maximization",
            "weekly_objectives": [
                "Practice Senior/Lead level system design interviews with senior mentors.",
                "Structure STAR stories emphasizing high-scale business impact, cross-functional collaboration, and technical leadership.",
                "Prepare targeted questions for engineering directors regarding infrastructure maturity and roadmaps."
            ],
            "hands_on_project": "Record a 5-minute technical presentation explaining a complex architectural decision to technical stakeholders.",
            "key_resources": ["System Design Interview Guides", "Engineering Leadership Frameworks", "Compensation Negotiation Guides"]
        }
    ]


def analyze_resume(resume_text: str, target_role: str) -> Dict[str, Any]:
    """
    Main orchestration function:
    1. Validates target role and resume text.
    2. Loads benchmark profile.
    3. Computes individual skill match scores (semantic + keyword).
    4. Computes weighted overall job readiness score.
    5. Determines interview readiness tier.
    6. Identifies missing/weak skills (< 50%).
    7. Generates an actionable 4-week learning roadmap.
    
    Returns structured analysis dictionary matching the required schema.
    """
    if not resume_text or len(resume_text.strip()) < 10:
        raise ValueError("Resume text is empty or too short for meaningful analysis.")

    job_profiles = load_job_profiles()
    if target_role not in job_profiles:
        available_roles = list(job_profiles.keys())
        raise ValueError(
            f"Invalid target role '{target_role}'. Available roles are: {available_roles}"
        )

    profile = job_profiles[target_role]
    model = get_embedding_model()

    # Compute individual skill scores
    skill_breakdown = compute_skill_scores(resume_text, profile, model=model)

    # Compute overall readiness score
    readiness_score = calculate_overall_readiness(skill_breakdown)

    # Assign readiness tier
    readiness_tier = determine_readiness_tier(readiness_score)

    # Identify weak skills (< 50%)
    missing_skills = identify_weak_skills(skill_breakdown)

    # Generate 4-week study plan
    study_plan = generate_study_plan(missing_skills, skill_breakdown, target_role)

    return {
        "readiness_score": readiness_score,
        "skill_breakdown": skill_breakdown,
        "missing_skills": missing_skills,
        "readiness_tier": readiness_tier,
        "study_plan": study_plan,
        "target_role": target_role,
        "total_skills_evaluated": len(skill_breakdown),
    }
