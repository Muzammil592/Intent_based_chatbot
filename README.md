Markdown
# Native Intent-Based AI Chatbot Platform

A production-grade, highly optimized Natural Language Processing (NLP) conversational engine engineered completely from scratch using pure Python, NLTK, and Scikit-Learn. By intentionally avoiding high-level orchestration frameworks (like LangChain), this project showcases low-level system mechanics, mathematical feature engineering, isolated microservice design patterns, and robust sandboxed deployment via multi-stage Docker builds.

---

## 🏗️ Architectural Topology

The system separates engineering pipelines into isolated, decoupled functional modules, guaranteeing scalability, determinism, and clear separation of concerns.

text
       [ Live User Text Input String Stream via REST POST ]
                                │
                                ▼
                    [ Advanced Preprocessor ]
       (Contractions -> Regex Clean -> Spellcheck -> Lemmas)
                                │
                                ▼
                    [ Sparse TF-IDF Vectorizer ]
                (Unigrams + Bigrams Feature Matrix)
                                │
                                ▼
                   [ Logistic Regression Core ]
               (Balanced Multi-Class Weight Space)
                 /                             \
                /                               \
               ▼                                 ▼
       [Confidence >= 0.45]              [Confidence < 0.45]
               │                                 │
               ▼                                 ▼
        [Regex Entity Match]            [Safe Fallback Logic]
               │                        "Cannot verify intent"
               ▼
  [Context-Aware Response Injection]
🛠️ Deep Technical Stack & Core Mechanics
1. Advanced Structural Preprocessing Pipeline
Contraction Expansion: Restructures conversational short-hands (e.g., matching i'm into standard structural i am) preceding tokenization to prevent syntax parsing failures.

Agnostic Typo Invariance: Integrates pyspellchecker on runtime strings to reconcile human typing mistakes prior to lemmatization calculations.

Morphological Lemmatization & Token Sanitization: Employs an NLTK WordNetLemmatizer engine to resolve tokens back to their base semantic root vectors while systematically filtering custom stopword layers.

2. Feature Extractor Optimization
N-gram Vocabulary Vectorization: Utilizes Scikit-Learn TfidfVectorizer mapping parameterized boundaries to extract both unigram and bigram relationships (ngram_range=(1,2)).

Sublinear Frequency Scaling: Implements logarithmic sublinear scale matching (sublinear_tf=True) to suppress word frequency skewing in dense inputs.

3. Balanced Convex Classifier & Fast Serialization
Logistic Regression Engine: Solves multi-class targets using cross-entropy planes optimized for fast execution speeds on resource-constrained micro-architectures.

Boundary Hyperparameter Tuning: Set strict margin separation scaling (C=5.0) combined with automated class weights distribution configurations (class_weight='balanced') to prevent accuracy collapse over generic single-word phrases.

Joblib Asset Persistence: Replaces baseline pickle files with memory-mapped joblib arrays to eliminate retraining constraints upon container runtime startup.

4. Intent State Tracker & Extraction System
Regex Parametric Extractions: Extracts deterministic real-time data nodes—such as specific timeline metrics (times) or region targets (locations)—using isolated pattern matches before text cleanup steps.

Stateful Volatile Context Memory: Tracks context changes between consecutive REST request strings, passing historical data variables across execution turns to drive responsive, contextual state overrides.

📁 Repository Directory Layout
Plaintext
chatbot/
│
├── data/
│   └── intents.json          # Enriched corpus containing balanced intent patterns
│
├── src/
│   ├── preprocessor.py       # Text purification, contraction mapping, spellcheck pipelines
│   ├── model.py              # Logistic Regression multiclass classifier wrapper
│   ├── train.py              # Stratified train/test validation pipeline & artifact generator
│   └── chatbot.py            # Multi-threaded Flask REST API application microservice
│
├── tests/
│   └── test_chatbot.py       # Automated validation checks (Contractions, API, Guardrails)
│
├── .dockerignore             # Excludes compiled runtime artifacts and cache layers
├── config.yaml               # Central infrastructure & architectural management parameters
├── Dockerfile                # Multi-stage lean production Docker environment setup
└── requirements.txt          # Python dependencies explicitly locked for reproducibility
⚙️ Execution and Deployment Guide
Local Native Execution (Development Setup)
Ensure your host machine has a compatible Python runtime, install dependencies, map local resources, and test:

Bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
python src/train.py
python src/chatbot.py
Local Automation Test Suite execution
To run the automated validation block outside of active container setups using pytest:

Bash
pytest -v
Sandbox Docker Deployment (Production-Ready)
This stack features a specialized multi-stage Dockerfile to optimize build layer caching and keep runtime images extremely lean.

1. Compile the Image Assets & Generate Binary Weights
Bash
docker build -t native-intent-service:v3 .
2. Run Isolated Automation Unit Testing directly inside the Sandbox
Bash
docker run --rm native-intent-service:v3 pytest tests/
3. Initialize Server Allocations Background Daemon
Map your machine's host ports into the running container infrastructure (using an alternative port if port 5000 is allocated to other active backends):

Bash
docker run -d -p 5001:5000 --name active-intent-service --restart unless-stopped native-intent-service:v3
📊 Sample REST API HTTP I/O Matrix
Outbound Network POST Request
PowerShell
Invoke-RestMethod -Uri "http://localhost:5001/chat" -Method Post -ContentType "application/json" -Body '{"message":"can you set a reminder for 4:30 pm to call my teacher"}'
Expected Production JSON Response Payload
JSON
{
  "intent": "reminder",
  "confidence": 0.8142,
  "entities": {
    "location": {},
    "time": "4:30 pm"
  },
  "response": "Understood. I have logged an automatic system alert for you exactly at 4:30 pm."
}
Guardrail Input Verification Error Handling
JSON
// POST Raw Input Payload Validation Check Failure
{
  "error": "Message structure failed sizing constraints (1-500 chars)."
}
