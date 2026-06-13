# Native Intent-Based AI Chatbot Platform

A production-grade, highly optimized Natural Language Processing (NLP) conversational engine engineered completely from scratch using pure Python, NLTK, and Scikit-Learn. By intentionally avoiding high-level orchestration frameworks (like LangChain), this project showcases low-level system mechanics, mathematical feature engineering, and robust sandboxed deployment via multi-stage Docker builds.

---

## 🏗️ Architectural Topology

The system separates engineering pipelines into isolated, decoupled functional modules, guaranteeing scalability, determinism, and clear separation of concerns.

```text
       [ Live User Text Input String Stream ]
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

Morphological Lemmatization: Employs an NLTK WordNetLemmatizer engine to resolve tokens back to their base semantic root vectors, optimizing contextual overlap.

2. Feature Extractor Optimization
N-gram Vocabulary Vectorization: Utilizes Scikit-Learn TfidfVectorizer mapping parameterized boundaries to extract both unigram and bigram relationships (ngram_range=(1,2)).

Sublinear Frequency Scaling: Implements logarithmic sublinear scale matching (sublinear_tf=True) to suppress word frequency skewing in dense inputs.

3. Balanced Convex Classifier Optimization
Logistic Regression Engine: Solves multi-class targets using cross-entropy planes optimized for fast execution speeds on resource-constrained micro-architectures.

Boundary Hyperparameter Tuning: Set strict margin separation scaling (C=5.0) combined with automated class weights distribution configurations (class_weight='balanced') to prevent accuracy collapse over generic single-word phrases.

4. Intent State Tracker & Extraction System
Regex Parametric Extractions: Extracts deterministic real-time data nodes—such as specific timeline metrics (times) or region targets (locations)—using isolated pattern matches before cleanup steps.

Context-Turn State Tracking: Maintains a stateful class tracker (SessionContext) between sequential execution turns, passing historical context to drive responsive, contextual state overrides.

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
│   ├── train.py              # 80/20 train/test evaluation validation splitter
│   └── chatbot.py            # TTY interactive CLI chat execution loop & context tracker
│
├── .dockerignore             # Excludes compiled runtime artifacts and cache layers
├── Dockerfile                # Multi-stage lean production Docker environment setup
└── requirements.txt          # Python dependencies explicitly locked for reproducibility
⚙️ Execution and Deployment Guide
Local Native Execution (Development)
Ensure your host machine has a Python 3.10+ runtime, install dependencies, and run:

Bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
python src/train.py
python src/chatbot.py
Sandbox Docker Deployment (Production-Ready)
This stack features a specialized multi-stage Dockerfile to optimize build layer caching and keep runtime images extremely lean.

1. Compile the Image Assets
Bash
docker build -t native-intent-bot:v2 .
During compilation, the builder stage sets an explicit environment layer (ENV PYTHONPATH=/app/src), fires the validation data split pipeline, outputs an industry-standard Classification Report, and commits binary serialized pickle representations straight into image data buffers.

2. Initialize the Interactive Conversational Session
Bash
docker run -it --rm native-intent-bot:v2
📊 Sample I/O Interaction Matrix
Plaintext
--- Day 2: Advanced NLP Intent & Entity Engine ---
Engine Status: ACTIVE. Threshold set to 0.45. Context memory tracking initialized.

You: hey hello there
Bot (Intent: greet | Conf: 0.61 | Entities: {}): Hi there! What can I do for you?

You: can you set a reminder to call my manager at 4:30 pm
Bot (Intent: reminder | Conf: 0.74 | Entities: {'time': '4:30 pm'}): Understood. I have logged an automatic system alert for you exactly at 4:30 pm.

You: check the weather report for Lahore
Bot (Intent: weather | Conf: 0.81 | Entities: {'time': '4:30 pm', 'location': 'Lahore'}): Checking real-time atmospheric updates for Lahore... It is currently sunny and clear.

You: what is the quantum string configuration of black holes
Bot: [Confidence Matrix Low (0.18)] I am not certain about your intent. Could you rephrase?

You: exit
System shutting down. Goodbye.