# Stage 1: Build & Cache Dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Install dependencies into a separate wheels directory to optimize memory footprints
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final Runtime Image Execution Layer
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy python user package site maps safely from stage 1
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

# Pre-download NLTK corporate data packets directly into container systems layer
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Execute data assembly and train weights inside container layers
RUN python src/train.py

# Launch interactive TTY terminal line loop upon initialization container run
CMD ["python", "src/chatbot.py"]