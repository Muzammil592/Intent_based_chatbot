# ==========================================
# STAGE 1: Build dependencies & download wheels
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Install packages inside a separate user cache space to keep image footprint small
RUN pip install --no-cache-dir --user -r requirements.txt


# ==========================================
# STAGE 2: Final minimal runtime execution layer
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy cached dependencies cleanly from stage 1
COPY --from=builder /root/.local /root/.local
COPY . .

# Wire paths correctly
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src

# Download internal linguistic packages for NLTK text streaming
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Execute advanced training matrix to build pkl models inside the image layer
RUN python src/train.py

# Launch interactive terminal shell chat application upon boot configuration
CMD ["python", "src/chatbot.py"]