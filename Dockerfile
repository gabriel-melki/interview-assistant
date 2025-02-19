# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and poetry
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create virtual environment inside container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY . .

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "uvicorn", "interview_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"] 