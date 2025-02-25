# Interview Assistant API

An AI-powered Interview Assistant API that generates interview questions and helpful tips for technical roles in data science, engineering, and analytics.

## Features

- Generate unique interview questions for technical roles
- Get tailored tips for specific interview questions
- Supports multiple question types (exercise, knowledge questions)
- Ensures uniqueness through embedding-based similarity checks
- Caches results for improved performance
- Uses FakeRedis for temporary storage

## Limitations
- Only English is supported for the language
- Only questions for the following job titles are supported: 
  - data analyst
  - data scientist
  - data engineer
- Only questions of type 'exercise' or 'knowledge question' are supported.

## Prerequisites
- Python 3.12+
- Poetry for dependency management
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gabriel-melki/interview-assistant.git
cd interview-assistant
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file in the project root with the following variables:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL="gpt-4o-mini-2024-07-18"
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
OPENAI_CHAT_TEMPERATURE=0.8
EXPIRATION_SECONDS=86400  # 24 hours
```

## Usage

1. Start the FastAPI server:
```bash
poetry run uvicorn interview_assistant.api:app --reload
```

2. The API will be available at `http://127.0.0.1:8000`

Some overall description of the API can be found at `http://127.0.0.1:8000/docs#`
### API Endpoints

#### Generate Questions
```http
POST /generate-questions
```

Request body:
```json
{
    "user_id": "uuid-string",
    "question_type": "exercise",
    "job_title": "data scientist",
    "skill_to_test": "python",
    "n": 3
}
```

#### Generate Tip
```http
POST /generate-tip
```

Request body:
```json
{
    "question_id": "uuid-string"
}
```

Here is an example of request for both question and tip:
```bash
question=$(curl -s -X POST "http://127.0.0.1:8000/generate-questions" \
  -H "Content-Type: application/json" \
  -d '{
        "user_id": "123e4567-e89b-12d3-a456-426614174000",
        "n": 1,
        "question_type": "knowledge question",
        "job_title": "data engineer",
        "skill_to_test": "Short definition of a integer"
      }')
echo $question

sanitized_question=$(echo "$question" | tr -d '\000-\037')
question_id=$(echo "$sanitized_question" | jq -r '.[0].id')

tip=$(curl -X POST "http://127.0.0.1:8000/generate-tip" \
  -H "Content-Type: application/json" \
  -d '{"question_id": "'"$question_id"'"}')
echo $tip
```

## Development

### Project Structure
```
interview_assistant/
├── api.py                 # FastAPI endpoints
├── service.py            # Business logic
├── schemas.py           # Pydantic models
├── storage.py           # Redis storage implementation
├── embedding.py         # Embedding generation and comparison
└── content_generation/  # Content generation implementations
    ├── question_content_generation.py
    └── tip_content_generation.py
```

### Quality Assurance Checklist

#### Before Committing
- [ ] Run all tests:
```bash
poetry run pytest
```

#### Before Deployment
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `schemas.py` (`app_version`)
- [ ] Run integration tests
- [ ] Check API documentation at `/docs`
- [ ] Verify environment variables
- [ ] Test with production settings

### Contributing

1. Create a new branch for your feature
2. Make your changes
3. Run the quality assurance checklist
4. Submit a pull request

## Testing

Run the test suite:
```bash
poetry run pytest
```

## RAG (Retrieval Augmented Generation)

This application includes a RAG mechanism that allows it to generate questions and answers based on existing documents. To use this feature:

1. Place your documents in the `documents` directory (supports .txt files)
2. The system will automatically index these documents using FAISS
3. When generating questions, the system will search for relevant context in these documents

The RAG system uses:
- FAISS for efficient similarity search
- LangChain for document processing and embeddings
- Sentence Transformers for generating embeddings