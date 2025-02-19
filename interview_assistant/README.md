# HR Assistant Services Documentation

## Overview
The HR Assistant provides two main services:
1. `QuestionService`: Generates unique interview questions
2. `TipService`: Generates unique tips for specific questions

Both services use OpenAI's APIs for content generation and embedding comparison to ensure uniqueness.

## Core Components

### 1. Storage System
- Implemented using `fakeredis` for development/testing
- Two main storage classes:
  - `QuestionStorage`: Manages question persistence
  - `TipStorage`: Manages tip persistence
- Key features:
  - Uses singleton pattern to ensure single instance
  - TTL-based expiration (configurable via `EXPIRATION_SECONDS`)
  - UUID-based identification
  - Redis-style key-value storage with hash maps and sets

### 2. Embedding System
- Uses OpenAI's embedding API to convert text to vector representations
- Key features:
  - Caching via `@lru_cache`
  - Configurable similarity threshold (default: 0.8)
  - Cosine similarity comparison
- Used to ensure content uniqueness by comparing vector representations

### 3. Content Generation
Two specialized generators:
- `QuestionContentGenerator`: Generates interview questions
- `TipContentGenerator`: Generates tips for questions

Both use OpenAI's chat API with:
- Configurable temperature
- Structured prompts
- Response parsing to Pydantic models

## Service Workflows

### QuestionService Workflow
```mermaid
A[Request] --> B[Generate Question]
B --> C{Check Uniqueness}
C -->|Not Unique| B
C -->|Unique| D[Store Question]
D --> E[Return Question]
```

1. **Input**: Receives `QuestionGenerationRequest` with:
   - User ID
   - Job title
   - Question type
   - Skill to test
   - Number of questions (n)

2. **Generation Process**:
   ```python
   for _ in range(request.n):
       candidate = generate_unique_question(request)
       store_question(candidate)
   ```

3. **Uniqueness Check**:
   - Retrieves existing questions for user
   - Generates embeddings for comparison
   - Maximum attempts configurable (default: 5)
   - Uses cosine similarity threshold

### TipService Workflow
```mermaid
A[Question ID] --> B[Generate Tip]
B --> C{Check Uniqueness}
C -->|Not Unique| B
C -->|Unique| D[Store Tip]
D --> E[Return Tip]
```

1. **Input**: Receives question ID (UUID)

2. **Generation Process**:
   - Validates question existence
   - Retrieves existing tips for comparison
   - Generates new tip content
   - Checks uniqueness via embeddings
   - Maximum attempts configurable (default: 5)

3. **Storage**:
   - Tips are linked to questions
   - Maintains question-tip relationships

## Error Handling

Both services implement:
- Maximum attempt limits
- Logging of generation failures
- Validation of inputs
- Storage error handling


## Configuration

Key configuration points:
- OpenAI API models (via environment variables)
- Similarity thresholds
- Maximum generation attempts
- Storage TTL
- Temperature for content generation

## Dependencies

- `openai`: API client for embeddings and chat
- `fakeredis`: Storage implementation
- `pydantic`: Data validation and serialization
- `numpy`: Vector operations for similarity checks
- `uuid`: Unique identifier generation
- `logging`: Error and debug logging

This system is designed to be extensible, with abstract base classes allowing for alternative implementations of:
- Storage backends
- Embedding providers
- Content generators