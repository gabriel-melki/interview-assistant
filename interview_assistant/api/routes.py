"""
Defining the routes for the API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from interview_assistant.core.schemas import (
    QuestionGenerationRequest,
    GeneratedQuestion,
    GeneratedTip,
    TipGenerationRequest,
)
from interview_assistant.core.services import QuestionService, TipService

app = FastAPI()

@app.post("/generate-questions", response_model=list[GeneratedQuestion])
async def generate_questions_endpoint(input_data: QuestionGenerationRequest):
    """API wrapper to QuestionService.generate_questions()"""
    try:
        return QuestionService().generate_questions(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/generate-tip", response_model=GeneratedTip)
async def generate_tip_endpoint(input_data: TipGenerationRequest):
    """API wrapper to QuestionService.generate_questions()"""
    try:
        return TipService().generate_tip(input_data.question_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/generate-tip-stream")
async def generate_tip_stream_endpoint(input_data: TipGenerationRequest):
    """Streaming API wrapper to TipService.generate_tip_stream()"""
    try:
        service = TipService()
        return StreamingResponse(
            service.generate_tip_stream(input_data.question_id),
            media_type="text/event-stream"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
