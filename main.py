import os
import sys
import traceback
from enum import Enum
from io import StringIO
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


app = FastAPI(title="Code Interpreter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python code to execute")


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        exec(code)
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
        return {"success": True, "output": f"{stdout_output}{stderr_output}"}
    except Exception:
        return {"success": False, "output": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def analyze_error_with_ai(code: str, tb_output: str) -> List[int]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)
    prompt = (
        "Analyze this Python code and traceback, then return only the line number(s) "
        "where the error occurred.\n\n"
        f"CODE:\n{code}\n\n"
        f"TRACEBACK:\n{tb_output}"
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER),
                    )
                },
                required=["error_lines"],
            ),
        ),
    )

    parsed = ErrorAnalysis.model_validate_json(response.text)
    return parsed.error_lines


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(payload: CodeRequest):
    execution = execute_python_code(payload.code)

    if execution["success"]:
        return CodeResponse(error=[], result=execution["output"])

    try:
        error_lines = analyze_error_with_ai(payload.code, execution["output"])
        return CodeResponse(error=error_lines, result=execution["output"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(exc)}")


# ── Sentiment Analysis Endpoint ──────────────────────────────────────────────

import anthropic
import json


class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="The comment to analyze")


class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)


SENTIMENT_TOOL = {
    "name": "sentiment_analysis",
    "description": "Return the sentiment analysis result for sample comment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "Overall sentiment of the comment",
            },
            "rating": {
                "type": "integer",
                "description": "Sentiment intensity: 5=highly positive, 4=positive, 3=neutral, 2=negative, 1=highly negative",
            },
        },
        "required": ["sentiment", "rating"],
    },
}


@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(payload: CommentRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            system=(
                "You are a sentiment analysis engine. "
                "Analyze the user's comment and call the sentiment_analysis tool "
                "with the result. "
                "sentiment must be exactly one of: positive, negative, neutral. "
                "rating must be an integer from 1 to 5 where "
                "5 = highly positive, 4 = positive, 3 = neutral, "
                "2 = negative, 1 = highly negative."
            ),
            messages=[{"role": "user", "content": payload.comment}],
            tools=[SENTIMENT_TOOL],
            tool_choice={"type": "tool", "name": "sentiment_analysis"},
        )

        # Extract the tool use block
        for block in message.content:
            if block.type == "tool_use" and block.name == "sentiment_analysis":
                result = SentimentResponse(**block.input)
                return result

        raise HTTPException(
            status_code=502, detail="Model did not return structured output"
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(exc)}")
