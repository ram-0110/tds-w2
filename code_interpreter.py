import os
import sys
import traceback
from io import StringIO
from typing import List

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="Code Interpreter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python code to execute")


class CodeResponse(BaseModel):
    error: List[int]
    result: str


# ── Tool: execute Python code ────────────────────────────────────────────────

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


# ── AI error analysis using Claude ───────────────────────────────────────────

ERROR_ANALYSIS_TOOL = {
    "name": "report_error_lines",
    "description": "Report the line numbers where errors occurred in the Python code.",
    "input_schema": {
        "type": "object",
        "properties": {
            "error_lines": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of 1-indexed line numbers where errors occurred",
            }
        },
        "required": ["error_lines"],
    },
}


def analyze_error_with_ai(code: str, tb_output: str) -> List[int]:
    """
    Use Claude with structured output (tool use) to identify error line numbers.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "Analyze this Python code and its error traceback. "
        "Identify the exact line number(s) where the error occurred. "
        "Use 1-based line numbering.\n\n"
        f"CODE:\n{code}\n\n"
        f"TRACEBACK:\n{tb_output}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=(
            "You are a Python error analysis engine. "
            "Analyze the code and traceback, then call the report_error_lines tool "
            "with the exact line numbers where errors occurred. "
            "Use 1-based line numbering matching the original code."
        ),
        messages=[{"role": "user", "content": prompt}],
        tools=[ERROR_ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "report_error_lines"},
    )

    for block in message.content:
        if block.type == "tool_use" and block.name == "report_error_lines":
            return block.input.get("error_lines", [])

    return []


# ── Endpoint ─────────────────────────────────────────────────────────────────

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
        raise HTTPException(status_code=502, detail=f"AI analysis error: {str(exc)}")
