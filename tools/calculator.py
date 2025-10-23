from __future__ import annotations
import math
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class CalculatorInput(BaseModel):
    """Arguments for the calculator tool."""
    expr: str = Field(
        ...,
        description="A Python evaluate-able mathematical expression to evaluate.",
    )
    precision: Optional[int] = Field(
        6,
        description="Number of decimal places to round the result to (default: 6).",
    )


@tool("calculator", args_schema=CalculatorInput, return_direct=False)
def calculator(expr: str, precision: int = 6) -> str:
    """
    Minimal calculator tool.
    Evaluates a simple mathematical expression safely using math functions.
    """
    # Allow basic math functions and constants
    safe_env = {
        "sqrt": math.sqrt,
        "log": math.log,   # natural logarithm
        "ln": math.log,    # alias
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "__builtins__": {},
    }

    try:
        result = eval(expr, safe_env, {})
        if isinstance(result, (int, float)):
            return str(round(result, precision))
        else:
            return "Calculator error: non-numeric result."
    except Exception as e:
        return f"Calculator error: {e}"
